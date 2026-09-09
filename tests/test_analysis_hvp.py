import pytest
import torch

from tiny_llm.analysis import hessian_quadratic, rademacher
from tiny_llm.analysis.core import QuadraticKernel
from tiny_llm.model import Llama


def test_functional_quadratic_and_padding(tiny_config):
    torch.manual_seed(19)
    model = Llama(tiny_config.model, "reference").double()
    x, y = torch.randint(17, (3, 4)), torch.randint(17, (3, 4))
    y[-1, -1] = -100
    direction = rademacher(tuple(model.parameters()), 42, 0)
    expected = hessian_quadratic(model, [(x, y)], 11, direction)
    kernel = QuadraticKernel(model, 4, compile=True, backend="aot_eager")
    assert kernel(x, y, direction).item() / 11 == pytest.approx(expected, rel=1e-9, abs=1e-9)
    # New parameter values and directions must not be captured as constants.
    with torch.no_grad():
        model.norm.weight.add_(0.1)
    direction = rademacher(tuple(model.parameters()), 42, 1)
    x = x.flip(0)
    expected = hessian_quadratic(model, [(x, y)], 11, direction)
    assert kernel(x, y, direction).item() / 11 == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_independent_hvp_batches_preserve_statistics(tiny_config, cache_dir):
    from dataclasses import replace

    from tiny_llm.analysis import AnalysisOptions, EpochData, measure
    from tiny_llm.data import TokenCache

    cfg = tiny_config
    cfg.training.epoch_tokens = 28  # Uneven end: 7 blocks, split as 5 + 2 for curvature.
    model = Llama(cfg.model, "reference").double()
    data = EpochData(TokenCache(cache_dir), cfg, "seen")
    opts = AnalysisOptions(device="cpu", dtype="float64", noise_samples=2, random_samples=2)
    baseline = measure(model, data, opts, torch.device("cpu"))
    sizes = []
    inner = QuadraticKernel(model, 5, compile=True, backend="aot_eager")

    def kernel(x, y, direction):
        sizes.append(len(x))
        return inner(x, y, direction)

    candidate = measure(model, data, replace(opts, hvp_batch_size=5), torch.device("cpu"), kernel)
    assert sizes == [5, 2] * 4
    for name, value in baseline["statistics"].items():
        assert candidate["statistics"][name] == pytest.approx(value, rel=1e-9, abs=1e-9)
    assert [sample["batch"] for sample in baseline["noise"]] == [
        sample["batch"] for sample in candidate["noise"]
    ]
    with pytest.raises(ValueError, match="retain training"):
        list(data.batches(torch.device("cpu"), data.samples[0], batch_size=5))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_cuda_tf32_against_ieee(tiny_config):
    from dataclasses import replace

    from tiny_llm.analysis import AnalysisOptions, analysis_runtime, gradient

    torch.manual_seed(13)
    source = Llama(tiny_config.model, "reference")
    x, y = torch.randint(17, (3, 4), device="cuda"), torch.randint(17, (3, 4), device="cuda")
    options = AnalysisOptions(device="cuda")
    with analysis_runtime(tiny_config, replace(options, tf32=False)) as (device, _):
        model = Llama(tiny_config.model, "reference").to(device)
        model.load_state_dict(source.state_dict())
        directions = [rademacher(tuple(model.parameters()), 42, 0)]
        g = gradient(model, [(x, y)], y.numel())
        local = gradient(model, [(x[:1], y[:1])], 4)
        directions.append(tuple(a - b for a, b in zip(local, g, strict=True)))
        directions.append(tuple(torch.zeros_like(p) for p in model.parameters()))
        reference = [hessian_quadratic(model, [(x, y)], 12, v) for v in directions]
    with analysis_runtime(tiny_config, options):
        kernel = QuadraticKernel(model, 4)
        for expected, direction in zip(reference, directions, strict=True):
            assert kernel(x, y, direction).item() / 12 == pytest.approx(
                expected, rel=0.03, abs=0.001
            )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_amp_functional_hvp(tiny_config):
    device = "cuda"
    from tiny_llm.analysis import AnalysisOptions, AutocastModel, analysis_runtime, gradient

    torch.manual_seed(17)
    base = Llama(tiny_config.model, "reference").to(device)
    amp = AutocastModel(base, True)
    fp32 = AutocastModel(base, False)
    x, y = torch.randint(17, (3, 4), device=device), torch.randint(17, (3, 4), device=device)
    with analysis_runtime(tiny_config, AnalysisOptions(device=device)):
        assert amp(x).dtype == torch.bfloat16
        assert fp32(x).dtype == torch.float32
        assert all(p.dtype == torch.float32 for p in amp.parameters())
        gradients = gradient(amp, [(x, y)], 12)
        assert all(g.dtype == torch.float32 for g in gradients)
        assert not torch.is_autocast_enabled(device)
        direction = rademacher(tuple(amp.parameters()), 42, 0)
        kernel = QuadraticKernel(
            amp,
            4,
            backend="inductor",
        )
        for _ in range(2):
            reference = hessian_quadratic(amp, [(x, y)], 12, direction)
            actual = kernel(x, y, direction).item() / 12
            assert actual == pytest.approx(reference, rel=0.04, abs=0.002)
            with torch.no_grad():
                base.norm.weight.add_(0.1)
