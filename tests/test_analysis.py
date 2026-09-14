import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from helpers import set_budget
from safetensors.torch import load_file, save_file
from torch.func import functional_call

from tiny_llm.analysis import (
    AnalysisOptions,
    EpochData,
    analysis_runtime,
    analyze,
    checkpoint_paths,
    gradient,
    hessian_quadratic,
    load_checkpoint,
    rademacher,
    summarize,
    tensor_dot,
)
from tiny_llm.cli import build_parser, main
from tiny_llm.config import Config, DecentralizedConfig, ModelConfig, save_config
from tiny_llm.data import BufferedTokenLoader, TokenCache, training_boundaries
from tiny_llm.model import Block, Llama, token_losses
from tiny_llm.packed import PackedLlama
from tiny_llm.runtime import precision_backend


def test_explicit_hessian_and_weighted_microbatches():
    from types import SimpleNamespace

    from tiny_llm.analysis.core import consensus_directions, consensus_statistics, measure_consensus

    torch.manual_seed(7)
    model = Llama(
        ModelConfig(vocab_size=3, layers=1, width=2, heads=1, ffn_width=2, context_length=2),
        "reference",
    ).double()
    x = torch.tensor([[1, 2], [2, 1], [1, 0]])
    y = torch.tensor([[2, 0], [1, 0], [0, -100]])
    params = tuple(model.parameters())
    names = [name for name, _ in model.named_parameters()]
    flat = torch.cat([p.detach().flatten() for p in params]).requires_grad_()

    def loss(vector):
        split = vector.split([p.numel() for p in params])
        weights = {n: v.view_as(p) for n, v, p in zip(names, split, params, strict=True)}
        return token_losses(functional_call(model, weights, (x,)), y).sum() / 5

    explicit = torch.autograd.functional.hessian(loss, flat, vectorize=True)
    assert isinstance(explicit, torch.Tensor)
    batches = [(x[:2], y[:2]), (x[2:], y[2:])]
    # Four workers with unequal deviations and one exactly at consensus.
    workers = [
        {name: p.detach() + offset * 0.001 for name, p in model.named_parameters()}
        for offset in (0, 1, 2, -3)
    ]
    data = SimpleNamespace(case="seen", tokens=5, batches=lambda *a, **kw: batches)
    consensus = measure_consensus(model, data, AnalysisOptions(), torch.device("cpu"), workers)
    expected_quadratics = []
    for row, direction in zip(consensus, consensus_directions(model, workers), strict=True):
        vector = torch.cat([p.flatten() for p in direction])
        expected = (vector @ explicit @ vector).item()
        expected_quadratics.append(expected)
        assert row["quadratic"] == pytest.approx(expected, rel=1e-9, abs=1e-12)
        assert row["norm_squared"] == pytest.approx(vector.square().sum().item())
    assert consensus[0]["quadratic"] == 0 and consensus[0]["normalized_alignment"] is None
    pooled = consensus_statistics(consensus)
    assert pooled["consensus_alignment"] == pytest.approx(sum(expected_quadratics) / 4)
    assert pooled["normalized_consensus_alignment"] == pytest.approx(
        sum(expected_quadratics) / sum(row["norm_squared"] for row in consensus)
    )
    assert pooled["average_consensus_norm"] == pytest.approx(
        sum(row["norm"] for row in consensus) / 4
    )
    full = gradient(model, [(x, y)], 5)
    micro = gradient(model, batches, 5)
    for a, b in zip(full, micro, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
    expected_gradient = torch.autograd.grad(loss(flat), flat)[0]
    torch.testing.assert_close(torch.cat([p.flatten() for p in full]), expected_gradient)
    noise_rows, random_rows = [], []
    for index, (bx, by) in enumerate(batches):
        g = gradient(model, [(bx, by)], int((by != -100).sum()))
        v = tuple(a - b for a, b in zip(g, full, strict=True))
        vector = torch.cat([p.flatten() for p in v])
        actual = hessian_quadratic(model, batches, 5, v)
        assert actual == pytest.approx((vector @ explicit @ vector).item(), rel=1e-9, abs=1e-12)
        noise_rows.append(
            dict(
                quadratic=actual,
                noise_norm_squared=tensor_dot(v, v).item(),
                gradient_norm=math.sqrt(tensor_dot(g, g).item()),
            )
        )
        s = rademacher(params, 42, index)
        assert all(set(p.unique().tolist()) <= {-1.0, 1.0} for p in s)
        assert tensor_dot(s, s).item() == model.parameter_count == flat.numel()
        vector = torch.cat([p.flatten() for p in s])
        q = hessian_quadratic(model, batches, 5, s)
        assert q == pytest.approx((vector @ explicit @ vector).item(), rel=1e-9, abs=1e-10)
        random_rows.append(dict(quadratic=q))
    stats = summarize(
        noise_rows, random_rows, model.parameter_count, math.sqrt(tensor_dot(full, full).item())
    )
    assert stats["average_batch_gradient_norm"] > stats["mean_gradient_norm"]
    assert stats["average_noise_norm"] == pytest.approx(
        sum(math.sqrt(row["noise_norm_squared"]) for row in noise_rows) / 2
    )
    assert stats["normalized_random_alignment"] == pytest.approx(
        sum(row["quadratic"] for row in random_rows) / (2 * flat.numel())
    )
    with pytest.raises(ValueError, match="token count"):
        gradient(model, batches, 9)


def test_signed_statistics_and_zero_noise():
    from tiny_llm.analysis.core import consensus_statistics

    stats = consensus_statistics(
        [
            dict(quadratic=-2, norm_squared=1, norm=1),
            dict(quadratic=3, norm_squared=9, norm=3),
        ]
    )
    assert stats["consensus_alignment"] == 0.5
    assert stats["normalized_consensus_alignment"] == 0.1
    assert stats["average_consensus_norm"] == 2
    assert (
        consensus_statistics([dict(quadratic=0, norm_squared=0, norm=0)])[
            "normalized_consensus_alignment"
        ]
        is None
    )
    rows = [
        dict(quadratic=-2, noise_norm_squared=1, gradient_norm=2),
        dict(quadratic=3, noise_norm_squared=9, gradient_norm=6),
    ]
    stats = summarize(rows, [dict(quadratic=-8)], 4, 3)
    assert stats["noise_alignment"] == 0.5
    assert stats["normalized_noise_alignment"] == 0.1
    assert stats["average_noise_norm"] == 2
    assert stats["average_batch_gradient_norm"] == 4
    assert stats["normalized_random_alignment"] == -2
    zero = summarize(
        [dict(quadratic=0, noise_norm_squared=0, gradient_norm=2)], [dict(quadratic=1)], 1, 2
    )
    assert zero["normalized_noise_alignment"] is None
    assert zero["average_noise_norm"] == 0


def test_epoch_replay_unseen_and_microbatch(tiny_config: Config, cache_dir: Path):
    cfg = tiny_config
    cfg.data.buffer_size_mib = 24 / 2**20  # Force several shuffled ranges.
    cfg.training.batch_tokens = 28
    set_budget(cfg, 56, 28)  # Seven blocks; uneven microbatches and curvature padding.
    cache = TokenCache(cache_dir)
    seen, unseen = [EpochData(cache, cfg, case) for case in ("seen", "unseen")]
    with BufferedTokenLoader(
        cache,
        "train",
        4,
        training_boundaries(cfg, cfg.model.parameter_count)[-1],
        cfg.data.buffer_size_mib,
        seed=cfg.runtime.seed,
    ) as loader:
        expected = loader.next_batch(seen.blocks, torch.device("cpu"))
    actual = list(seen.batches(torch.device("cpu")))
    assert [len(x) for x, _ in actual] == [2, 2, 2, 1]
    for index in (0, 1):
        torch.testing.assert_close(torch.cat([batch[index] for batch in actual]), expected[index])
    assert [sample.local_blocks for sample in seen.samples] == [7]
    assert unseen.start == seen.training_blocks
    assert unseen.start >= seen.blocks
    # Verify actual cache reads start after the entire physical training prefix.
    reads = []
    original = cache.read_into

    def record(split, start, stop, destination):
        reads.append((start, stop))
        original(split, start, stop, destination)

    cache.read_into = record
    first = list(unseen.batches(torch.device("cpu")))
    second = list(unseen.batches(torch.device("cpu")))
    assert all(start >= unseen.start * 4 for start, _ in reads)
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    set_budget(cfg, 140, 28)
    with pytest.raises(ValueError, match="cache too small"):
        EpochData(cache, cfg, "unseen")


def test_packed_batches_and_average_loading(tiny_config: Config, cache_dir: Path):
    from tiny_llm.analysis.core import result_key

    cfg = tiny_config
    cfg.decentralized = DecentralizedConfig(num_models=2)
    cfg.training.micro_batch_size = 3
    cfg.training.batch_tokens = 24
    set_budget(cfg, 48, 24)
    data = EpochData(TokenCache(cache_dir), cfg, "seen")
    assert [(s.cursor, s.blocks, s.worker) for s in data.samples] == [
        (0, 6, 0),
        (0, 6, 1),
    ]
    with data.loader() as loader:
        full = loader.next_batch(data.blocks, torch.device("cpu"))
    for sample in data.samples:
        actual = list(data.batches(torch.device("cpu"), sample))
        for index in (0, 1):
            expected = full[index][
                sample.cursor + sample.worker : sample.cursor + sample.blocks : sample.workers
            ]
            torch.testing.assert_close(torch.cat([b[index] for b in actual]), expected)
    packed = PackedLlama(cfg.model, 2)
    path = cfg.runtime.output_dir / "final.pt"
    path.parent.mkdir()
    torch.save(
        dict(
            config=cfg.model_dump(mode="json"),
            model=packed.packed_state_dict(),
            cursor=6,
            step=1,
            completed_epochs=1,
        ),
        path,
    )
    model, info = load_checkpoint(cfg, path)
    for entry in packed.layout:
        torch.testing.assert_close(
            model.get_parameter(entry.name), packed.get_parameter(entry.name).mean(0)
        )
    assert info["tokens"] == 24
    assert all(b.attention.backend == "reference" for b in cast(Iterable[Block], model.blocks))
    assert len(info["workers"]) == 2
    root = path.parent / "epoch-001.safetensors"
    save_file(model.state_dict(), root)
    for index in range(2):
        directory = path.parent / f"node-{index:03d}"
        directory.mkdir()
        save_file(packed.local_state_dict(index), directory / root.name)
    assert result_key(load_checkpoint(cfg, root)[1]) == result_key(info)
    # A changed local state must not alias another checkpoint with the same mean.
    for index, delta in enumerate((-0.25, 0.25)):
        state = packed.local_state_dict(index)
        state["norm.weight"].add_(delta)
        save_file(state, path.parent / f"node-{index:03d}" / root.name)
    _, changed = load_checkpoint(cfg, root)
    assert changed["weights_hash"] == info["weights_hash"]
    assert result_key(changed) != result_key(info)
    save_config(cfg, path.parent / "resolved.yaml")
    with pytest.raises(ValueError, match="submission assignments"):
        analyze(
            path.parent,
            [root.name],
            path.parent / "stale",
            AnalysisOptions(device="cuda"),
            expected_checkpoints=[dict(info, name=root.name, path=str(root))],
        )
    assert not (path.parent / "stale").exists()
    worker = path.parent / "node-001" / root.name
    valid = load_file(str(worker))
    for damage in ("shape", "name", "nonfinite", "mean"):
        state = {k: v.clone() for k, v in valid.items()}
        if damage == "shape":
            state["norm.weight"] = state["norm.weight"][:1]
        elif damage == "name":
            state.pop("norm.weight")
        elif damage == "nonfinite":
            state["norm.weight"][0] = float("nan")
        else:
            state["norm.weight"].add_(1)
        save_file(state, worker)
        with pytest.raises(ValueError, match="worker"):
            load_checkpoint(cfg, root)
    worker.unlink()
    with pytest.raises(ValueError, match="missing worker"):
        analyze(path.parent, ["all"], path.parent / "missing", AnalysisOptions(device="cuda"))
    assert not (path.parent / "missing").exists()
    assert "workers" not in load_checkpoint(cfg, root, consensus=False)[1]


def test_runtime_restoration(tiny_config: Config):
    before = (
        precision_backend.fp32_precision,
        torch.backends.cuda.matmul.fp32_precision,
        torch.get_num_threads(),
        torch.random.get_rng_state().clone(),
    )
    tiny_config.runtime.amp = True
    tiny_config.runtime.sdpa_backend = "flash"
    with pytest.raises(RuntimeError, match="test failure"):
        with analysis_runtime(tiny_config, AnalysisOptions(device="cpu")) as (_, policy):
            assert policy["attention_backend"] == "reference"
            assert policy["matmul_precision"] == "ieee"
            assert not policy["tf32_effective"]
            raise RuntimeError("test failure")
    assert (
        precision_backend.fp32_precision,
        torch.backends.cuda.matmul.fp32_precision,
        torch.get_num_threads(),
    ) == before[:3]
    assert torch.equal(torch.random.get_rng_state(), before[3])


def test_analysis_cli_resume_and_aliases(
    tiny_config: Config, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    import tiny_llm.analysis.core as module

    cfg = tiny_config
    run = cfg.runtime.output_dir
    run.mkdir()
    save_config(cfg, run / "resolved.yaml")
    model = Llama(cfg.model)
    save_file(model.state_dict(), str(run / "epoch-001.safetensors"))
    with torch.no_grad():
        model.norm.weight.add_(0.01)
    save_file(model.state_dict(), str(run / "epoch-002.safetensors"))
    state = dict(
        config=cfg.model_dump(mode="json"),
        model=model.state_dict(),
        cursor=16,
        step=4,
        completed_epochs=2,
    )
    for name in ("alias.pt", "final.pt"):
        torch.save(state, run / name)
    monkeypatch.setattr(
        module,
        "environment",
        lambda: dict(source_hash="fixture", versions={"torch": torch.__version__}, gpu=None),
    )
    output = run / "analysis"
    argv = [
        "analyze",
        "--run",
        str(run),
        "--checkpoints",
        "all",
        "--output",
        str(output),
        "--device",
        "cpu",
        "--no-plots",
        "--noise-samples",
        "1",
        "--random-samples",
        "1",
    ]
    main(argv)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["micro_batch_size"] == cfg.training.micro_batch_size
    assert len(manifest["checkpoints"]) == 4
    assert len({row["weights_hash"] for row in manifest["checkpoints"]}) == 2
    assert not list(output.glob("*.png"))
    monkeypatch.setattr(
        module, "measure", lambda *args: pytest.fail("recomputed completed results")
    )
    main(argv)
    with pytest.raises(ValueError, match="differ"):
        analyze(run, ["all"], output, AnalysisOptions(device="cpu", random_samples=2))
    assert len(checkpoint_paths(run, ["all"])) == 4
    with pytest.raises(ValueError, match="root checkpoint"):
        checkpoint_paths(run, ["node-000/epoch-001.safetensors"])


def test_sampling_reproducibility_and_training_microbatch(tiny_config: Config, cache_dir: Path):
    from tiny_llm.analysis import measure

    cfg = tiny_config
    data = EpochData(TokenCache(cache_dir), cfg, "seen")
    model = Llama(cfg.model, "reference").double()
    sizes = []
    handle = model.register_forward_pre_hook(lambda module, args: sizes.append(len(args[0])))
    options = AnalysisOptions(noise_samples=1, random_samples=1, device="cpu", dtype="float64")
    first = measure(model, data, options, torch.device("cpu"))
    second = measure(model, data, options, torch.device("cpu"))
    handle.remove()
    assert first["statistics"] == second["statistics"]
    assert first["noise"][0]["batch_index"] == second["noise"][0]["batch_index"]
    assert set(sizes) == {cfg.training.micro_batch_size}


def test_cli_defaults_and_invalid_options():
    args = build_parser().parse_args(
        ["analyze", "--run", "run", "--checkpoints", "all", "--output", "out"]
    )
    assert args.tf32 and args.noise_samples == "32" and args.random_samples == 32
    assert args.consensus
    for command in ("analyze", "submit-analysis"):
        argv = [command, "--run", "run", "--output", "out", "--no-consensus"]
        if command == "submit-analysis":
            argv.extend(["--jobs", "2"])
        assert not build_parser().parse_args(argv).consensus
    assert not hasattr(args, "micro_batch_size")
    for kwargs in (
        dict(noise_samples=0),
        dict(random_samples=0),
        dict(seed=-1),
        dict(amp=True, dtype="float64"),
    ):
        with pytest.raises(ValueError):
            AnalysisOptions(**cast(dict[str, Any], kwargs))
