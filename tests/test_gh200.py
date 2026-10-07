"""Canonical storage and numerical contracts of first-order GH200 training."""

import copy
import json
from typing import Any

import pytest
import torch

from tiny_llm.config import ModelConfig
from tiny_llm.gh200.model import GH200Model
from tiny_llm.packed import PackedLlama, local_mean_losses


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_fused_views_preserve_canonical_layout_initialization_and_rng(workers):
    cfg = ModelConfig(vocab_size=17, layers=1, width=8, heads=2, ffn_width=16, context_length=4)
    torch.manual_seed(71)
    reference = PackedLlama(cfg, workers)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(71)
    fused = GH200Model(cfg, workers, torch.device("cpu"))
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert fused.layout_metadata() == reference.layout_metadata()
    assert fused.parameter_count == sum(p.numel() for p in fused.parameters())
    torch.testing.assert_close(fused.parameter_storage, reference.parameter_storage, rtol=0, atol=0)
    for worker in range(workers):
        for key, value in fused.local_state_dict(worker).items():
            torch.testing.assert_close(value, reference.local_state_dict(worker)[key], rtol=0, atol=0)
    state = copy.deepcopy(fused.packed_state_dict())
    pointers = [p.data_ptr() for p in fused.parameters()]
    fused.parameter_storage.zero_()
    fused.load_packed_state_dict(state)
    assert pointers == [p.data_ptr() for p in fused.parameters()]
    with pytest.raises(RuntimeError, match="arena"):
        fused.to("cpu")


@pytest.mark.cuda
@pytest.mark.parametrize("width,heads,ffn", [(320, 5, 896), (512, 8, 1408), (640, 10, 1792)])
@pytest.mark.parametrize("workers", [1, 4])
def test_gh200_model_and_gradients_against_compiled_native(width, heads, ffn, workers):
    from tiny_llm.gh200.loss import local_cross_entropy

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("GH200 required")
    cfg = ModelConfig(layers=2, width=width, heads=heads, ffn_width=ffn, context_length=512)
    torch.manual_seed(71)
    native = PackedLlama(cfg, workers).cuda()
    fused = GH200Model(cfg, workers, torch.device("cuda"))
    fused.load_packed_state_dict(native.packed_state_dict())
    x = torch.randint(0, cfg.vocab_size, (workers, 4, 512), device="cuda")
    y = torch.randint(0, cfg.vocab_size, x.shape, device="cuda")

    def native_loss(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return local_mean_losses(native(x), y)

    def fused_loss(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return local_cross_entropy(fused(x), y)

    a = torch.compile(native_loss, options={"triton.cudagraphs": False})(x, y)
    b = torch.compile(fused_loss, options={"triton.cudagraphs": False})(x, y)
    a.sum().backward()
    b.sum().backward()
    torch.testing.assert_close(a, b, atol=0.002, rtol=0.002)
    gradients = torch.zeros_like(fused.parameter_storage)
    for (_, p), entry in zip(fused.named_parameters(), fused.fused_layout, strict=True):
        assert p.grad is not None
        gradients[:, entry.start:entry.start + entry.numel].copy_(p.grad.flatten(1))
    for worker in range(workers):
        error, scale = 0.0, 0.0
        for entry in native.layout:
            grad = native.get_parameter(entry.name).grad
            assert grad is not None
            expected = grad[worker]
            actual = fused.parameter_view(gradients, entry)[worker]
            error += (actual - expected).double().square().sum().item()
            scale += expected.double().square().sum().item()
        assert (error / scale) ** 0.5 < 0.05


@pytest.mark.cuda
def test_segmented_loss_ignored_targets_and_distinct_workers():
    from tiny_llm.gh200.loss import local_cross_entropy

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("GH200 required")
    x = torch.randn(4, 2, 8, 32, device="cuda", requires_grad=True)
    y = torch.randint(0, 32, (4, 2, 8), device="cuda")
    y[0].fill_(-100)
    y[1, 0].fill_(-100)
    a = local_cross_entropy(x, y)
    b = local_mean_losses(x, y)
    torch.testing.assert_close(a, b)
    ga, = torch.autograd.grad(a.sum(), x, retain_graph=True)
    gb, = torch.autograd.grad(b.sum(), x)
    torch.testing.assert_close(ga, gb, atol=1e-7, rtol=1e-5)


def require_gh200():
    if (not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0)
            or "GH200" not in torch.cuda.get_device_name()):
        pytest.skip("GH200 required")


@pytest.mark.cuda
def test_cross_entropy_addresses_beyond_signed_32_bit_elements():
    import math

    from tiny_llm.gh200.loss import local_cross_entropy

    require_gh200()
    rows, vocab = 65537, 32768
    x = torch.zeros((1, 1, rows, vocab), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = torch.zeros(x.shape[:-1], device="cuda", dtype=torch.int64)
    loss = local_cross_entropy(x, y)
    assert loss.item() == pytest.approx(math.log(vocab), rel=1e-6)
    loss.sum().backward()
    assert x.grad is not None
    expected = torch.full((vocab,), 1. / (rows * vocab), device="cuda", dtype=torch.bfloat16)
    expected[0] = (1. / vocab - 1.) / rows
    for row in (0, rows // 2, rows - 1):
        torch.testing.assert_close(x.grad[0, 0, row], expected, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("scheme", ["awc", "atc"])
@pytest.mark.parametrize("topology", ["complete", "one_peer_ring", "one_peer_exponential"])
@pytest.mark.parametrize("betas,clip", [((0.9, 0.99), 1.), ((0., 0.), None),
                                       ((0.98, 0.999999), 1.)])
@pytest.mark.parametrize("invalid", ["gradient", "loss"])
def test_arena_optimizer_matches_native_with_supplied_gradients(scheme, topology, betas, clip, invalid):
    from tiny_llm.config import Config, DecentralizedConfig
    from tiny_llm.gh200.optimizer import ArenaAdamW
    from tiny_llm.packed_optimizer import PackedAdamW

    require_gh200()
    cfg = Config()
    cfg.model = ModelConfig(vocab_size=17, layers=1, width=8, heads=2, ffn_width=16, context_length=4)
    cfg.training.micro_batch_size = 2
    cfg.training.batch_tokens = 32
    cfg.decentralized = DecentralizedConfig(num_models=4, topology=topology, scheme=scheme)
    cfg.optimizer.beta1, cfg.optimizer.beta2 = betas
    cfg.optimizer.grad_clip = clip
    torch.manual_seed(12)
    native = PackedLlama(cfg.model, 4).cuda()
    fused = GH200Model(cfg.model, 4, torch.device("cuda"))
    fused.load_packed_state_dict(native.packed_state_dict())
    native_opt, fused_opt = PackedAdamW(native, cfg), ArenaAdamW(fused, cfg)
    for step, gamma in enumerate([0., 0.25, 1., 0.7, 1.]):
        gradients = torch.randn_like(native.parameter_storage)
        for entry in native.layout:
            native.get_parameter(entry.name).grad = native.parameter_view(gradients, entry).clone()
        for entry, p in zip(fused.fused_layout, fused.parameters(), strict=True):
            p.grad = gradients[:, entry.start:entry.start + entry.numel].reshape(p.shape).contiguous()
        norms = native_opt.clip_grad_norm_(clip)
        if scheme == "atc":
            native_opt.step()
        native.mix_(topology, step, gamma)
        if scheme == "awc":
            native_opt.step()
        fused_opt.gamma.fill_(gamma)
        fused_opt.step(torch.ones(4, device="cuda"))
        fused_opt.inspect(step + 1)
        torch.testing.assert_close(norms, fused_opt.norms, rtol=1e-6, atol=1e-8)
        for a, b in ((native.parameter_storage, fused.parameter_storage),
                     (native_opt.first_moment_storage, fused_opt.first_moment_storage),
                     (native_opt.second_moment_storage, fused_opt.second_moment_storage)):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-8)
    saved = copy.deepcopy(fused_opt.state_dict())
    pointers = [value.data_ptr() for value in (fused.parameter_storage,
                fused_opt.first_moment_storage, fused_opt.second_moment_storage)]
    fused_opt.load_state_dict(native_opt.state_dict())
    assert pointers == [value.data_ptr() for value in (fused.parameter_storage,
                        fused_opt.first_moment_storage, fused_opt.second_moment_storage)]
    for name in ("first_moment_storage", "second_moment_storage"):
        torch.testing.assert_close(saved[name], getattr(fused_opt, name), rtol=1e-6, atol=1e-8)
    before = [value.clone() for value in (fused.parameter_storage, fused_opt.first_moment_storage,
                                        fused_opt.second_moment_storage, fused_opt.completed,
                                        fused_opt.clip_counts, fused_opt.loss_sum, fused_opt.losses,
                                        fused_opt.norms)]
    grad = next(fused.parameters()).grad
    assert grad is not None
    losses = torch.ones(4, device="cuda")
    if invalid == "gradient":
        grad[2].fill_(float("inf"))
    else:
        losses[2] = float("nan")
    fused_opt.step(losses)
    with pytest.raises(FloatingPointError, match="5 successful"):
        fused_opt.inspect()
    grad.zero_()
    fused_opt.step(torch.ones(4, device="cuda"))
    for expected, actual in zip(before, (fused.parameter_storage, fused_opt.first_moment_storage,
                                       fused_opt.second_moment_storage, fused_opt.completed,
                                       fused_opt.clip_counts, fused_opt.loss_sum, fused_opt.losses,
                                       fused_opt.norms), strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("workers,blocks,micro", [(1, 4, 2), (1, 5, 2), (4, 8, 2)])
def test_complete_graph_capture_restore_and_resume(workers, blocks, micro):
    from tiny_llm.checkpoints import compact_cpu
    from tiny_llm.config import Config, DecentralizedConfig
    from tiny_llm.gh200.engine import GH200Update

    require_gh200()
    cfg = Config()
    cfg.model = ModelConfig(vocab_size=32, layers=1, width=320, heads=5,
                            ffn_width=896, context_length=16)
    cfg.training.micro_batch_size = micro
    cfg.training.batch_tokens = blocks * 16
    cfg.decentralized = DecentralizedConfig(num_models=workers) if workers > 1 else None
    engine = GH200Update(cfg, torch.device("cuda"))
    engine.model.parameter_storage.add_(torch.randn_like(engine.model.parameter_storage) * 0.001)
    engine.optimizer.first_moment_storage.fill_(0.003)
    engine.optimizer.second_moment_storage.fill_(0.004)
    engine.optimizer.completed.fill_(7)
    before = compact_cpu(engine._persistent())
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    pointers = {name: value.data_ptr() for name, value in engine._persistent().items()}
    engine.prepare()
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    for name, value in engine._persistent().items():
        torch.testing.assert_close(value.cpu(), before[name], rtol=0, atol=0)
        assert value.data_ptr() == pointers[name]
    x = torch.randint(0, 32, (blocks, 16), device="cuda")
    y = torch.randint(0, 32, x.shape, device="cuda")

    def update():
        cursor = 0

        def next_batch(count, device):
            nonlocal cursor
            result = x[cursor:cursor + count], y[cursor:cursor + count]
            cursor += count
            return result

        engine.execute(next_batch, 0.001, 0.3)
        assert cursor == blocks

    for _ in range(5):
        update()
    engine.inspect(12)
    saved = compact_cpu(engine.canonical_state())
    update()
    expected = compact_cpu(engine.canonical_state())
    engine.load_canonical_state(saved)
    update()
    engine.inspect(13)
    actual = compact_cpu(engine.canonical_state())

    def compare(a, b):
        if isinstance(a, torch.Tensor):
            # CUDA embedding backward uses atomic sums. Retain the existing
            # strict FP32 optimizer/resume tolerances; storage restoration above is exact.
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-8)
        elif isinstance(a, dict):
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            for aa, bb in zip(a, b, strict=True):
                compare(aa, bb)
        else:
            assert a == b

    compare(expected, actual)
    assert pointers == {name: value.data_ptr() for name, value in engine._persistent().items()}


@pytest.mark.cuda
@pytest.mark.parametrize("workers", [1, 4, 8])
def test_production_training_native_checkpoint_transition_and_gh200_resume(
    tiny_config, cache_dir, tmp_path, workers,
):
    from tiny_llm.checkpoints import load_training_checkpoint
    from tiny_llm.config import DecentralizedConfig, TrainingConfig
    from tiny_llm.train import train

    require_gh200()
    cfg = tiny_config
    cfg.model = ModelConfig(vocab_size=17, layers=1, width=320, heads=5,
                            ffn_width=896, context_length=4)
    cfg.training = TrainingConfig(tokens_per_parameters=0.0001,
                                  epoch_tokens_per_parameters=0.00005,
                                  batch_tokens=max(4, workers) * 4,
                                  micro_batch_size=1 if workers > 1 else 3,
                                  checkpoint_policy="interval", log_every=2)
    if workers > 1:
        cfg.decentralized = DecentralizedConfig(num_models=workers)
    cfg.runtime.device = "cuda:0"
    cfg.runtime.amp = cfg.runtime.compile = True
    cfg.runtime.deterministic = False
    cfg.runtime.output_dir = tmp_path / "native"
    native = train(cfg)
    source = cfg.runtime.output_dir / "epoch-001.pt"
    cfg.runtime.output_dir = tmp_path / "gh200-import"
    cfg.runtime.training_backend = "gh200"
    candidate = train(cfg, source)
    assert candidate["tokens"] == native["tokens"]
    meta = json.loads((cfg.runtime.output_dir / "environment-resume.json").read_text())
    assert meta["resume_backend_transition"]["source"] == "native"
    assert meta["resume_backend_transition"]["target"] == "gh200"
    state = load_training_checkpoint(cfg.runtime.output_dir / "final.pt")
    assert state["step"] == candidate["step"]
    # The newly written packed worker format must remain canonically readable.
    epoch = load_training_checkpoint(cfg.runtime.output_dir / "epoch-002.pt")
    assert epoch["cursor"] == state["cursor"]
    cfg.runtime.output_dir = tmp_path / "gh200-full"
    full = train(cfg)
    source = cfg.runtime.output_dir / "epoch-001.pt"
    cfg.runtime.output_dir = tmp_path / "gh200-resume"
    resumed = train(cfg, source)
    assert full["tokens"] == resumed["tokens"]
    assert full["step"] == resumed["step"]
    assert full["final_validation"]["loss"] == pytest.approx(
        resumed["final_validation"]["loss"], rel=0.002, abs=0.002,
    )


@pytest.mark.cuda
def test_captured_nonfinite_worker_freezes_all_committed_state(monkeypatch):
    from tiny_llm.checkpoints import compact_cpu
    from tiny_llm.config import Config, DecentralizedConfig
    from tiny_llm.gh200.engine import GH200Update

    require_gh200()
    cfg = Config()
    cfg.model = ModelConfig(vocab_size=32, layers=1, width=320, heads=5,
                            ffn_width=896, context_length=16)
    cfg.training.micro_batch_size, cfg.training.batch_tokens = 2, 128
    cfg.decentralized = DecentralizedConfig(num_models=4, scheme="atc", topology="one_peer_ring")
    engine = GH200Update(cfg, torch.device("cuda"))
    injection = torch.zeros(4, device="cuda")
    original = engine.optimizer.step
    monkeypatch.setattr(engine.optimizer, "step", lambda means: original(means + injection))
    engine.prepare()
    x = torch.randint(32, (8, 16), device="cuda")
    y = torch.randint(32, x.shape, device="cuda")
    engine.execute(lambda count, device: (x, y), 0.001, 0.5)
    engine.inspect(1)
    before = compact_cpu(engine._persistent())
    injection[2] = float("nan")
    for _ in range(4):
        engine.execute(lambda count, device: (x, y), 0.001, 0.5)
    with pytest.raises(FloatingPointError, match="1 successful"):
        engine.inspect()
    for name, value in engine._persistent().items():
        if name != "finite":
            torch.testing.assert_close(value.cpu(), before[name], rtol=0, atol=0)


@pytest.mark.cuda
def test_training_reports_only_successful_updates_after_delayed_failure(
    tiny_config, cache_dir, monkeypatch,
):
    from tiny_llm.config import DecentralizedConfig, TrainingConfig
    from tiny_llm.gh200.engine import GH200Update
    from tiny_llm.train import train

    require_gh200()
    cfg = tiny_config
    cfg.model = ModelConfig(vocab_size=17, layers=1, width=320, heads=5,
                            ffn_width=896, context_length=4)
    cfg.training = TrainingConfig(tokens_per_parameters=0.0001,
                                  epoch_tokens_per_parameters=0.0001,
                                  batch_tokens=16, micro_batch_size=1, log_every=4)
    cfg.decentralized = DecentralizedConfig(num_models=4)
    cfg.runtime.device, cfg.runtime.training_backend = "cuda:0", "gh200"
    cfg.runtime.amp = cfg.runtime.compile = True
    original_init, original_execute = GH200Update.__init__, GH200Update.execute
    calls = []

    def initialize(self, config, device):
        original_init(self, config, device)
        self.test_injection = torch.zeros(4, device=device)
        original_step = self.optimizer.step
        self.optimizer.step = lambda means: original_step(means + self.test_injection)

    def execute(self, next_batch, lr, gamma=1.):
        if calls:
            self.test_injection[2] = float("nan")
        calls.append(1)
        original_execute(self, next_batch, lr, gamma)

    monkeypatch.setattr(GH200Update, "__init__", initialize)
    monkeypatch.setattr(GH200Update, "execute", execute)
    with pytest.raises(FloatingPointError, match="1 successful"):
        train(cfg)
    assert len(calls) == cfg.training.log_every
    status = json.loads((cfg.runtime.output_dir / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["step"] == 1 and status["tokens"] == 16
    assert not (cfg.runtime.output_dir / "final.pt").exists()


@pytest.mark.cuda
@pytest.mark.parametrize("topology", ["complete", "one_peer_ring", "one_peer_exponential"])
def test_consensus_in_isolation_preserves_moments_and_step(topology):
    from tiny_llm.config import Config, DecentralizedConfig
    from tiny_llm.gh200.optimizer import ArenaAdamW

    require_gh200()
    cfg = Config()
    cfg.model = ModelConfig(vocab_size=17, layers=1, width=8, heads=2, ffn_width=16, context_length=4)
    cfg.training.micro_batch_size, cfg.training.batch_tokens = 2, 64
    cfg.decentralized = DecentralizedConfig(num_models=8, topology=topology)
    native = PackedLlama(cfg.model, 8).cuda()
    fused = GH200Model(cfg.model, 8, torch.device("cuda"))
    optimizer = ArenaAdamW(fused, cfg)
    optimizer.first_moment_storage.normal_(std=0.01)
    optimizer.second_moment_storage.uniform_(0, 0.01)
    first, second = optimizer.first_moment_storage.clone(), optimizer.second_moment_storage.clone()
    for phase in range(6):
        for gamma in (0., 0.1, 0.9999, 1.):
            native.parameter_storage.normal_(std=0.02)
            fused.load_packed_state_dict(native.packed_state_dict())
            optimizer.completed.fill_(phase)
            optimizer.gamma.fill_(gamma)
            native.mix_(topology, phase, gamma)
            optimizer.mix()
            torch.testing.assert_close(native.parameter_storage, fused.parameter_storage,
                                       rtol=1e-6, atol=1e-8)
            torch.testing.assert_close(optimizer.first_moment_storage, first, rtol=0, atol=0)
            torch.testing.assert_close(optimizer.second_moment_storage, second, rtol=0, atol=0)
            assert optimizer.completed.item() == phase


@pytest.mark.cuda
def test_synchronous_optimizer_canonical_import_export_with_both_moments():
    from tiny_llm.config import Config
    from tiny_llm.gh200.optimizer import ArenaAdamW
    from tiny_llm.model import Llama
    from tiny_llm.train import make_optimizer

    require_gh200()
    cfg = Config()
    cfg.model = ModelConfig(vocab_size=17, layers=1, width=8, heads=2, ffn_width=16, context_length=4)
    native = Llama(cfg.model).cuda()
    native_opt = make_optimizer(native, cfg, torch.device("cuda"))
    for _ in range(3):
        for p in native.parameters():
            p.grad = torch.randn_like(p)
        native_opt.step()
    fused = GH200Model(cfg.model, 1, torch.device("cuda"))
    fused.load_local_state_dict(0, native.state_dict())
    optimizer = ArenaAdamW(fused, cfg)
    optimizer.load_state_dict(native_opt.state_dict())
    canonical: Any = optimizer.state_dict()
    for key, value in native_opt.state_dict()["state"].items():
        for name in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(value[name], canonical["state"][key][name].to(value[name].device),
                                       rtol=0, atol=0)
    for _ in range(5):
        for p in fused.parameters():
            p.grad = torch.randn_like(p).contiguous()
        for entry, p in zip(fused.fused_layout, fused.parameters(), strict=True):
            assert p.grad is not None
            optimizer.gradients[:, entry.start:entry.start + entry.numel].copy_(p.grad.flatten(1))
        for entry in fused.layout:
            native.get_parameter(entry.name).grad = fused.parameter_view(optimizer.gradients, entry)[0].clone()
        assert cfg.optimizer.grad_clip is not None
        torch.nn.utils.clip_grad_norm_(native.parameters(), cfg.optimizer.grad_clip)
        native_opt.step()
        optimizer.step(torch.ones(1, device="cuda"))
        for entry in fused.layout:
            torch.testing.assert_close(native.get_parameter(entry.name), fused.get_parameter(entry.name)[0],
                                       rtol=1e-6, atol=1e-8)
            for key, storage in (("exp_avg", optimizer.first_moment_storage),
                                 ("exp_avg_sq", optimizer.second_moment_storage)):
                torch.testing.assert_close(native_opt.state[native.get_parameter(entry.name)][key],
                                           fused.parameter_view(storage, entry)[0], rtol=1e-6, atol=1e-8)


@pytest.mark.cuda
@pytest.mark.parametrize("workers", [1, 4])
def test_real_data_benchmark_uses_complete_updates(tiny_config, cache_dir, tmp_path, workers):
    from tiny_llm.benchmark import benchmark_worker
    from tiny_llm.config import AdaptiveConsensusConfig, DecentralizedConfig, TrainingConfig

    require_gh200()
    cfg = tiny_config
    cfg.model = ModelConfig(vocab_size=17, layers=1, width=320, heads=5,
                            ffn_width=896, context_length=4)
    cfg.training = TrainingConfig(tokens_per_parameters=0.0001,
                                  epoch_tokens_per_parameters=0.0001,
                                  batch_tokens=16, micro_batch_size=1 if workers > 1 else 3,
                                  log_every=2)
    if workers > 1:
        cfg.decentralized = DecentralizedConfig(num_models=4, scheme="atc",
                                                adaptive_consensus=AdaptiveConsensusConfig(start_frac=0.5, p=1))
    cfg.runtime.device, cfg.runtime.training_backend = "cuda:0", "gh200"
    cfg.runtime.amp = cfg.runtime.compile = True
    result = benchmark_worker(cfg, tmp_path / "timing.json", warmup=2, steps=2, windows=2,
                               data_mode="real")
    assert result["status"] == "ok"
    assert result["backend"] == "gh200"
    assert result["successful_updates"] == 6 and result["cursor"] == 24
    assert len(result["measurements"]) == 2
    assert result["peak_memory_bytes"] > 0
    metrics = [json.loads(row) for row in (tmp_path / "timing.metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in metrics] == [2, 4, 6]
