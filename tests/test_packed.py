import json
from pathlib import Path

import pytest
import torch
from helpers import assert_nested_equal, set_budget
from safetensors.torch import load_file
from torch.func import functional_call

from tiny_llm.config import Config, ModelConfig, load_config
from tiny_llm.data import BufferedTokenLoader, TokenCache
from tiny_llm.model import Llama, token_losses
from tiny_llm.packed import PackedLlama, local_mean_losses
from tiny_llm.packed_benchmark import benchmark_packed_worker
from tiny_llm.packed_optimizer import PackedAdamW
from tiny_llm.runtime import preserve_rng, rng_state, setup_runtime
from tiny_llm.state import LoaderState, TrainingState
from tiny_llm.train import (
    adaptive_consensus_schedule,
    evaluate,
    evaluate_checkpoint,
    learning_rate,
    make_optimizer,
    recipe_identity,
    train,
)


@pytest.mark.parametrize("seed,n", [(0, 1), (42, 4), (123, 8)])
def test_initialization(tiny_config: Config, seed, n):
    torch.manual_seed(seed)
    ordinary = Llama(tiny_config.model)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(seed)
    packed = PackedLlama(tiny_config.model, n)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for worker in range(n):
        for name, value in packed.local_state_dict(worker).items():
            assert torch.equal(value, ordinary.state_dict()[name])
    assert packed.parameter_count == n * ordinary.parameter_count
    optimizer = PackedAdamW(packed, tiny_config)
    assert not optimizer.first_moment_storage.count_nonzero()
    assert not optimizer.second_moment_storage.count_nonzero()
    assert torch.equal(torch.get_rng_state(), expected_rng)


def matrix(n, topology, step, device="cpu", dtype=torch.float64):
    if topology == "complete":
        return torch.full((n, n), 1 / n, device=device, dtype=dtype)
    offset = (
        (1 if step % 2 == 0 else -1)
        if topology == "one_peer_ring"
        else (1 << (step % max(1, (n - 1).bit_length())))
    )
    result = torch.eye(n, device=device, dtype=dtype) * 0.5
    for i in range(n):
        result[i, (i - offset) % n] += 0.5
    return result


@pytest.mark.parametrize(
    "backend,topology",
    [
        ("reference", "complete"),
        ("reference", "one_peer_ring"),
        ("reference", "one_peer_exponential"),
        ("sdpa", "complete"),
    ],
)
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("scheme", ["awc", "atc"])
@pytest.mark.parametrize("grad_clip", [0.5, None])
def test_independent_worker_parity(
    tiny_config: Config, backend, topology, adaptive, scheme, grad_clip
):
    n = 3
    schedule = None
    if adaptive:
        raw = tiny_config.model_dump()
        raw["decentralized"] = dict(num_models=n, adaptive_consensus=dict(start_frac=0.2, p=2))
        raw["training"]["batch_tokens"] = 24
        raw["optimizer"]["min_lr_ratio"] = 0
        tiny_config = Config.model_validate(raw)
        schedule = adaptive_consensus_schedule(tiny_config, [9, 15])
    torch.manual_seed(13)
    packed = PackedLlama(tiny_config.model, n, backend).double()
    # Distinct parameters detect mixing order and cross-worker contamination.
    with torch.no_grad():
        for p in packed.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    locals_ = [Llama(tiny_config.model, backend).double() for _ in range(n)]
    for i, model in enumerate(locals_):
        model.load_state_dict(packed.local_state_dict(i))
    optimizer = PackedAdamW(packed, tiny_config)
    optimizers = [make_optimizer(model, tiny_config, torch.device("cpu")) for model in locals_]
    consumed = 0
    for step, batch in enumerate((2, 1, 2)):
        consumed += n * batch * 4
        lr = learning_rate(tiny_config, consumed, 60)
        for opt in [optimizer, *optimizers]:
            for group in opt.param_groups:
                group["lr"] = lr
        gamma = schedule.gamma(step, lr) if schedule else 1.0
        x = torch.randint(17, (n, batch, 4))
        y = torch.randint(17, x.shape)
        y[0, 0, -1] = -100
        optimizer.zero_grad()
        logits = packed(x)
        means = local_mean_losses(logits, y)
        means.sum().backward()
        for i, (model, local_optimizer) in enumerate(zip(locals_, optimizers, strict=True)):
            local_optimizer.zero_grad()
            actual = model(x[i])
            torch.testing.assert_close(logits[i], actual, rtol=1e-10, atol=1e-12)
            loss = token_losses(actual, y[i]).sum() / (y[i] != -100).sum()
            torch.testing.assert_close(means[i], loss)
            loss.backward()
            for p, q in zip(packed.parameters(), model.parameters(), strict=True):
                assert p.grad is not None and q.grad is not None
                torch.testing.assert_close(p.grad[i], q.grad, rtol=1e-9, atol=1e-11)
        optimizer.clip_grad_norm_(grad_clip)
        if grad_clip is not None:
            for model in locals_:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if scheme == "atc":
            optimizer.step()
            for local_optimizer in optimizers:
                local_optimizer.step()
        moments_before = optimizer.first_moment_storage.clone()
        packed.mix_(topology, step, gamma=gamma)
        assert torch.equal(moments_before, optimizer.first_moment_storage)
        w = gamma * matrix(n, topology, step) + (1 - gamma) * torch.eye(n, dtype=torch.float64)
        with torch.no_grad():
            for name, _ in locals_[0].named_parameters():
                old = torch.stack([model.get_parameter(name) for model in locals_])
                mixed = (w @ old.flatten(1)).view_as(old)
                for i, model in enumerate(locals_):
                    model.get_parameter(name).copy_(mixed[i])
        if scheme == "awc":
            optimizer.step()
            for local_optimizer in optimizers:
                local_optimizer.step()
        for i, (model, local_optimizer) in enumerate(zip(locals_, optimizers, strict=True)):
            for entry, p in zip(packed.layout, model.parameters(), strict=True):
                torch.testing.assert_close(
                    packed.get_parameter(entry.name)[i], p, rtol=1e-9, atol=1e-11
                )
                for arena, key in (
                    (optimizer.first_moment_storage, "exp_avg"),
                    (optimizer.second_moment_storage, "exp_avg_sq"),
                ):
                    torch.testing.assert_close(
                        packed.parameter_view(arena, entry)[i],
                        local_optimizer.state[p][key],
                        rtol=1e-9,
                        atol=1e-11,
                    )
        assert_storage(packed, optimizer)


def assert_storage(model, optimizer):
    arenas = (
        model.parameter_storage,
        optimizer.first_moment_storage,
        optimizer.second_moment_storage,
    )
    for arena in arenas:
        assert arena.is_contiguous()
        assert arena.shape == (model.num_models, model.storage_numel)
        for entry in model.layout:
            assert entry.start % 64 == 0
            assert not arena[
                :, entry.start + entry.numel : entry.start + entry.padded_numel
            ].count_nonzero()
    for i, parameters in enumerate(optimizer.local_parameters):
        for entry, p in zip(model.layout, parameters, strict=True):
            packed = model.get_parameter(entry.name)
            assert packed.untyped_storage().data_ptr() == arenas[0].untyped_storage().data_ptr()
            assert p.data_ptr() == packed[i].data_ptr()
            state = optimizer.optimizers[i].state[p]
            assert (
                state["exp_avg"].data_ptr() == model.parameter_view(arenas[1], entry)[i].data_ptr()
            )
            assert (
                state["exp_avg_sq"].data_ptr()
                == model.parameter_view(arenas[2], entry)[i].data_ptr()
            )


@pytest.mark.parametrize("n", [1, 2, 3, 8])
@pytest.mark.parametrize("topology", ["complete", "one_peer_ring", "one_peer_exponential"])
def test_topologies(tiny_config: Config, n, topology):
    model = PackedLlama(tiny_config.model, n).double()
    with torch.no_grad():
        model.parameter_storage.normal_()
    original = model.parameter_storage.clone()
    for step in range(max(2, (n - 1).bit_length())):
        expected = matrix(n, topology, step) @ model.parameter_storage
        model.mix_(topology, step)
        torch.testing.assert_close(model.parameter_storage, expected)
    if topology == "one_peer_exponential" and n & (n - 1) == 0:
        torch.testing.assert_close(model.parameter_storage, original.mean(0).expand_as(original))


def test_storage_roundtrip_and_moment_edits(tiny_config: Config, tmp_path: Path):
    model = PackedLlama(tiny_config.model, 2).double()
    model.load_state_dict(model.state_dict())
    with pytest.raises(ValueError, match="aliases"):
        model.load_state_dict(model.state_dict(), assign=True)
    optimizer = PackedAdamW(model, tiny_config)
    assert_storage(model, optimizer)
    # A direct arena edit must influence AdamW, even with zero gradients.
    optimizer.first_moment_storage.fill_(0.1)
    optimizer.second_moment_storage.fill_(0.2)
    for entry in model.layout:
        for arena in (optimizer.first_moment_storage, optimizer.second_moment_storage):
            arena[:, entry.start + entry.numel : entry.start + entry.padded_numel].zero_()
    before = model.parameter_storage.clone()
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert not torch.equal(before, model.parameter_storage)
    path = tmp_path / "state.pt"
    torch.save({"model": model.packed_state_dict(), "optimizer": optimizer.state_dict()}, path)
    other = PackedLlama(tiny_config.model, 2).double()
    other_optimizer = PackedAdamW(other, tiny_config)
    state = torch.load(path, weights_only=False)
    other.load_packed_state_dict(state["model"])
    other_optimizer.load_state_dict(state["optimizer"])
    assert_storage(other, other_optimizer)
    assert torch.equal(other.parameter_storage, model.parameter_storage)
    assert torch.equal(other_optimizer.first_moment_storage, optimizer.first_moment_storage)
    assert torch.equal(other_optimizer.second_moment_storage, optimizer.second_moment_storage)
    model.float()
    with pytest.raises(RuntimeError, match="placement"):
        optimizer.step()
    assert_storage(model, PackedAdamW(model, tiny_config))


def test_causality_and_worker_independence(tiny_config: Config):
    packed = PackedLlama(tiny_config.model, 4, "reference").double()
    x = torch.randint(17, (4, 2, 4))
    before = packed(x)
    changed = x.clone()
    changed[0, :, 2:] = (changed[0, :, 2:] + 1) % 17
    after = packed(changed)
    torch.testing.assert_close(before[0, :, :2], after[0, :, :2], rtol=0, atol=0)
    torch.testing.assert_close(before[1:], after[1:], rtol=0, atol=0)
    single = PackedLlama(tiny_config.model, 1, "reference").double()
    single.load_local_state_dict(0, packed.local_state_dict(0))
    local_mean_losses(packed(x), x).sum().backward()
    local_mean_losses(single(x[:1]), x[:1]).sum().backward()
    for p, q in zip(packed.parameters(), single.parameters(), strict=True):
        assert p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad[0], q.grad[0], rtol=1e-10, atol=1e-12)


def test_second_derivatives():
    cfg = ModelConfig(vocab_size=5, layers=1, width=4, heads=1, ffn_width=4, context_length=2)
    model = PackedLlama(cfg, 2, "reference").double()
    x = torch.tensor([[[1, 2]], [[3, 4]]])
    names, parameters = zip(*model.named_parameters(), strict=True)

    def loss(*values):
        return local_mean_losses(
            functional_call(model, dict(zip(names, values, strict=True)), (x,)), x
        ).sum()

    assert torch.autograd.gradcheck(loss, parameters, fast_mode=True)
    assert torch.autograd.gradgradcheck(loss, parameters, fast_mode=True)


def test_average_evaluation(tiny_config: Config, cache_dir: Path):
    setup_runtime(tiny_config)
    packed = PackedLlama(tiny_config.model, 3, "reference")
    optimizer = PackedAdamW(packed, tiny_config)
    with torch.no_grad():
        for p in packed.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    before = packed.parameter_storage.clone()
    rng = rng_state()
    with preserve_rng():
        averaged = Llama(tiny_config.model, "reference")
    packed.copy_average_to(averaged)
    for name, p in averaged.named_parameters():
        torch.testing.assert_close(p, packed.get_parameter(name).mean(0), rtol=0, atol=0)
    batches = []
    handle = averaged.register_forward_pre_hook(lambda model, args: batches.append(args[0].shape))
    result = evaluate(averaged, TokenCache(cache_dir), tiny_config, torch.device("cpu"), True)
    handle.remove()
    assert all(
        len(shape) == 2 and shape[0] <= tiny_config.evaluation.batch_size for shape in batches
    )
    assert result["tokens"] == 29
    assert torch.equal(before, packed.parameter_storage)
    assert torch.equal(rng["torch"], rng_state()["torch"])
    assert not optimizer.first_moment_storage.count_nonzero()


def decentralized_config(config, n=2, topology="one_peer_ring"):
    raw = config.model_dump()
    raw["decentralized"] = {"num_models": n, "topology": topology}
    raw["training"]["batch_tokens"] = (
        n * config.training.micro_batch_size * config.model.context_length
    )
    return Config.model_validate(raw)


@pytest.mark.parametrize("scheme", ["awc", "atc"])
@pytest.mark.parametrize("grad_clip", [1.0, None])
def test_training_resume(
    tiny_config: Config, cache_dir: Path, monkeypatch: pytest.MonkeyPatch, scheme, grad_clip
):
    tiny_config.training.checkpoint_policy = "interval"
    topology = "one_peer_exponential"
    config = decentralized_config(tiny_config, n=4, topology=topology)
    assert config.decentralized is not None
    config.decentralized.scheme = scheme
    config.optimizer.grad_clip = grad_clip
    config.runtime.deterministic = True
    # Cross buffer boundaries inside packed steps and switch prefetch on resume.
    config.data.buffer_size_mib = 24 / 2**20
    # 12-block epochs: full local batch 2, followed by local batch 1.
    set_budget(config, 128, 64)
    complete = config.model_copy(deep=True)
    complete.runtime.output_dir = cache_dir.parent / "complete"
    result = train(complete)
    config.data.prefetch = False
    resumed = train(config, complete.runtime.output_dir / "epoch-001.pt")
    assert result["scheme"] == resumed["scheme"] == scheme
    metadata = json.loads((complete.runtime.output_dir / "environment.json").read_text())
    assert metadata["scheme"] == scheme
    assert result["final_validation"]["loss"] == resumed["final_validation"]["loss"]
    a = torch.load(complete.runtime.output_dir / "final.pt", weights_only=False)
    b = torch.load(config.runtime.output_dir / "final.pt", weights_only=False)
    assert_nested_equal(a["model"], b["model"])
    assert_nested_equal(a["optimizer"], b["optimizer"])
    assert a["step"] == b["step"] == 4
    assert a["version"] == b["version"] == 3
    assert a["config"]["decentralized"]["scheme"] == scheme
    assert b["config"]["decentralized"]["scheme"] == scheme
    assert a["loader"] == b["loader"]
    assert a["loader"]["cursor"] == a["cursor"] == 32
    expected = evaluate_checkpoint(config, config.runtime.output_dir / "final.pt", True)
    assert expected["loss"] == result["final_validation"]["loss"]
    averaged = load_file(str(config.runtime.output_dir / "epoch-002.safetensors"))
    locals_ = [
        load_file(str(config.runtime.output_dir / f"node-{i:03d}/epoch-002.safetensors"))
        for i in range(4)
    ]
    for key, value in averaged.items():
        torch.testing.assert_close(
            value, torch.stack([local[key] for local in locals_]).mean(0), rtol=0, atol=0
        )
    assert json.loads((config.runtime.output_dir / "best.json").read_text())["epoch"] in (1, 2)


def test_config_and_buffered_identity(tiny_config: Config, cache_dir: Path):
    raw = tiny_config.model_dump()
    raw["decentralized"] = {"num_models": 8}
    with pytest.raises(ValueError, match="no accumulation"):
        Config.model_validate(raw)


def test_buffered_worker_partition(
    tiny_config: Config, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    import tiny_llm.train as module

    config = decentralized_config(tiny_config, 4)
    assert config.decentralized is not None
    config.runtime.deterministic = True
    config.data.buffer_size_mib = 24 / 2**20
    set_budget(config, 96, 48)
    cache = TokenCache(cache_dir)
    with BufferedTokenLoader(
        cache, "train", 4, 24, config.data.buffer_size_mib, seed=config.runtime.seed
    ) as loader:
        expected = loader.next_batch(24, torch.device("cpu"))
    original = module.loss_function
    batches = []

    def record_loss(model, config, device):
        loss = original(model, config, device)

        def compute(x, y):
            batches.append((x.clone(), y.clone()))
            return loss(x, y)

        return compute

    def random_access_forbidden(*args, **kwargs):
        raise AssertionError("packed training/evaluation must use buffered reads")

    monkeypatch.setattr(module, "loss_function", record_loss)
    monkeypatch.setattr(TokenCache, "batch", random_access_forbidden)
    assert train(config)["status"] == "complete"
    assert [x.shape[1] for x, _ in batches] == [2, 2, 2]
    for index, wanted in enumerate(expected):
        actual = torch.cat([batch[index] for batch in batches], dim=1)
        assert torch.equal(actual, wanted.view(6, 4, 4).transpose(0, 1))


@pytest.mark.parametrize("execution", ["packed", "sequential"])
@pytest.mark.parametrize("scheme", ["awc", "atc"])
@pytest.mark.parametrize("grad_clip", [1.0, None])
def test_benchmark_worker(
    tiny_config: Config,
    tmp_path: Path,
    execution,
    scheme,
    monkeypatch: pytest.MonkeyPatch,
    grad_clip,
):
    config = decentralized_config(tiny_config)
    assert config.decentralized is not None
    config.decentralized.scheme = scheme
    config.optimizer.grad_clip = grad_clip
    events = []
    original_mix = PackedLlama.mix_
    original_step = PackedAdamW.step if execution == "packed" else torch.optim.AdamW.step

    def mix(self, *args, **kwargs):
        events.append("mix")
        return original_mix(self, *args, **kwargs)

    def update(self, *args, **kwargs):
        events.append("update")
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(PackedLlama, "mix_", mix)
    monkeypatch.setattr(PackedAdamW if execution == "packed" else torch.optim.AdamW, "step", update)
    destination = tmp_path / f"{execution}.json"
    result = benchmark_packed_worker(config, destination, execution, warmup=1, steps=1)
    assert result["status"] == "ok"
    assert result["scheme"] == scheme
    updates = ["update"] * (1 if execution == "packed" else config.decentralized.num_models)
    pair = ["mix", *updates] if scheme == "awc" else [*updates, "mix"]
    assert events == pair * 3  # Warmup, throughput, and component timing passes.
    assert result["tokens_per_second"] > 0
    assert result["global_batch_tokens"] == config.training.batch_tokens
    assert set(result["component_seconds"]) == {"compute", "clipping", "mixing", "optimizer"}
    assert destination.with_suffix(".trace.json").exists()
    if execution == "packed":
        assert any(
            op["name"] == "aten::bmm" and op["input_shapes"][0][0] == 2
            for op in result["operators"]
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_bf16_and_fused_optimizer(tiny_config: Config):
    from tiny_llm.checkpoints import load_training_checkpoint, save_epoch_checkpoint
    from tiny_llm.config import DecentralizedConfig

    cfg = tiny_config
    cfg.runtime.device, cfg.runtime.amp = "cuda:0", True
    cfg.runtime.deterministic = False
    cfg.training.batch_tokens = 64
    cfg.model = ModelConfig(
        vocab_size=128, layers=2, width=64, heads=2, ffn_width=128, context_length=16
    )
    torch.use_deterministic_algorithms(False)
    packed = PackedLlama(cfg.model, 2).cuda()
    optimizer = PackedAdamW(packed, cfg)
    locals_ = [Llama(cfg.model).cuda() for _ in range(2)]
    for i, model in enumerate(locals_):
        model.load_state_dict(packed.local_state_dict(i))
    optimizers = [make_optimizer(model, cfg, torch.device("cuda")) for model in locals_]

    def compute(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return local_mean_losses(packed(x), y)

    loss = torch.compile(compute, fullgraph=True)
    for _step in range(2):
        x = torch.randint(128, (2, 2, 16), device="cuda")
        optimizer.zero_grad()
        actual = loss(x, x)
        actual.sum().backward()
        for i, (model, local_optimizer) in enumerate(zip(locals_, optimizers, strict=True)):
            local_optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                expected = token_losses(model(x[i]), x[i]).mean()
            torch.testing.assert_close(actual[i], expected, rtol=0.003, atol=0.003)
            expected.backward()
            for p, q in zip(packed.parameters(), model.parameters(), strict=True):
                assert p.grad is not None and q.grad is not None
                error = (p.grad[i] - q.grad).norm() / q.grad.norm().clamp_min(1e-8)
                assert error < 0.06
                # Isolate fused-optimizer/storage parity from BF16 kernel rounding.
                # Adam can amplify sign differences in near-zero gradients.
                q.grad.copy_(p.grad[i])
        optimizer.clip_grad_norm_(cfg.optimizer.grad_clip)
        assert cfg.optimizer.grad_clip is not None
        for model in locals_:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
        gamma = 1.0 if _step == 0 else 0.25
        packed.mix_("complete", _step, gamma=gamma)
        with torch.no_grad():
            for name, _ in locals_[0].named_parameters():
                average = torch.stack([model.get_parameter(name) for model in locals_]).mean(0)
                for model in locals_:
                    model.get_parameter(name).mul_(1 - gamma).add_(average, alpha=gamma)
        optimizer.step()
        for opt in optimizers:
            opt.step()
        assert_storage(packed, optimizer)
        if _step == 0:
            # Continue the compiled BF16/fused update after restoring separate worker files.
            cfg.decentralized = DecentralizedConfig(num_models=2)
            cfg.training.tokens_per_parameters = 128 / cfg.model.parameter_count
            cfg.training.epoch_tokens_per_parameters = 64 / cfg.model.parameter_count
            directory = cfg.runtime.output_dir
            directory.mkdir()
            path = directory / "epoch-001.pt"
            state = TrainingState(
                version=3,
                config=cfg.model_dump(mode="json"),
                recipe_identity="cuda-fixture",
                model=packed.packed_state_dict(),
                optimizer=optimizer.state_dict(),
                cursor=4,
                step=1,
                completed_epochs=1,
                loader=LoaderState(
                    version=1,
                    cache_identity="cuda-fixture",
                    split="train",
                    length=16,
                    blocks=8,
                    buffer_blocks=2097152,
                    seed=42,
                    cursor=4,
                ),
                rng=rng_state(),
                best_loss=1.0,
                best_epoch=1,
            )
            save_epoch_checkpoint(path, state, packed, optimizer)
            restored = load_training_checkpoint(path)
            with torch.no_grad():
                packed.parameter_storage.zero_()
                optimizer.first_moment_storage.zero_()
                optimizer.second_moment_storage.zero_()
            packed.load_packed_state_dict(restored["model"])
            optimizer.load_state_dict(restored["optimizer"])
            assert_storage(packed, optimizer)
    for i, model in enumerate(locals_):
        for name, q in model.named_parameters():
            torch.testing.assert_close(packed.get_parameter(name)[i], q, rtol=1e-6, atol=1e-8)
        for entry, q in zip(packed.layout, model.parameters(), strict=True):
            for arena, key in (
                (optimizer.first_moment_storage, "exp_avg"),
                (optimizer.second_moment_storage, "exp_avg_sq"),
            ):
                torch.testing.assert_close(
                    packed.parameter_view(arena, entry)[i],
                    optimizers[i].state[q][key],
                    rtol=1e-6,
                    atol=1e-8,
                )


def test_scheme_config_and_identity(tiny_config: Config, cache_dir: Path, tmp_path: Path):
    config = decentralized_config(tiny_config)
    assert config.decentralized is not None
    path = tmp_path / "config.json"
    raw = config.model_dump(mode="json")
    raw["decentralized"].pop("scheme")
    path.write_text(json.dumps(raw))
    assert load_config(path) == config
    assert config.decentralized.scheme == "awc"
    assert load_config(path, ["decentralized.scheme=awc"]) == config
    atc = load_config(path, ["decentralized.scheme=atc"])
    assert atc.decentralized is not None
    assert atc.decentralized.scheme == "atc"
    with pytest.raises(ValueError):
        load_config(path, ["decentralized.scheme=invalid"])
    cache = TokenCache(cache_dir)
    assert recipe_identity(atc, cache) != recipe_identity(config, cache)
    config.training.checkpoint_policy = "interval"
    train(config)
    with pytest.raises(ValueError, match="recipe"):
        train(atc, config.runtime.output_dir / "final.pt")


@pytest.mark.parametrize("scheme", ["awc", "atc"])
@pytest.mark.parametrize("grad_clip", [1.0, None])
def test_benchmark_scheme_propagation(
    tiny_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scheme, grad_clip
):
    from types import SimpleNamespace

    import tiny_llm.packed_benchmark as module

    config = decentralized_config(tiny_config)
    assert config.decentralized is not None
    config.decentralized.scheme = scheme
    config.optimizer.grad_clip = grad_clip
    executions = []

    def run(command, **kwargs):
        candidate = load_config(command[command.index("--config") + 1])
        assert candidate.decentralized is not None
        assert candidate.decentralized.scheme == scheme
        assert candidate.optimizer.grad_clip == grad_clip
        execution = command[command.index("--execution") + 1]
        executions.append(execution)
        destination = module.Path(command[command.index("--output") + 1])
        destination.write_text(
            json.dumps(
                dict(status="ok", seconds_per_step=1.0, scheme=candidate.decentralized.scheme)
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    result = module.benchmark_packed(config, tmp_path / "benchmark", num_models=[1, 2])
    assert executions == ["sequential", "packed"] * 2
    for row in result["comparisons"]:
        assert row["packed"]["scheme"] == row["sequential"]["scheme"] == scheme
