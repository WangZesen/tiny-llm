import json
import os
import signal

import pytest
import torch
from safetensors.torch import load_file
from torch.func import functional_call

from tiny_llm.config import Config, ModelConfig
from tiny_llm.data import BufferedTokenLoader, TokenCache, fingerprint
from tiny_llm.model import Llama, token_losses
from tiny_llm.packed import PackedLlama, local_mean_losses
from tiny_llm.packed_benchmark import benchmark_packed_worker
from tiny_llm.packed_optimizer import PackedAdamW
from tiny_llm.runtime import preserve_rng, rng_state, setup_runtime
from tiny_llm.train import evaluate, evaluate_checkpoint, make_optimizer, recipe_identity, train


@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize("n", [1, 4, 8])
def test_initialization(tiny_config, seed, n):
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


@pytest.mark.parametrize("backend", ["reference", "sdpa"])
@pytest.mark.parametrize("topology", ["complete", "one_peer_ring", "one_peer_exponential"])
def test_independent_worker_parity(tiny_config, backend, topology):
    n = 3
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
    for step, batch in enumerate((2, 1, 2)):
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
                torch.testing.assert_close(p.grad[i], q.grad, rtol=1e-9, atol=1e-11)
        optimizer.clip_grad_norm_(0.5)
        for model in locals_:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        moments_before = optimizer.first_moment_storage.clone()
        packed.mix_(topology, step)
        assert torch.equal(moments_before, optimizer.first_moment_storage)
        w = matrix(n, topology, step)
        with torch.no_grad():
            for name, _ in locals_[0].named_parameters():
                old = torch.stack([model.get_parameter(name) for model in locals_])
                mixed = (w @ old.flatten(1)).view_as(old)
                for i, model in enumerate(locals_):
                    model.get_parameter(name).copy_(mixed[i])
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


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("topology", ["complete", "one_peer_ring", "one_peer_exponential"])
def test_topologies(tiny_config, n, topology):
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


def test_storage_roundtrip_and_moment_edits(tiny_config, tmp_path):
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


def test_causality_and_worker_independence(tiny_config):
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


def test_average_evaluation(tiny_config, cache_dir):
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


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("topology", ["complete", "one_peer_ring", "one_peer_exponential"])
def test_training_resume(tiny_config, cache_dir, monkeypatch, topology):
    config = decentralized_config(tiny_config, n=4, topology=topology)
    config.runtime.deterministic = True
    # Cross buffer boundaries inside packed steps and switch prefetch on resume.
    config.data.buffer_size_mib = 24 / 2**20
    # 12-block epochs: full local batch 2, followed by local batch 1.
    config.training.max_tokens = 96
    config.training.epoch_tokens = 48
    complete = config.model_copy(deep=True)
    complete.runtime.output_dir = cache_dir.parent / "complete"
    result = train(complete)
    original = PackedAdamW.step
    original_mix = PackedLlama.mix_
    events = []

    def record_mix(self, topology, step):
        assert all(parameter.grad is not None for parameter in self.parameters())
        events.append(("mix", step))
        original_mix(self, topology, step)

    def record_update(self):
        events.append(("update", None))
        original(self)

    monkeypatch.setattr(PackedLlama, "mix_", record_mix)

    def interrupted(self):
        original(self)
        os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(PackedAdamW, "step", interrupted)
    assert train(config)["status"] == "interrupted"
    monkeypatch.setattr(PackedAdamW, "step", record_update)
    config.data.prefetch = False
    resumed = train(config, config.runtime.output_dir / "latest.pt")
    assert events == [
        ("mix", 0),
        ("mix", 1),
        ("update", None),
        ("mix", 2),
        ("update", None),
        ("mix", 3),
        ("update", None),
    ]
    assert result["final_validation"]["loss"] == resumed["final_validation"]["loss"]
    a = torch.load(complete.runtime.output_dir / "final.pt", weights_only=False)
    b = torch.load(config.runtime.output_dir / "final.pt", weights_only=False)
    assert_nested_equal(a["model"], b["model"])
    assert_nested_equal(a["optimizer"], b["optimizer"])
    assert a["step"] == b["step"] == 4
    assert a["version"] == b["version"] == 3
    assert a["loader"] == b["loader"]
    assert a["loader"]["cursor"] == a["cursor"] == 24
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


def test_config_and_buffered_identity(tiny_config, cache_dir):
    raw = tiny_config.model_dump()
    raw["decentralized"] = {"num_models": 8}
    with pytest.raises(ValueError, match="no accumulation"):
        Config.model_validate(raw)
    config = decentralized_config(tiny_config, 4)
    config.training.epoch_tokens = 20
    with pytest.raises(ValueError, match="epoch boundaries"):
        train(config)
    value = tiny_config.model_dump(mode="json")
    value.pop("decentralized")
    for key in ("device", "output_dir", "cpu_threads", "compile_mode", "sdpa_backend"):
        value["runtime"].pop(key)
    value["data"].pop("cache_dir")
    value["data"].pop("prefetch")
    cache = TokenCache(cache_dir)
    value["cache_identity"] = cache.manifest["identity"]
    value["loader_version"] = BufferedTokenLoader.VERSION
    assert recipe_identity(tiny_config, cache) == fingerprint(value)


def test_buffered_worker_partition(tiny_config, cache_dir, monkeypatch):
    import tiny_llm.train as module

    config = decentralized_config(tiny_config, 4)
    config.runtime.deterministic = True
    config.data.buffer_size_mib = 24 / 2**20
    config.training.max_tokens, config.training.epoch_tokens = 96, 48
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
    assert [x.shape[1] for x, _ in batches] == [2, 1, 2, 1]
    for index, wanted in enumerate(expected):
        actual = torch.cat([batch[index] for batch in batches], dim=1)
        assert torch.equal(actual, wanted.view(6, 4, 4).transpose(0, 1))


def test_checkpoint_format_detection(tiny_config, cache_dir):
    config = decentralized_config(tiny_config)
    config.runtime.deterministic = True
    packed = PackedLlama(config.model, 2, "reference")
    averaged = Llama(config.model, "reference")
    packed.copy_average_to(averaged)
    path = cache_dir.parent / "legacy-packed.pt"
    torch.save(
        {
            "version": 2,
            "config": config.model_dump(mode="json"),
            "model": packed.packed_state_dict(),
        },
        path,
    )
    expected = evaluate(averaged, TokenCache(cache_dir), config, torch.device("cpu"), True)
    assert evaluate_checkpoint(config, path, True)["loss"] == expected["loss"]
    with pytest.raises(ValueError, match="legacy checkpoint.*weights remain usable"):
        train(config, path)
    # Upstream ordinary checkpoints also use version 2, with an ordinary state dict.
    torch.save({"version": 2, "model": averaged.state_dict()}, path)
    assert evaluate_checkpoint(config, path, True)["loss"] == expected["loss"]


@pytest.mark.parametrize("execution", ["packed", "sequential"])
def test_benchmark_worker(tiny_config, tmp_path, execution):
    config = decentralized_config(tiny_config)
    destination = tmp_path / f"{execution}.json"
    result = benchmark_packed_worker(config, destination, execution, warmup=1, steps=1)
    assert result["status"] == "ok"
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
@pytest.mark.parametrize("compiled", [False, True])
def test_cuda_bf16_and_fused_optimizer(tiny_config, compiled):
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

    loss = torch.compile(compute, fullgraph=True) if compiled else compute
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
                error = (p.grad[i] - q.grad).norm() / q.grad.norm().clamp_min(1e-8)
                assert error < 0.06
                # Isolate fused-optimizer/storage parity from BF16 kernel rounding.
                # Adam can amplify sign differences in near-zero gradients.
                q.grad.copy_(p.grad[i])
        optimizer.clip_grad_norm_(cfg.optimizer.grad_clip)
        for model in locals_:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
        packed.mix_("complete", _step)
        with torch.no_grad():
            for name, _ in locals_[0].named_parameters():
                average = torch.stack([model.get_parameter(name) for model in locals_]).mean(0)
                for model in locals_:
                    model.get_parameter(name).copy_(average)
        optimizer.step()
        for opt in optimizers:
            opt.step()
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
