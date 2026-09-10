import json
import os
import signal
from dataclasses import asdict

import pytest
import torch
from test_packed import assert_nested_equal, assert_storage, matrix

from tiny_llm.checkpoints import RETENTION_FIELDS, load_training_checkpoint
from tiny_llm.config import AdaptiveConsensusConfig, Config
from tiny_llm.data import BufferedTokenLoader, TokenCache, fingerprint, training_boundaries
from tiny_llm.packed import PackedLlama
from tiny_llm.packed_benchmark import benchmark_packed, benchmark_packed_worker
from tiny_llm.packed_optimizer import PackedAdamW
from tiny_llm.train import adaptive_consensus_schedule, learning_rate, recipe_identity, train


def adaptive_config(config, start_frac=0.3, p=1.5):
    raw = config.model_dump()
    raw["decentralized"] = dict(
        num_models=4,
        topology="one_peer_exponential",
        adaptive_consensus=dict(start_frac=start_frac, p=p),
    )
    raw["training"].update(batch_tokens=32, max_tokens=128, epoch_tokens=48)
    return Config.model_validate(raw)


def test_config_validation(tiny_config):
    for raw in (
        {},
        {"start_frac": 0},
        {"p": 1},
        {"start_frac": -0.1, "p": 1},
        {"start_frac": 1.1, "p": 1},
        {"start_frac": 0.5, "p": -1},
        {"start_frac": float("nan"), "p": 1},
        {"start_frac": 0.5, "p": float("inf")},
    ):
        with pytest.raises(ValueError):
            AdaptiveConsensusConfig(**raw)
    assert adaptive_consensus_schedule(tiny_config, [8, 16]) is None


@pytest.mark.parametrize(
    "start_frac,warmup,p",
    [(0, 0.5, 1), (0.3, 0.6, 2), (0.6, 0.2, 0.5), (1, 0.5, 1), (0.99, 0, 1), (0, 0, 0)],
)
def test_realized_schedule(tiny_config, start_frac, warmup, p):
    config = adaptive_config(tiny_config, start_frac, p)
    config.optimizer.warmup_fraction = warmup
    before = torch.get_rng_state()
    schedule = adaptive_consensus_schedule(
        config, training_boundaries(config, config.model.parameter_count)
    )
    assert torch.equal(before, torch.get_rng_state())
    # Three epochs contain 12, 12, 8 blocks: two full and two shortened steps,
    # followed by one full step. The naive global ceil would give only four.
    assert schedule.total_steps == 5
    expected_start = {0: 0, 0.3: 2, 0.6: 3, 1: 5, 0.99: 5}[start_frac]
    assert schedule.start_step == expected_start
    rates = [learning_rate(config, tokens, 128) for tokens in (32, 48, 80, 96, 128)]
    assert schedule.lr_max == (max(rates[expected_start:]) if expected_start < 5 else None)
    for r, lr in enumerate(rates):
        expected = 1 if r < expected_start or p == 0 else (lr / max(rates[expected_start:])) ** p
        assert schedule.gamma(r, lr) == expected
    if start_frac == 0 and warmup == 0.5:
        assert schedule.lr_max > rates[0]  # Maximum occurs after activation.


def test_zero_lr_and_empty_window(tiny_config):
    config = adaptive_config(tiny_config, 0)
    config.optimizer.min_lr_ratio = 0
    with pytest.raises(ValueError, match="positive maximum LR"):
        adaptive_consensus_schedule(config, [8])
    config.decentralized.adaptive_consensus.p = 0
    assert adaptive_consensus_schedule(config, [8]).gamma(0, 0) == 1
    config.decentralized.adaptive_consensus.p = 1
    config.decentralized.adaptive_consensus.start_frac = 0.1
    schedule = adaptive_consensus_schedule(config, [8])
    assert schedule.start_step == 1 and schedule.lr_max is None
    assert schedule.gamma(0, 0) == 1


@pytest.mark.parametrize("topology", ["complete", "one_peer_ring", "one_peer_exponential"])
@pytest.mark.parametrize("n", [1, 3, 4])
@pytest.mark.parametrize("exclude_embeddings", [False, True])
def test_weighted_mixing(tiny_config, topology, n, exclude_embeddings):
    model = PackedLlama(tiny_config.model, n).double()
    opt = PackedAdamW(model, tiny_config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_()
            parameter.grad = torch.randn_like(parameter)
        for arena in (opt.first_moment_storage, opt.second_moment_storage):
            for entry in model.layout:
                arena[:, entry.start : entry.start + entry.numel].copy_(
                    torch.arange(n, dtype=arena.dtype)[:, None] + 0.25
                )
    before = model.parameter_storage.clone()
    moments = (opt.first_moment_storage.clone(), opt.second_moment_storage.clone())
    gradients = [p.grad.clone() for p in model.parameters()]
    for step, gamma in enumerate((0, 0.2, 0.75, 1)):
        model.parameter_storage.copy_(before)
        pointer = model.parameter_storage.data_ptr()
        expected = (
            gamma * matrix(n, topology, step) + (1 - gamma) * torch.eye(n, dtype=before.dtype)
        ) @ before
        embedding = next(entry for entry in model.layout if entry.name == "embedding.weight")
        start, end = embedding.start, embedding.start + embedding.numel
        if exclude_embeddings:
            expected[:, start:end] = matrix(n, topology, step) @ before[:, start:end]
        model.mix_(topology, step, gamma, exclude_embeddings=exclude_embeddings)
        torch.testing.assert_close(model.parameter_storage, expected, rtol=1e-14, atol=1e-14)
        assert model.parameter_storage.data_ptr() == pointer
        if gamma == 0:
            assert torch.equal(model.parameter_storage[:, :start], before[:, :start])
            assert torch.equal(model.parameter_storage[:, end:], before[:, end:])
            if not exclude_embeddings:
                assert torch.equal(model.parameter_storage, before)
        if gamma == 1 or not exclude_embeddings:
            if n == 1 or gamma == 0:
                legacy = before
            elif topology == "complete":
                legacy = before.mean(0).expand_as(before).clone()
            else:
                offset = (
                    (1 if step % 2 == 0 else -1)
                    if topology == "one_peer_ring"
                    else 1 << (step % (n - 1).bit_length())
                )
                legacy = (before[(torch.arange(n) - offset) % n] + before) * 0.5
            if n > 1 and 0 < gamma < 1:
                legacy.mul_(gamma).add_(before, alpha=1 - gamma)
            assert torch.equal(model.parameter_storage, legacy)
        assert_storage(model, opt)
        assert torch.equal(opt.first_moment_storage, moments[0])
        assert torch.equal(opt.second_moment_storage, moments[1])
        for local in opt.optimizers:
            assert all(state["step"].item() == 0 for state in local.state.values())
        for parameter, grad in zip(model.parameters(), gradients, strict=True):
            assert torch.equal(parameter.grad, grad)
    for gamma in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="gamma"):
            model.mix_(topology, 0, gamma)


def test_tied_embedding_after_conversion_and_restore(tiny_config, monkeypatch):
    import tiny_llm.packed as module
    from tiny_llm.model import Llama

    model = PackedLlama(tiny_config.model, 3).double()
    with torch.no_grad():
        model.embedding.weight.add_(torch.arange(3, dtype=torch.float64)[:, None, None] * 0.1)
    model.mix_("one_peer_ring", 0, 0.3, exclude_embeddings=True)
    restored = PackedLlama(tiny_config.model, 3).double()
    restored.load_packed_state_dict(model.packed_state_dict())
    assert_storage(restored, PackedAdamW(restored, tiny_config))
    assert restored.local_parameter_count == tiny_config.model.parameter_count
    assert not any("lm_head" in key for key in restored.state_dict())
    weights = []
    original = module.linear

    def observe(inputs, weight):
        weights.append(weight)
        return original(inputs, weight)

    monkeypatch.setattr(module, "linear", observe)
    tokens = torch.randint(tiny_config.model.vocab_size, (3, 2, 4))
    actual = restored(tokens)
    assert weights[-1] is restored.embedding.weight
    for worker in range(3):
        ordinary = Llama(tiny_config.model).double()
        ordinary.load_state_dict(restored.local_state_dict(worker))
        torch.testing.assert_close(actual[worker], ordinary(tokens[worker]), rtol=1e-10, atol=1e-12)


def test_identity_and_benchmark_guards(tiny_config, cache_dir, tmp_path):
    config = adaptive_config(tiny_config)
    cache = TokenCache(cache_dir)
    enabled = recipe_identity(config, cache)
    config.decentralized.adaptive_consensus = None
    disabled = recipe_identity(config, cache)
    legacy = config.model_dump(mode="json")
    legacy["decentralized"].pop("adaptive_consensus")
    for key in RETENTION_FIELDS:
        legacy["training"].pop(key)
    for key in ("output_dir", "device", "cpu_threads", "compile_mode", "sdpa_backend"):
        legacy["runtime"].pop(key)
    for key in ("cache_dir", "prefetch"):
        legacy["data"].pop(key)
    legacy.update(
        cache_identity=cache.manifest["identity"], loader_version=BufferedTokenLoader.VERSION
    )
    assert disabled == fingerprint(legacy) != enabled
    # This is the pre-exclusion adaptive recipe: the new field must not alter its hash.
    legacy["decentralized"]["adaptive_consensus"] = dict(start_frac=0.3, p=1.5)
    assert enabled == fingerprint(legacy)
    config = adaptive_config(tiny_config)
    assert not config.decentralized.adaptive_consensus.exclude_embeddings
    config.decentralized.adaptive_consensus.exclude_embeddings = False
    assert recipe_identity(config, cache) == enabled
    config.decentralized.adaptive_consensus.exclude_embeddings = True
    assert recipe_identity(config, cache) != enabled
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="training only"):
        benchmark_packed(config, destination)
    for execution in ("packed", "sequential"):
        with pytest.raises(ValueError, match="training only"):
            benchmark_packed_worker(config, destination, execution)
    assert not destination.exists()


@pytest.mark.parametrize(
    "resume_at,exclude_embeddings",
    [
        ("before", False),
        ("before", True),
        ("after", False),
        ("after", True),
        ("epoch", False),
        ("epoch", True),
        ("legacy", False),
    ],
)
def test_adaptive_resume(tiny_config, cache_dir, monkeypatch, resume_at, exclude_embeddings):
    config = adaptive_config(tiny_config)
    config.decentralized.adaptive_consensus.exclude_embeddings = exclude_embeddings
    config.training.checkpoint_policy = "all"
    config.runtime.deterministic = True
    config.optimizer.warmup_fraction = 0.6
    baseline = config.model_copy(deep=True)
    baseline.runtime.output_dir = cache_dir.parent / "baseline"
    expected = train(baseline)
    if resume_at in ("epoch", "legacy"):
        source = baseline.runtime.output_dir / "epoch-001.pt"
        if resume_at == "legacy":
            state = load_training_checkpoint(source)
            state["config"]["decentralized"]["adaptive_consensus"].pop("exclude_embeddings")
            source = baseline.runtime.output_dir / "legacy.pt"
            torch.save(state, source)
            config = Config.model_validate(state["config"])
            assert not config.decentralized.adaptive_consensus.exclude_embeddings
    else:
        original = PackedAdamW.step
        updates = 0

        def interrupt(self):
            nonlocal updates
            original(self)
            updates += 1
            if updates == (1 if resume_at == "before" else 4):
                os.kill(os.getpid(), signal.SIGTERM)

        monkeypatch.setattr(PackedAdamW, "step", interrupt)
        assert train(config)["status"] == "interrupted"
        monkeypatch.setattr(PackedAdamW, "step", original)
        source = config.runtime.output_dir / "latest.pt"
    config.runtime.output_dir = cache_dir.parent / "resumed"
    assert train(config, source)["final_validation"]["loss"] == expected["final_validation"]["loss"]
    a = load_training_checkpoint(baseline.runtime.output_dir / "final.pt")
    b = load_training_checkpoint(config.runtime.output_dir / "final.pt")
    for key in ("model", "optimizer", "rng", "loader", "step", "cursor"):
        if key == "rng":
            assert torch.equal(a[key]["torch"], b[key]["torch"])
            assert a[key]["python"] == b[key]["python"]
            assert (a[key]["numpy"][1] == b[key]["numpy"][1]).all()
            assert a[key]["numpy"][0] == b[key]["numpy"][0]
            assert a[key]["numpy"][2:] == b[key]["numpy"][2:]
            assert_nested_equal(a[key]["cuda"], b[key]["cuda"])
        else:
            assert_nested_equal(a[key], b[key])
    schedule = adaptive_consensus_schedule(
        config, training_boundaries(config, config.model.parameter_count)
    )
    for directory in (baseline.runtime.output_dir, config.runtime.output_dir):
        rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        for row in rows:
            if row["event"] == "train":
                assert row["mixing_gamma"] == schedule.gamma(row["mixing_step"], row["lr"])
                assert row["embedding_mixing_gamma"] == (
                    1 if exclude_embeddings else row["mixing_gamma"]
                )
    meta = json.loads((config.runtime.output_dir / "environment-resume.json").read_text())
    assert meta["adaptive_consensus"] == asdict(schedule) | {
        "exclude_embeddings": exclude_embeddings
    }
    config.decentralized.adaptive_consensus.exclude_embeddings = not exclude_embeddings
    config.runtime.output_dir = cache_dir.parent / "incompatible-exclusion"
    with pytest.raises(ValueError, match="incompatible"):
        train(config, source)
    config.decentralized.adaptive_consensus.exclude_embeddings = exclude_embeddings
    config.decentralized.adaptive_consensus.p += 1
    config.runtime.output_dir = cache_dir.parent / "incompatible"
    with pytest.raises(ValueError, match="incompatible"):
        train(config, source)
