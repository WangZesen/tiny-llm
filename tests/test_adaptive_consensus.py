import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from helpers import assert_nested_equal
from test_packed import assert_storage, matrix

from tiny_llm.checkpoints import load_training_checkpoint
from tiny_llm.config import AdaptiveConsensusConfig, Config
from tiny_llm.data import TokenCache, training_boundaries
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
    raw["training"].update(
        batch_tokens=32, tokens_per_parameters=128 / 800, epoch_tokens_per_parameters=32 / 800
    )
    return Config.model_validate(raw)


def test_config_validation(tiny_config: Config):
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
def test_realized_schedule(tiny_config: Config, start_frac, warmup, p):
    config = adaptive_config(tiny_config, start_frac, p)
    assert config.decentralized is not None and config.decentralized.adaptive_consensus is not None
    config.optimizer.warmup_fraction = warmup
    before = torch.get_rng_state()
    schedule = adaptive_consensus_schedule(
        config, training_boundaries(config, config.model.parameter_count)
    )
    assert schedule is not None
    assert torch.equal(before, torch.get_rng_state())
    assert schedule.total_steps == 4
    expected_start = {0: 0, 0.3: 2, 0.6: 3, 1: 4, 0.99: 4}[start_frac]
    assert schedule.start_step == expected_start
    rates = [learning_rate(config, tokens, 128) for tokens in (32, 64, 96, 128)]
    assert schedule.lr_max == (max(rates[expected_start:]) if expected_start < 4 else None)
    for r, lr in enumerate(rates):
        expected = 1 if r < expected_start or p == 0 else (lr / max(rates[expected_start:])) ** p
        assert schedule.gamma(r, lr) == expected
    if start_frac == 0 and warmup == 0.5:
        assert schedule.lr_max is not None
        assert schedule.lr_max > rates[0]  # Maximum occurs after activation.


def test_zero_lr_and_empty_window(tiny_config: Config):
    config = adaptive_config(tiny_config, 0)
    assert config.decentralized is not None and config.decentralized.adaptive_consensus is not None
    config.optimizer.min_lr_ratio = 0
    with pytest.raises(ValueError, match="positive maximum LR"):
        adaptive_consensus_schedule(config, [8])
    config.decentralized.adaptive_consensus.p = 0
    zero = adaptive_consensus_schedule(config, [8])
    assert zero is not None and zero.gamma(0, 0) == 1
    config.decentralized.adaptive_consensus.p = 1
    config.decentralized.adaptive_consensus.start_frac = 0.1
    schedule = adaptive_consensus_schedule(config, [8])
    assert schedule is not None
    assert schedule.start_step == 1 and schedule.lr_max is None
    assert schedule.gamma(0, 0) == 1


@pytest.mark.parametrize("topology", ["complete", "one_peer_ring", "one_peer_exponential"])
@pytest.mark.parametrize("n", [1, 3, 4])
def test_weighted_mixing(tiny_config: Config, topology, n):
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
    gradients = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    assert len(gradients) == len(tuple(model.parameters()))
    for step, gamma in enumerate((0, 0.2, 0.75, 1)):
        model.parameter_storage.copy_(before)
        pointer = model.parameter_storage.data_ptr()
        expected = (
            gamma * matrix(n, topology, step) + (1 - gamma) * torch.eye(n, dtype=before.dtype)
        ) @ before
        model.mix_(topology, step, gamma)
        torch.testing.assert_close(model.parameter_storage, expected, rtol=1e-14, atol=1e-14)
        assert model.parameter_storage.data_ptr() == pointer
        if gamma == 0:
            assert torch.equal(model.parameter_storage, before)
        if gamma == 1:
            if n == 1:
                legacy = before
            elif topology == "complete":
                legacy = before.mean(0).expand_as(before)
            else:
                offset = (
                    (1 if step % 2 == 0 else -1)
                    if topology == "one_peer_ring"
                    else 1 << (step % (n - 1).bit_length())
                )
                legacy = (before[(torch.arange(n) - offset) % n] + before) * 0.5
            assert torch.equal(model.parameter_storage, legacy)
        assert_storage(model, opt)
        assert torch.equal(opt.first_moment_storage, moments[0])
        assert torch.equal(opt.second_moment_storage, moments[1])
        for local in opt.optimizers:
            assert all(state["step"].item() == 0 for state in local.state.values())
        for parameter, grad in zip(model.parameters(), gradients, strict=True):
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, grad)
    for gamma in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="gamma"):
            model.mix_(topology, 0, gamma)


def test_identity_and_benchmark_guards(tiny_config: Config, cache_dir: Path, tmp_path: Path):
    config = adaptive_config(tiny_config)
    assert config.decentralized is not None and config.decentralized.adaptive_consensus is not None
    cache = TokenCache(cache_dir)
    enabled = recipe_identity(config, cache)
    config.decentralized.adaptive_consensus = None
    disabled = recipe_identity(config, cache)
    assert disabled != enabled
    config = adaptive_config(tiny_config)
    assert config.decentralized is not None and config.decentralized.adaptive_consensus is not None
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="training only"):
        benchmark_packed(config, destination)
    for execution in ("packed", "sequential"):
        with pytest.raises(ValueError, match="training only"):
            benchmark_packed_worker(config, destination, execution)
    assert not destination.exists()


@pytest.mark.parametrize("resume_epoch", [1, 3])
@pytest.mark.parametrize("scheme", ["awc", "atc"])
def test_adaptive_resume(tiny_config: Config, cache_dir: Path, resume_epoch, scheme):
    config = adaptive_config(tiny_config)
    assert config.decentralized is not None and config.decentralized.adaptive_consensus is not None
    config.decentralized.scheme = scheme
    config.training.checkpoint_policy = "interval"
    config.runtime.deterministic = True
    config.optimizer.warmup_fraction = 0.6
    baseline = config.model_copy(deep=True)
    baseline.runtime.output_dir = cache_dir.parent / "baseline"
    expected = train(baseline)
    source = baseline.runtime.output_dir / f"epoch-{resume_epoch:03d}.pt"
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
    assert schedule is not None
    for directory in (baseline.runtime.output_dir, config.runtime.output_dir):
        rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        for row in rows:
            if row["event"] == "train":
                assert row["mixing_gamma"] == schedule.gamma(row["mixing_step"], row["lr"])
    meta = json.loads((config.runtime.output_dir / "environment-resume.json").read_text())
    assert meta["adaptive_consensus"] == asdict(schedule)
    config.decentralized.adaptive_consensus.p += 1
    config.runtime.output_dir = cache_dir.parent / "incompatible"
    with pytest.raises(ValueError, match="incompatible"):
        train(config, source)
