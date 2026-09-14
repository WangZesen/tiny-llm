import json
import os
import signal
from pathlib import Path

import numpy as np
import pytest
import torch

from tiny_llm.config import Config, CosineScheduleConfig, WSDScheduleConfig
from tiny_llm.data import TokenCache
from tiny_llm.model import Llama, token_losses
from tiny_llm.runtime import rng_state, setup_runtime
from tiny_llm.train import evaluate, learning_rate, train


def test_accumulation_matches_full(tiny_config: Config):
    torch.manual_seed(42)
    full = Llama(tiny_config.model, "reference").double()
    accumulated = Llama(tiny_config.model, "reference").double()
    accumulated.load_state_dict(full.state_dict())
    x, y = torch.randint(17, (3, 4)), torch.randint(17, (3, 4))
    token_losses(full(x), y).mean().backward()
    # Uneven microbatches exercise accumulation remainders.
    for offset, count in ((0, 2), (2, 1)):
        (
            token_losses(accumulated(x[offset : offset + count]), y[offset : offset + count]).sum()
            / y.numel()
        ).backward()
    for a, b in zip(full.parameters(), accumulated.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-10, atol=1e-12)


def test_schedule(tiny_config: Config):
    tiny_config.lr_schedule.warmup_steps = 2
    assert learning_rate(tiny_config, 0, 640) == 0
    assert learning_rate(tiny_config, 16, 640) == tiny_config.optimizer.lr * 0.5
    assert learning_rate(tiny_config, 32, 640) == tiny_config.optimizer.lr
    assert learning_rate(tiny_config, 640, 640) == pytest.approx(tiny_config.optimizer.lr * 0.1)


@pytest.mark.parametrize("full", [False, True])
def test_evaluation_state_and_weighting(tiny_config: Config, cache_dir: Path, full):
    tiny_config.evaluation.subset_blocks = 8
    tiny_config.data.shuffle_group_size = 1
    setup_runtime(tiny_config)
    model = Llama(tiny_config.model, "reference")
    cache = TokenCache(cache_dir)
    before = rng_state()
    result = evaluate(model, cache, tiny_config, torch.device("cpu"), full=full)
    after = rng_state()
    assert model.training
    assert before["python"] == after["python"]
    np.testing.assert_array_equal(before["numpy"][1], after["numpy"][1])
    assert torch.equal(before["torch"], after["torch"])
    # Subset sampling uses complete blocks; full validation includes the padded tail.
    assert result["tokens"] == (29 if full else 28)
    for batch_size in (1, 3, 128):
        tiny_config.evaluation.batch_size = batch_size
        repeated = evaluate(model, cache, tiny_config, torch.device("cpu"), full=full)
        assert repeated["tokens"] == result["tokens"]
        assert repeated["loss"] == pytest.approx(result["loss"], abs=1e-7)
    with pytest.raises(InterruptedError):
        evaluate(
            model, cache, tiny_config, torch.device("cpu"), full=full, should_stop=lambda: True
        )
    assert model.training
    assert torch.equal(before["torch"], rng_state()["torch"])


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("grad_clip", [1.0, None])
@pytest.mark.parametrize("lr_schedule", ["cosine", "wsd"])
def test_offline_train_and_resume(
    tiny_config: Config,
    cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    device,
    grad_clip,
    lr_schedule,
):
    if lr_schedule == "wsd":
        tiny_config.lr_schedule = WSDScheduleConfig(decay_fraction=0.5)
    tiny_config.lr_schedule.warmup_steps = 3
    tiny_config.training.checkpoint_policy = "interval"
    tiny_config.data.shuffle_group_size = 1
    tiny_config.optimizer.grad_clip = grad_clip
    tiny_config.runtime.device = device
    tiny_config.runtime.amp = device != "cpu"
    tiny_config.runtime.deterministic = device == "cpu"
    complete = tiny_config.model_copy(deep=True)
    complete.runtime.output_dir = cache_dir.parent / "complete"
    result = train(complete)
    assert result["tokens"] == 64 and result["epochs"] == 2
    assert result["final_validation"]["tokens"] == 29
    rows = [
        json.loads(line)
        for line in (complete.runtime.output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["lr"] / tiny_config.optimizer.lr for row in rows if row["event"] == "train"] == (
        pytest.approx([1 / 3, 2 / 3, 1, 0 if lr_schedule == "wsd" else 0.1])
    )
    tiny_config.data.prefetch = False
    resumed = train(tiny_config, complete.runtime.output_dir / "epoch-001.pt")
    assert resumed["final_validation"]["loss"] == pytest.approx(
        result["final_validation"]["loss"], abs=1e-4 if device != "cpu" else 0
    )
    a = torch.load(complete.runtime.output_dir / "final.pt", weights_only=False)
    b = torch.load(tiny_config.runtime.output_dir / "final.pt", weights_only=False)
    assert a["version"] == b["version"] == 2
    assert a["loader"] == b["loader"]
    assert a["loader"]["version"] == 2
    assert a["loader"]["cursor"] == a["cursor"] == 16
    assert all(isinstance(v, (str, int)) for v in a["loader"].values())
    for key in a["model"]:
        if device == "cpu":
            assert torch.equal(a["model"][key], b["model"][key])
        else:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=1e-3, atol=1e-5)
    assert (tiny_config.runtime.output_dir / "epoch-002.safetensors").exists()
    assert json.loads((tiny_config.runtime.output_dir / "best.json").read_text())["epoch"] in (1, 2)
    changed = tiny_config.model_copy(deep=True)
    changed.optimizer.lr *= 2
    with pytest.raises(ValueError, match="incompatible"):
        train(changed, complete.runtime.output_dir / "epoch-001.pt")
    changed = tiny_config.model_copy(deep=True)
    changed.training.tokens_per_parameters *= 2
    with pytest.raises(ValueError, match="incompatible"):
        train(changed, complete.runtime.output_dir / "epoch-001.pt")
    changed = tiny_config.model_copy(deep=True)
    changed.lr_schedule = CosineScheduleConfig() if lr_schedule == "wsd" else WSDScheduleConfig()
    with pytest.raises(ValueError, match="incompatible"):
        train(changed, complete.runtime.output_dir / "epoch-001.pt")
    if lr_schedule == "wsd":
        changed.lr_schedule = WSDScheduleConfig(warmup_steps=3, decay_fraction=0.25)
        with pytest.raises(ValueError, match="incompatible"):
            train(changed, complete.runtime.output_dir / "epoch-001.pt")


def test_throughput_excludes_startup_evaluation_and_checkpoints(
    tiny_config: Config, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    import time
    from types import SimpleNamespace

    import tiny_llm.train as module

    tiny_config.training.checkpoint_policy = "interval"
    offset = 0.0
    real_time = time.monotonic
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: real_time() + offset))
    original_checkpoint = module.atomic_checkpoint
    original_evaluate = module.evaluate
    original_subset = TokenCache.validation_subset

    def checkpoint(*args, **kwargs):
        nonlocal offset
        result = original_checkpoint(*args, **kwargs)
        offset += 10
        return result

    def evaluate(*args, **kwargs):
        nonlocal offset
        result = original_evaluate(*args, **kwargs)
        offset += 20
        return result

    def subset(*args, **kwargs):
        nonlocal offset
        result = original_subset(*args, **kwargs)
        offset += 30
        return result

    monkeypatch.setattr(module, "atomic_checkpoint", checkpoint)
    monkeypatch.setattr(module, "evaluate", evaluate)
    monkeypatch.setattr(TokenCache, "validation_subset", subset)
    result = train(tiny_config)
    assert result["training_seconds"] < 10
    assert result["training_elapsed_seconds"] >= 80
    assert result["seconds_this_session"] > result["training_elapsed_seconds"]
    assert result["training_tokens_per_second"] > result["elapsed_tokens_per_second"]
    rows = [
        json.loads(line)
        for line in (tiny_config.runtime.output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert all(row["training_seconds"] < 10 for row in rows if row["event"] == "train")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_microbatch_remainder_resume(
    tiny_config: Config, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    tiny_config.runtime.compile = True
    # A full four-sequence global batch is accumulated as microbatches of three and one.
    tiny_config.training.micro_batch_size = 3
    test_offline_train_and_resume(tiny_config, cache_dir, monkeypatch, "cuda:0", 1.0, "cosine")


@pytest.mark.parametrize("workers", [1, 4, 8])
@pytest.mark.parametrize("interrupt", [False, True])
@pytest.mark.parametrize("policy", ["final", "none"])
def test_minimal_checkpoint_policy(
    tiny_config: Config,
    cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    workers,
    interrupt,
    policy,
):
    import tiny_llm.train as module
    from tiny_llm.config import Config
    from tiny_llm.train import evaluate_checkpoint, recipe_identity

    raw = tiny_config.model_dump()
    raw["training"]["batch_tokens"] = 32
    raw["training"]["micro_batch_size"] = 8 // workers
    if workers > 1:
        raw["decentralized"] = dict(num_models=workers, topology="one_peer_ring")
    config = Config.model_validate(raw)
    assert config.training.checkpoint_policy == "final"
    config.training.checkpoint_policy = policy
    identity = recipe_identity(config, TokenCache(cache_dir))
    other = config.model_copy(deep=True)
    other.training.checkpoint_policy = "interval"
    other.training.checkpoint_epochs = [2]
    assert recipe_identity(other, TokenCache(cache_dir)) == identity
    saved = []
    original = module.atomic_checkpoint

    def checkpoint(path, state):
        saved.append(path.name)
        return original(path, state)

    monkeypatch.setattr(module, "atomic_checkpoint", checkpoint)
    if interrupt:
        evaluate = module.evaluate

        def interrupted(*args, **kwargs):
            os.kill(os.getpid(), signal.SIGTERM)
            return evaluate(*args, **kwargs)

        monkeypatch.setattr(module, "evaluate", interrupted)
    result = train(config)
    output = config.runtime.output_dir
    assert not list(output.rglob("*.safetensors"))
    assert not (output / "latest.pt").exists()
    assert saved == ([] if interrupt or policy == "none" else ["final.pt"])
    if policy == "none":
        assert not list(output.rglob("*.pt"))
    if interrupt:
        assert result["status"] == "interrupted"
    else:
        assert result["status"] == "complete"
        assert json.loads((output / "best.json").read_text())["weights"] is None
        assert result["final_validation"]["validation_complete"]
        assert json.loads((output / "result.json").read_text()) == result
        if policy == "none":
            return
        evaluated = evaluate_checkpoint(config, output / "final.pt", full=True)
        assert evaluated["loss"] == pytest.approx(result["final_validation"]["loss"])
        resumed = train(config, output / "final.pt")
        assert resumed["final_validation"]["loss"] == pytest.approx(evaluated["loss"])


@pytest.mark.parametrize("packed", [False, True])
def test_longer_training_preserves_early_batches(
    tiny_config: Config, cache_dir: Path, monkeypatch: pytest.MonkeyPatch, packed
):
    from helpers import set_budget

    import tiny_llm.train as module
    from tiny_llm.config import DecentralizedConfig

    if packed:
        tiny_config.decentralized = DecentralizedConfig(num_models=2)
    tiny_config.training.checkpoint_policy = "none"
    original = module.loss_function
    batches = []

    def record_loss(model, config, device):
        loss = original(model, config, device)

        def compute(x, y):
            batches.append((x.clone(), y.clone()))
            return loss(x, y)

        return compute

    monkeypatch.setattr(module, "loss_function", record_loss)
    set_budget(tiny_config, 32, 32)
    assert train(tiny_config)["status"] == "complete"
    short_batches = batches.copy()
    batches.clear()
    tiny_config.runtime.output_dir = cache_dir.parent / "long"
    tiny_config.data.prefetch = False
    set_budget(tiny_config, 64, 32)
    assert train(tiny_config)["status"] == "complete"
    assert len(batches) == 2 * len(short_batches)
    for short, long in zip(short_batches, batches[: len(short_batches)], strict=True):
        for actual, expected in zip(short, long, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
