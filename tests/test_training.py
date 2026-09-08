import json
import os
import signal

import numpy as np
import pytest
import torch

from tiny_llm.data import TokenCache
from tiny_llm.model import Llama, token_losses
from tiny_llm.runtime import rng_state, setup_runtime
from tiny_llm.train import evaluate, learning_rate, train


def test_accumulation_matches_full(tiny_config):
    torch.manual_seed(42)
    full = Llama(tiny_config.model, "reference").double()
    accumulated = Llama(tiny_config.model, "reference").double()
    accumulated.load_state_dict(full.state_dict())
    x, y = torch.randint(17, (3, 4)), torch.randint(17, (3, 4))
    token_losses(full(x), y).mean().backward()
    # Uneven microbatches exercise the epoch-boundary case.
    for offset, count in ((0, 2), (2, 1)):
        (
            token_losses(accumulated(x[offset : offset + count]), y[offset : offset + count]).sum()
            / y.numel()
        ).backward()
    for a, b in zip(full.parameters(), accumulated.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-10, atol=1e-12)


def test_schedule(tiny_config):
    assert learning_rate(tiny_config, 0, 1000) == 0
    assert learning_rate(tiny_config, 50, 1000) == tiny_config.optimizer.lr
    assert learning_rate(tiny_config, 1000, 1000) == pytest.approx(tiny_config.optimizer.lr * 0.1)


@pytest.mark.parametrize("full", [False, True])
def test_evaluation_state_and_weighting(tiny_config, cache_dir, full):
    tiny_config.evaluation.subset_blocks = 8
    tiny_config.data.buffer_size_mib = 24 / 2**20
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
def test_offline_train_and_resume(tiny_config, cache_dir, monkeypatch, device):
    import tiny_llm.train as module

    tiny_config.data.buffer_size_mib = 24 / 2**20
    tiny_config.runtime.device = device
    tiny_config.runtime.amp = device != "cpu"
    tiny_config.runtime.deterministic = device == "cpu"
    complete = tiny_config.model_copy(deep=True)
    complete.runtime.output_dir = cache_dir.parent / "complete"
    result = train(complete)
    assert result["tokens"] == 64 and result["epochs"] == 2
    assert result["final_validation"]["tokens"] == 29
    original = module.make_optimizer

    def interrupted_optimizer(*args):
        optimizer = original(*args)
        original_step = optimizer.step
        calls = 0

        def step(*args, **kwargs):
            nonlocal calls
            value = original_step(*args, **kwargs)
            calls += 1
            if calls == 1:
                os.kill(os.getpid(), signal.SIGTERM)
            return value

        optimizer.step = step
        return optimizer

    monkeypatch.setattr(module, "make_optimizer", interrupted_optimizer)
    interrupted = train(tiny_config)
    assert interrupted["status"] == "interrupted"
    monkeypatch.setattr(module, "make_optimizer", original)
    tiny_config.data.prefetch = False
    resumed = train(tiny_config, tiny_config.runtime.output_dir / "latest.pt")
    assert resumed["final_validation"]["loss"] == pytest.approx(
        result["final_validation"]["loss"], abs=1e-4 if device != "cpu" else 0
    )
    a = torch.load(complete.runtime.output_dir / "final.pt", weights_only=False)
    b = torch.load(tiny_config.runtime.output_dir / "final.pt", weights_only=False)
    assert a["version"] == b["version"] == 2
    assert a["loader"] == b["loader"]
    assert a["loader"]["version"] == 1
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
        train(changed, tiny_config.runtime.output_dir / "latest.pt")


def test_throughput_excludes_startup_evaluation_and_checkpoints(
    tiny_config, cache_dir, monkeypatch
):
    import time
    from types import SimpleNamespace

    import tiny_llm.train as module

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
def test_compiled_partial_epoch_resume(tiny_config, cache_dir, monkeypatch):
    tiny_config.runtime.compile = True
    # Nine blocks in epoch one exercise one-block and uneven accumulated updates.
    tiny_config.training.epoch_tokens = 36
    test_offline_train_and_resume(tiny_config, cache_dir, monkeypatch, "cuda:0")
