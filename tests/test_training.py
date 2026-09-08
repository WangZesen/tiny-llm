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


def test_evaluation_state_and_weighting(tiny_config, cache_dir):
    setup_runtime(tiny_config)
    model = Llama(tiny_config.model, "reference")
    cache = TokenCache(cache_dir)
    before = rng_state()
    result = evaluate(model, cache, tiny_config, torch.device("cpu"), full=True)
    after = rng_state()
    assert model.training
    assert before["python"] == after["python"]
    np.testing.assert_array_equal(before["numpy"][1], after["numpy"][1])
    assert torch.equal(before["torch"], after["torch"])
    assert result["tokens"] == 29
    tiny_config.evaluation.batch_size = 1
    repeated = evaluate(model, cache, tiny_config, torch.device("cpu"), full=True)
    assert repeated["loss"] == pytest.approx(result["loss"], abs=1e-7)
    with pytest.raises(InterruptedError):
        evaluate(
            model, cache, tiny_config, torch.device("cpu"), full=True, should_stop=lambda: True
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
    resumed = train(tiny_config, tiny_config.runtime.output_dir / "latest.pt")
    assert resumed["final_validation"]["loss"] == pytest.approx(
        result["final_validation"]["loss"], abs=1e-4 if device != "cpu" else 0
    )
    a = torch.load(complete.runtime.output_dir / "final.pt", weights_only=False)
    b = torch.load(tiny_config.runtime.output_dir / "final.pt", weights_only=False)
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
