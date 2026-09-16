import copy
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from helpers import assert_nested_equal

from tiny_llm.config import Config, load_config, save_config
from tiny_llm.data import TokenCache, fingerprint
from tiny_llm.model import Llama
from tiny_llm.optimizers import AccumAdamW
from tiny_llm.packed import PackedLlama
from tiny_llm.packed_optimizer import PackedAccumAdamW
from tiny_llm.train import make_optimizer, recipe_identity


@pytest.mark.parametrize("s", [1, 2, 4])
@pytest.mark.parametrize("foreach", [False, True])
def test_accumulated_history_reference(s, foreach):
    """Reconstruct moments from complete gradient history, independently of state updates."""
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    opt = AccumAdamW(
        [p], betas=(0.7, 0.8), eps=0.03, weight_decay=0.2, accum_iter=s, foreach=foreach
    )
    history = [[g, -0.3 * g + 0.1] for g in (2, -2, 1, 3, -4, 0, 0.5, -1, 2)]
    expected = [1.0, -2.0]
    for t, grad in enumerate(history, 1):
        lr = 0.01 / t if t < len(history) else 0.0
        opt.param_groups[0]["lr"] = lr
        p.grad = torch.tensor(grad, dtype=p.dtype)
        opt.step()
        for coordinate in range(2):
            windows = [
                sum(row[coordinate] for row in history[j : j + s]) / s
                for j in range(0, ((t - 1) // s) * s, s)
            ]
            m = 0.3 * (
                grad[coordinate]
                + sum(0.7 ** (len(windows) - j) * mean for j, mean in enumerate(windows))
            )
            v = 0.2 * (
                grad[coordinate] ** 2
                + sum(0.8 ** (len(windows) - j) * mean**2 for j, mean in enumerate(windows))
            )
            k = math.ceil(t / s)
            direction = (m / (1 - 0.7**k)) / (math.sqrt(v / (1 - 0.8**k)) + 0.03)
            expected[coordinate] = (1 - lr * 0.2) * expected[coordinate] - lr * direction
        torch.testing.assert_close(p, torch.tensor(expected, dtype=p.dtype), rtol=1e-14, atol=1e-14)
        if s == 2 and t == 2:
            # The first coordinate cancels exactly; mean-of-squares would be nonzero.
            assert opt.state[p]["exp_avg_sq"][0] == 0
        saved_buffer = opt.state[p]["accum_grad"].clone()
        opt.zero_grad()
        assert torch.equal(saved_buffer, opt.state[p]["accum_grad"])


@pytest.mark.parametrize("foreach", [False, True])
def test_window_one_matches_adamw(foreach):
    torch.manual_seed(7)
    parameters = [
        torch.nn.Parameter(torch.randn(2, 3, dtype=torch.float64)),
        torch.nn.Parameter(torch.randn(3, dtype=torch.float64)),
    ]
    other = [torch.nn.Parameter(p.detach().clone()) for p in parameters]

    def groups(params):
        return [
            {"params": params[:1], "weight_decay": 0.3},
            {"params": params[1:], "weight_decay": 0.0},
        ]

    args: dict[str, Any] = dict(lr=0.02, betas=(0.8, 0.93), eps=0.002, foreach=foreach)
    actual = AccumAdamW(groups(parameters), **args, accum_iter=1)
    expected = torch.optim.AdamW(groups(other), **args)
    for step in range(7):
        for opt in (actual, expected):
            for group in opt.param_groups:
                group["lr"] = 0.02 * (6 - step) / 6
        for i, (p, q) in enumerate(zip(parameters, other, strict=True)):
            p.grad = None if step == 2 and i == 1 else torch.randn_like(p)
            q.grad = None if p.grad is None else p.grad.clone()
        actual.step()
        expected.step()
        for p, q in zip(parameters, other, strict=True):
            torch.testing.assert_close(p, q, rtol=1e-13, atol=1e-14)
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(actual.state[p][key], expected.state[q][key])


def test_foreach_missing_gradients_and_resume():
    torch.manual_seed(3)
    p = [torch.nn.Parameter(torch.randn(3, dtype=torch.float64)) for _ in range(3)]
    q = [torch.nn.Parameter(x.detach().clone()) for x in p]
    a = AccumAdamW(p, accum_iter=3, foreach=True)
    b = AccumAdamW(q, accum_iter=3, foreach=False)
    for step in range(10):
        for i, (x, y) in enumerate(zip(p, q, strict=True)):
            x.grad = None if (step + i) % 3 == 0 else torch.randn_like(x)
            y.grad = x.grad
        a.step()
        b.step()
        for x, y in zip(p, q, strict=True):
            torch.testing.assert_close(x, y, rtol=1e-14, atol=1e-14)
            for key in a.state[x]:
                torch.testing.assert_close(a.state[x][key], b.state[y][key])
        if step == 4:
            saved = copy.deepcopy(a.state_dict())
            a = AccumAdamW(p, accum_iter=3, foreach=True)
            a.load_state_dict(saved)
            assert all(state["step"].device.type == "cpu" for state in a.state.values())
    before = copy.deepcopy(a.state_dict())
    weights = [x.clone() for x in p]
    a.zero_grad()
    a.step()
    assert_nested_equal(before, a.state_dict())
    for x, y in zip(p, weights, strict=True):
        assert torch.equal(x, y)


@pytest.mark.parametrize(
    "damage", ["identity", "window", "shape", "counter", "nan", "negative", "boundary"]
)
def test_reject_invalid_state_without_mutating(damage):
    p = torch.nn.Parameter(torch.ones(2))
    opt = AccumAdamW([p])
    p.grad = torch.ones_like(p)
    opt.step()
    before = copy.deepcopy(opt.state_dict())
    saved = copy.deepcopy(before)
    state = saved["state"][0]
    if damage == "identity":
        saved["optimizer"] = "adamw"
    elif damage == "window":
        saved["param_groups"][0]["accum_iter"] = 2
    elif damage == "shape":
        state["accum_grad"] = torch.zeros(3)
    elif damage == "counter":
        state["step"] = torch.tensor(1.5)
    elif damage == "nan":
        state["accum_grad"][0] = float("nan")
    elif damage == "negative":
        state["exp_avg_sq"][0] = -1
    else:
        state["step"].fill_(4)
    with pytest.raises(ValueError):
        opt.load_state_dict(saved)
    assert_nested_equal(before, opt.state_dict())


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_accumulation_window(value):
    with pytest.raises(ValueError, match="accum_iter"):
        Config.model_validate({"optimizer": {"name": "accumadamw", "accum_iter": value}})
    with pytest.raises(ValueError, match="accum_iter"):
        AccumAdamW([torch.nn.Parameter(torch.ones(1))], accum_iter=value)


def test_config_factory_and_identity(tiny_config: Config, cache_dir: Path, tmp_path: Path):
    cache = TokenCache(cache_dir)
    old = tiny_config.model_dump(mode="json")
    old["optimizer"].pop("name")
    old["optimizer"].pop("accum_iter")
    for key in ("checkpoint_policy", "checkpoint_epochs", "save_epoch_training_state"):
        old["training"].pop(key)
    for key in ("output_dir", "device", "cpu_threads"):
        old["runtime"].pop(key)
    for key in ("cache_dir", "prefetch", "prepare_workers"):
        old["data"].pop(key)
    from tiny_llm.data import BufferedTokenLoader

    old.update(
        cache_identity=cache.manifest["identity"], loader_version=BufferedTokenLoader.VERSION
    )
    legacy = fingerprint(old)
    assert recipe_identity(tiny_config, cache) == legacy
    tiny_config.optimizer.accum_iter = 7
    assert recipe_identity(tiny_config, cache) == legacy
    assert isinstance(
        make_optimizer(Llama(tiny_config.model), tiny_config, torch.device("cpu")),
        torch.optim.AdamW,
    )
    tiny_config.optimizer.name = "accumadamw"
    accum_identity = recipe_identity(tiny_config, cache)
    assert accum_identity != legacy
    tiny_config.optimizer.accum_iter = 4
    assert recipe_identity(tiny_config, cache) != accum_identity
    assert isinstance(
        make_optimizer(Llama(tiny_config.model), tiny_config, torch.device("cpu")), AccumAdamW
    )
    assert isinstance(
        make_optimizer(PackedLlama(tiny_config.model, 2), tiny_config, torch.device("cpu")),
        PackedAccumAdamW,
    )
    saved = tmp_path / "config.yaml"
    save_config(tiny_config, saved)
    assert load_config(saved) == tiny_config
    base = load_config("configs/20m.yaml")
    config = load_config(["configs/20m.yaml", "configs/accumadamw.yaml"])
    assert config.optimizer.name == "accumadamw" and config.optimizer.accum_iter == 4
    assert config.optimizer.beta2 == base.optimizer.beta2
    assert config.optimizer.lr == base.optimizer.lr
    with pytest.raises(ValueError):
        load_config(overrides=["optimizer.name=unknown"])


def test_packed_buffer_aliases_and_restore(tiny_config: Config):
    tiny_config.optimizer.name = "accumadamw"
    model = PackedLlama(tiny_config.model, 2).double()
    opt = PackedAccumAdamW(model, tiny_config)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    assert opt.accum_grad_storage is not None
    buffer = opt.accum_grad_storage.clone()
    model.mix_("complete", 0)
    opt.zero_grad()
    assert torch.equal(buffer, opt.accum_grad_storage)
    saved = copy.deepcopy(opt.state_dict())
    restored = PackedAccumAdamW(model, tiny_config)
    restored.load_state_dict(saved)
    assert restored.accum_grad_storage is not None
    for worker, parameters in enumerate(restored.local_parameters):
        for entry, p in zip(model.layout, parameters, strict=True):
            arena = model.parameter_view(restored.accum_grad_storage, entry)[worker]
            assert restored.optimizers[worker].state[p]["accum_grad"].data_ptr() == arena.data_ptr()
    assert_nested_equal(saved, restored.state_dict())
    broken = copy.deepcopy(saved)
    broken["accum_grad_storage"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="accum_grad"):
        restored.load_state_dict(broken)
    assert_nested_equal(saved, restored.state_dict())
