import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from helpers import assert_nested_equal
from pydantic import ValidationError

from tiny_llm.benchmark import benchmark_worker
from tiny_llm.config import Config, load_config
from tiny_llm.data import BufferedTokenLoader, TokenCache, fingerprint
from tiny_llm.optimizer import AccumAdamW
from tiny_llm.packed import PackedLlama
from tiny_llm.packed_benchmark import benchmark_packed_worker
from tiny_llm.packed_optimizer import PackedOptimizer
from tiny_llm.runtime import clip_grad_norm_
from tiny_llm.train import make_optimizer, recipe_identity


@pytest.mark.parametrize("s", [1, 2, 4])
@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("betas", [(0.6, 0.85), (0.0, 0.0)])
@pytest.mark.parametrize("grad_clip", [None, 0.5])
def test_history_reference(s, foreach, betas, grad_clip):
    """Compute moments from explicit weighted history, independently of optimizer state."""
    p = torch.nn.Parameter(torch.tensor([2.0, -3.0], dtype=torch.float64))
    opt = AccumAdamW([p], betas=betas, eps=0.02, accumulation_steps=s, foreach=foreach)
    expected = p.detach().numpy().copy()
    history = np.array([[1, -2], [-1, 4], [3, 1], [0, -3], [2, 5], [-4, 0], [1, 2], [3, -1]])
    clipped_history = history.astype(np.float64)
    if grad_clip is not None:
        clipped_history *= np.minimum(
            1.0, grad_clip / (np.linalg.norm(clipped_history, axis=1, keepdims=True) + 1e-6)
        )
    beta1, beta2 = betas

    def window_mean(j):
        return clipped_history[j * s : (j + 1) * s].mean(0)

    for t, g in enumerate(clipped_history, 1):
        lr = 0.03 / t
        opt.param_groups[0]["lr"] = lr
        completed = (t - 1) // s
        means = [window_mean(j) for j in range(completed)]
        m = (1 - beta1) * (g + sum(beta1 ** (completed - j) * v for j, v in enumerate(means)))
        v = (1 - beta2) * (g**2 + sum(beta2 ** (completed - j) * v**2 for j, v in enumerate(means)))
        expected *= 1 - lr * 0.1
        expected -= (
            lr
            * (m / (1 - beta1 ** (completed + 1)))
            / (np.sqrt(v / (1 - beta2 ** (completed + 1))) + 0.02)
        )
        p.grad = torch.tensor(history[t - 1], dtype=p.dtype)
        clip_grad_norm_([p], grad_clip)
        opt.step()
        np.testing.assert_allclose(p.detach().numpy(), expected, rtol=1e-13, atol=1e-14)
        state = opt.state[p]
        full = t // s
        for key, beta, power in (("exp_avg", beta1, 1), ("exp_avg_sq", beta2, 2)):
            moment = sum(
                (1 - beta) * beta ** (full - 1 - j) * window_mean(j) ** power for j in range(full)
            )
            np.testing.assert_allclose(state[key].numpy(), moment, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(
            state["accum_grad"].numpy(), clipped_history[full * s : t].sum(0) / s
        )
        torch.testing.assert_close(p.grad, torch.tensor(g, dtype=p.dtype), rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("s", [2, 4])
def test_accumulate_clipped_gradients_across_groups_before_cancellation(foreach, s):
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    q = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    opt = AccumAdamW(
        [{"params": [p], "weight_decay": 0.2}, {"params": [q], "weight_decay": 0}],
        betas=(0.6, 0.85),
        accumulation_steps=s,
        foreach=foreach,
    )
    raw = np.array([[6.0, 100.0, 3.0], [0.0, -100.0, 5.0], [-3.0, 0.0, 4.0], [5.0, 12.0, -1.0]])[:s]
    clipped = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-6)
    for t, g in enumerate(raw, 1):
        p.grad, q.grad = torch.tensor(g[:2]), torch.tensor(g[2:])
        norm = clip_grad_norm_([p, q], 1.0)
        assert norm > 1
        np.testing.assert_allclose(p.grad.numpy(), clipped[t - 1, :2])
        np.testing.assert_allclose(q.grad.numpy(), clipped[t - 1, 2:])
        opt.step()
        if t < s:
            np.testing.assert_allclose(
                opt.state[p]["accum_grad"].numpy(), clipped[:t, :2].sum(0) / s
            )
            np.testing.assert_allclose(
                opt.state[q]["accum_grad"].numpy(), clipped[:t, 2:].sum(0) / s
            )
            assert not opt.state[p]["exp_avg"].count_nonzero()
    mean = clipped.mean(0)
    raw_mean = raw.mean(0)
    clipped_mean = raw_mean * min(1, 1 / (np.linalg.norm(raw_mean) + 1e-6))
    assert not np.allclose(mean, clipped_mean)
    for parameter, expected in ((p, mean[:2]), (q, mean[2:])):
        np.testing.assert_allclose(opt.state[parameter]["exp_avg"].numpy(), 0.4 * expected)
        np.testing.assert_allclose(opt.state[parameter]["exp_avg_sq"].numpy(), 0.15 * expected**2)
        assert not opt.state[parameter]["accum_grad"].count_nonzero()


@pytest.mark.parametrize("foreach", [False, True])
def test_clipping_leaves_missing_gradient_state_untouched(foreach):
    p, q = [torch.nn.Parameter(torch.zeros(1, dtype=torch.float64)) for _ in range(2)]
    opt = AccumAdamW([p, q], betas=(0.0, 0.0), foreach=foreach)
    p.grad, q.grad = torch.full_like(p, 8.0), None
    clip_grad_norm_([p, q], 1.0)
    opt.step()
    p.grad, q.grad = torch.full_like(p, 4.0), torch.full_like(q, 100.0)
    clip_grad_norm_([p, q], 1.0)
    opt.step()
    scale = 1 / (np.sqrt(4**2 + 100**2) + 1e-6)
    torch.testing.assert_close(
        opt.state[p]["exp_avg"], torch.full_like(p, (8 / (8 + 1e-6) + 4 * scale) / 2)
    )
    torch.testing.assert_close(opt.state[q]["accum_grad"], torch.full_like(q, 50 * scale))
    assert not opt.state[q]["exp_avg"].count_nonzero()
    before = copy.deepcopy(opt.state[p])
    p_before = p.detach().clone()
    p.grad, q.grad = None, torch.full_like(q, -100.0)
    clip_grad_norm_([p, q], 1.0)
    opt.step()
    assert_nested_equal(opt.state[p], before)
    torch.testing.assert_close(p, p_before, rtol=0, atol=0)
    torch.testing.assert_close(
        opt.state[q]["exp_avg"], torch.full_like(q, (100 * scale - 100 / (100 + 1e-6)) / 2)
    )
    assert not opt.state[q]["accum_grad"].count_nonzero()


@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("grad_clip", [None, 0.5])
def test_s1_matches_adamw_with_groups_and_missing_gradients(foreach, grad_clip):
    params = [torch.nn.Parameter(torch.ones(3, dtype=torch.float64)) for _ in range(3)]
    refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
    opts = []
    for cls, ps in ((AccumAdamW, params), (torch.optim.AdamW, refs)):
        groups = [{"params": ps[:2], "weight_decay": 0.3}, {"params": ps[2:], "weight_decay": 0}]
        if cls is AccumAdamW:
            opts.append(
                AccumAdamW(
                    groups,
                    betas=(0.7, 0.9),
                    eps=0.01,
                    foreach=foreach,
                    accumulation_steps=1,
                )
            )
        else:
            opts.append(torch.optim.AdamW(groups, betas=(0.7, 0.9), eps=0.01, foreach=foreach))
    for t in range(7):
        for opt in opts:
            for group in opt.param_groups:
                group["lr"] = 0.01 / (t + 1)
        for i, (p, q) in enumerate(zip(params, refs, strict=True)):
            g = torch.tensor([t - 1, i + 1, t + i], dtype=p.dtype)
            p.grad = g.clone() if (t + i) % 3 else None
            q.grad = g.clone() if p.grad is not None else None
        for ps, opt in zip((params, refs), opts, strict=True):
            clip_grad_norm_(ps, grad_clip)
            opt.step()
        for p, q in zip(params, refs, strict=True):
            torch.testing.assert_close(p, q, rtol=1e-13, atol=1e-14)
            for key in ("exp_avg", "exp_avg_sq", "step"):
                if p in opts[0].state:
                    torch.testing.assert_close(
                        opts[0].state[p][key],
                        opts[1].state[q][key],
                        check_dtype=False,
                        rtol=1e-13,
                        atol=1e-14,
                    )


@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("betas", [(0.0, 0.0), (0.6, 0.85)])
def test_epsilon_outside_sqrt_after_bias_correction_for_small_gradients(foreach, betas):
    p = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    opt = AccumAdamW([p], lr=0.1, betas=betas, eps=1e-8, weight_decay=0, foreach=foreach)
    gradient = np.array([0.0, 1e-8, -1e-4])
    p.grad = torch.tensor(gradient)
    opt.step()
    expected = -0.1 * gradient / (np.sqrt(gradient**2) + 1e-8)
    inside_sqrt = -0.1 * gradient / np.sqrt(gradient**2 + 1e-8)
    assert not np.allclose(expected, inside_sqrt)
    np.testing.assert_allclose(p.detach().numpy(), expected, rtol=1e-13, atol=1e-14)
    assert torch.isfinite(p).all()


@pytest.mark.parametrize("grad_clip", [None, 0.5])
def test_foreach_matches_scalar_with_different_parameter_counters(grad_clip):
    params = [
        torch.nn.Parameter(torch.ones(3, dtype=dtype))
        for dtype in (torch.float64, torch.float64, torch.float32)
    ]
    refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
    a, b = (
        AccumAdamW(params),
        AccumAdamW(refs, foreach=False),
    )
    for t in range(9):
        for i, (p, q) in enumerate(zip(params, refs, strict=True)):
            p.grad = torch.full_like(p, t - 2.5) if (t + i) % 3 else None
            q.grad = None if p.grad is None else p.grad.clone()
        clip_grad_norm_(params, grad_clip)
        clip_grad_norm_(refs, grad_clip)
        a.step()
        b.step()
        for p, q in zip(params, refs, strict=True):
            torch.testing.assert_close(p, q)
            if p in a.state:
                assert_nested_equal(a.state[p], b.state[q])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"accumulation_steps": 0},
        {"accumulation_steps": -1},
        {"accumulation_steps": True},
        {"accumulation_steps": 1.5},
        {"lr": -1},
        {"lr": float("nan")},
        {"betas": (1, 0.9)},
        {"betas": (0.9, -0.1)},
        {"eps": 0},
        {"weight_decay": -1},
    ],
)
def test_invalid_optimizer_settings(kwargs):
    with pytest.raises(ValueError):
        AccumAdamW([torch.nn.Parameter(torch.ones(2))], **kwargs)


def test_sparse_rejection_is_atomic_and_closure_works():
    p, q = [torch.nn.Parameter(torch.ones(2)) for _ in range(2)]
    opt = AccumAdamW([p, q])
    p.grad = torch.ones_like(p)
    q.grad = torch.ones_like(q).to_sparse()
    with pytest.raises(RuntimeError, match="sparse"):
        opt.step()
    assert not opt.state
    assert torch.equal(p, torch.ones_like(p))
    opt.zero_grad()

    def closure():
        loss = p.square().sum()
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert isinstance(loss, torch.Tensor) and loss.item() == 2
    assert q not in opt.state


@pytest.mark.parametrize("grad_clip", [None, 0.5])
def test_ordinary_midwindow_resume_and_atomic_validation(grad_clip):
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    a = AccumAdamW([p])
    p.grad = torch.tensor([1.0, -2.0], dtype=p.dtype)
    clip_grad_norm_([p], grad_clip)
    a.step()
    saved = copy.deepcopy(a.state_dict())
    q = torch.nn.Parameter(p.detach().clone())
    b = AccumAdamW([q], foreach=False)
    b.load_state_dict(saved)
    assert b.param_groups[0]["foreach"] is False
    for g in ([2.0, 1.0], [-3.0, 4.0], [1.0, 0.0]):
        p.grad = torch.tensor(g, dtype=p.dtype)
        q.grad = p.grad.clone()
        clip_grad_norm_([p], grad_clip)
        clip_grad_norm_([q], grad_clip)
        a.step()
        b.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        assert_nested_equal(a.state[p], b.state[q])
    before = copy.deepcopy(b.state_dict())
    for damage in (
        "name",
        "steps",
        "buffer",
        "counter",
        "moment",
        "window_mean",
        "inside_epsilon",
    ):
        bad = copy.deepcopy(saved)
        state = bad["state"][0]
        if damage == "name":
            bad["optimizer_name"] = "adamw"
        elif damage == "steps":
            bad["param_groups"][0]["accumulation_steps"] = 3
        elif damage == "buffer":
            state["accum_grad"] = torch.zeros(1, dtype=p.dtype)
        elif damage == "counter":
            state["step"] = torch.tensor(0.5)
        elif damage == "inside_epsilon":
            bad["version"] = 3
        elif damage == "window_mean":
            bad["version"] = 2
            bad["grad_clip"] = grad_clip
        else:
            state["exp_avg_sq"].fill_(-1)
        with pytest.raises(ValueError):
            b.load_state_dict(bad)
        assert_nested_equal(before, b.state_dict())


def test_config_and_legacy_identity(tiny_config: Config, cache_dir: Path):
    cfg = load_config(overrides=["optimizer.name=accum_adamw"])
    assert cfg.optimizer.accumulation_steps == 2
    for value in ("0", "true", "1.5"):
        with pytest.raises(ValidationError):
            load_config(overrides=[f"optimizer.accumulation_steps={value}"])
    with pytest.raises(ValidationError):
        load_config(overrides=["optimizer.name=unknown"])
    cache = TokenCache(cache_dir)
    raw = tiny_config.model_dump(mode="json")
    raw["optimizer"].pop("name")
    raw["optimizer"].pop("accumulation_steps")
    for key in ("checkpoint_policy", "checkpoint_epochs", "save_epoch_training_state"):
        raw["training"].pop(key)
    for key in ("output_dir", "device", "cpu_threads"):
        raw["runtime"].pop(key)
    for key in ("cache_dir", "prefetch", "prepare_workers"):
        raw["data"].pop(key)
    raw.update(
        cache_identity=cache.manifest["identity"], loader_version=BufferedTokenLoader.VERSION
    )
    legacy = recipe_identity(tiny_config, cache)
    assert fingerprint(raw) == legacy
    tiny_config.optimizer.name = "accum_adamw"
    accum = recipe_identity(tiny_config, cache)
    assert accum != legacy
    raw["optimizer"].update(name="accum_adamw", accumulation_steps=2)
    assert accum == fingerprint(raw)  # Original per-update recipes remain compatible.
    raw["optimizer"]["clipping"] = "window_mean"
    assert accum != fingerprint(raw)
    raw["optimizer"].pop("clipping")
    raw["optimizer"]["epsilon_placement"] = "inside_sqrt"
    assert accum != fingerprint(raw)  # Epsilon-inside recipes must not resume.
    tiny_config.optimizer.accumulation_steps = 3
    assert recipe_identity(tiny_config, cache) != accum


def test_packed_arenas_and_atomic_restore(tiny_config: Config):
    tiny_config.optimizer.name = "accum_adamw"
    model = PackedLlama(tiny_config.model, 2)
    opt = make_optimizer(model, tiny_config, torch.device("cpu"))
    assert isinstance(opt, PackedOptimizer)
    assert opt.accumulated_gradient_storage is not None
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    saved = copy.deepcopy(opt.state_dict())
    opt.step()
    before = copy.deepcopy(opt.state_dict())
    for damage in ("shape", "settings", "counter", "version", "inside_epsilon", "boundary"):
        bad = copy.deepcopy(saved)
        if damage == "shape":
            bad["accumulated_gradient_storage"] = torch.zeros(2)
        elif damage == "settings":
            bad["workers"][-1]["groups"][-1]["accumulation_steps"] = 3
        elif damage == "counter":
            bad["workers"][-1]["steps"][-1] = torch.tensor(-1)
        elif damage == "inside_epsilon":
            bad["version"] = 4
        elif damage == "version":
            bad["version"] = 3  # Completed-mean clipping state.
        else:
            bad["workers"][-1]["steps"][-1].fill_(2)
        with pytest.raises(ValueError):
            opt.load_state_dict(bad)
        assert_nested_equal(before, opt.state_dict())
    opt.load_state_dict(saved)
    assert_nested_equal(saved, opt.state_dict())
    for worker, local in enumerate(opt.local_parameters):
        for entry, p in zip(model.layout, local, strict=True):
            for key, arena in (
                ("exp_avg", opt.first_moment_storage),
                ("exp_avg_sq", opt.second_moment_storage),
                ("accum_grad", opt.accumulated_gradient_storage),
            ):
                view = model.parameter_view(arena, entry)[worker]
                assert opt.optimizers[worker].state[p][key].data_ptr() == view.data_ptr()
                assert not arena[
                    :, entry.start + entry.numel : entry.start + entry.padded_numel
                ].count_nonzero()


@pytest.mark.parametrize("scheme", ["awc", "atc"])
def test_benchmarks_use_accum_adamw(tiny_config: Config, tmp_path: Path, scheme):
    tiny_config.optimizer.name = "accum_adamw"
    tiny_config.runtime.deterministic = True
    ordinary = benchmark_worker(
        tiny_config, tmp_path / "ordinary.json", warmup=1, steps=3, windows=1
    )
    assert ordinary["status"] == "ok"
    raw = tiny_config.model_dump()
    raw["decentralized"] = dict(num_models=2, scheme=scheme)
    config = Config.model_validate(raw)
    results = [
        benchmark_packed_worker(config, tmp_path / f"{mode}.json", mode, warmup=1, steps=3)
        for mode in ("packed", "sequential")
    ]
    assert all(result["status"] == "ok" for result in results)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_foreach_matches_cpu_and_resume():
    p = torch.nn.Parameter(torch.ones(8, device="cuda"))
    q = torch.nn.Parameter(p.detach().cpu())
    a, b = (
        AccumAdamW([p], accumulation_steps=2),
        AccumAdamW([q], accumulation_steps=2, foreach=False),
    )
    for t in range(5):
        p.grad = torch.arange(8, device="cuda", dtype=p.dtype) - t
        q.grad = p.grad.cpu()
        clip_grad_norm_([p], 1.0)
        clip_grad_norm_([q], 1.0)
        a.step()
        b.step()
        torch.testing.assert_close(p.cpu(), q)
        if t == 2:
            saved = copy.deepcopy(a.state_dict())
            a = AccumAdamW([p], accumulation_steps=2)
            a.load_state_dict(saved)
            assert a.state[p]["step"].device.type == "cpu"
