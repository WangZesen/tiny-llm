import copy
import json

import pytest
import torch
from pydantic import ValidationError

from tiny_llm.benchmark import benchmark_identity, benchmark_worker
from tiny_llm.config import Config
from tiny_llm.data import BufferedTokenLoader, TokenCache, fingerprint
from tiny_llm.model import Llama
from tiny_llm.runtime import setup_runtime
from tiny_llm.train import loss_function, make_optimizer, optimizer_update, recipe_identity


def test_execution_config_and_legacy_identity(tiny_config, cache_dir):
    cache = TokenCache(cache_dir)
    old = tiny_config.model_dump(mode="json")
    for key in ("output_dir", "device", "cpu_threads", "compile_mode", "sdpa_backend"):
        old["runtime"].pop(key)
    old["data"].pop("cache_dir")
    old["data"].pop("prefetch")
    old["cache_identity"] = cache.manifest["identity"]
    old["loader_version"] = BufferedTokenLoader.VERSION
    original = recipe_identity(tiny_config, cache)
    assert original == fingerprint(old)
    tiny_config.runtime.compile_mode = "reduce-overhead"
    assert recipe_identity(tiny_config, cache) != original
    with pytest.raises(ValidationError):
        Config(runtime={"sdpa_backend": "unknown"})
    tiny_config.runtime.sdpa_backend = "flash"
    with pytest.raises(ValueError, match="require CUDA"):
        setup_runtime(tiny_config)


def test_benchmark_reuse_identity(tiny_config):
    metadata = dict(
        source_hash="a",
        versions={"torch": "1"},
        gpu="GH200",
        gpu_status="",
        hostname="a",
        cpu_affinity=[1, 2],
    )
    protocol = dict(warmup=20, steps=100, windows=3, data_mode="real")
    baseline = benchmark_identity(tiny_config, metadata, protocol)
    assert baseline == benchmark_identity(tiny_config, metadata | {"hostname": "b"}, protocol)
    for key, value in (
        ("source_hash", "b"),
        ("compiler_cache", {"TORCHINDUCTOR_CACHE_DIR": "/tmp/fresh"}),
        ("gpu", "RTX"),
        ("versions", {"torch": "2"}),
        ("cpu_affinity", [1]),
    ):
        assert baseline != benchmark_identity(tiny_config, metadata | {key: value}, protocol)
    assert baseline != benchmark_identity(tiny_config, metadata, protocol | {"steps": 101})


@pytest.mark.parametrize("data_mode", ["synthetic", "real"])
def test_worker_windows(tiny_config, cache_dir, tmp_path, data_mode):
    tiny_config.training.max_tokens = 128
    result = benchmark_worker(
        tiny_config,
        tmp_path / "bench.json",
        warmup=1,
        steps=1,
        windows=3,
        data_mode=data_mode,
        profile=True,
    )
    assert result["status"] == "ok"
    assert len(result["measurements"]) == 3
    assert result["tokens_per_second"] > 0
    assert (tmp_path / "bench.trace.json").exists()
    assert json.loads((tmp_path / "bench.json").read_text())["protocol"]["data_mode"] == data_mode
    with pytest.raises(ValueError, match="positive"):
        benchmark_worker(tiny_config, tmp_path / "bad.json", steps=0)


@pytest.mark.parametrize("bad_gradient", [False, True])
def test_nonfinite_update_never_steps(tiny_config, monkeypatch, bad_gradient):
    setup_runtime(tiny_config)
    model = Llama(tiny_config.model)
    optimizer = make_optimizer(model, tiny_config, torch.device("cpu"))
    before = copy.deepcopy(model.state_dict())

    def forbidden_step():
        pytest.fail("invalid update reached optimizer.step")

    monkeypatch.setattr(optimizer, "step", forbidden_step)
    compute = loss_function(model, tiny_config, torch.device("cpu"))
    if bad_gradient:
        model.embedding.weight.register_hook(lambda gradient: gradient * float("nan"))
    else:
        original = compute

        def compute(x, y):
            return original(x, y) * float("nan")

    def batch(count, device):
        x = torch.ones((count, 4), dtype=torch.long)
        return x, x

    with pytest.raises((FloatingPointError, RuntimeError), match="non.?finite"):
        optimizer_update(model, optimizer, compute, batch, tiny_config, torch.device("cpu"), 4)
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "compiled,mode,backend",
    [
        (False, "default", "flash"),
        (False, "default", "cudnn"),
        (True, "default", "auto"),
        (True, "reduce-overhead", "auto"),
        (True, "max-autotune", "auto"),
    ],
)
def test_cuda_execution_parity(compiled, mode, backend):
    from tiny_llm.config import ModelConfig
    from tiny_llm.model import token_losses

    config = Config(
        model=ModelConfig(
            vocab_size=128, layers=2, width=64, heads=2, ffn_width=176, context_length=32
        )
    )
    config.training.micro_batch_size = 2
    config.runtime.compile = compiled
    config.runtime.compile_mode = mode
    config.runtime.sdpa_backend = backend
    config.runtime.cpu_threads = 1
    device = setup_runtime(config)
    model = Llama(config.model).to(device)
    reference = Llama(config.model, "reference").to(device)
    reference.load_state_dict(model.state_dict())
    compute = loss_function(model, config, device)
    for _ in range(3):
        x = torch.randint(128, (2, 32), device=device)
        y = torch.randint(128, x.shape, device=device)
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = token_losses(reference(x), y).sum()
        expected.backward()
        actual = compute(x, y)
        actual.backward()
        torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
        left = torch.cat([p.grad.flatten() for p in model.parameters()])
        right = torch.cat([p.grad.flatten() for p in reference.parameters()])
        assert (left - right).norm() / right.norm() < 0.05


def test_tuning_timeout_kills_worker_group(tiny_config, cache_dir, tmp_path, monkeypatch):
    import os
    import subprocess

    import tiny_llm.benchmark as module

    monkeypatch.setattr(module, "setup_runtime", lambda config: torch.device("cpu"))
    monkeypatch.setattr(module, "environment", lambda: {"gpu": "GH200", "gpu_status": ""})
    killed = []
    monkeypatch.setattr(os, "killpg", lambda pid, signum: killed.append(pid))

    class Process:
        pid = 987654321
        returncode = None

        def __init__(self, args, **kwargs):
            assert kwargs["start_new_session"]
            self.args = args

        def wait(self, timeout=None):
            if timeout is not None:
                assert 0 < timeout <= 120
                raise subprocess.TimeoutExpired(self.args, timeout)
            self.returncode = -9

    monkeypatch.setattr(subprocess, "Popen", Process)
    with pytest.raises(RuntimeError, match="no successful candidates"):
        module.tune_gh200(tiny_config, tmp_path / "tuning", budget_minutes=2)
    progress = json.loads((tmp_path / "tuning" / "progress.json").read_text())
    assert len(killed) == len(progress) == 2
    assert all(row["status"] == "timeout" for row in progress)


def test_compile_dispatch_keeps_partial_batches_eager(tiny_config, monkeypatch):
    from tiny_llm.model import token_losses

    seen = []

    def compile_function(function, mode):
        assert mode == "default"

        def compiled(x, y):
            seen.append(x.shape[0])
            return function(x, y)

        return compiled

    monkeypatch.setattr(torch, "compile", compile_function)
    tiny_config.runtime.compile = True
    model = Llama(tiny_config.model, "reference")
    compute = loss_function(model, tiny_config, torch.device("cpu"))
    for count in (2, 1, 2):
        x = torch.ones((count, 4), dtype=torch.long)
        torch.testing.assert_close(compute(x, x), token_losses(model(x), x).sum())
    assert seen == [2, 2]
