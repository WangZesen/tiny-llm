"""Explicit runtime policy, run metadata, and atomic artifacts."""

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from tiny_llm.config import Config


def setup_logging(path: Path | None = None):
    logger.remove()
    fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}"
    logger.add(sys.stderr, format=fmt, colorize=False)
    if path:
        logger.add(path, format=fmt, colorize=False)


def setup_runtime(config: Config) -> torch.device:
    cfg = config.runtime
    if cfg.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(cfg.cpu_threads)
    torch.use_deterministic_algorithms(cfg.deterministic)
    torch.backends.cudnn.deterministic = cfg.deterministic
    torch.backends.cudnn.benchmark = not cfg.deterministic
    torch.set_float32_matmul_precision("highest" if cfg.deterministic else "high")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; set runtime.device=cpu for fixtures")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(cfg.seed)
        if cfg.amp and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 AMP requested on a GPU without BF16 support")
    elif cfg.amp:
        raise ValueError("training AMP requires CUDA; set runtime.amp=false for CPU runs")
    backend = cfg.sdpa_backend
    if backend != "auto" and actual_backend(config) == "sdpa" and device.type != "cuda":
        raise ValueError("forced SDPA kernels require CUDA")
    torch.backends.cuda.enable_flash_sdp(backend in ("auto", "flash"))
    torch.backends.cuda.enable_cudnn_sdp(backend in ("auto", "cudnn"))
    torch.backends.cuda.enable_math_sdp(backend == "auto")
    torch.backends.cuda.enable_mem_efficient_sdp(backend == "auto")
    return device


def actual_backend(config: Config) -> str:
    return "reference" if config.runtime.deterministic else config.runtime.attention_backend


def autocast(config: Config, device: torch.device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.runtime.amp)


def attention_kernels(model, config: Config, device: torch.device) -> list[str]:
    """Record SDPA dispatch for the configured context and precision."""
    if actual_backend(config) == "reference":
        return ["reference_matmul_softmax"]
    with (
        preserve_rng(),
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile,
    ):
        with autocast(config, device):
            inputs = torch.zeros((1, config.model.context_length), device=device, dtype=torch.long)
            if hasattr(model, "num_models"):
                inputs = inputs.unsqueeze(0).expand(model.num_models, -1, -1)
            model(inputs).sum().backward()
    model.zero_grad(set_to_none=True)
    return sorted({event.key for event in profile.key_averages() if "attention" in event.key})


def rng_state() -> dict:
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    )


def restore_rng(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        # Local device numbering can change when a sweep resumes on another physical GPU.
        for index, value in enumerate(state["cuda"][: torch.cuda.device_count()]):
            torch.cuda.set_rng_state(value.cpu(), index)


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_checkpoint(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_metric(path: Path, value):
    with path.open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def source_digest(package: Path) -> str:
    """Hash all Python sources, including subpackages, with stable relative paths."""
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment() -> dict:
    def git(*args):
        process = subprocess.run(["git", *args], capture_output=True, text=True)
        return process.stdout.strip() if process.returncode == 0 else None

    def command(*args):
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=10)
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    def package_version(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return dict(
        python=sys.version,
        platform=platform.platform(),
        hostname=platform.node(),
        versions={
            name: package_version(name)
            for name in ("torch", "triton", "numpy", "pydantic", "datasets", "transformers")
        },
        cuda=torch.version.cuda,
        git_revision=git("rev-parse", "HEAD"),
        git_status=git("status", "--short"),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        source_hash=source_digest(Path(__file__).parent),
        compiler_cache={
            key: os.environ.get(key) for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
        },
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        cpu_threads=torch.get_num_threads(),
        cudnn=torch.backends.cudnn.version(),
        slurm={key: value for key, value in os.environ.items() if key.startswith("SLURM_")},
        gpu_status=command(
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,memory.total,power.limit,power.draw,clocks.sm,clocks.mem,utilization.gpu",
            "--format=csv",
        ),
        gpu_topology=command("nvidia-smi", "topo", "-m"),
    )
