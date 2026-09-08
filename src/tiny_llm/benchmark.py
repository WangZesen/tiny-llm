"""Repeatable production-update benchmarks; profiling is outside timed windows."""

import statistics
import time
from pathlib import Path

import torch

from tiny_llm.config import Config
from tiny_llm.data import BufferedTokenLoader, TokenCache, fingerprint, training_boundaries
from tiny_llm.model import Llama
from tiny_llm.runtime import actual_backend, atomic_json, environment, setup_runtime
from tiny_llm.train import learning_rate, loss_function, make_optimizer, optimizer_update


def benchmark_identity(config, metadata, protocol):
    """Exclude transient utilization, hostname, job IDs, and timestamps from reuse."""
    stable = {
        key: metadata.get(key)
        for key in (
            "source_hash",
            "compiler_cache",
            "versions",
            "python",
            "cuda",
            "cudnn",
            "gpu",
            "cpu_affinity",
            "cpu_threads",
            "platform",
        )
    }
    # The first GPU status fields include UUID, driver, memory and power limit.
    status = metadata.get("gpu_status") or ""
    stable["gpu_configuration"] = [line.split(",")[:5] for line in status.splitlines()[1:]]
    return fingerprint(
        dict(config=config.model_dump(mode="json"), environment=stable, protocol=protocol)
    )


def benchmark_worker(
    config: Config,
    destination: Path,
    warmup: int = 20,
    steps: int = 100,
    windows: int = 3,
    data_mode: str = "synthetic",
    profile: bool = False,
):
    if config.decentralized is not None:
        raise ValueError("use benchmark-packed for decentralized training")
    if min(warmup, steps, windows) < 1 or data_mode not in ("synthetic", "real"):
        raise ValueError("positive benchmark lengths and synthetic/real data mode required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    device = setup_runtime(config)
    metadata = environment()
    protocol = dict(
        version=2, warmup=warmup, steps=steps, windows=windows, data_mode=data_mode, profile=profile
    )
    identity = benchmark_identity(config, metadata, protocol)
    model = Llama(config.model, actual_backend(config)).to(device)
    optimizer = make_optimizer(model, config, device)
    compute_loss = loss_function(model, config, device)
    length = config.model.context_length
    step_blocks = config.training.batch_tokens // length
    batch = config.training.micro_batch_size
    loader = None
    cursor = 0
    if data_mode == "real":
        cache = TokenCache(config.data.cache_dir)
        cache.validate_config(config)
        cache.verify()
        # Use the production prefix so range ordering matches real training.
        blocks = training_boundaries(config, model.parameter_count)[-1]
        needed = (warmup + steps * windows + (3 if profile else 1)) * step_blocks
        if needed > blocks:
            raise ValueError(f"benchmark needs {needed} blocks but recipe has {blocks}")
        loader = BufferedTokenLoader(
            cache,
            "train",
            length,
            blocks,
            config.data.buffer_size_mib,
            seed=config.runtime.seed,
            prefetch=config.data.prefetch,
        )
        next_batch = loader.next_batch
        metadata["cache_identity"] = cache.manifest["identity"]
        protocol["cache_identity"] = cache.manifest["identity"]
        identity = benchmark_identity(config, metadata, protocol)
    else:
        x = torch.randint(config.model.vocab_size, (batch, length), device=device)
        y = torch.randint(config.model.vocab_size, x.shape, device=device)

        def next_batch(count, device):
            return x[:count], y[:count]

    total_tokens = training_boundaries(config, model.parameter_count)[-1] * length

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def update(tracing=False):
        nonlocal cursor
        lr = learning_rate(config, (cursor + step_blocks) * length, total_tokens)
        for group in optimizer.param_groups:
            group["lr"] = lr
        value, norm = optimizer_update(
            model, optimizer, compute_loss, next_batch, config, device, step_blocks, profile=tracing
        )
        cursor += step_blocks
        return value, norm

    try:
        synchronize()
        start = time.monotonic()
        update()
        synchronize()
        first_update_seconds = time.monotonic() - start
        for _ in range(warmup - 1):
            update()
        synchronize()
        warmup_seconds = time.monotonic() - start
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        measurements = []
        for window in range(windows):
            synchronize()
            start = time.monotonic()
            timestamp = time.time()
            loss = 0.0
            for _ in range(steps):
                value, norm = update()
                loss += value.item()
            synchronize()
            seconds = time.monotonic() - start
            measurements.append(
                dict(
                    window=window,
                    timestamp=timestamp,
                    seconds=seconds,
                    tokens_per_second=steps * config.training.batch_tokens / seconds,
                    loss=loss / (steps * config.training.batch_tokens),
                    grad_norm=norm.item(),
                )
            )
        memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        total_memory = (
            torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 1
        )
        # This probe uses the actual compiled update and actual microbatch shape.
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=profile) as trace:
            for _ in range(3 if profile else 1):
                update(tracing=profile)
            synchronize()
        kernels = sorted(
            {
                event.key
                for event in trace.key_averages()
                if any(name in event.key.lower() for name in ("attention", "flash", "fmha", "sdpa"))
            }
        )
        if actual_backend(config) == "reference":
            kernels = ["reference_matmul_softmax"]
        if profile:
            trace.export_chrome_trace(str(destination.with_suffix(".trace.json")))
            destination.with_suffix(".profile.txt").write_text(
                trace.key_averages().table(
                    sort_by="self_cuda_time_total"
                    if device.type == "cuda"
                    else "self_cpu_time_total",
                    row_limit=40,
                )
            )
        rate = statistics.median(row["tokens_per_second"] for row in measurements)
        result = dict(
            status="ok" if memory < 0.9 * total_memory else "memory_limit",
            synthetic=data_mode == "synthetic",
            tokens_per_second=rate,
            seconds_per_step=config.training.batch_tokens / rate,
            first_update_seconds=first_update_seconds,
            warmup_seconds=warmup_seconds,
            peak_memory_bytes=memory,
            total_memory_bytes=total_memory,
            micro_batch_size=batch,
            compile=config.runtime.compile,
            compile_mode=config.runtime.compile_mode,
            sdpa_backend=config.runtime.sdpa_backend,
            cpu_threads=config.runtime.cpu_threads,
            attention_kernels=kernels,
            environment=metadata,
            protocol=protocol,
            measurements=measurements,
            config_identity=identity,
        )
        atomic_json(destination, result)
        return result
    finally:
        if loader is not None:
            loader.close()


def tune_gh200(config: Config, output: Path, budget_minutes: float = 75):
    """Bounded, sequential tuning; all candidates use real C4 production updates."""
    import json
    import os
    import signal
    import subprocess
    import sys

    from tiny_llm.config import save_config

    if config.decentralized is not None:
        raise ValueError("use benchmark-packed for decentralized training")
    if config.runtime.deterministic:
        raise ValueError("GH200 tuning requires the nondeterministic fast path")
    if budget_minutes <= 0:
        raise ValueError("budget_minutes must be positive")
    output.mkdir(parents=True, exist_ok=True)
    setup_runtime(config)
    metadata = environment()
    if "GH200" not in (metadata["gpu"] or ""):
        raise ValueError("GH200 tuning requires a GH200 GPU")
    protocol = dict(
        version=2,
        warmup=20,
        steps=100,
        windows=3,
        data_mode="real",
        profile=False,
        cache_identity=TokenCache(config.data.cache_dir).manifest["identity"],
    )
    deadline = time.monotonic() + budget_minutes * 60
    results = []
    seen = set()

    def candidate(batch, compiled, backend="auto", mode="default", threads=8):
        key = (batch, compiled, backend, mode, threads)
        if key in seen or deadline - time.monotonic() < 60:
            return
        seen.add(key)
        cfg = config.model_copy(deep=True)
        cfg.training.micro_batch_size = batch
        cfg.runtime.compile = compiled
        cfg.runtime.sdpa_backend = backend
        cfg.runtime.compile_mode = mode
        cfg.runtime.cpu_threads = threads
        name = f"micro-{batch}-compile-{int(compiled)}-{backend}-{mode}-threads-{threads}"
        path = output / f"{name}.yaml"
        target = path.with_suffix(".json")
        save_config(cfg, path)
        identity = benchmark_identity(cfg, metadata | {"cpu_threads": threads}, protocol)
        if target.exists():
            cached = json.loads(target.read_text())
            if cached.get("config_identity") == identity and cached.get("status") == "ok":
                results.append(cached | {"config_path": str(path)})
                return
        with path.with_suffix(".log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tiny_llm",
                    "benchmark-worker",
                    "--config",
                    str(path),
                    "--output",
                    str(target),
                    "--data-mode",
                    "real",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                process.wait(timeout=min(600, max(0.01, deadline - time.monotonic())))
                status = "failed" if process.returncode else "ok"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                status = "timeout"
            except BaseException:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        if status == "ok":
            row = json.loads(target.read_text())
        else:
            row = dict(status=status, config_identity=identity, log=str(path.with_suffix(".log")))
            atomic_json(target, row)
        results.append(row | {"config_path": str(path)})
        atomic_json(output / "progress.json", results)

    # Establish the configuration mismatch first, then complete the batch grid.
    for batch, compiled in (
        (8, False),
        (8, True),
        (16, True),
        (4, False),
        (4, True),
        (16, False),
        (32, False),
        (32, True),
    ):
        if batch * config.model.context_length <= config.training.batch_tokens:
            candidate(batch, compiled, threads=config.runtime.cpu_threads)

    def best_rows():
        return sorted(
            (row for row in results if row["status"] == "ok"),
            key=lambda row: row["tokens_per_second"],
            reverse=True,
        )

    for row in best_rows()[:2]:
        for backend in ("flash", "cudnn"):
            candidate(
                row["micro_batch_size"], row["compile"], backend, threads=config.runtime.cpu_threads
            )
    if best_rows():
        best = best_rows()[0]
        for mode in ("reduce-overhead", "max-autotune"):
            candidate(
                best["micro_batch_size"],
                True,
                best["sdpa_backend"],
                mode,
                config.runtime.cpu_threads,
            )
    if best_rows():
        best = best_rows()[0]
        for threads in (1, 4, 8):
            candidate(
                best["micro_batch_size"],
                best["compile"],
                best["sdpa_backend"],
                best["compile_mode"],
                threads,
            )
    if not best_rows():
        raise RuntimeError(f"no successful candidates; see {output}")
    best = best_rows()[0]
    from tiny_llm.config import load_config

    selected = load_config(best["config_path"])
    save_config(selected, output / "selected.yaml")
    summary = dict(
        selected=best,
        candidates=results,
        budget_minutes=budget_minutes,
        budget_exhausted=time.monotonic() >= deadline,
    )
    atomic_json(output / "summary.json", summary)
    return summary
