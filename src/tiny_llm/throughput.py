"""Synthetic, complete GH200 updates with explicit timing and failure boundaries."""

from __future__ import annotations

import math
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from tiny_llm.config import Config
from tiny_llm.runtime import atomic_json, environment, setup_runtime


class CapturedUpdateTimer:
    """Benchmark-only event nodes around the production graph and optimizer step.

    A replay overwrites the events. Read them only after synchronization, before
    the next replay; each sample therefore describes the last update in a burst.
    External events are required to retain timing nodes in the captured graph.
    """

    def __init__(self, engine: Any):
        self.begin = torch.cuda.Event(enable_timing=True, external=True)
        self.optimizer_begin = torch.cuda.Event(enable_timing=True, external=True)
        self.end = torch.cuda.Event(enable_timing=True, external=True)
        update, step = engine._update, engine.optimizer.step

        def timed_update():
            if torch.cuda.is_current_stream_capturing():
                self.begin.record()
            update()

        def timed_step(losses):
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing:
                self.optimizer_begin.record()
            step(losses)
            if capturing:
                self.end.record()

        engine._update = timed_update
        engine.optimizer.step = timed_step

    def sample(self) -> dict:
        graph_ms = self.begin.elapsed_time(self.end)
        optimizer_ms = self.optimizer_begin.elapsed_time(self.end)
        if not (0 < optimizer_ms <= graph_ms and math.isfinite(graph_ms)):
            raise RuntimeError(f"invalid captured event times: {optimizer_ms=}, {graph_ms=}")
        return dict(optimizer_ms=optimizer_ms, graph_ms=graph_ms)


def optimizer_window(samples: list[dict]) -> dict:
    """Weight each sampled final replay by the number of updates in its burst."""
    steps = sum(row["steps"] for row in samples)
    return dict(
        optimizer_ms=sum(row["steps"] * row["optimizer_ms"] for row in samples) / steps,
        graph_ms=sum(row["steps"] * row["graph_ms"] for row in samples) / steps,
        optimizer_samples=samples,
    )


def optimizer_summary(measurements: list[dict]) -> dict:
    times = [row["optimizer_ms"] for row in measurements]
    cv = statistics.stdev(times) / statistics.mean(times) if len(times) > 1 else 0.0
    return dict(
        optimizer_milliseconds_per_update=statistics.median(times),
        graph_milliseconds_per_update=statistics.median(row["graph_ms"] for row in measurements),
        optimizer_fraction_instrumented=statistics.median(
            row["optimizer_ms"] / (1000 * row["seconds"] / row["steps"]) for row in measurements),
        optimizer_fraction_graph=statistics.median(
            row["optimizer_ms"] / row["graph_ms"] for row in measurements),
        optimizer_coefficient_of_variation=cv,
        optimizer_stable=cv <= 0.05,
    )


def measurement_summary(measurements: list[dict], batch: int, length: int) -> dict:
    rates = [row["steps"] * batch * length / row["seconds"] for row in measurements]
    rate = statistics.median(rates)
    cv = statistics.stdev(rates) / statistics.mean(rates) if len(rates) > 1 else 0.0
    return dict(
        tokens_per_second=rate,
        sequences_per_second=rate / length,
        milliseconds_per_update=1000 * batch * length / rate,
        coefficient_of_variation=cv,
        stable=cv <= 0.05,
    )


def classify_failure(error: BaseException) -> str:
    """Do not confuse host OOM, compilation errors, or timeouts with device OOM."""
    chain, pending, seen = [], [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        for nested in (current.__cause__, current.__context__):
            if nested is not None:
                pending.append(nested)
    if any(
        isinstance(item, torch.cuda.OutOfMemoryError)
        or (isinstance(item, RuntimeError) and any(
            text in str(item) for text in ("CUDA out of memory", "CUDA error: out of memory")
        ))
        for item in chain
    ):
        return "cuda_oom"
    if any(isinstance(item, FloatingPointError) for item in chain):
        return "nonfinite"
    if any(isinstance(item, TimeoutError) for item in chain):
        return "timeout"
    if any(type(item).__module__.startswith(("torch._inductor", "torch._dynamo", "triton"))
           for item in chain):
        return "compiler_error"
    return "error"


def benchmark_throughput(
    config: Config,
    destination: Path,
    *,
    warmup: int = 20,
    windows: int = 5,
    window_seconds: float = 10.0,
    optimizer_timing: bool = False,
) -> dict[str, Any]:
    """One shape per process. The parent enforces a deadline including compilation."""
    batch, length = config.training.micro_batch_size, config.model.context_length
    if config.decentralized is not None or config.training.batch_tokens != batch * length:
        raise ValueError("throughput requires one model and no gradient accumulation")
    if config.runtime.training_backend != "gh200":
        raise ValueError("throughput requires the production GH200 backend")
    if min(warmup, windows) < 1 or not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("positive warmup, windows and finite window_seconds required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    result: dict[str, Any] = dict(
        status="running", stage="runtime", batch_size=batch, context_length=length,
        parameters=config.model.parameter_count, config=config.model_dump(mode="json"),
        protocol=dict(version=1, warmup=warmup, windows=windows, window_seconds=window_seconds,
                      minimum_window_steps=20, calibration_steps=5, stability_cv=0.05,
                      max_windows=2 * windows, synthetic=True, accumulation_steps=1,
                      timing="synchronized wall time around GH200Update.execute; GPU-resident data",
                      optimizer_lr="constant", excludes="setup, compilation, logging, inspection"),
        measurements=[],
    )
    if optimizer_timing:
        result["protocol"].update(
            optimizer_timing=True, sample_burst_seconds=0.1,
            optimizer_scope="ArenaAdamW.step: gradient gather, norm/clipping, finite checks, "
                            "coefficients, AdamW moments/weights, metric commit",
            optimizer_clock="CUDA external timing events inside the complete-update graph",
            optimizer_sampling="last replay per synchronized burst; weighted by burst updates",
            timing="sum of synchronized burst wall times; sample reads excluded",
        )
    progress = destination.with_suffix(".progress.json")

    def stage(name: str):
        result["stage"] = name
        result["elapsed_seconds"] = time.monotonic() - began
        atomic_json(progress, result)

    device = None
    try:
        stage("runtime")
        device = setup_runtime(config)
        from tiny_llm.gh200.engine import GH200Update

        result["environment"] = environment()
        result["total_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
        result["initial_free_memory_bytes"] = torch.cuda.mem_get_info(device)[0]
        torch.cuda.reset_peak_memory_stats(device)
        stage("model_optimizer_allocation")
        engine = GH200Update(config, device)
        timer = CapturedUpdateTimer(engine) if optimizer_timing else None
        stage("synthetic_data")
        x = torch.randint(config.model.vocab_size, (batch, length), device=device)
        y = torch.randint(config.model.vocab_size, x.shape, device=device)
        engine.inputs.copy_(x)
        engine.targets.copy_(y)

        def next_batch(count, target_device):
            return x[:count], y[:count]

        def update():
            engine.execute(next_batch, config.optimizer.lr)

        stage("compilation_and_capture")
        engine.prepare()
        result["preparation_seconds"] = engine.preparation_seconds
        stage("warmup")
        for _ in range(warmup):
            update()
        completed = warmup
        torch.cuda.synchronize(device)
        engine.inspect(completed)
        stage("calibration")
        start = time.perf_counter()
        for _ in range(5):
            update()
        torch.cuda.synchronize(device)
        seconds_per_update = (time.perf_counter() - start) / 5
        completed += 5
        engine.inspect(completed)
        steps = max(20, math.ceil(window_seconds / seconds_per_update))
        burst_steps = max(1, int(0.1 / seconds_per_update)) if timer else steps
        result.update(
            calibrated_steps=steps, calibration_seconds_per_update=seconds_per_update,
            burst_steps=burst_steps,
            setup_seconds=time.monotonic() - began,
            setup_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            setup_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
        torch.cuda.reset_peak_memory_stats(device)
        stage("measurement")
        for window in range(2 * windows):
            # Inspection and JSON writes happen outside these timing boundaries.
            seconds, samples = 0.0, []
            for offset in range(0, steps, burst_steps):
                count = min(burst_steps, steps - offset)
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                for _ in range(count):
                    update()
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                seconds += elapsed
                if timer:
                    samples.append(dict(steps=count, seconds=elapsed, **timer.sample()))
            completed += steps
            engine.inspect(completed)
            measurement = dict(
                window=window, steps=steps, seconds=seconds,
                tokens_per_second=steps * batch * length / seconds,
            )
            if timer:
                measurement.update(optimizer_window(samples))
            result["measurements"].append(measurement)
            result.update(measurement_summary(result["measurements"], batch, length))
            if timer:
                result["throughput_stable"] = result["stable"]
                result.update(optimizer_summary(result["measurements"]))
                result["stable"] = result["stable"] and result["optimizer_stable"]
            result.update(
                successful_updates=completed,
                measurement_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                measurement_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
            stage("measurement")
            if window + 1 == windows and result["stable"]:
                break
        result.update(status="ok", stage="complete", final_loss=engine.optimizer.losses.item(),
                      final_grad_norm=engine.optimizer.norms.item())
    except Exception as error:
        result.update(status=classify_failure(error), error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        if device is not None and device.type == "cuda":
            # Allocator counters are host-side; do not synchronize a failed CUDA context.
            try:
                result["failure_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                result["failure_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            except Exception:
                pass
    result["elapsed_seconds"] = time.monotonic() - began
    atomic_json(destination, result)
    return result
