"""Isolated packed/sequential comparisons with identical decentralized updates."""

import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from tiny_llm.config import Config, save_config
from tiny_llm.model import Llama
from tiny_llm.packed import PackedLlama
from tiny_llm.runtime import (
    actual_backend,
    atomic_json,
    clip_grad_norm_,
    environment,
    preserve_rng,
    setup_runtime,
)
from tiny_llm.train import loss_function, make_optimizer


def benchmark_packed_worker(
    config: Config, destination: Path, execution: str, warmup: int = 3, steps: int = 8
) -> dict:
    if config.decentralized and config.decentralized.adaptive_consensus is not None:
        raise ValueError("adaptive consensus is supported by training only, not packed benchmarks")
    config = Config.model_validate(config.model_dump())
    if config.decentralized is None or execution not in ("packed", "sequential"):
        raise ValueError("benchmark requires decentralized config and packed/sequential execution")
    if warmup < 1 or steps < 1:
        raise ValueError("warmup and steps must be positive")
    device = setup_runtime(config)
    n, batch, length = (
        config.decentralized.num_models,
        config.training.micro_batch_size,
        config.model.context_length,
    )
    packed = PackedLlama(config.model, n, actual_backend(config)).to(device)
    if execution == "packed":
        models = [packed]
    else:
        # Ordinary forward/backward, sharing the same mixing arena. The temporary
        # ordinary initializations must not change benchmark input RNG.
        models = []
        with preserve_rng():
            for i in range(n):
                model = Llama(config.model, actual_backend(config)).to(device)
                for name, parameter in model.named_parameters():
                    parameter.data = packed.get_parameter(name)[i].detach()
                models.append(model)
    optimizers = [make_optimizer(model, config, device) for model in models]
    losses = [loss_function(model, config, device) for model in models]
    x = torch.randint(config.model.vocab_size, (n, batch, length), device=device)
    y = torch.randint(config.model.vocab_size, x.shape, device=device)

    def compute():
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        if execution == "packed":
            value = losses[0](x, y).sum()
            value.backward()
        else:
            values = []
            for i, loss in enumerate(losses):
                value = loss(x[i], y[i]) / (batch * length)
                value.backward()
                values.append(value.detach())
            value = torch.stack(values).sum()
        return value.detach()

    def clip():
        if execution == "packed":
            optimizers[0].clip_grad_norm_(config.optimizer.grad_clip)
        else:
            for model in models:
                clip_grad_norm_(model.parameters(), config.optimizer.grad_clip)

    def update():
        for optimizer in optimizers:
            optimizer.step()

    def combine(step):
        packed.mix_(config.decentralized.topology, step)

    def update_operations(step):
        operations = (("mixing", lambda: combine(step)), ("optimizer", update))
        return operations if config.decentralized.scheme == "awc" else operations[::-1]

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    start = time.monotonic()
    for step in range(warmup):
        compute()
        clip()
        for _, operation in update_operations(step):
            operation()
    synchronize()
    warmup_seconds = time.monotonic() - start
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    for step in range(warmup, warmup + steps):
        value = compute()
        clip()
        for _, operation in update_operations(step):
            operation()
    synchronize()
    elapsed = time.monotonic() - start
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    if not torch.isfinite(value).item():
        raise FloatingPointError("benchmark produced nonfinite loss")

    # Component measurements are a separate pass, so synchronization does not
    # inflate the end-to-end throughput measurement above.
    timings = {key: 0.0 for key in ("compute", "clipping", "mixing", "optimizer")}
    for step in range(warmup + steps, warmup + 2 * steps):
        for key, operation in (
            ("compute", compute),
            ("clipping", clip),
            *update_operations(step),
        ):
            synchronize()
            start = time.monotonic()
            operation()
            synchronize()
            timings[key] += (time.monotonic() - start) / steps

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
        with torch.profiler.record_function(f"{execution}_forward_backward"):
            compute()
        synchronize()
    operators = [
        {"name": event.key, "count": event.count, "input_shapes": event.input_shapes}
        for event in profile.key_averages(group_by_input_shape=True)
        if event.key in ("aten::bmm", "aten::mm", "aten::matmul") or "attention" in event.key
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    profile.export_chrome_trace(str(destination.with_suffix(".trace.json")))
    result = dict(
        status="ok",
        synthetic=True,
        execution=execution,
        num_models=n,
        topology=config.decentralized.topology,
        scheme=config.decentralized.scheme,
        global_batch_tokens=config.training.batch_tokens,
        local_micro_batch_size=batch,
        steps=steps,
        warmup=warmup,
        warmup_seconds=warmup_seconds,
        seconds_per_step=elapsed / steps,
        tokens_per_second=steps * config.training.batch_tokens / elapsed,
        peak_memory_bytes=peak,
        component_seconds=timings,
        operators=operators,
        environment=environment(),
    )
    atomic_json(destination, result)
    return result


def benchmark_packed(
    config: Config, output: Path, num_models: list[int] = (4, 8), warmup: int = 3, steps: int = 8
) -> dict:
    if config.decentralized and config.decentralized.adaptive_consensus is not None:
        raise ValueError("adaptive consensus is supported by training only, not packed benchmarks")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for n in num_models:
        raw = config.model_dump(mode="json")
        if n < 1 or config.training.batch_tokens % (n * config.model.context_length):
            raise ValueError("benchmark global batch must divide evenly across num_models")
        raw["decentralized"] = {
            "num_models": n,
            "topology": (config.decentralized.topology if config.decentralized else "complete"),
            "scheme": config.decentralized.scheme if config.decentralized else "awc",
        }
        raw["training"]["micro_batch_size"] = config.training.batch_tokens // (
            n * config.model.context_length
        )
        candidate = Config.model_validate(raw)
        path = output / f"n-{n}.yaml"
        save_config(candidate, path)
        pair = {}
        for execution in ("sequential", "packed"):
            destination = output / f"n-{n}-{execution}.json"
            with destination.with_suffix(".log").open("w") as log:
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "tiny_llm",
                        "benchmark-packed-worker",
                        "--config",
                        str(path),
                        "--output",
                        str(destination),
                        "--execution",
                        execution,
                        "--warmup",
                        str(warmup),
                        "--steps",
                        str(steps),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if process.returncode == 0:
                pair[execution] = json.loads(destination.read_text())
            else:
                pair[execution] = dict(
                    status="failed",
                    returncode=process.returncode,
                    log=str(destination.with_suffix(".log")),
                )
        row = dict(num_models=n, **pair)
        if all(value["status"] == "ok" for value in pair.values()):
            row["speedup"] = (
                pair["sequential"]["seconds_per_step"] / pair["packed"]["seconds_per_step"]
            )
        results.append(row)
    summary = dict(comparisons=results)
    atomic_json(output / "summary.json", summary)
    if any(row[mode]["status"] != "ok" for row in results for mode in ("packed", "sequential")):
        raise RuntimeError(f"some benchmark workers failed; see {output / 'summary.json'}")
    return summary
