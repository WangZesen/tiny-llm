"""Isolated throughput benchmarks and a resumable, bounded two-GPU sweep."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch
from loguru import logger

from tiny_llm.config import PRESETS, Config, ModelConfig, load_config, save_config
from tiny_llm.data import TokenCache, fingerprint, training_boundaries
from tiny_llm.model import Llama
from tiny_llm.runtime import (
    actual_backend,
    atomic_json,
    attention_kernels,
    environment,
    setup_runtime,
)
from tiny_llm.train import loss_function, make_optimizer


def benchmark_worker(config: Config, destination: Path, warmup: int = 3, steps: int = 8):
    if config.decentralized is not None:
        raise ValueError("use benchmark-packed for decentralized training")
    device = setup_runtime(config)
    model = Llama(config.model, actual_backend(config)).to(device)
    optimizer = make_optimizer(model, config, device)
    kernels = attention_kernels(model, config, device)
    loss = loss_function(model, config, device)
    batch = config.training.micro_batch_size
    x = torch.randint(config.model.vocab_size, (batch, config.model.context_length), device=device)
    y = torch.randint(config.model.vocab_size, x.shape, device=device)
    accum = config.training.batch_tokens // (batch * config.model.context_length)
    if accum < 1 or config.training.batch_tokens % (batch * config.model.context_length):
        raise ValueError("benchmark microbatch must divide the effective batch")

    def update():
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            value = loss(x, y) / config.training.batch_tokens
            value.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.optimizer.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        return value.detach()

    start = time.monotonic()
    for _ in range(warmup):
        update()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    warmup_seconds = time.monotonic() - start
    start = time.monotonic()
    for _ in range(steps):
        value = update()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - start
    if not torch.isfinite(value).item():
        raise FloatingPointError("benchmark produced nonfinite loss")
    memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    total_memory = (
        torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 1
    )
    result = dict(
        status="ok" if memory < 0.9 * total_memory else "memory_limit",
        synthetic=True,
        tokens_per_second=steps * config.training.batch_tokens / elapsed,
        seconds_per_step=elapsed / steps,
        warmup_seconds=warmup_seconds,
        peak_memory_bytes=memory,
        total_memory_bytes=total_memory,
        micro_batch_size=batch,
        compile=config.runtime.compile,
        attention_kernels=kernels,
        environment=environment(),
    )
    atomic_json(destination, result)
    return result


def benchmark(config: Config, output: Path) -> dict:
    if config.decentralized is not None:
        raise ValueError("use benchmark-packed for decentralized training")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for batch in (4, 8, 16, 32):
        if batch * config.model.context_length > config.training.batch_tokens:
            continue
        for compiled in (False,) if config.runtime.deterministic else (False, True):
            name = f"micro-{batch}-compile-{int(compiled)}"
            candidate = config.model_copy(deep=True)
            candidate.training.micro_batch_size = batch
            candidate.runtime.compile = compiled
            path = output / f"{name}.yaml"
            save_config(candidate, path)
            result_path = output / f"{name}.json"
            identity = fingerprint(candidate.model_dump(mode="json"))
            if result_path.exists():
                cached = json.loads(result_path.read_text())
                if cached.get("config_identity") == identity:
                    results.append(cached)
                    continue
            logger.info("Benchmark {} on {}", name, candidate.runtime.device)
            with (output / f"{name}.log").open("w") as log:
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "tiny_llm",
                        "benchmark-worker",
                        "--config",
                        str(path),
                        "--output",
                        str(result_path),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if process.returncode == 0:
                result = json.loads(result_path.read_text())
            else:
                result = dict(
                    status="failed",
                    returncode=process.returncode,
                    micro_batch_size=batch,
                    compile=compiled,
                    log=str(output / f"{name}.log"),
                )
            result["config_identity"] = identity
            atomic_json(result_path, result)
            results.append(result)
    successful = [row for row in results if row["status"] == "ok"]
    if not successful:
        raise RuntimeError(f"all benchmarks failed; see {output}")
    best = max(successful, key=lambda row: row["tokens_per_second"])
    selected = config.model_copy(deep=True)
    selected.training.micro_batch_size = best["micro_batch_size"]
    selected.runtime.compile = best["compile"]
    save_config(selected, output / "selected.yaml")
    summary = dict(candidates=results, selected=best)
    atomic_json(output / "summary.json", summary)
    logger.success(
        "Selected microbatch={}, compile={}: {:,.0f} tokens/s",
        best["micro_batch_size"],
        best["compile"],
        best["tokens_per_second"],
    )
    return summary


def rank_results(runs: list[Path]) -> list[Path]:
    def key(path):
        result = json.loads((path / "result.json").read_text())
        config = load_config(path / "resolved.yaml")
        return (
            result["final_validation"]["loss"],
            config.optimizer.lr,
            config.optimizer.weight_decay,
            config.optimizer.beta2,
        )

    return sorted(runs, key=key)


def run_stage(configs: list[Config], gpus: list[str], stage_dir: Path):
    stage_dir.mkdir(parents=True, exist_ok=True)
    queue, running = [], {}
    for index, config in enumerate(configs):
        directory = config.runtime.output_dir
        if (directory / "result.json").exists():
            result = json.loads((directory / "result.json").read_text())
            if result.get("status") == "complete":
                continue
        path = stage_dir / f"run-{index:02d}.yaml"
        save_config(config, path)
        queue.append((config, path))
    failures = []
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        for process, _, _ in running.values():
            process.send_signal(signal.SIGTERM)

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while queue or running:
            for gpu in gpus:
                if stopping or gpu in running or not queue:
                    continue
                config, path = queue.pop(0)
                directory = config.runtime.output_dir
                directory.mkdir(parents=True, exist_ok=True)
                log = (directory / "process.log").open("a")
                command = [
                    sys.executable,
                    "-m",
                    "tiny_llm",
                    "train",
                    "--config",
                    str(path),
                    "--set",
                    "runtime.device=cuda:0",
                ]
                if (directory / "latest.pt").exists():
                    command += ["--resume", str(directory / "latest.pt")]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, TOKENIZERS_PARALLELISM="false")
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                running[gpu] = process, config, log
                logger.info("Started {} on GPU {} (PID {})", directory, gpu, process.pid)
            for gpu, (process, config, log) in list(running.items()):
                if process.poll() is None:
                    continue
                log.close()
                if (
                    process.returncode != 0
                    or not (config.runtime.output_dir / "result.json").exists()
                ):
                    failures.append(str(config.runtime.output_dir))
                logger.info(
                    "Finished {} with exit code {}", config.runtime.output_dir, process.returncode
                )
                del running[gpu]
            if stopping and not running:
                raise InterruptedError(
                    "campaign interrupted; rerun the same sweep command to resume"
                )
            if queue or running:
                time.sleep(5)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        for process, _, log in running.values():
            process.terminate()
            log.close()
    if failures:
        raise RuntimeError(f"stage failed; inspect and resume these runs: {failures}")


def sweep(base: Config, root: Path, benchmarks: Path, gpus: list[str]):
    if base.decentralized is not None:
        raise ValueError("the tuning campaign supports ordinary training only")
    import fcntl

    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("GPU list must be nonempty and unique")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".sweep.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _sweep(base, root, benchmarks, gpus)


def _sweep(base: Config, root: Path, benchmarks: Path, gpus: list[str]):
    cache = TokenCache(base.data.cache_dir)
    cache.validate_config(base)
    if not cache.manifest["validation_complete"]:
        raise ValueError("the tuning campaign requires the complete official validation split")
    templates = {}
    estimate = 0
    for preset, runs in (("20m", 6), ("50m", 4), ("90m", 2)):
        template = base.model_copy(deep=True)
        template.model = ModelConfig(**(base.model.model_dump() | PRESETS[preset]))
        selection = json.loads((benchmarks / preset / "summary.json").read_text())["selected"]
        template.training.micro_batch_size = selection["micro_batch_size"]
        template.runtime.compile = selection["compile"]
        templates[preset] = template
        tokens = (
            training_boundaries(template, template.model.parameter_count)[-1]
            * template.model.context_length
        )
        estimate += runs * tokens / selection["tokens_per_second"]
    campaign = dict(
        version=1,
        base=base.model_dump(mode="json"),
        gpus=gpus,
        estimated_training_gpu_hours=estimate / 3600,
        estimated_training_wall_hours=estimate / 3600 / len(gpus),
        note="Synthetic benchmark estimate excludes evaluation, compilation, and checkpoint overhead.",
    )
    identity = fingerprint(
        {
            "base": campaign["base"],
            "cache_identity": cache.manifest["identity"],
            "templates": {k: v.model_dump(mode="json") for k, v in templates.items()},
        }
    )
    existing = root / "campaign.json"
    if existing.exists() and json.loads(existing.read_text())["identity"] != identity:
        raise ValueError("campaign settings changed; use a new output directory")
    atomic_json(existing, campaign | {"identity": identity})
    logger.info(
        "Estimated training: {:.1f} GPU hours / {:.1f} wall hours on {} GPUs, excluding evaluation",
        estimate / 3600,
        estimate / 3600 / len(gpus),
        len(gpus),
    )
    configs = []
    for lr in (3e-4, 1e-3, 3e-3):
        for decay in (0.0, 0.1):
            config = templates["20m"].model_copy(deep=True)
            config.optimizer.lr, config.optimizer.weight_decay = lr, decay
            config.optimizer.beta2 = 0.95
            config.runtime.output_dir = root / f"20m-lr{lr:g}-wd{decay:g}"
            configs.append(config)
    run_stage(configs, gpus, root / "stage-20m")
    report(root)
    promoted = rank_results([c.runtime.output_dir for c in configs])[:2]
    configs = []
    for index, path in enumerate(promoted):
        source = load_config(path / "resolved.yaml")
        for beta2 in (0.95, 0.99):
            config = templates["50m"].model_copy(deep=True)
            config.optimizer = source.optimizer.model_copy(deep=True)
            config.optimizer.beta2 = beta2
            config.runtime.output_dir = root / f"50m-parent{index}-beta2{beta2:g}"
            configs.append(config)
    run_stage(configs, gpus, root / "stage-50m")
    report(root)
    promoted = rank_results([c.runtime.output_dir for c in configs])[:2]
    configs = []
    for index, path in enumerate(promoted):
        config = templates["90m"].model_copy(deep=True)
        config.optimizer = load_config(path / "resolved.yaml").optimizer.model_copy(deep=True)
        config.runtime.output_dir = root / f"90m-parent{index}"
        configs.append(config)
    run_stage(configs, gpus, root / "stage-90m")
    report(root)
    atomic_json(root / "complete.json", dict(status="complete", runs=12, identity=identity))


def report(root: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for preset, ax in zip(PRESETS, axes, strict=True):
        completed = []
        for directory in sorted(root.glob(f"{preset}-*")):
            metrics = directory / "metrics.jsonl"
            if metrics.exists():
                values = [
                    json.loads(line) for line in metrics.read_text().splitlines() if line.strip()
                ]
                curve = {
                    value["tokens"]: value["loss"]
                    for value in values
                    if value["event"] == "validation"
                }
                if curve:
                    ax.plot(list(curve), list(curve.values()), label=directory.name)
            if (directory / "result.json").exists():
                completed.append(directory)
        for directory in rank_results(completed):
            result = json.loads((directory / "result.json").read_text())
            config = load_config(directory / "resolved.yaml")
            training_metrics = [
                value
                for value in (
                    json.loads(line)
                    for line in (directory / "metrics.jsonl").read_text().splitlines()
                )
                if value["event"] == "train"
            ]
            throughput = sorted(value["tokens_per_second"] for value in training_metrics)
            median_throughput = throughput[len(throughput) // 2] if throughput else None
            rows.append(
                dict(
                    run=directory.name,
                    preset=preset,
                    parameters=result["parameters"],
                    lr=config.optimizer.lr,
                    weight_decay=config.optimizer.weight_decay,
                    beta2=config.optimizer.beta2,
                    loss=result["final_validation"]["loss"],
                    tokens=result["tokens"],
                    peak_memory_bytes=result["peak_memory_bytes"],
                    median_tokens_per_second=median_throughput,
                    seed=config.runtime.seed,
                    deterministic=config.runtime.deterministic,
                    checkpoint=str(directory / "final.pt"),
                )
            )
        if completed:
            best = rank_results(completed)[0]
            save_config(load_config(best / "resolved.yaml"), root / f"selected-{preset}.yaml")
        ax.set(title=preset, xlabel="Training targets", ylabel="Subset validation CE (nats)")
        if ax.lines:
            ax.legend(fontsize=6)
    fig.savefig(root / "learning-curves.png", dpi=160)
    plt.close(fig)
    atomic_json(root / "comparison.json", rows)
    text = "# C4 training comparison\n\nFinal-checkpoint full-validation losses. Exact seeds and runtime settings are recorded in comparison.json and each run's resolved.yaml.\n\n"
    text += "| Run | Parameters | LR | Decay | Beta2 | Final loss | Tokens | Median tok/s | Peak GiB |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for row in rows:
        speed = (
            f"{row['median_tokens_per_second']:,.0f}"
            if row["median_tokens_per_second"] is not None
            else "—"
        )
        text += f"| {row['run']} | {row['parameters']:,} | {row['lr']:g} | {row['weight_decay']:g} | {row['beta2']} | {row['loss']:.5f} | {row['tokens']:,} | {speed} | {row['peak_memory_bytes'] / 2**30:.2f} |\n"
    text += "\n![Learning curves](learning-curves.png)\n\nThis is a bounded single-seed recipe comparison, not a test of seed robustness or batch-size optimality.\n"
    (root / "report.md").write_text(text)
    return rows
