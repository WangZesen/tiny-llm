"""Benchmark the same complete updates and logging boundaries as GH200 training."""

import statistics
import time

import torch
from loguru import logger

from tiny_llm.data import BufferedTokenLoader, TokenCache, training_boundaries
from tiny_llm.runtime import append_metric, atomic_json, environment, setup_runtime
from tiny_llm.train import adaptive_consensus_schedule, learning_rate

from .engine import GH200Update


def benchmark_update(config, destination, warmup=20, steps=100, windows=3,
                     data_mode="synthetic", profile=False):
    from tiny_llm.benchmark import benchmark_identity

    if min(warmup, steps, windows) < 1 or data_mode not in ("synthetic", "real"):
        raise ValueError("positive benchmark lengths and synthetic/real data mode required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    device = setup_runtime(config)
    engine = GH200Update(config, device)
    metadata = environment()
    boundaries = training_boundaries(config, config.model.parameter_count)
    schedule = adaptive_consensus_schedule(config, boundaries)
    length, batch = config.model.context_length, config.training.micro_batch_size
    step_blocks = config.training.batch_tokens // length
    loader = None
    protocol = dict(version=3, backend="gh200", warmup=warmup, steps=steps, windows=windows,
                    data_mode=data_mode, profile=profile, logging_cadence=config.training.log_every)
    if data_mode == "real":
        needed = (warmup + steps * windows + int(profile)) * step_blocks
        if needed > boundaries[-1]:
            raise ValueError(f"benchmark needs {needed} blocks but recipe has {boundaries[-1]}")
        cache = TokenCache(config.data.cache_dir)
        cache.validate_config(config)
        loader = BufferedTokenLoader(cache, "train", length, boundaries[-1],
                                     config.data.shuffle_group_size, seed=config.runtime.seed,
                                     prefetch=config.data.prefetch)
        next_batch = loader.next_batch
        metadata["cache_identity"] = protocol["cache_identity"] = cache.manifest["identity"]
    else:
        count = step_blocks if config.decentralized else batch
        x = torch.randint(config.model.vocab_size, (count, length), device=device)
        y = torch.randint(config.model.vocab_size, x.shape, device=device)

        def next_batch(count, device):
            return x[:count], y[:count]

    cursor = step = window_tokens = 0
    window_start = time.monotonic()

    def update():
        nonlocal cursor, step, window_tokens, window_start
        lr = learning_rate(config, (cursor + step_blocks) * length, boundaries[-1] * length)
        gamma = schedule.gamma(step, lr) if schedule else 1.
        engine.execute(next_batch, lr, gamma)
        norm = engine.optimizer.norms.max()
        step += 1
        cursor += step_blocks
        window_tokens += config.training.batch_tokens
        if step % config.training.log_every == 0 or cursor in boundaries:
            engine.inspect(step)
            row = dict(event="train", step=step, tokens=cursor * length,
                       loss=engine.optimizer.loss_sum.item() / window_tokens, lr=lr,
                       grad_norm=norm.item(), local_losses=engine.optimizer.losses.tolist(),
                       local_grad_norms=engine.optimizer.norms.tolist(),
                       tokens_per_second=window_tokens / (time.monotonic() - window_start))
            append_metric(destination.with_suffix(".metrics.jsonl"), row)
            logger.info("Benchmark step {} | loss {:.4f} | {:,.0f} tok/s",
                        step, row["loss"], row["tokens_per_second"])
            engine.reset_window()
            window_tokens, window_start = 0, time.monotonic()

    try:
        engine.prepare()
        window_start = time.monotonic()
        for _ in range(warmup):
            update()
        engine.inspect(step)
        torch.cuda.synchronize(device)
        startup_seconds = time.monotonic() - began
        torch.cuda.reset_peak_memory_stats(device)
        measurements = []
        for window in range(windows):
            torch.cuda.synchronize(device)
            start = time.monotonic()
            for _ in range(steps):
                update()
            torch.cuda.synchronize(device)
            seconds = time.monotonic() - start
            engine.inspect(step)
            measurements.append(dict(window=window, seconds=seconds,
                                     tokens_per_second=steps * config.training.batch_tokens / seconds))
        memory = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        if profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as trace:
                update()
                engine.inspect(step)
            trace.export_chrome_trace(str(destination.with_suffix(".trace.json")))
        rate = statistics.median(row["tokens_per_second"] for row in measurements)
        result = dict(status="ok", backend="gh200", synthetic=data_mode == "synthetic",
                      tokens_per_second=rate, seconds_per_step=config.training.batch_tokens / rate,
                      startup_seconds=startup_seconds, preparation_seconds=engine.preparation_seconds,
                      elapsed_seconds=time.monotonic() - began, peak_memory_bytes=memory,
                      peak_reserved_bytes=reserved,
                      successful_updates=engine.inspect(), cursor=cursor,
                      clipping_counts=engine.optimizer.clip_counts.tolist(),
                      total_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
                      micro_batch_size=batch, local_micro_batch_size=batch, num_models=engine.workers,
                      compile=True, compile_mode="default",
                      requested_compile_mode=config.runtime.compile_mode,
                      sdpa_backend=config.runtime.sdpa_backend, cpu_threads=config.runtime.cpu_threads,
                      environment=metadata, protocol=protocol, measurements=measurements,
                      config_identity=benchmark_identity(config, metadata, protocol))
        atomic_json(destination, result)
        return result
    finally:
        if loader is not None:
            loader.close()
