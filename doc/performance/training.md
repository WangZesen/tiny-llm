---
title: Training and packed performance
description: GH200 throughput and memory measurements, with historical recipes and timing boundaries kept explicit.
---

These measurements explain execution choices in the implementation. They use historical 32,768-target batches, not the current 131,072-target tuning recipes. Compare throughput within the matched experiment that produced it.

## Ordinary training on GH200

The experiment uses a 20,403,520-parameter model, context 1,024, BF16 autocast with FP32 weights, fused AdamW, and real C4 loading. Each trial warms up for 20 updates and measures three 100-update windows; the reported rate is their median. Compilation and initialization are outside these steady-state windows.

| Execution | Microbatch | Targets/second | Peak allocated GiB |
|---|---:|---:|---:|
| Original computation, eager | 8 | 245,035 | 4.53 |
| Original computation, compiled | 8 | 713,010 | 1.70 |
| Original computation, compiled | 16 | 1,194,163 | 2.94 |
| Compiled, uncached RoPE | 32 | 1,267,517 | 5.38 |
| Compiled, cached RoPE | 32 | 1,636,281 | 5.39 |

Compilation increases throughput approximately 2.9× at microbatch 8. Larger microbatches reduce accumulation and launch overhead while preserving the effective batch in this experiment. Cached RoPE tables improve the controlled microbatch-32 measurement approximately 28%.

The end-to-end effect is smaller. In matched shortened training over 100,000,768 targets, session time including full validation falls from 750.27 to 369.98 seconds, approximately 2.03×. Full-validation loss differs by −0.001231 nats in this single-seed comparison; it does not establish better model quality.

These records used evaluation batch size 8. Current evaluation defaults use 128 sequences. The synchronous 20M preset now uses a 128-sequence training microbatch; 50M/90M retain 32. Historical timings are not fresh measurements of these newer defaults.

## Packed versus sequential workers

Measured on September 8, 2026, in Slurm job 2151572 on GH200, with complete parameter mixing and a 32,768-target global batch. Four workers each receive eight sequences; eight workers each receive four. Each candidate runs separately with ten warmup updates and 50 timed updates.

| Execution | Workers | Sequential targets/s | Packed targets/s | Speedup | Sequential GiB | Packed GiB |
|---|---:|---:|---:|---:|---:|---:|
| Uncompiled | 4 | 248,201 | 545,765 | 2.20× | 5.69 | 17.82 |
| Uncompiled | 8 | 127,633 | 492,920 | 3.86× | 5.18 | 19.19 |
| Compiled | 4 | 692,118 | 1,044,051 | 1.51× | 2.86 | 6.52 |
| Compiled | 8 | 352,963 | 885,148 | 2.51× | 3.78 | 8.01 |

These synthetic measurements exclude loading, evaluation, checkpoint writes, compilation, and physical network latency. Packing uses more activation memory; compilation reduces that overhead in this workload. Separate synchronized component timings include instrumentation overhead and should not be summed to reconstruct end-to-end throughput.

## Reproduction and evidence

The [benchmarking guide](../guides/benchmarking.md) describes current commands. Reproducing historical measurements requires their documented batch, topology, warmup, and timing settings rather than every current preset default.

- [GH200 data and profiling records](../data/performance/gh200_results.json)
- [Complete GH200 training investigation](../archive/gh200_performance.md)
- [Packed timings, correctness checks, and profiler evidence](../archive/packed_benchmarks.md)
- [Earlier RTX and buffered-loader validation](../archive/implementation_validation.md)

The archived reports retain original environments and measurement context. Synthetic update throughput, real-data update throughput, and complete-session time answer different questions and remain labeled separately.
