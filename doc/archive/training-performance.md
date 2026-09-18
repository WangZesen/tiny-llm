---
archive: true
title: "Earlier training performance"
description: GH200 throughput and memory measurements, with historical recipes and timing boundaries kept explicit.
---

These measurements explain execution choices in the implementation. The ordinary
and packed throughput experiments use historical 32,768-target batches; the loading
and Slurm startup experiments use the current 131,072-target packed-8 recipe.
Compare results within the matched experiment that produced them.

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

## Data loading on GH200

On September 16, 2026, jobs **2532756**, **2532892**, and **2533005** profiled real
C4 loading during packed-8 training using account `naiss2026-4-1590-gpu`. Each
trial completed 2,100 production updates (275,251,200 targets), crossing four
128 MiB shard-group boundaries. The first 100 updates were excluded from steady
timings. The diagnostic harness then requested a clean stop through the training
signal handler, avoiding final validation and checkpoint writes.

Prefetching worked in all 20 measured transitions with prefetch enabled: the next
group was already ready. In the initial trial, reading each group took 0.16–0.22
seconds in the background, while consuming it took about 46 seconds. Activating
the prefetched group took 0.8–1.7 ms. Disabling prefetch in the control run caused
18–40 ms synchronous pauses at group boundaries, even with warm file caches.
The first group is still read synchronously, and a sufficiently delayed filesystem
read can still make training wait.

The remaining measurable cost was batch assembly: ordinary `int64` CPU arrays
were filled, then copied into pinned memory for GPU transfer. The updated loader
allocates pinned tensors first and fills their NumPy views directly, eliminating
that extra host copy. Sample order, group size, two-buffer prefetching, and loader
checkpoint identity are unchanged.

The two comparisons used opposite run orders within separate single-GPU
allocations. Batch timings are instrumented wall time through copy enqueue;
GPU-copy completion is excluded. CUDA events sampled transfers every 64 batches
without adding per-batch synchronization.

| Job / run order | Baseline batch mean | Updated batch mean | Reduction | Training throughput change |
|---|---:|---:|---:|---:|
| 2532892: updated, baseline | 0.457 ms | 0.390 ms | 14.7% | +0.004% |
| 2533005: baseline, updated | 0.370 ms | 0.314 ms | 15.3% | −0.024% |

Batch p99 times fell from 0.842 to 0.597 ms and from 0.698 to 0.426 ms.
Job **2533594** additionally alternated baseline/updated/updated/baseline for four
500-update windows within one training process and one compiled model, after
100 warmup updates. Mean batch preparation fell from 0.381 to 0.326 ms (14.3%);
throughput differed by −0.33%. All four additional prefetched groups were ready.
Loading was already under 1% of the roughly 85–90 ms update time, so the saved
host work did not produce a material overall throughput change. These runs found
no disk bottleneck with default prefetching; they do not rule out intermittent
shared-filesystem stalls outside the measured windows.

All 63 targeted CPU tests passed. Four CUDA cases passed for each source variant,
checking variable batch sizes, group crossings, validation padding, and asynchronous
copies on a nondefault stream after temporary host buffers are released. The
[loading measurements](../data/performance/slurm-loader.json) preserve stage
distributions, group timings, source hashes, versions, and Slurm accounting.
All four jobs completed successfully, totaling 14,700 updates and 31:17 GPU-minutes.

## Slurm training startup

On September 16, 2026, jobs **2529889** and **2529890** compared training with and
without its automatic full-cache checksum scan on GH200, using account
`naiss2026-4-1590-gpu`. Every baseline launch hashed **42.9 GiB across 687 shards**,
even though the shortened run trained on only 10,223,616 targets. The scan took
122.01 seconds in the first process and 30.23 seconds after the node's file cache
was warm. Metadata-only checks took 0.23 and 0.04 seconds, respectively.

Each allocation ran both variants in fresh processes, with their order reversed
between allocations. Both allocations landed on `n135`. Inductor and Triton
caches were fresh for every process; OS page caches were not cleared. The table
compares the same process position across allocations. Times start at Python
entry, exclude queue/launcher time, and have one-second timestamp resolution.

| Process position | Baseline to training announcement | Updated to training announcement | Reduction | Baseline to first update | Updated to first update |
|---|---:|---:|---:|---:|---:|
| First | 166 s | 53 s | 68% | 183 s | 101 s |
| Second | 37 s | 7 s | 81% | 54 s | 25 s |

The remaining first-process setup included about 29 seconds of imports and
12 seconds of runtime initialization. The first update, including data loading
and compilation, took 17–47 seconds across these runs. These costs are now visible
in startup logs. Training and resume skip token-content checksums; manifest,
shard-size, and configuration checks remain. Rerun `prepare` with matching settings
for explicit full-cache verification.

The experiment used `configs/20m.yaml` followed by
`configs/packed8-20m-awc.yaml`, overriding both token-per-parameter ratios to 0.5
and checkpoint policy to `none`: one virtual epoch, 78 updates, followed by full
validation. Steps 21–78 sustained 1.50–1.57 million targets/s. Within each
allocation, updated throughput differed by +0.78% and +0.39%; these short runs
show no throughput regression. Production BF16 execution was nondeterministic:
paired final validation losses differed by −0.00433 and +0.00893 nats. Deterministic
CPU tests matched fresh and resumed weights exactly, including a run with
corruption confined to an unused training shard.

Both jobs completed successfully in 7:15 and 6:36, totaling 13:51 GPU-minutes.
The [startup measurements](../data/performance/slurm-startup.json) record stage
timings, source hashes, versions, loss windows, and comparison methodology.

## Reproduction and evidence

The [benchmarking guide](../guides/benchmarking.md) describes current commands. Reproducing historical measurements requires their documented batch, topology, warmup, and timing settings rather than every current preset default.

- [GH200 data and profiling records](../data/performance/gh200_results.json)
- [Complete GH200 training investigation](../archive/gh200_performance.md)
- [Packed timings, correctness checks, and profiler evidence](../archive/packed_benchmarks.md)
- [Earlier RTX and buffered-loader validation](../archive/implementation_validation.md)

The archived reports retain original environments and measurement context. Synthetic update throughput, real-data update throughput, and complete-session time answer different questions and remain labeled separately.
