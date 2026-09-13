---
title: "Packed training: GH200 validation and benchmarks"
description: "Historical report; retained as an experimental record."
archive: true
---

> **Historical report.** Settings and conclusions describe the recorded experiment. See the [current documentation](../performance/training.md) for maintained guidance.

# Packed training: GH200 validation and benchmarks

Validated on 2026-09-08 in SLURM job `2151572`, account `naiss2026-3-205-gpu`,
with `--gpus 1` on aarch64 node `n472`. The environment sourced `~/.bashrc`
before running UV. Hardware: NVIDIA GH200 120GB; Python 3.12.13, PyTorch 2.14.0,
CUDA 13.0. The allocation reported approximately 95.6 GiB of device memory.

## Throughput

The 20,403,520-parameter preset uses BF16 autocast, SDPA, fused AdamW, complete
parameter mixing before each optimizer update, and a 32,768-token global batch.
Local batches contain eight sequences at N=4 and four at N=8, each 1,024 tokens.
Packed and sequential workers have identical initial weights, inputs, and update
semantics. Every candidate runs in a separate process. Measurements use 10 warmup
updates followed by 50 timed updates, with CUDA synchronization at measurement
boundaries. Compilation and initialization are excluded.

| Execution | Workers | Sequential tok/s | Packed tok/s | Speedup | Sequential peak GiB | Packed peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| Uncompiled | 4 | 248,201 | 545,765 | 2.20× | 5.69 | 17.82 |
| Uncompiled | 8 | 127,633 | 492,920 | 3.86× | 5.18 | 19.19 |
| Compiled | 4 | 692,118 | 1,044,051 | 1.51× | 2.86 | 6.52 |
| Compiled | 8 | 352,963 | 885,148 | 2.51× | 3.78 | 8.01 |

These are synthetic single-run measurements, not confidence intervals. They
exclude data loading, evaluation, checkpoint writes, and physical network latency.
Packing uses more activation memory; compilation substantially reduces that
overhead in this workload. No performance claim is made for other model sizes.

## Component timings

A separate 50-update pass synchronizes around each component. These timings
include launch/synchronization overhead and should not be summed to reconstruct
end-to-end throughput. Compute includes zeroing gradients, forward, loss, and backward.

| Execution | Workers | Mode | Compute ms | Clip ms | Mix ms | AdamW ms |
|---|---:|---|---:|---:|---:|---:|
| Uncompiled | 4 | sequential | 126.23 | 3.18 | 0.59 | 1.57 |
| Uncompiled | 4 | packed | 55.32 | 3.58 | 0.60 | 2.60 |
| Uncompiled | 8 | sequential | 248.01 | 6.30 | 1.10 | 3.00 |
| Uncompiled | 8 | packed | 56.92 | 7.25 | 1.10 | 5.10 |
| Compiled | 4 | sequential | 41.96 | 3.26 | 0.59 | 1.68 |
| Compiled | 4 | packed | 26.17 | 3.50 | 0.61 | 2.66 |
| Compiled | 8 | sequential | 82.69 | 6.48 | 1.10 | 3.29 |
| Compiled | 8 | packed | 26.65 | 7.07 | 1.11 | 5.15 |

## Correctness and profiling

- The full test suite passed on the compute node: **63 tests**. Ruff checks and
  formatting checks passed, and the SLURM script passed `bash -n`.
- Tests cover seeded initialization equality, FP64 first/second derivatives,
  independent local gradients, mixing order/topologies, arena aliasing and moment
  edits, averaged evaluation, deterministic resume, compiled BF16 execution, and
  fused AdamW parameter/moment parity. The CUDA optimizer comparison uses matched
  gradients after separately checking BF16 gradient agreement, to isolate
  optimizer correctness from rounding near zero gradients.
- Profiler traces show packed projections as `aten::bmm`, including operands
  `[8, 4096, 320] × [8, 320, 320]`. Their leading dimension represents workers;
  forward/backward does not loop over local models. SDPA dispatches to cuDNN
  attention on this platform.
- A 20M, four-worker CLI smoke run completed two updates over 65,536 C4 targets.
  It wrote averaged/local weight exports and resume checkpoints. Its validation
  cache contains only 8,193 targets and is explicitly marked incomplete.

## Reproduction

Submit from the repository root:

```bash
mkdir -p runs
sbatch scripts/slurm.sh benchmark-packed --config configs/20m.yaml \
  --num-models 4 8 --warmup 10 --steps 50 --output runs/packed-benchmarks-gh200
sbatch scripts/slurm.sh benchmark-packed --config configs/20m.yaml \
  --num-models 4 8 --warmup 10 --steps 50 --set runtime.compile=true \
  --output runs/packed-benchmarks-gh200-compiled
```

Each output directory contains resolved configurations, per-worker JSON results,
logs, Chrome profiler traces, and `summary.json`. These generated artifacts live
under gitignored `runs/`; this report retains the measured results.
