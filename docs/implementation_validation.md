# Implementation validation — 2026-09-08

Environment: Python 3.12.14, PyTorch 2.14.0+cu130, NVIDIA RTX 5000 Ada Generation
(30 GiB), driver 595.71.05. Dependencies are locked by UV.

## Correctness and packaging

The test suite passes on this machine, including CUDA checks:

- Causality, parameter counts, tied weights, FP64 model/loss/gradient parity.
- Parameter-level gradcheck, gradgradcheck, and finite-difference Hessian-vector
  products through the complete miniature reference model.
- BF16 CUDA fast/reference parity: relative logit error below 2%, aggregate
  gradient error below 5%, and close cross-entropy.
- Uneven gradient accumulation and token-based schedule/epoch accounting.
- Exact CPU replay after checkpoint interruption, plus numerical CUDA replay
  with BF16 AMP and fused AdamW.
- EOS insertion, deterministic preparation, cross-shard packing, validation
  padding, cache checksums, and incompatible resume rejection.
- A simulated twelve-run campaign checks recipe promotion and report generation.

Ruff lint/format checks pass. UV builds both a source distribution and wheel.

## Real-data smoke checks

The 20M model completed a 262,144-target C4 smoke run with two virtual epochs.
Its subset validation loss declined from 9.41180 to 9.15760. Final loss on the
**truncated smoke validation cache** was 9.1655853; reloading `final.pt` and
evaluating again reproduced this value.

A compiled run with 261,120 targets and an uneven 133,120-target first epoch also
completed. Its smoke validation loss was 9.03376. These runs have different
budgets and update boundaries and are not a recipe or backend quality comparison.
Neither is a full-C4 validation result.

## Throughput benchmark

All candidates preserve an effective batch of 32,768 targets and context 1024.
Each candidate runs in an isolated process, warms up for three optimizer updates,
and measures eight updates. Compilation time is excluded from steady-state
throughput. The profiler confirms FlashAttention forward and backward dispatch.

| Preset | Selected microbatch | Compilation | Targets/s | Peak allocated GiB |
|---|---:|---|---:|---:|
| 20M | 16 | enabled | 352,068 | 2.90 |
| 50M | 16 | enabled | 204,059 | 4.90 |
| 90M | 8 | enabled | 114,909 | 4.80 |

Exact measurements are in [benchmark_results.json](benchmark_results.json).
Synthetic rates exclude data loading, evaluation, checkpointing, and compilation;
the full campaign records observed training throughput separately.

## Full campaign

The runner is configured for twelve complete runs with full final validation,
one process per GPU. The synthetic estimate is 16.1 GPU hours / 8.0 wall hours
on two GPUs, plus evaluation and checkpoint overhead.

Live status and results are stored under `runs/campaign/`. A finished campaign
has `complete.json`; until that file exists, the complete training/tuning campaign
has not finished. `report.md`, `comparison.json`, `learning-curves.png`, and
`selected-{20m,50m,90m}.yaml` are generated as stages finish.
