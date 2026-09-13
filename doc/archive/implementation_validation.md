---
title: "Implementation validation \u2014 2026-09-08"
description: "Historical report; retained as an experimental record."
archive: true
---

> **Historical report.** Settings and conclusions describe the recorded experiment. See the [current documentation](../results/overview.md) for maintained guidance.

# Implementation validation — 2026-09-08

Environment: Python 3.12.14, PyTorch 2.14.0+cu130, NVIDIA RTX 5000 Ada Generation
(30 GiB), driver 595.71.05. Dependencies are locked by UV.

## Correctness and packaging

All 28 tests pass on this machine, including CUDA checks:

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

Loading the trained epoch-2 safetensors weights directly into the reference
attention model gave loss 9.1655901 on the same 32,768 targets, a difference of
approximately 0.0000048 nats under BF16 AMP.

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

Exact measurements are in [benchmark_results.json](../data/performance/benchmark_results.json).
Synthetic rates exclude data loading, evaluation, checkpointing, and compilation;
the full campaign records observed training throughput separately.

## Sequential buffered loader checks

The buffered loader uses the unchanged token binaries and original sequence
boundaries. New tests cover complete prefix coverage, cross-file targets, a short
final range, microbatches spanning several buffers, equal/different seeds, and
prefetch-independent ordering. Resume is tested at every cursor on a fixture;
read traces confirm earlier ranges are skipped. Training replay tests use tiny
buffers so optimizer updates and virtual epochs cross ranges, and switch prefetch
off on resume. CPU final weights match exactly; CUDA weights pass numerical checks.

Instrumented short file reads confirm contiguous access within each range.
Weak references to allocated token arrays confirm no more than two are resident,
including transitions. Reader errors propagate; interruption and normal completion
release buffers and worker threads. Validation tests preserve the original fixed
subset, reuse it without further reads, and match inputs, targets, losses, and
last-block padding. Legacy training checkpoint resumes are rejected explicitly. Evaluating the old
`runs/smoke/final.pt` checkpoint with the new full-validation reader reproduces
its original loss exactly: 9.165585288210423 over 32,768 targets.

Data-only measurements used the real C4 cache, context 1024, microbatch 16,
seed 42, one CPU thread, and 98,311 blocks (100,670,464 targets): three complete
64 MiB ranges plus a seven-block final range. Wall time includes reading,
allocation, shuffling, gathering, and conversion to CPU int64 microbatches. It
excludes checksum verification, H2D transfer, and model computation.

| Prefetch | Trial times (s) | Encoded token throughput (MiB/s) |
|---|---|---|
| disabled | 0.164, 0.135 | 1,174, 1,418 |
| enabled | 0.156, 0.123 | 1,232, 1,560 |

Raw results and settings are in [buffered_loader_results.json](../data/performance/buffered_loader_results.json).
The OS page cache was not cleared. These short local measurements are not an
old-loader comparison and do not establish a speedup on slower disks.

A compiled BF16 C4 smoke run used 64 KiB buffers to exercise eight ranges over
262,144 training targets with microbatch 16 and two virtual epochs. It completed
with subset losses 9.38870 and 9.13163, and final loss 9.13656 over 32,768 targets
in the **truncated smoke validation cache**. Artifacts are under
`runs/smoke-buffered`. The changed ordering means its loss is not a controlled
performance comparison with the earlier smoke run.

## Full campaign

The runner is configured for twelve complete runs with full final validation,
one process per GPU. The synthetic estimate is 16.1 GPU hours / 8.0 wall hours
on two GPUs, plus evaluation and checkpoint overhead.

The original campaign was gracefully stopped before editing source. Its six
completed 20M results and two interrupted 50M checkpoints remain under
`runs/campaign/`. The interrupted 50M runs committed 529,579,008 and 547,278,848
targets respectively. No old results or checkpoints were removed.

The fresh twelve-run buffered campaign uses `runs/campaign-buffered/`, with the
same search, full-validation promotion protocol, and existing model microbatch /
compilation selections. Live status and results are stored in that new directory. A finished campaign
has `complete.json`; until that file exists, the complete training/tuning campaign
has not finished. `report.md`, `comparison.json`, `learning-curves.png`, and
`selected-{20m,50m,90m}.yaml` are generated as stages finish.
