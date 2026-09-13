---
title: Checkpoint-analysis performance
description: Historical GH200 measurements behind the retained Hessian-vector-product execution settings.
---

## Hessian-vector products

The tuning experiment uses the 20M checkpoint `20m-lr0.001-wd0.1`, epoch 1, context 1,024, reference attention, and FP32 with TF32. Gradient microbatches remain 32. Slurm job 2190245 measured the average of five warm calls, excluding tracing and compilation. Peak memory is allocated memory in decimal GB across startup and timed calls.

| Method | HVP batch | Seconds/batch | Tokens/second | Peak GB |
|---|---:|---:|---:|---:|
| Eager | 32 | 0.3661 | 89,510 | 61.7 |
| Forward-forward | 32 | 0.1747 | 187,615 | 21.3 |
| Forward-forward | 64 | 0.3417 | 191,791 | 42.3 |
| Forward-forward | 128 | 0.6762 | 193,847 | 84.3 |
| Forward-reverse | 32 | 0.1654 | 198,131 | 35.8 |
| Forward-reverse | 64 | 0.3233 | 202,737 | 71.2 |

The selected setting is compiled forward-over-reverse at batch 64, approximately 2.26× the eager throughput. It computes Pearlmutter's Hessian-vector product with `jvp(grad(loss))`, followed by a scalar dot product. Both derivatives and the reduction are compiled together. No production method constructs a dense Hessian.

The selected scalar differed from the TF32 eager reference by 0.0020%. Failed candidates and candidates exceeding 90% of GPU memory were excluded from selection. The historical forward-forward alternative is not a retained production API.

## Precision and interpretation

The separate paired BF16 investigation keeps FP32 weights and gradient accumulation while enabling BF16 autocast in model forward calls. Reference attention scores and softmax, normalization, cross-entropy, and scalar reductions follow the analysis precision policy. Autocast weight caching is disabled for functional derivatives.

Matched batch selections and random directions compare FP32 and BF16 calculations. The study examines scalar statistics, gradient/noise vector errors, and a counterfactual that applies the AMP Hessian to FP32 noise vectors. That comparison helps separate curvature error from a change in the evaluated direction.

The [complete historical report](../archive/analysis_performance.md) preserves per-checkpoint/case comparisons, runtime profiles, and validation checks. These are measurements on a particular model, checkpoint recipe, and GPU, not precision guarantees for future experiments.

## Current usage

Direct `analyze` supports FP32 execution and explicit `--amp`. The Slurm submitter defaults to the measured BF16 forward-AMP settings, FP32 parameters, TF32, compiled HVPs, and HVP batch 64. Gradient microbatch size comes from the run's resolved recipe.

A noise or random direction requires a full-epoch HVP pass. Default analysis uses 32 noise samples and 32 random directions per data case, so a full workload is substantially larger than one timed HVP batch. Seen/unseen cases, checkpoint count, compilation, and scheduling overhead affect complete-job runtime.

Use the [analysis guide](../guides/analysis.md) and [Slurm guide](../guides/slurm.md) for commands and resource estimates. The [analysis reference](../methods/analysis.md) defines precision, sampling, token weighting, provenance, and which settings invalidate reuse.
