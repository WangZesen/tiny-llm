---
title: Optimizer tuning
description: Learning-rate response, AdamW moments, and the difference between tuned winners and matched mixing-order comparisons.
---

Learning rate and AdamW moment settings interact strongly in these small-model experiments. The useful region is not captured by a single learning rate shared across all methods. The current studies search synchronous training, adapt-while-combine (AWC), and adapt-then-combine (ATC) under the [128K-token protocol](../methods/protocol.md).

## Synchronous baseline

The synchronous grid contains 24 configurations and 72 runs: eight learning rates and three beta2 values, with beta1 fixed at 0.9. LR 0.0056 and beta2 0.98 give the lowest mean final loss, 3.558689 ± 0.004887 nats. At the same beta2, LR 0.004 measures 3.562837 ± 0.004298.

The 0.004148-nat difference is comparable to the seed variation. The ranking identifies a practical preset without establishing a statistically reliable improvement over every nearby candidate. Larger learning rates generally worsen loss and can increase variation. All finite outcomes, including poor ones, remain in the [source dataset](../data/recipe_sweep_20m_128k/results.json).

## Four-worker AWC

The combined AWC study contains 43 configurations and 129 unique runs. Its best tested recipe uses LR 0.008, beta1 0.95, and beta2 0.99, reaching 3.595559 ± 0.005025. The runner-up uses the same LR with beta1 0.9 and beta2 0.98, reaching 3.601910 ± 0.006842.

Search coverage is uneven. Beta1 values above 0.9 were evaluated only at LR 0.008. The complete 4×4 beta1/beta2 grid at that learning rate is informative about that slice, but it does not constitute a complete three-dimensional search. Beta2=0.99 also has some learning rates absent from the other slices. Heatmaps explicitly mark unmeasured cells.

At beta1 0.9, the best learning rate for each tested beta2 falls between 0.0048 and 0.008. All four response curves worsen above 0.008. Beta2 0.999 is particularly sensitive to larger learning rates: mean loss reaches 4.626482 at LR 0.016.

## The beta2=0.99 slice

The former standalone beta99 report covers 30 runs already included in AWC-4. It contributes useful comparisons but no additional unique experiments. At fixed beta1 0.9, LR 0.0056 gives the lowest tested mean, 3.604343 ± 0.007428. That conditional recipe remains available in [packed4-20m-awc-beta99.yaml](../../configs/packed4-20m-awc-beta99.yaml).

At LR 0.008, beta1 0.95 gives the lowest mean. The available evidence does not resolve whether that beta1 also improves other learning rates. Use the explorer's LR and beta filters to inspect this slice alongside the full [AWC dataset](../data/recipe_sweep_packed4_20m_128k/results.json).

## AWC versus ATC

Both methods evaluate gradients at the current local parameters and retain local AdamW moments. AWC mixes parameters before applying the local update; ATC applies the local update before mixing. Their [equations](../methods/algorithms.md) distinguish the parameter location receiving weight decay and the point at which updated models are combined.

ATC has 24 configurations and 72 runs, with beta1 fixed at 0.9. Its winner uses LR 0.0056 and beta2 0.99, reaching 3.595506 ± 0.003221. That is almost the same mean as the independently tuned AWC winner, but the optimizer settings differ.

The [ATC records](../data/recipe_sweep_packed4_atc_20m_128k/results.json) retain 21 matched comparisons with AWC at beta1 0.9. In the beta2=0.99 slice, ATC has lower mean loss at the three matched learning rates below 0.008; AWC has lower means at 0.008 and 0.0112. The direction depends on the learning rate, so a single global claim about mixing order would hide the measured interaction.

## Practical interpretation

Start with a committed preset, then inspect nearby measured settings and individual seeds. Keep the distinction between a conditional winner, a full-study winner, and a matched comparison. The explorer reports sample SD and shows the actual grid; it does not interpolate missing measurements or label validation-selected winners as global optima.
