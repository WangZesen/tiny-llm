---
archive: true
title: Clipping & worker-count comparisons
description: What changes when gradients are unclipped or the global batch is split among more local models.
---

The current ablations examine two choices within packed AWC: clipping each worker's gradient and increasing the number of workers from four to eight. Both studies share the 20M architecture and 128K-token global batch with the main AWC experiment. Each contains a complete 80-configuration LR/beta1/beta2 grid with three seeds, for 240 runs.

## Removing gradient clipping

With clipping disabled, the best tested recipe is LR 0.0056, beta1 0.9, beta2 0.98. It reaches 3.668404 ± 0.046502 nats on final full validation. The selected clipped recipe reaches 3.595559 ± 0.005025 with different hyperparameters. The unclipped winner therefore has a higher mean and much larger observed seed variation.

That comparison describes separately tuned recipes. A more focused check holds the optimizer settings and runtime seeds fixed. Across all 28 published matched configurations, unclipped training has higher mean loss in 27 cases. The [paired seed records](../data/recipe_sweep_packed4_awc_noclip_20m_128k/matched_runs.csv) retain 84 individual comparisons, including the source of each clipped reference measurement.

The 28 matched configurations are a subset of the consolidated 129-run AWC dataset. The comparison records retain which archived source supplies each clipped measurement; they do not add extra primary tuning runs. The [full provenance](../data/recipe_sweep_packed4_awc_noclip_20m_128k/results.json) distinguishes those sources.

The sign convention is unclipped loss minus clipped loss. Positive differences favor clipping. For a matched configuration, the variability of the paired difference is computed from the three seedwise differences, not by treating two independent standard deviations as a paired estimate.

## What clipping changes

The default applies a norm threshold of 1.0 separately to each worker's raw gradient. Disabling clipping leaves the logging and nonfinite-gradient checks active. AWC mixing order, local moments, token budget, and evaluation convention otherwise remain the same in this comparison.

Removing clipping does not imply that every gradient would have exceeded the threshold. The result is empirical evidence about the tested complete recipes. It does not establish a universal threshold or prove that clipping benefits every possible learning rate and momentum setting. See the [algorithm definitions](../methods/algorithms.md) for the transformation being changed.

## Four versus eight workers

Eight-worker AWC selects LR 0.0056, beta1 0.95, and beta2 0.999, with mean loss 3.609816 ± 0.005523. The nearby LR 0.008 recipe with the same betas measures 3.610121 ± 0.003780. This very small mean difference is below the observed seed variation.

The selected learning rate is the lowest tested, and beta2 is the highest tested. The grid therefore does not locate an interior optimum in those dimensions. The [eight-worker dataset](../data/recipe_sweep_packed8_awc_20m_128k/results.json) retains every measured point and 28 matched comparisons against four-worker references.

At a fixed global batch of 131,072 targets, four workers receive 32 sequences each, while eight receive 16 each. Each sequence has context length 1,024. The global token budget remains approximately fixed, so the eight-worker local models each see fewer targets. Local optimizer state count and communication topology dynamics also change.

This is a comparison of packed training systems under a shared global budget, not a physical multi-node scaling benchmark. All workers execute within a single GPU allocation. For measured execution throughput and memory, use the separately labeled [performance measurements](../performance/training.md), whose historical recipes and timing boundaries differ from these tuning studies.

## Inspecting trajectories

The website explorer supports both final-loss comparisons and available learning trajectories. Start with the chosen recipes, then compare matched hyperparameters. Use the validation view to inspect averaged-model progress and the training view for logged local-model training loss. No smoothing is applied, and unavailable logs are labeled rather than reconstructed. Three seeds and validation-based selection limit the claims that can be drawn from close rankings.
