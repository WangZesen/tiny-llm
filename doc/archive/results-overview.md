---
archive: true
title: "Earlier results overview"
description: WSD horizon campaigns and earlier cosine studies, with separate evidence and practical recipes.
---

## WSD horizon campaigns

The [synchronous and packed-8 AWC WSD campaigns](../results/wsd-horizon-tuning.md) add **594 seed/horizon results across 198 configuration/horizon comparisons**, at 20, 40, 80, 120, and 160 global tokens per parameter. They reuse pre-decay checkpoints, so continued results are not independent training runs. The [WSD explorer](/tiny-llm/explorer/wsd/) provides rankings, individual seeds, matched comparisons, and reconstructed trajectories.

Synchronous training has lower selected mean full-validation loss at every horizon. The AWC-minus-synchronous gap narrows from **0.036429** to **0.015171** nats/token. AWC selects beta1 0.95 and beta2 0.999 throughout, with LR 0.006 at horizons 20–40 and 0.003 at 80–160. Later winners are conditional on LR pruning; the horizon-80 AWC decision over LR 0.004 is only 0.000015 nats.

These campaigns use a different schedule, search grid, and token-allocation implementation from the earlier studies below. Their differences do not isolate the effect of WSD versus cosine scheduling. Committed training presets retain their existing settings.

## Earlier cosine studies

The synchronous 20M recipe achieves the lowest mean final full-validation loss among the five earlier cosine studies. Four-worker AWC and ATC finish very close to one another after separate tuning. Eight-worker AWC is slightly higher, and removing gradient clipping produces a substantially worse best tested result. These statements describe the measured grids; they do not establish universal rankings of the algorithms.

## Best tested cosine recipes

All results below use runtime seeds 42, 43, and 44. Loss is token-weighted cross-entropy in nats on the complete cached C4 validation split. The ± values are sample standard deviations across seeds, not confidence intervals.

| Method                       |     LR | Beta 1 | Beta 2 | Mean loss ± sample SD |
| ---------------------------- | -----: | -----: | -----: | --------------------: |
| Synchronous                  | 0.0056 |    0.9 |   0.98 |   3.558689 ± 0.004887 |
| Four-worker AWC              |  0.008 |   0.95 |   0.99 |   3.595559 ± 0.005025 |
| Four-worker ATC              | 0.0056 |    0.9 |   0.99 |   3.595506 ± 0.003221 |
| Eight-worker AWC             | 0.0056 |   0.95 |  0.999 |   3.609816 ± 0.005523 |
| Four-worker AWC, no clipping | 0.0056 |    0.9 |   0.98 |   3.668404 ± 0.046502 |

The synchronous, AWC-4, ATC-4, and AWC-8 winners are available as [committed configuration files](../../configs/20m.yaml), with [AWC-4](../../configs/packed4-20m-awc.yaml), [ATC-4](../../configs/packed4-20m-atc.yaml), and [AWC-8](../../configs/packed8-20m-awc.yaml) presets. The unclipped result is an ablation and does not replace a recommended preset.

## What is being compared

Each local model has 20,403,520 unique trainable parameters, including the tied embedding once. The global batch contains 131,072 prediction targets. Packed training divides that batch among four or eight local models and maintains independent optimizer moments. Final evaluation uses their averaged parameters.

The global training budget is approximately 408 million targets in each earlier cosine study. Packed workers each process only their allocated share of that budget. This differs from training four independent models for the full synchronous budget. All packed workers are simulated on one GPU; the measurements do not include physical inter-node communication costs.

The comparison therefore changes more than parallel execution. It changes local batches, optimizer states, and the relationship between trained and evaluated parameters. The [shared protocol](../methods/protocol.md) records exact budgets and evaluation conventions.

## Reading the differences

The selected four-worker AWC mean exceeds the selected ATC mean by only 0.000053 nats, far below the observed seed variation. Those recipes use different learning rates and beta1 values. Their proximity cannot establish equivalence or isolate the effect of mixing order. The [optimizer article](../results/optimizer-tuning.md) discusses matched settings as well as tuned winners.

The synchronous winner is 0.036870 nats below the AWC-4 winner. That is a useful comparison of the tested systems, but it is not a controlled test of one optimizer operation. Similarly, the eight-worker result must be interpreted with its own search bounds and per-worker token allocation.

Clipping has more consistent directional evidence: removing it increases mean loss in 27 of 28 published matched configurations. The [ablation article](../results/ablations.md) retains the pairing and explains the limits of that result.

## Inspecting the evidence

The website's [earlier cosine explorer](/tiny-llm/explorer/) includes all 753 historical cosine runs across 251 configurations. The separate beta2=0.99 report is a view of the AWC-4 dataset and is not counted again. Full rankings and individual seed losses are available as tables and downloads.

Recorded trajectories are available for 726 runs. The 27 missing logs belong to an earlier AWC beta sweep; their final losses remain available. Trajectory views distinguish logged training-window loss from epoch subset-validation loss. The final full-validation metric remains the selection criterion throughout.

Use the [training guide](../guides/training.md) to reproduce a selected recipe. Earlier 32K-token studies and the original 50M/90M selection evidence remain in the [historical campaign](../archive/campaign_results.md). Validation was used for tuning, and no independent test estimate is reported.
