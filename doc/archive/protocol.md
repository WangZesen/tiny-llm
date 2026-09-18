---
archive: true
title: "Earlier experiment protocols"
description: Measurement conventions for WSD horizon campaigns and earlier cosine tuning studies.
---

The model, validation metric, and seed aggregation conventions are shared. Schedule and realized training budgets differ between the WSD campaigns and the earlier cosine studies.

## WSD horizon campaigns

The [WSD publication](../results/wsd-horizon-tuning.md) covers synchronous and eight-worker AWC training at 20, 40, 80, 120, and 160 global tokens per single-model parameter. Both use 131,072-token global batches, 312 warmup updates, and square-root decay over the final 10% of each realized budget. Global targets are 408,027,136; 816,185,344; 1,632,239,616; 2,448,424,960; and 3,264,610,304, respectively. Eight AWC workers each process one eighth of these tokens; full evaluation uses their averaged parameters.

Each eligible optimizer triple requires complete, finite final full-validation losses for seeds 42–44. Rank by arithmetic mean, breaking exact ties by lower LR, then lower beta1, then lower beta2. The global winner sets the next horizon’s LR ceiling; retain every beta pair at eligible LRs, and never restore a pruned LR. These campaigns contain 594 seed/horizon results across 198 comparisons, including 90 comparisons matched across methods.

Each configuration and seed continues its own trajectory from the previous horizon’s pre-decay checkpoint. The 40, 80, 160, and 240 epoch horizons save at epochs 35, 72, 143, and 216, respectively; the final horizon has 320 epochs. Boundaries follow realized tokens and the unchanged scheduler. Warmup, AdamW moments, RNG state, and the committed data cursor continue. Each horizon still completes its own decay and full validation before selection.

Published curves splice parent history only through that saved cursor and append child observations strictly after it, excluding earlier terminal decays. All 594 results have reconstructed curves; continued horizons share training history. Native receipts, recipe/cache identities, continuation metadata, retirement records, and retained checksums are recorded in the [publication provenance](../data/wsd-horizon-tuning/provenance.json). Website builds use the committed export.

## Earlier cosine studies: model and data

Each local Llama-style model has 20,403,520 unique trainable parameters: eight layers, width 320, five attention heads, feed-forward width 896, and a 32,000-token vocabulary. Context length is 1,024. Tied embeddings count once. Training uses the prepared English C4 cache with preprocessing seed 42; runtime seeds 42, 43, and 44 vary initialization and sample order.

The intended budget is 20 prediction targets per unique parameter. Actual complete-block budgets differ slightly because epoch and worker partitions must divide into sequences:

| System              | Global training targets | Targets per local model | Updates | Epochs |
| ------------------- | ----------------------: | ----------------------: | ------: | -----: |
| Synchronous         |             408,071,168 |             408,071,168 |   3,120 |     40 |
| Four-worker packed  |             408,068,096 |             102,017,024 |   3,119 |     40 |
| Eight-worker packed |             408,068,096 |              51,008,512 |   3,119 |     40 |

The earlier cosine studies use 131,072 global targets per full update, without accumulation. The microbatch contains 128 sequences for synchronous training, 32 per worker for four workers, and 16 per worker for eight. Short epoch-ending updates use their actual target counts. Virtual epochs partition a single token stream; they are not repeated passes over the entire C4 dataset.

## Optimizer and execution

AdamW uses epsilon $10^{-8}$, weight decay 0.1 on matrix weights, and no decay on normalization scales. The token-based schedule in the earlier cosine studies warmed up linearly for 5% of the budget, then followed cosine decay to 10% of the peak learning rate. Raw-gradient clipping uses threshold 1.0 except in the explicit no-clipping study.

Current scheduler configs use `lr_schedule.warmup_steps`, defaulting to 1000 optimizer updates. See the [schedule guide](../guides/training.md#learning-rate-schedules) for current configuration; the study schedules above retain their historical settings.

Packed studies use `one_peer_exponential` topology, independent optimizer moments, and no adaptive consensus. GH200 execution uses BF16 autocast, FP32 parameters, compiled SDPA, fused AdamW, and eight CPU threads. Runtime seeds do not guarantee bitwise replay under nondeterministic kernels.

## Evaluation and selection

Epoch evaluation uses a fixed 1,024-block subset with validation seed 12345. Final evaluation covers the full cached C4 validation split: 197,411,295 prediction targets. Packed evaluation first averages worker parameters in FP32; it does not average the workers' losses.

For valid target positions $T$, cross-entropy is

$$
\mathcal L = -\frac{1}{|T|}\sum_{t\in T}\log p_\theta(x_t\mid x_{<t}).
$$

Losses are summed over valid targets and divided by their count, including partial batches and excluding padding. Selection minimizes final full-validation mean across the three runtime seeds. Exact ties use lower LR, then beta1 where varied, then beta2. With seed losses $L_1,\ldots,L_n$, the reported mean and sample SD are

$$
\bar L=\frac{1}{n}\sum_{i=1}^{n}L_i,
\qquad s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(L_i-\bar L)^2}.
$$

The SD describes seed variation. It is not a confidence interval. Validation is used for tuning, and there is no independent test estimate. Keep conditional optima, separately tuned winners, and matched comparisons distinct.

## Curves and coverage

The five earlier cosine datasets contain 753 runs and 251 configurations. The beta99 report reuses AWC measurements and adds no unique primary runs. Exported curves cover 726 runs; 27 AWC logs are unavailable at their recorded paths. Final results for all runs remain available.

Training curves use the logged token-weighted loss over each logging window, plotted at its ending global token count. Packed training loss concerns local models. Validation curves use the epoch subset, evaluated at averaged weights for packed methods. Final full-validation losses are displayed separately.

Curve means and sample SDs are computed only from observations at matching recorded token positions. The contributing seed count appears in chart tooltips. Lines join recorded points without smoothing or interpolation of additional observations. Missing logs and unmeasured heatmap cells are explicit.

## Reproduction and provenance

Follow the [training guide](../guides/training.md), select a committed preset, and repeat it with runtime seeds 42, 43, and 44 in separate output directories. Current token allocation rounds cumulative epochs to complete global batches,
and checkpoint defaults differ from some historical runs; record the resolved recipe and retention policy when reproducing them.

The study data bundles retain CSV summaries, seed records, original result JSON, and plotting tools. Original paths and source hashes inside provenance describe historical execution and are preserved after relocation. Curve snapshots add hashes of the raw metrics and result files. Website builds use only committed artifacts; refreshing curves requires access to the source logs through the [maintenance workflow](../guides/website.md).
