---
title: Experimental protocol
description: The common measurement contract behind the five current tuning studies.
---

## Model and data

Each local Llama-style model has 20,403,520 unique trainable parameters: eight layers, width 320, five attention heads, feed-forward width 896, and a 32,000-token vocabulary. Context length is 1,024. Tied embeddings count once. Training uses the prepared English C4 cache with preprocessing seed 42; runtime seeds 42, 43, and 44 vary initialization and sample order.

The intended budget is 20 prediction targets per unique parameter. Actual complete-block budgets differ slightly because epoch and worker partitions must divide into sequences:

| System | Global training targets | Targets per local model | Updates | Epochs |
|---|---:|---:|---:|---:|
| Synchronous | 408,071,168 | 408,071,168 | 3,120 | 40 |
| Four-worker packed | 408,068,096 | 102,017,024 | 3,119 | 40 |
| Eight-worker packed | 408,068,096 | 51,008,512 | 3,119 | 40 |

All studies use 131,072 global targets per full update, without accumulation. The microbatch contains 128 sequences for synchronous training, 32 per worker for four workers, and 16 per worker for eight. Short epoch-ending updates use their actual target counts. Virtual epochs partition a single token stream; they are not repeated passes over the entire C4 dataset.

## Optimizer and execution

AdamW uses epsilon $10^{-8}$, weight decay 0.1 on matrix weights, and no decay on normalization scales. The token-based schedule warms up linearly for 5% of the budget, then follows cosine decay to 10% of the peak learning rate. Raw-gradient clipping uses threshold 1.0 except in the explicit no-clipping study.

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

The five primary datasets contain 753 runs and 251 configurations. The beta99 report reuses AWC measurements and adds no unique primary runs. Exported curves cover 726 runs; 27 AWC logs are unavailable at their recorded paths. Final results for all runs remain available.

Training curves use the logged token-weighted loss over each logging window, plotted at its ending global token count. Packed training loss concerns local models. Validation curves use the epoch subset, evaluated at averaged weights for packed methods. Final full-validation losses are displayed separately.

Curve means and sample SDs are computed only from observations at matching recorded token positions. The contributing seed count appears in chart tooltips. Lines join recorded points without smoothing or interpolation of additional observations. Missing logs and unmeasured heatmap cells are explicit.

## Reproduction and provenance

Follow the [training guide](../guides/training.md), select a committed preset, and repeat it with runtime seeds 42, 43, and 44 in separate output directories. Current checkpoint defaults differ from some historical runs; record the resolved recipe and retention policy when reproducing them.

The study data bundles retain CSV summaries, seed records, original result JSON, and plotting tools. Original paths and source hashes inside provenance describe historical execution and are preserved after relocation. Curve snapshots add hashes of the raw metrics and result files. Website builds use only committed artifacts; refreshing curves requires access to the source logs through the [maintenance workflow](../guides/website.md).
