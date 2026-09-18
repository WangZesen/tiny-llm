---
title: Experimental protocol
description: How the current schedule and worker-count experiments are trained, evaluated, and compared.
---

## Model and data

Each model has 20,403,520 unique trainable parameters: eight Llama-style layers,
width 320, five attention heads, feed-forward width 896, a 32,000-token vocabulary,
and context length 1,024. Tied embeddings count once.

The campaigns use the same prepared English C4 cache identity, with pinned dataset
and tokenizer revisions. Preprocessing seed 42 and validation seed 12345 are
fixed. Runtime seeds 42, 43, and 44 vary initialization and training sample order.
Virtual epochs partition a token stream; they are not repeated passes over C4.

## Global training budgets

Tokens per parameter means **global prediction targets divided by one model's
parameter count**, including for packed training.

| Tokens/parameter | Global targets | Updates | Virtual epochs |
| ---: | ---: | ---: | ---: |
| 20 | 408,027,136 | 3,113 | 40 |
| 40 | 816,185,344 | 6,227 | 80 |
| 80 | 1,632,239,616 | 12,453 | 160 |
| 120 | 2,448,424,960 | 18,680 | 240 |
| 160 | 3,264,610,304 | 24,907 | 320 |

Only WSD measures the last two horizons. Cumulative epoch boundaries round to
complete global batches of 131,072 targets. Synchronous training uses 128 sequences
per update; four workers use 32 sequences each, and eight workers use 16 each.
Each worker processes one fourth or one eighth of the global targets.

## Schedules and search

Both schedules warm up for **312 optimizer updates**. Cosine then anneals to
zero; WSD holds the peak LR and applies square-root decay over the final 10% of
the realized token budget.

| Campaign | Peak LR grid | β₁ | β₂ |
| --- | --- | --- | --- |
| Cosine · synchronous | 0.004–0.011 in 0.001 increments | 0.9 | 0.98, 0.99, 0.999 |
| Cosine · four workers | 0.005–0.012 in 0.001 increments | 0.9, 0.95, 0.974 | 0.99, 0.999 |
| Cosine · eight workers | 0.005–0.014 in 0.001 increments | 0.9, 0.95, 0.974 | 0.99, 0.999 |
| WSD · synchronous | 0.002, 0.003, 0.004, 0.006, 0.008 | 0.9, 0.95 | 0.9, 0.98, 0.999 |
| WSD · eight workers | 0.002, 0.003, 0.004, 0.006, 0.008, 0.01 | 0.9, 0.95 | 0.9, 0.98, 0.999 |

Cosine trains every listed configuration from a fresh start at each horizon.
Each WSD horizon's winning LR sets the next horizon's LR ceiling; all beta pairs
at eligible rates continue, and pruned rates do not return. Four-worker WSD has
not been measured.

## WSD continuation

Each configuration and seed continues its own parent trajectory. Saved pre-decay
states come from epochs 35, 72, 143, and 216 at the preceding horizons. Warmup,
optimizer moments, RNG state, and committed data cursor continue.

Curves retain parent observations only through the saved cursor and append child
observations strictly after it, excluding the parent's terminal decay. Every
horizon still completes its own decay and full validation. Thus 594 WSD
seed/horizon results share training history; the 1,188 cosine runs start independently.

## Optimizer and execution

AdamW uses epsilon $10^{-8}$, weight decay 0.1 on matrix weights, and gradient
clipping at 1.0. Packed training uses AWC with one-peer exponential communication,
independent optimizer moments, and no adaptive consensus.

All workers execute on **one GH200 120GB**. Execution uses BF16 autocast, FP32
parameters, compiled SDPA, fused AdamW, and eight CPU threads. These experiments
do not measure physical multi-node speedups. Nondeterministic kernels mean runtime
seeds do not promise bitwise replay.

The cosine and WSD campaigns have different frozen trainer versions, recorded in
the retained environment records. Search grids, continuation, and checkpoint
policies also differ. Cosine saved no weights or checkpoints.

## Evaluation and selection

Epoch validation uses a fixed 1,024-block subset. Final evaluation covers the
complete cached validation split: **197,411,295 prediction targets**. Packed
evaluation averages worker parameters in FP32 before evaluating that model.

Cross-entropy sums loss over valid prediction targets and divides by their count,
including partial batches and excluding padding:

$$
\mathcal L=-\frac{1}{|T|}\sum_{t\in T}\log p_\theta(x_t\mid x_{<t}).
$$

Rank complete, finite three-seed configurations by mean final full-validation
loss, breaking exact ties by lower LR, then beta1, then beta2. Report sample SD:

$$
\bar L=\frac{1}{n}\sum_i L_i,
\qquad s=\sqrt{\frac{\sum_i(L_i-\bar L)^2}{n-1}}.
$$

SD is seed variation, not a confidence interval. Validation is used for tuning,
with no independent test estimate. Keep separately tuned winners and matched
hyperparameter comparisons distinct.

## Curves and coverage

Training curves show recorded token-weighted logging-window loss at its ending
token count. Packed training losses concern the local models. Validation curves
show the fixed epoch subset, while final full-validation losses are reported
separately. Curve statistics combine only matching recorded token positions.
No smoothing or invented observations are used.

The [current explorer](/tiny-llm/results/explorer/) exposes all measured groups,
unavailable combinations, and WSD LR pruning.

## Reproduction and provenance

[Prepare data](../guides/data.md), then use the
[training guide](../guides/training.md#published-configurations) with explicit
schedule and optimizer overrides. Repeat seeds 42–44 in separate output directories.
Retain checkpoints in a new run when analysis is required.

Metrics, results, configurations, receipts, environment records, and WSD continuation
metadata are retained under the publication's source bundle, grouped into containers of
one line per artifact. [Source paths and hashes](../data/current-training/sources.json)
preserve historical identity while the publication regenerates entirely from portable
local files. Retained metrics omit the per-worker `local_grad_norms` and `local_losses`
arrays, which no published result uses; the complete logs are archived at
[`zesen-kth/tiny-llm`](https://huggingface.co/datasets/zesen-kth/tiny-llm).
See [publication maintenance](../guides/website.md).
