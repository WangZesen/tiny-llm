---
title: Prepare the data
description: Build and reuse the pinned English C4 token cache with enough capacity for training and analysis.
---

## Install and prepare

Use Python 3.12+ and UV. Preparation downloads the pinned English C4 data and
tokenizer, caches training tokens, and retains the official validation split.
Run from the repository root:

```bash
uv sync --locked
uv run tiny-llm prepare --config configs/90m.yaml
```

The 90M preset prepares enough training data for the 20M model at 20, 40, and
80 tokens per parameter. Smaller models reuse cache prefixes.

For a specific larger budget, use that same budget when preparing:

```bash
uv run tiny-llm prepare --config configs/20m.yaml \
  --set training.tokens_per_parameters=160 --set data.cache_dir=data/c4-long
```

Use the same cache path and compatible data settings in training. A completed
cache is immutable; select a new directory when more capacity or different
preprocessing is required.

## Capacity for unseen analysis

Unseen analysis starts after the entire final training shuffle group, including
candidates beyond the training budget. Prepare room for the training budget,
one unseen epoch, complete shard groups, and sequence lookahead.

For a 20M run through 80 tokens per parameter, a two-billion-token preparation
minimum supplies that capacity with the default two-shard groups:

```bash
uv run tiny-llm prepare --config configs/20m.yaml \
  --set training.tokens_per_parameters=80 \
  --set data.prepare_train_tokens=2000000000 \
  --set data.cache_dir=data/c4-analysis
```

Preparation rounds capacity up to full groups and lookahead. The analysis tool
checks capacity before computing. Keep the matching preparation settings when
launching training with this cache.

## Reuse and preprocessing

Preparation defaults to eight tokenizer processes; change
`data.prepare_workers` for the available CPU/memory resources. Worker count
does not change token contents. The training-document shuffle buffer contains
100,000 documents and uses preprocessing seed 42.

Each shard contains 33,554,432 tokens. Training reads consecutive groups of two
shards, shuffles sequences within each group, and prefetches the next group.
The same seed and data settings preserve sample prefixes at longer horizons.

Training checks manifest identity, shard sizes, and configuration compatibility.
Rerun `prepare` with matching settings to verify all cached shard checksums.
Older caches prepared with a different shuffle buffer require matching settings
or a new cache directory.

Continue with [local training](training.md) or [Slurm submission](slurm.md).
