---
title: "C4 campaign recipe selection"
description: "Historical report; retained as an experimental record."
archive: true
---

> **Historical report.** Settings and conclusions describe the recorded experiment. See the [current documentation](../results/overview.md) for maintained guidance.

# C4 campaign recipe selection

This earlier single-seed campaign covers three model sizes. The later
[three-seed 20M benchmark](recipe_sweep_20m.md) records 324 synchronous and packed
training runs; its selected recipes are documented without changing the presets.

All twelve runs in `runs/campaign` completed 40 virtual epochs and their full
20-targets-per-parameter training budgets. Each final checkpoint was evaluated
on the same complete C4 validation split: 197,411,295 prediction targets.

The audit checked per-run `resolved.yaml`, `result.json`, `environment.json`,
and validation/training metrics against `comparison.json`. Every run used seed 42,
loader v1, the same cache identity and source hash, and matching fixed settings
within its model size. All 40 subset evaluations were present and recorded losses
were finite. Exact results and provenance are in [campaign_results.json](../data/campaign/results.json).

## Default recipe decisions

| Model | LR | Weight decay | Beta2 | Full-validation loss | Decision |
|---|---:|---:|---:|---:|---|
| 20M | 0.001 | 0.1 | 0.95 | 3.609748 | Keep existing optimizer recipe |
| 50M | 0.001 | 0.1 | 0.99 | 3.257811 | Change beta2 from 0.95 to 0.99 |
| 90M | 0.001 | 0.1 | 0.99 | 3.049379 | Adopt the best tested recipe; beta2=0.95 was not tested |

At 20M, LR 0.001 beats 0.003 by only 0.003827 nats at weight decay 0.1.
Weight decay 0.1 beats zero decay at every tested learning rate. LR 0.0003
finishes substantially worse, so the existing 20M settings are the best tested
within this earlier grid.

At 50M, beta2=0.99 lowers loss from 3.293054 to 3.257811 at LR 0.001:
a 0.035243-nat reduction, or approximately 3.46% lower perplexity. It also wins
at LR 0.003, lowering loss from 3.328142 to 3.291123. LR 0.001 wins with either beta2.

At 90M, both promoted recipes use beta2=0.99. LR 0.001 beats 0.003 by
0.073556 nats. Adopting beta2=0.99 follows the best completed recipe and the
50M evidence; this campaign does not establish that it beats beta2=0.95 at 90M.
A matched 90M run at LR 0.001, weight decay 0.1, beta2=0.95 would resolve that gap.

## All candidates

| Run | LR | Weight decay | Beta2 | Full-validation loss |
|---|---:|---:|---:|---:|
| 20m-lr0.001-wd0.1 | 0.001 | 0.1 | 0.95 | 3.609748 |
| 20m-lr0.003-wd0.1 | 0.003 | 0.1 | 0.95 | 3.613574 |
| 20m-lr0.001-wd0 | 0.001 | 0 | 0.95 | 3.621143 |
| 20m-lr0.003-wd0 | 0.003 | 0 | 0.95 | 3.622036 |
| 20m-lr0.0003-wd0.1 | 0.0003 | 0.1 | 0.95 | 3.742181 |
| 20m-lr0.0003-wd0 | 0.0003 | 0 | 0.95 | 3.749784 |
| 50m-parent0-beta20.99 | 0.001 | 0.1 | 0.99 | 3.257811 |
| 50m-parent1-beta20.99 | 0.003 | 0.1 | 0.99 | 3.291123 |
| 50m-parent0-beta20.95 | 0.001 | 0.1 | 0.95 | 3.293054 |
| 50m-parent1-beta20.95 | 0.003 | 0.1 | 0.95 | 3.328142 |
| 90m-parent0 | 0.001 | 0.1 | 0.99 | 3.049379 |
| 90m-parent1 | 0.003 | 0.1 | 0.99 | 3.122935 |

## Interpretation

All runs reached their best recorded subset loss at epoch 40, with decreasing
loss across the last five evaluations. There is no late validation deterioration
in these measurements that would justify shortening the schedule. Longer training
was not tested, so retain the current token budgets and learning-rate schedule.

This is one seed and a staged search, with validation used for selection.
The 50M stage did not test zero weight decay or LR 0.0003, and the 90M stage did
not compare beta2 values. The results select practical defaults within the tested
grid; they do not establish seed robustness, general optimality, or an independent
test-set improvement.

Only `configs/50m.yaml` and `configs/90m.yaml` change optimizer settings.
The generic Python defaults still describe the 20M recipe. Existing checkpoints
retain their saved recipe; use each run's `resolved.yaml` when resuming.
