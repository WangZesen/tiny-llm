"""Write the comprehensive eight-worker AWC report from audited results."""

import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT / "results.json").read_text())
    groups, runs = data["groups"], data["runs"]
    w = data["winner"]
    rank = "\n".join(
        f"| {i} | {r['lr']:g} | {r['beta1']:g} | {r['beta2']:g} | "
        f"{r['mean_loss']:.6f} | {r['std_loss']:.6f} |"
        for i, r in enumerate(groups, 1)
    )
    best_lrs = [next(g for g in groups if g["lr"] == lr) for lr in data["protocol"]["lrs"]]
    lr_table = "\n".join(
        f"| {r['lr']:g} | {r['beta1']:g} | {r['beta2']:g} | "
        f"{r['mean_loss']:.6f} ± {r['std_loss']:.6f} |"
        for r in best_lrs
    )
    winner_runs = [r for r in runs if all(r[k] == w[k] for k in ("lr", "beta1", "beta2"))]
    seed_table = "\n".join(f"| {r['seed']} | {r['loss']:.9f} |" for r in winner_runs)
    four = data["packed4_reference"]["winner"]
    same = next(g for g in groups if all(g[k] == four[k] for k in ("lr", "beta1", "beta2")))
    seconds = [r["seconds"] for r in runs]
    text = f"""# AWC packed-8 20M tuning at 128K tokens per batch

Measured on **2026-09-12**: **240 successful runs** covering the complete
**80-configuration LR/beta1/beta2 grid**, with runtime seeds **42, 43, and 44**
for every configuration. Each run trains **eight local 20M models** using AWC.

The best tested recipe is **LR {w["lr"]:g}, beta1 {w["beta1"]:g}, beta2 {w["beta2"]:g}**,
with mean final full-validation cross-entropy **{w["mean_loss"]:.6f} ± {w["std_loss"]:.6f}**
nats. All ± values and error bars are **sample standard deviations across the
three seeds**, not confidence intervals. Lower loss is better.

LR 0.008 with the same betas measures **{groups[1]["mean_loss"]:.6f} ± {groups[1]["std_loss"]:.6f}**,
only **{groups[1]["mean_loss"] - w["mean_loss"]:.6f} nats higher**. This gap is small relative to
seed variation. The winner is at the **lowest tested LR and highest tested beta2**;
the search does not locate an interior optimum in those dimensions. Selection
uses validation data, with no independent test estimate. These results do not
establish statistical significance, equivalence, or global optimality.

## Search space

| Parameter | Values |
|---|---|
| Learning rate | 0.0056, 0.008, 0.010, 0.0112, 0.016 |
| AdamW beta1 | 0.9, 0.925, 0.95, 0.975 |
| AdamW beta2 | 0.95, 0.98, 0.99, 0.999 |
| Runtime seeds | 42, 43, 44 |

The repeated 0.010 in the requested LR list is represented once. Every unique
combination has three runs: 5 × 4 × 4 × 3 = 240. All finite outcomes are included.
Ranking minimizes mean final full-validation loss, breaking exact ties by lower
LR, then beta1, then beta2. There are no missing or failed configurations.

## Beta1/beta2 heatmaps

![Complete beta1/beta2 heatmaps at all five learning rates](recipe_sweep_packed8_awc_20m_128k/heatmaps.png)

[PDF](recipe_sweep_packed8_awc_20m_128k/heatmaps.pdf) ·
[PNG](recipe_sweep_packed8_awc_20m_128k/heatmaps.png).
Each panel contains 16 configurations and 48 seeded runs. All five panels share
a color scale; cells show mean ± sample SD. Orange outlines mark the lowest mean
within each LR, so the outlines are conditional selections.

The best configurations at each learning rate are:

| LR | Beta1 | Beta2 | Mean loss ± sample SD |
|---|---:|---:|---:|
{lr_table}

Beta1 0.95 and beta2 0.999 lead at LR 0.0056, 0.008, and 0.010. At LR 0.0112,
the lowest mean uses beta1 0.925 and beta2 0.99. At LR 0.016, beta1 0.95 and
beta2 0.98 perform best, but even that selected mean is
{best_lrs[-1]["mean_loss"] - w["mean_loss"]:.6f} nats above the overall minimum. The preferred
beta2 depends on LR; a single marginal average would obscure that relationship.

## Learning-rate response

![Learning-rate response for all four beta1 values](recipe_sweep_packed8_awc_20m_128k/tuning.png)

[PDF](recipe_sweep_packed8_awc_20m_128k/tuning.pdf) ·
[PNG](recipe_sweep_packed8_awc_20m_128k/tuning.png).
Each panel fixes beta1 and shows all four beta2 curves across the five LRs.
Error bars show sample SD; all panels share the same loss scale. Lines connect
measured points. The star identifies the best tested configuration.

The three leading configurations use beta1 0.95 and beta2 0.999 across LRs
0.0056–0.010. Their closely spaced means support considering this region together
when interpreting the sweep. Higher LR does not improve the best achievable
mean within the measured beta grid; the selected minimum at LR 0.016 is clearly
higher in these measurements.

## Complete ranking

All 80 configurations are included below and in the
[summary CSV](recipe_sweep_packed8_awc_20m_128k/summary.csv).

| Rank | LR | Beta1 | Beta2 | Mean loss (nats) | Sample SD |
|---|---:|---:|---:|---:|---:|
{rank}

The selected recipe's individual final full-validation losses are:

| Runtime seed | Loss (nats) |
|---|---:|
{seed_table}

## Comparison with four-worker AWC

The [four-worker AWC study](recipe_sweep_packed4_20m_128k.md) selects LR
{four["lr"]:g}, beta1 {four["beta1"]:g}, beta2 {four["beta2"]:g}, with
**{four["mean_loss"]:.6f} ± {four["std_loss"]:.6f}**. The best eight-worker mean is
**{w["mean_loss"] - four["mean_loss"]:.6f} nats higher**. At that same four-worker recipe,
eight workers measure **{same["mean_loss"]:.6f} ± {same["std_loss"]:.6f}**,
a difference of **{same["mean_loss"] - four["mean_loss"]:.6f} nats**.

There are **28 shared LR/beta1/beta2 configurations**, with the same runtime
seeds. Eight workers have lower means in 13 and higher means in 15. The complete
[matched comparison CSV](recipe_sweep_packed8_awc_20m_128k/matched_packed4.csv)
contains means, sample SDs, and signed differences (eight minus four).
The eight-worker winner has no exact four-worker counterpart in the measured grid.

Both studies use the same model architecture, global batch and training-token
budget, cache identity, topology family, and averaged-model validation. Eight
workers use microbatch 16 and 51,008,512 targets per local model; four workers
use microbatch 32 and 102,017,024 targets per local model. Changing worker count
also changes the mixing schedule and number of local optimizer states. The
separately selected best recipes differ in LR and beta2, and the search spaces
are unequal. This comparison is descriptive, not a controlled estimate of a
single mechanism or a distributed scaling benchmark.

## Shared training protocol

AWC computes forward and backward passes and clips local gradients, mixes
parameters, then applies local AdamW updates. Optimizer moments remain local;
adaptive consensus is disabled. Validation evaluates the globally averaged model.

| Setting | Value |
|---|---|
| Local models | 8 × 20,403,520 parameters; 163,228,160 total local parameters |
| Architecture per model | 8 layers, width 320, 5 heads, FFN 896, vocabulary 32,000 |
| Topology / scheme | one_peer_exponential / AWC |
| Context / local microbatch | 1,024 tokens / 16 sequences |
| Global batch | 8 × 16 × 1,024 = 131,072 targets; no accumulation |
| Global / per-model training budget | 408,068,096 / 51,008,512 targets |
| Epochs / updates | 40 / 3,119, including shortened epoch-ending steps |
| Other AdamW settings | Weight decay 0.1, epsilon 1e-8, local gradient clipping 1.0 |
| Schedule | 5% token-based linear warmup; cosine decay to 10% of peak LR |
| Epoch evaluation | Fixed 1,024-block validation subset |
| Final evaluation | Full C4 validation split: 197,411,295 targets |
| Preprocessing / validation seeds | 42 / 12345 |
| Execution | GH200 120GB, BF16 autocast, FP32 weights, compiled SDPA, fused AdamW, 8 CPU threads |
| Runtime | Python 3.12.13, PyTorch 2.14.0, CUDA 13.0 |
| Checkpoint policy | none; no model, optimizer, or weight files saved |

Each job simulates eight local workers on one GPU. Physical inter-node
communication costs are not measured. Runtime seeds vary initialization and
sample order; nondeterministic execution does not guarantee bitwise replay.

Median run time, including validation, was **{statistics.median(seconds):.1f} seconds**
(range {min(seconds):.1f}–{max(seconds):.1f} seconds). Runs used at most 24 concurrent
GPUs with a 30-minute allocation limit. No training checkpoints were saved;
configuration, environment, logs, metrics, and final validation results are retained.

## Audit and artifacts

All **240 Slurm tasks in array 2318595 completed with exit code 0**. The audit
verified immutable source/config hashes, resolved configurations, runtime source
digests and versions, worker counts, training-token and update budgets, complete
full validation, finite losses, and absence of checkpoint files. All 80 groups
contain exactly runtime seeds 42, 43, and 44; means and sample SDs are recomputed
from individual results.

Frozen inputs and run artifacts are under
`runs/packed8-awc-20m-batch128k-20260912`. The source is based on commit
`d0d0d00` plus the archived checkpoint-policy change. The full source digest is
`{data["source_hash"]}`. The results JSON records the frozen input hashes,
per-run result/config/environment hashes, and the four-worker reference hash.

- [Per-run CSV](recipe_sweep_packed8_awc_20m_128k/runs.csv): all 240 measurements and artifact locations.
- [Summary CSV](recipe_sweep_packed8_awc_20m_128k/summary.csv): all 80 ranked configurations.
- [Results and provenance](recipe_sweep_packed8_awc_20m_128k/results.json).
- [Matched four-worker comparisons](recipe_sweep_packed8_awc_20m_128k/matched_packed4.csv).
- [Slurm accounting](recipe_sweep_packed8_awc_20m_128k/slurm-accounting.txt).
- [Collection script](recipe_sweep_packed8_awc_20m_128k/collect.py).
- [Plot script](recipe_sweep_packed8_awc_20m_128k/plot.py).
- [Report generator](recipe_sweep_packed8_awc_20m_128k/write_report.py).

## Reproduction

The winning recipe is [configs/packed8-20m-awc.yaml](../configs/packed8-20m-awc.yaml).
With the prepared C4 cache, run:

```bash
uv run tiny-llm train --config configs/packed8-20m-awc.yaml
```

The recipe saves a final checkpoint for subsequent evaluation or analysis.
Add `--set training.checkpoint_policy=none` to reproduce the sweep's checkpoint-free
retention. Repeat with runtime seeds 43 and 44 in separate output directories.
The preset explicitly selects eight AWC workers and supplies the measured optimizer
settings, architecture, global batch, and training budget. For exact frozen
configurations, use the archived config files and source. Checkpoint-free runs
retain evaluation metrics but cannot be resumed or used for checkpoint analysis.

Regenerate figures and this report from the audited data:

```bash
uv run python doc/data/recipe_sweep_packed8_awc_20m_128k/plot.py
uv run python doc/data/recipe_sweep_packed8_awc_20m_128k/write_report.py
```
"""
    text = re.sub(r"\]\((recipe_sweep_[^/)]+/)", r"](../data/\1", text)
    text = text.replace("](../configs/", "](../../configs/")
    title = text.splitlines()[0].removeprefix("# ")
    header = (
        f"---\ntitle: {json.dumps(title)}\narchive: true\n---\n\n"
        "> **Historical report.** See the [current results](../results/overview.md) "
        "for maintained guidance.\n\n"
    )
    (ROOT.parent.parent / "archive" / f"{ROOT.name}.md").write_text(header + text)


if __name__ == "__main__":
    main()
