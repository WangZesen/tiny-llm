"""Write the no-clipping variant report and its matched clipping comparison."""

import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT / "results.json").read_text())
    groups, runs, matches = data["groups"], data["runs"], data["matched_clipped"]
    winner = data["winner"]
    clipped = data["clipped_reference"]["winner"]
    keys = ("lr", "beta1", "beta2")
    at_clipped = next(g for g in groups if all(g[k] == clipped[k] for k in keys))
    at_winner = next(g for g in matches if all(g[k] == winner[k] for k in keys))
    best_lrs = [next(g for g in groups if g["lr"] == lr) for lr in data["protocol"]["lrs"]]
    negatives = [m for m in matches if m["difference_unclipped_minus_clipped"] < 0]
    assert len(negatives) == 1
    exception = negatives[0]
    rank_table = "\n".join(
        f"| {i} | {r['lr']:g} | {r['beta1']:g} | {r['beta2']:g} | "
        f"{r['mean_loss']:.6f} | {r['std_loss']:.6f} |"
        for i, r in enumerate(groups, 1)
    )
    lr_table = "\n".join(
        f"| {r['lr']:g} | {r['beta1']:g} | {r['beta2']:g} | "
        f"{r['mean_loss']:.6f} ± {r['std_loss']:.6f} |"
        for r in best_lrs
    )
    seed_table = "\n".join(
        f"| {r['seed']} | {r['loss']:.9f} |" for r in runs if all(r[k] == winner[k] for k in keys)
    )
    match_table = "\n".join(
        f"| {r['lr']:g} | {r['beta1']:g} | {r['beta2']:g} | "
        f"{r['unclipped_mean']:.6f} ± {r['unclipped_std']:.6f} | "
        f"{r['clipped_mean']:.6f} ± {r['clipped_std']:.6f} | "
        f"{r['difference_unclipped_minus_clipped']:+.6f} ± {r['paired_std']:.6f} |"
        for r in sorted(matches, key=lambda r: tuple(r[k] for k in keys))
    )
    seconds = [r["seconds"] for r in runs]
    text = f"""# Four-worker AWC without gradient clipping: 20M tuning at 128K tokens

Measured on **2026-09-12–2026-09-13**: **240 successful runs** covering a complete
**80-configuration LR/beta1/beta2 grid**, with runtime seeds **42, 43, and 44**
for every configuration. This report describes AWC with gradient clipping disabled
and compares it with the [four-worker clipped AWC study](recipe_sweep_packed4_20m_128k.md).

**Clipping at norm 1.0 is favored within the tested settings.** Removing clipping
increases mean final full-validation loss in **27 of the 28 matched configurations**.
The best tested no-clipping recipe is **LR {winner["lr"]:g}, beta1 {winner["beta1"]:g},
beta2 {winner["beta2"]:g}**, with loss **{winner["mean_loss"]:.6f} ± {winner["std_loss"]:.6f}** nats.
The tuned clipped recipe measures **{clipped["mean_loss"]:.6f} ± {clipped["std_loss"]:.6f}**,
{winner["mean_loss"] - clipped["mean_loss"]:.6f} nats lower. These selected recipes use different
hyperparameters; the matched comparisons below hold LR and betas fixed.

All ± values are **sample standard deviations across three seeds**, not confidence
intervals. In the difference column and comparison plot, the SD is computed from
the three seed-paired differences. Lower loss is better.

## Variant and search space

AWC first computes local forward and backward passes, then mixes parameters,
then applies local AdamW updates. With clipping enabled, each worker's gradients
are clipped to norm 1.0 before mixing. Here `optimizer.grad_clip=null` skips that
scaling operation. Gradient norms are still computed and logged, and nonfinite
gradients still stop training. Optimizer moments remain local; adaptive consensus
is disabled. Both variants evaluate the globally averaged model.

| Parameter | Values |
|---|---|
| Learning rate | 0.0056, 0.008, 0.010, 0.0112, 0.016 |
| AdamW beta1 | 0.9, 0.925, 0.95, 0.975 |
| AdamW beta2 | 0.95, 0.98, 0.99, 0.999 |
| Runtime seeds | 42, 43, 44 |
| Gradient clipping | Disabled (`optimizer.grad_clip=null`) |

Every unique combination has three runs: 5 × 4 × 4 × 3 = 240. All completed with
finite full-validation losses; no failed or divergent jobs were discarded.
Selection minimizes mean final full-validation cross-entropy over all three seeds,
breaking exact ties by lower LR, then beta1, then beta2.

## No-clipping response surfaces

![No-clipping beta1/beta2 heatmaps at all learning rates](recipe_sweep_packed4_awc_noclip_20m_128k/heatmaps.png)

[PDF](recipe_sweep_packed4_awc_noclip_20m_128k/heatmaps.pdf) ·
[PNG](recipe_sweep_packed4_awc_noclip_20m_128k/heatmaps.png).
Each panel covers 16 configurations and 48 runs, with a shared color scale.
Cells show mean ± sample SD; orange outlines mark the best mean within each LR.

| LR | Best beta1 | Best beta2 | Mean loss ± sample SD |
|---|---:|---:|---:|
{lr_table}

The best mean at each LR worsens as LR increases across this grid. The overall
winner lies at the **lowest tested LR and lowest tested beta1**, so this sweep
does not rule out better unclipped settings below those bounds. The next-best
configuration is only {groups[1]["mean_loss"] - winner["mean_loss"]:.6f} nats behind, compared with
substantial seed variation in both groups. The top-ranked hyperparameters should
therefore be interpreted as the best tested mean, not a precisely resolved optimum.

![No-clipping LR response for all beta1 and beta2 values](recipe_sweep_packed4_awc_noclip_20m_128k/tuning.png)

[PDF](recipe_sweep_packed4_awc_noclip_20m_128k/tuning.pdf) ·
[PNG](recipe_sweep_packed4_awc_noclip_20m_128k/tuning.png).
Each panel fixes beta1 and includes four beta2 curves; error bars show sample SD.
All panels share the same loss scale, and lines connect measured points.

## Comparison with clipping enabled

### Selected recipes and fixed-hyperparameter checks

| Comparison | No clipping | Clipping at 1.0 | No clipping − clipping |
|---|---:|---:|---:|
| Each variant's best tested recipe | {winner["mean_loss"]:.6f} ± {winner["std_loss"]:.6f} | {clipped["mean_loss"]:.6f} ± {clipped["std_loss"]:.6f} | {winner["mean_loss"] - clipped["mean_loss"]:+.6f} |
| Fixed LR {winner["lr"]:g}, beta1 {winner["beta1"]:g}, beta2 {winner["beta2"]:g} | {winner["mean_loss"]:.6f} ± {winner["std_loss"]:.6f} | {at_winner["clipped_mean"]:.6f} ± {at_winner["clipped_std"]:.6f} | {at_winner["difference_unclipped_minus_clipped"]:+.6f} |
| Fixed LR {clipped["lr"]:g}, beta1 {clipped["beta1"]:g}, beta2 {clipped["beta2"]:g} | {at_clipped["mean_loss"]:.6f} ± {at_clipped["std_loss"]:.6f} | {clipped["mean_loss"]:.6f} ± {clipped["std_loss"]:.6f} | {at_clipped["mean_loss"] - clipped["mean_loss"]:+.6f} |

The clipped winner uses LR {clipped["lr"]:g}, beta1 {clipped["beta1"]:g}, beta2 {clipped["beta2"]:g}.
Clipping improves the measured mean at both variants' selected settings. The
selected-recipe comparison covers unequal search spaces: 80 configurations without
clipping and 43 with clipping. It does not isolate the effect of clipping by itself.

### All matched configurations

There are **28 shared configurations and 84 pairs of seeded runs**. Both variants
match LR, beta1, beta2, runtime seeds, local worker count, architecture, global and
local batch sizes, token budget, topology, optimizer settings, data cache, and
validation protocol. This overlap consists of all 16 beta pairs at LR 0.008,
plus the four beta2 choices at beta1 0.9 for LR 0.0056, 0.0112, and 0.016.

![Seed-paired loss differences with and without clipping](recipe_sweep_packed4_awc_noclip_20m_128k/comparison.png)

[PDF](recipe_sweep_packed4_awc_noclip_20m_128k/comparison.pdf) ·
[PNG](recipe_sweep_packed4_awc_noclip_20m_128k/comparison.png).
Positive differences favor clipping. Error bars show sample SD of the three
seed-paired differences, not uncertainty bounds on the mean.

The median difference across the 28 configuration means is
**{statistics.median(r["difference_unclipped_minus_clipped"] for r in matches):+.6f} nats**.
Clipping has the lower mean in 27 configurations. The only negative difference,
at LR {exception["lr"]:g}, beta1 {exception["beta1"]:g}, beta2 {exception["beta2"]:g}, is
**{exception["difference_unclipped_minus_clipped"]:+.6f} ± {exception["paired_std"]:.6f}**,
which is negligible relative to its seed variation.

Across the matched configurations, median within-configuration sample SD is
**{statistics.median(r["unclipped_std"] for r in matches):.6f} without clipping**, versus
**{statistics.median(r["clipped_std"] for r in matches):.6f} with clipping**. This describes
greater sensitivity to runtime seed without clipping in this experiment; all
240 unclipped jobs nevertheless completed with finite results. The median
difference and win count describe this particular grid and are not an average
effect over arbitrary training recipes.

| LR | Beta1 | Beta2 | No clipping mean ± SD | Clipped mean ± SD | Paired difference mean ± SD |
|---|---:|---:|---:|---:|---:|
{match_table}

These results support retaining clipping for the measured four-worker protocol.
They do not establish that clipping is necessary at every LR or for every model.
Only three runtime seeds were tested; validation-based selection has no independent
test estimate. The unclipped winner is at a search boundary, and 52 of its
configurations have no clipped counterpart. Both implementations use the same AWC
order, but source revisions differ to support the scheme and no-clipping options.
Checkpoint retention also differs (none versus final). Matching seeds does not
guarantee bitwise-identical execution or remove these provenance differences.

## Complete no-clipping ranking

All 80 configurations are ranked below and in the
[summary CSV](recipe_sweep_packed4_awc_noclip_20m_128k/summary.csv).

| Rank | LR | Beta1 | Beta2 | Mean loss (nats) | Sample SD |
|---|---:|---:|---:|---:|---:|
{rank_table}

The selected no-clipping recipe's individual full-validation losses are:

| Runtime seed | Loss (nats) |
|---|---:|
{seed_table}

## Shared training protocol

| Setting | Value |
|---|---|
| Local models | 4 × 20,403,520 parameters; 81,614,080 total local parameters |
| Architecture per model | 8 layers, width 320, 5 heads, FFN 896, vocabulary 32,000 |
| Topology / scheme | one_peer_exponential / AWC |
| Context / local microbatch | 1,024 tokens / 32 sequences |
| Global batch | 4 × 32 × 1,024 = 131,072 targets; no accumulation |
| Global / per-model training budget | 408,068,096 / 102,017,024 targets |
| Epochs / updates | 40 / 3,119, including shortened epoch-ending steps |
| Other AdamW settings | Weight decay 0.1, epsilon 1e-8 |
| Schedule | 5% token-based linear warmup; cosine decay to 10% of peak LR |
| Epoch evaluation | Fixed 1,024-block subset, globally averaged model |
| Final evaluation | Full C4 validation split: 197,411,295 targets, globally averaged model |
| Preprocessing / validation seeds | 42 / 12345 |
| Adaptive consensus | Disabled |
| Execution | GH200 120GB, BF16 autocast, FP32 weights, compiled SDPA, fused AdamW, 8 CPU threads |
| Runtime | Python 3.12.13, PyTorch 2.14.0, CUDA 13.0 |
| No-clipping checkpoint policy | none; no model, optimizer, or weight files saved |

Each job simulates four workers on one GPU, so these measurements do not include
physical inter-node communication costs. Median no-clipping run time including
validation was {statistics.median(seconds):.1f} seconds (range {min(seconds):.1f}–{max(seconds):.1f}).
Jobs used at most 24 concurrent GPUs, isolated compiler caches, and a 30-minute
allocation limit. This report makes no speedup claim from removing clipping.

## Audit and data

All **240 tasks in Slurm array 2341796 completed with exit code 0**. The audit
verified frozen source and config hashes, resolved recipes, runtime source digests
and versions, training and validation counts, update and epoch counts, finite
losses, and absence of checkpoint files. Every group has exactly seeds 42, 43,
and 44; means and sample SDs are recomputed from per-run measurements.

For the clipped reference, all 84 matching individual losses are preserved in the
published [AWC results dataset](recipe_sweep_packed4_20m_128k/results.json), and its
28 matched means and SDs were recomputed. **57 original result/config pairs were
re-audited**; after allowing for clipping, checkpoint retention, output path, and
the explicit default AWC field, the resolved configurations match. The original
archive for **27 clipped runs** at higher beta1 is unavailable, so those comparisons
use the published per-run records and protocol/source metadata. They cannot be
re-audited against the original resolved files in the current workspace. Each
paired row records which evidence was available; no results were imputed.

The no-clipping archive is `runs/packed4-awc-noclip-20m-batch128k-20260912`.
It contains the frozen source based on commit `11fb614`, the archived no-clipping
patch, all configurations, and run artifacts. Its source digest is
`{data["source_hash"]}`. The results JSON includes input and artifact hashes and
the SHA-256 of the clipped reference dataset.

- [Per-run CSV](recipe_sweep_packed4_awc_noclip_20m_128k/runs.csv): all 240 no-clipping outcomes.
- [Summary CSV](recipe_sweep_packed4_awc_noclip_20m_128k/summary.csv): all 80 ranked configurations.
- [Matched configuration CSV](recipe_sweep_packed4_awc_noclip_20m_128k/matched_clipped.csv): all 28 comparisons.
- [Matched seed CSV](recipe_sweep_packed4_awc_noclip_20m_128k/matched_runs.csv): all 84 paired outcomes and evidence sources.
- [Results and provenance](recipe_sweep_packed4_awc_noclip_20m_128k/results.json).
- [Slurm accounting](recipe_sweep_packed4_awc_noclip_20m_128k/slurm-accounting.txt).
- [Collection script](recipe_sweep_packed4_awc_noclip_20m_128k/collect.py), [plot script](recipe_sweep_packed4_awc_noclip_20m_128k/plot.py), and [report generator](recipe_sweep_packed4_awc_noclip_20m_128k/write_report.py).

## Reproduction

With the prepared C4 cache, reproduce the best tested no-clipping recipe:

```bash
uv run tiny-llm train --config configs/packed4-20m-awc.yaml \\
  --set optimizer.lr={winner["lr"]:g} \\
  --set optimizer.beta1={winner["beta1"]:g} \\
  --set optimizer.beta2={winner["beta2"]:g} \\
  --set optimizer.grad_clip=null \\
  --set training.checkpoint_policy=none \\
  --set runtime.output_dir=runs/packed4-20m-awc-noclip-seed42
```

Repeat with runtime seeds 43 and 44 in separate output directories. For a matched
clipped run, use `optimizer.grad_clip=1.0` at the same LR and betas. The existing
AWC winner preset is unchanged. Checkpoint-free runs retain metrics but cannot
be resumed or used for checkpoint analysis. Regenerate the figures and report:

```bash
uv run python doc/data/recipe_sweep_packed4_awc_noclip_20m_128k/plot.py
uv run python doc/data/recipe_sweep_packed4_awc_noclip_20m_128k/write_report.py
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
