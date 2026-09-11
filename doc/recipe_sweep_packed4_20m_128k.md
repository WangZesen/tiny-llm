# Packed-4 20M tuning at 128K tokens per batch

Measured on **2026-09-11**: **99 successful runs** covering **33 distinct
LR/beta1/beta2 configurations**, each with runtime seeds **42, 43, and 44**.
The search varies LR at beta1 0.9 and explores beta1 values from 0.9 to 0.975
at LR 0.008. The exact coverage is listed below.

The best tested recipe is [`configs/packed-20m.yaml`](../configs/packed-20m.yaml):
**LR 0.008, beta1 0.9, beta2 0.98**, with mean final full-validation loss
**3.601910 ± 0.006842**. The runner-up, **LR 0.008, beta1 0.95, beta2 0.98**,
measures **3.602794 ± 0.012887**, a difference of 0.000884 nats.

All runs use four local 20M models, `one_peer_exponential` topology, global
batch **131,072 prediction targets**, and microbatch **32 sequences per model**,
without accumulation. Adaptive consensus is disabled. Losses evaluate the
**globally averaged model** on the full C4 validation split. Error bars and
± values are **sample standard deviations** across seeds, not confidence intervals.

## Results

Selection minimizes mean final-checkpoint full-validation cross-entropy, with
exact ties resolved by lower LR, then beta1, then beta2.

| Rank | LR | Beta1 | Beta2 | Mean loss (nats) | Sample SD |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0080 | 0.900 | 0.98 | 3.601910 | 0.006842 |
| 2 | 0.0080 | 0.950 | 0.98 | 3.602794 | 0.012887 |
| 3 | 0.0080 | 0.925 | 0.98 | 3.604153 | 0.004039 |
| 4 | 0.0056 | 0.900 | 0.98 | 3.606178 | 0.005628 |
| 5 | 0.0048 | 0.900 | 0.999 | 3.610736 | 0.005867 |

The top mean differences are small relative to seed variation. This ranking
alone does not establish a statistically reliable improvement or equivalence.
The winner is inside the tested LR and beta2 ranges; no beta1 below 0.9 was
searched. Beta1 values above 0.9 were tested only at LR 0.008, so their
performance at other learning rates remains unmeasured. These validation-based
selections have no independent test estimate and do not establish global optimality.

## Higher beta1 at fixed LR 0.008

![Fixed-LR beta1 response and beta1/beta2 heatmap](recipe_sweep_packed4_20m_128k/betas.png)

This figure shows the **36 runs in 12 configurations at LR 0.008**. The left
panel shows mean loss and seed SD; the right panel annotates the same
statistics for every beta pair. The star and outlined cell identify the winner.
[PDF](recipe_sweep_packed4_20m_128k/betas.pdf) ·
[PNG](recipe_sweep_packed4_20m_128k/betas.png).

Each table cell is mean ± sample SD at **LR 0.008**:

| Beta1 | Beta2 = 0.95 | Beta2 = 0.98 | Beta2 = 0.999 |
|---|---:|---:|---:|
| 0.900 | 3.620824 ± 0.008201 | **3.601910 ± 0.006842** | 3.698712 ± 0.062199 |
| 0.925 | 3.612169 ± 0.008923 | 3.604153 ± 0.004039 | 3.627433 ± 0.008334 |
| 0.950 | 3.619770 ± 0.011060 | 3.602794 ± 0.012887 | 3.641030 ± 0.024592 |
| 0.975 | 3.693064 ± 0.006425 | 3.662906 ± 0.010097 | 3.673407 ± 0.014273 |

Beta2 0.98 gives the lowest mean for every tested beta1 at this LR. Increasing
beta1 improves the means for some beta2 choices—especially beta2 0.999—but
the lowest mean loss is achieved at beta1 0.9 and beta2 0.98. Beta1 0.975
performs worse than 0.925 and 0.95 for each beta2 in this fixed-LR comparison.

## Learning-rate response at beta1 0.9

![LR and beta2 response with seed variation](recipe_sweep_packed4_20m_128k/tuning.png)

These panels show the **72 runs in 24 configurations at beta1 0.9**. The left
panel covers the full LR range; the right panel shows LRs up to
0.008 on a narrower loss scale. Higher LRs generally worsen mean loss and can
increase seed variation. All outcomes, including poor finite-loss runs, remain
in the results.
[PDF](recipe_sweep_packed4_20m_128k/tuning.pdf) ·
[PNG](recipe_sweep_packed4_20m_128k/tuning.png).

The [synchronous 128K-token winner](recipe_sweep_20m_128k.md) measured
3.558689 ± 0.004887, 0.043221 nats below the packed winner. Local optimizer states,
topology mixing, and averaged-model evaluation distinguish packed training;
this comparison does not isolate one algorithmic effect.

## Tuning process

| LR | Beta1 | Beta2 | Seeds | Configurations | Runs |
|---|---|---|---|---:|---:|
| 0.0024, 0.0048, 0.0056, 0.008, 0.0112, 0.016, 0.0224, 0.032 | 0.9 | 0.95, 0.98, 0.999 | 42, 43, 44 | 24 | 72 |
| 0.008 | 0.925, 0.95, 0.975 | 0.95, 0.98, 0.999 | 42, 43, 44 | 9 | 27 |

The plots show overlapping views of this search: the nine runs at LR 0.008
and beta1 0.9 appear in both figures, while the dataset contains 99 unique runs.
All runs use identical frozen training-source files, dependency metadata,
recorded runtime versions, cache identity, and token budgets. Preprocessing
seed 42 and validation seed 12345 remain fixed; runtime seeds vary initialization
and training order. `deterministic=false` does not promise bitwise replay.

| Setting | Shared value |
|---|---|
| Local models | 4 × 20,403,520 unique parameters; 81,614,080 total |
| Architecture per model | 8 layers, width 320, 5 heads, FFN width 896, vocabulary 32,000 |
| Context / local microbatch | 1,024 tokens / 32 sequences |
| Global batch | 4 × 32 × 1,024 = 131,072 targets; no accumulation |
| Global / per-model budget | 408,068,096 / 102,017,024 targets |
| Epochs / updates | 40 virtual epochs / 3,119 optimizer updates |
| Other local AdamW settings | Epsilon 1e-8, weight decay 0.1, per-model clipping 1.0 |
| Schedule | 5% token-based linear warmup, cosine decay to 10% of peak LR |
| Epoch validation | Averaged model, fixed 1,024-block subset |
| Final validation | Averaged model, full C4 split: 197,411,295 targets |
| Execution | BF16 autocast, FP32 parameters, compiled SDPA, fused local AdamW, 8 CPU threads |
| Retention | One final packed checkpoint per run with all local weights and optimizer states |

Epoch-ending updates are shortened, keeping equal local batches and using
actual target counts. The schedule follows globally consumed tokens. All runs
used NVIDIA GH200 120GB GPUs, Python 3.12.13, PyTorch 2.14.0, and CUDA 13.0.
Each allocation simulates all four models on one GPU; physical inter-node
communication costs are not represented.

Jobs allowed up to 24 concurrent GPUs with a 30-minute allocation limit and
isolated compiler caches. All **99 Slurm jobs completed with exit code 0**.
The audit checked training and validation counts, configuration and source
provenance, and checkpoint completion. Every mean and sample SD was recomputed
from individual runs. The data contains 99 unique `(LR, beta1, beta2, seed)`
records and 33 complete three-seed groups.

## Artifacts and reproduction

- [Per-run CSV](recipe_sweep_packed4_20m_128k/runs.csv): all **99 runs**, with
  hyperparameters, seed, job ID, loss, and artifact locations.
- [Summary CSV](recipe_sweep_packed4_20m_128k/summary.csv): all **33 configurations**,
  sorted by mean loss and grouped by LR, beta1, and beta2.
- [Results and provenance](recipe_sweep_packed4_20m_128k/results.json): complete
  metrics, source commits, input hashes, environment, cache identity, and rankings.
- [Plot script](recipe_sweep_packed4_20m_128k/plot.py): regenerates both figure
  pairs from the summary CSV, without access to training checkpoints.

Per-run provenance and archive locations are retained in the CSV and JSON.
Frozen inputs and large training artifacts remain gitignored; the tracked
report contains the data needed to reproduce the statistics and figures.

With the cache prepared as described in the [README](../README.md#training),
reproduce a configuration by setting its LR and betas:

```bash
uv run tiny-llm train --config configs/packed-20m.yaml \
  --set optimizer.lr=0.008 --set optimizer.beta1=0.95 --set optimizer.beta2=0.98 \
  --set runtime.seed=42 --set runtime.output_dir=runs/packed4-beta1-0.95-beta2-0.98-seed42
```

Repeat with the other tested configurations and seeds 43/44 in separate output
directories. Omit the beta overrides to reproduce the winner at beta1 0.9.
Regenerate both plot pairs with:

```bash
uv run python doc/recipe_sweep_packed4_20m_128k/plot.py
```

The packed baseline and the separate adaptive-consensus preset are unchanged.
