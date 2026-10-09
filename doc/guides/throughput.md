---
title: "Single-GH200 batch throughput sweeps"
---

# Single-GH200 batch throughput sweeps

Run the login-side watcher from the repository root. It submits independent
14-minute, single-GH200 jobs for 20M, 90M, 250M and 500M models at contexts
512, 1024, 2048 and 4096. At most twelve jobs run concurrently. Each curve doubles
the batch from one sequence until a confirmed CUDA out-of-memory failure.
No prepared dataset or tokenizer download is required.

```bash
.venv-x86_64/bin/python scripts/throughput_sweep.py run \
  --output runs/gh200-throughput --dry-run
.venv-x86_64/bin/python scripts/throughput_sweep.py run \
  --output runs/gh200-throughput
```

Use `--contexts 512` to measure only the new context, or pass several lengths.
The selected contexts are frozen in the campaign manifest. Existing campaigns
resume with their original contexts; adding a context requires a new campaign.

Keep the watcher running in a terminal multiplexer. Interrupting it leaves
submitted GPU jobs running. Resume with the same output directory:

```bash
.venv-x86_64/bin/python scripts/throughput_sweep.py run \
  --output runs/gh200-throughput --resume
.venv-x86_64/bin/python scripts/throughput_sweep.py status \
  --output runs/gh200-throughput
.venv-x86_64/bin/python scripts/throughput_sweep.py collect \
  --output runs/gh200-throughput
```

Resume reconciles recorded Slurm job IDs and unresolved submission intents
before submitting anything else. Successful points are retained. Failed curves
remain blocked until their failure is investigated; `--resume --retry-failed`
retries the current batch in a fresh attempt directory. An ambiguous submission
intent must be reconciled before retrying. A timeout, host-memory failure,
compiler error or nonfinite update never establishes the CUDA memory boundary.

The campaign freezes package source, its driver, resolved configurations and
the lockfile. GPU jobs use the existing `.venv-aarch64` environment, eight CPU
threads, 64 GiB host memory, and disposable node-local compiler caches. A
13-minute parent watchdog includes Python imports and compilation, leaving
time to record a timeout within the 14-minute Slurm allocation. The default
account and partition match the repository's Slurm launcher.

## What is measured

Batch size is the number of sequences in one complete forward, cross-entropy,
backward, gradient-clipping and AdamW update. The microbatch equals the optimizer
batch: there is no accumulation. Inputs and targets are generated once on the
GPU. The timed production entrypoint includes its device-to-device copies.

The backend uses BF16 autocast, FP32 weights and optimizer moments, compiled
SDPA and a complete-update CUDA graph. The synthetic protocol fixes seed 42,
constant LR 0.001, AdamW betas (0.9, 0.99), epsilon 1e-8, weight decay 0.1,
and gradient clipping at 1.0. The repeated synthetic batch measures execution
speed, not convergence or training quality.

Compilation, graph capture, initialization, 20 warmup updates and five
calibration updates precede measurement. Calibration chooses a fixed number
of updates targeting ten seconds per window, with at least twenty updates.
Five synchronized wall-clock windows are measured without disk logging or
host-side finite checks inside the timer. Every window is followed by a finite
state check and an exact optimizer-update count check. Validation and
checkpoints are absent.

Reported throughput is the median window rate. If the sample coefficient of
variation exceeds 5%, five additional windows are included in the same result.
Remaining instability is flagged; only stable points compete for fastest batch.
Allocated and reserved GPU-memory peaks are recorded separately for setup and
measurement. Preparation or CUDA-graph memory is part of the fitting limit;
there is no artificial percentage-of-capacity cutoff.

For an individual point inside a GH200 allocation:

```bash
.venv-aarch64/bin/python -m tiny_llm benchmark-throughput-worker \
  --config configs/250m.yaml --set model.context_length=2048 \
  --set training.micro_batch_size=8 --set training.batch_tokens=16384 \
  --set optimizer.lr=0.001 --set optimizer.beta2=0.99 \
  --output runs/throughput-point.json \
  --warmup 20 --windows 5 --window-seconds 10
```

The individual worker uses the supplied optimizer settings; the campaign
driver applies the common protocol above to every model. Direct worker use
does not provide the campaign's watchdog or Slurm resource requests.

## Optimizer time within each update

Remeasure all fitting batches from a completed sweep with:

```bash
.venv-x86_64/bin/python scripts/throughput_sweep.py run \
  --optimizer-timing-from runs/gh200-throughput-20261008 \
  --output runs/gh200-optimizer-timing-20261009
```

The same `status`, `collect`, `--resume`, `--retry-failed` and `--dry-run`
commands apply. The follow-up stops at each curve's previously largest fitting
batch. An unexpected OOM blocks that curve for investigation. The original
throughput campaign remains unchanged.

Contexts are inherited from the throughput campaign, including a campaign that
contains only context 512:

```bash
.venv-x86_64/bin/python scripts/throughput_sweep.py run \
  --contexts 512 --output runs/gh200-throughput-c512-20261009
.venv-x86_64/bin/python scripts/throughput_sweep.py run \
  --optimizer-timing-from runs/gh200-throughput-c512-20261009 \
  --parallel-optimizer-points \
  --output runs/gh200-optimizer-timing-c512-20261009
```

`--parallel-optimizer-points` schedules already validated fitting points
independently, up to twelve jobs at once. It requires a completed memory sweep.
Resume retains every point's submission receipt; a failure pauses further
submissions on that curve until the failed point is successfully retried.
Memory-discovery sweeps always advance sequentially by doubling after success.

Each 14-minute allocation runs two fresh workers with identical configuration
on the same GPU: an uninstrumented baseline followed by an instrumented update
graph. Their combined deadline is 13 minutes. The second worker captures three
external CUDA timing events around the graph and `ArenaAdamW.step`. This step
includes gathering gradients, norm/clipping and finite checks, coefficients,
AdamW moment/weight updates and metric/counter commit. Both forward and backward
precede the optimizer measurement inside the actual full-update graph.

Event timestamps describe the last replay in each synchronized burst targeting
0.1 seconds. Each window weights those samples by burst update count. Event
reads happen outside timed bursts. Median window optimizer milliseconds are
divided by the paired baseline's full-iteration milliseconds for the primary
optimizer percentage. This is an estimate of average per-update cost, rather
than a trace of every update. Shares of instrumented wall time and captured
graph time are also retained.

Both workers use the usual warmup and five ten-second windows; five more
windows are added when throughput or optimizer CV exceeds 5%. The report flags
remaining instability and instrumented wall-time changes exceeding 5% versus
baseline. That comparison includes both instrumentation overhead and possible
drift between workers. Each point keeps `baseline.json`, `result.json`, and raw
burst samples. The follow-up report adds `optimizer-time.png` and
`optimizer-share.png`, plus detailed CSV/JSON.

For a single instrumented worker, append `--optimizer-timing` to the individual
worker command above. Paired baseline measurements are managed by the campaign
driver.

## Combined report and training-time estimates

Combine the original and context-512 campaign pairs and project
**20 tokens per parameter** with:

```bash
.venv-x86_64/bin/python scripts/throughput_report.py \
  --throughput runs/gh200-throughput-20261008 runs/gh200-throughput-c512-20261009 \
  --optimizer runs/gh200-optimizer-timing-20261009 runs/gh200-optimizer-timing-c512-20261009 \
  --output runs/gh200-training-cost-c512-4096-20261009
```

Campaigns are paired by argument order; context overlap or incomplete grids are
rejected. A previously audited output directory cannot be overwritten. To
regenerate the report, choose a new output directory.

The [combined report](../../runs/gh200-training-cost-c512-4096-20261009/README.md) provides
a six-page PDF, five figures in PDF/SVG/450-dpi PNG, joined CSV/JSON data, and
exact input hashes. No GPU allocation is needed. It independently checks raw
timing arithmetic, event weighting, parameter counts, update counts, matching
configurations and the original memory boundaries.

The [previous three-context report](../../runs/gh200-training-cost-20261009/README.md)
remains available as a historical artifact.

The primary projection uses the optimizer campaign's uninstrumented baseline,
keeping the original sweep as independent comparison data. For parameter count
`P`, sequence batch `B`, context `L`, and measured rate `R`, the ideal steady-state
time is `T = 20P / R`. Each model's budget stays fixed across setups; budgets
differ between models. Tables also provide time per billion tokens and the
rounding required for complete final updates.

Context comparisons additionally hold `B × L` fixed, avoiding different token
counts per update. Counterfactual optimizer speedups hold all remaining work
constant. These estimates exclude startup, data delivery, evaluation and
checkpointing; they do not predict convergence or equal model quality. Window
ranges describe within-run variation rather than confidence intervals across
independent runs.

## Larger architecture presets

| Preset | Layers | Width | Heads | FFN width | Parameters |
|---|---:|---:|---:|---:|---:|
| 250m | 20 | 960 | 15 | 2560 | 251,943,360 |
| 500m | 24 | 1280 | 20 | 3456 | 516,814,080 |

Both use the existing 32,000-token vocabulary, tied embeddings, and head
dimension 64. Their YAML files provide conservative starting batches; optimizer
hyperparameters are not tuned. Existing 20M/50M/90M recipes and the legacy
three-model tuning campaign retain their previous behavior.

## Artifacts

`README.md`, `results.csv`, `results.json`, `throughput.png` and `memory.png`
summarize the campaign. Raw per-point JSON includes all timing windows, setup
duration, memory peaks, hardware/software metadata, and the resolved recipe.
Submission receipts retain job IDs, final Slurm accounting and all failed
attempts. Each curve reports its fastest stable batch, largest fitting batch,
first OOM batch, and unstable points. An incomplete campaign is labeled as such.

Audit a completed campaign, including its raw result arithmetic and actual
Slurm resources, with:

```bash
CUDA_VISIBLE_DEVICES='' .venv-x86_64/bin/python scripts/throughput_audit.py \
  --output runs/gh200-throughput-c512-20261009
```

The same command accepts an optimizer campaign. It writes `audit.json` and
`accounting.txt`, verifies contiguous batches and memory boundaries (or the
exact inherited optimizer grid), and checks job limits and peak concurrency.
