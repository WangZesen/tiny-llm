---
title: "Slurm and analysis jobs"
---

## Slurm jobs

Submit from the repository root. Create `runs/` first because Slurm opens logs
before the launcher starts:

```bash
mkdir -p runs
sbatch scripts/slurm.sh prepare --config configs/90m.yaml
sbatch scripts/slurm.sh train --config configs/20m.yaml
sbatch scripts/slurm.sh train --config configs/packed4-20m-awc.yaml
sbatch scripts/slurm.sh evaluate --config runs/20m/resolved.yaml \
  --checkpoint runs/20m/final.pt --full
```

Wait for preparation to complete before submitting training with that cache.
To request a GH200 explicitly and use the selected synchronous cosine recipe:

```bash
sbatch --gpus=nvidia_gh200_120gb:1 scripts/slurm.sh train \
  --config configs/20m.yaml --config configs/cosine.yaml \
  --set lr_schedule.min_lr_ratio=0 --set optimizer.lr=0.01 \
  --set optimizer.beta1=0.9 --set optimizer.beta2=0.99 \
  --set runtime.output_dir=runs/sync-cosine
```

Use the [local training guide](training.md#published-configurations) for packed
worker recipes, WSD, other horizons, and checkpoint policies. Override the
account and partition with Slurm options before the script path when needed.

The launcher defaults to account `naiss2026-3-205-gpu`, partition `gpu`, one GPU,
one task, and **two hours**. Slurm allocates CPUs automatically. Combined output
goes to `runs/slurm-<job-id>.log`. The launcher sources `~/.bashrc` to select the
compute node's architecture-specific UV environment, synchronizes locked
dependencies, and runs the command with `srun`.

Startup logs timestamp batch entry, shell setup, dependency synchronization, and
the `srun` launch. Training then reports Python entry, CLI import time, runtime
initialization, cache checks, model/optimizer setup, checkpoint/loader setup,
metadata collection, validation-subset loading, and the first update (including
compilation). Python stage durations use a monotonic clock. The stage totals start
at training initialization; compare earlier timestamps to account for imports and
launcher overhead. Existing throughput metrics retain their timing boundaries.

Place resource overrides before the script path. For example, request six hours
for a longer training run:

```bash
sbatch --time=06:00:00 scripts/slurm.sh train --config configs/90m.yaml
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,Timelimit,ExitCode
```

Benchmark jobs use the same launcher:

```bash
sbatch scripts/slurm.sh benchmark --config configs/20m.yaml \
  --gh200 --budget-minutes 75 --output runs/gh200-tuning
sbatch scripts/slurm.sh benchmark-packed --config configs/20m.yaml \
  --num-models 4 8 --output runs/packed-benchmarks
```

See [benchmarking and sweeps](benchmarking.md) for tuning, profiling,
selected configurations, and campaign commands.

### Analysis jobs

Run the submitter on the login node. It divides unique checkpoint states among
`--jobs N` independent single-GPU allocations, waits for successful jobs, then
collects results and plots locally. Identical checkpoints stay together, with
all filenames retained as aliases. Preview assignments before submitting:

```bash
uv run tiny-llm submit-analysis \
  --run runs/campaign/20m-lr0.001-wd0.1 --checkpoints all --jobs 4 \
  --output runs/campaign/20m-lr0.001-wd0.1/analysis/new-analysis --dry-run
```

Remove `--dry-run` to submit and wait. The equivalent script entrypoint is
`uv run python scripts/submit_analysis.py`. Submission defaults to **BF16 forward
AMP, FP32 parameters, TF32, compiled HVPs at batch 64**, both cases, seed 42, and
32 noise plus 32 random samples. Gradient microbatches come from `resolved.yaml`.
The same numerical options as `analyze` are available.

For the measured ordinary 20M recipe (context 1024, training batch/microbatch 32,
default analysis settings), jobs explicitly request GH200 GPUs. Each wall time
is twice estimated runtime plus 30 minutes, rounded up to an hour: **25 hours
for ten checkpoints** with default sample counts. Other architectures or
execution settings require `--walltime-hours N`. Requests must fit the **72-hour**
partition limit; increase `--jobs` if the estimated work will not fit. Each job
uses the same account and partition as the launcher, with no shard dependencies.

Keep the login-side watcher running in `tmux`, or detach it with `nohup`:

```bash
nohup uv run tiny-llm submit-analysis \
  --run runs/campaign/20m-lr0.001-wd0.1 --checkpoints all --jobs 4 \
  --output runs/campaign/20m-lr0.001-wd0.1/analysis/new-analysis \
  > analysis-orchestrator.log 2>&1 < /dev/null &
```

`OUTPUT/submission.json` records assignments, source snapshots, job IDs,
resource requests, and log paths. The watcher checks `squeue` and `sacct` every
60 seconds and requires successful status plus valid output before collection.
Stopping it leaves GPU jobs running. Repeat the original arguments with
`--resume` to reattach and retry terminal failed jobs, preserving completed
results. If submission was interrupted before a job ID was recorded, reconcile
that job with Slurm before retrying. Older receipts must use their original
[frozen source entrypoint](../methods/analysis.md#execution-and-provenance).

Each shard writes statistics to its own directory; the watcher publishes the
combined results only after all shards succeed. Compiler caches use a unique
temporary directory under `SLURM_TMPDIR`, `TMPDIR`, or `/tmp`, with cleanup on
normal exit, failure, or catchable termination.
