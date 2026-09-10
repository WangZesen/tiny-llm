# tiny-llm

Train small Llama-style language models from scratch on English C4, evaluate
checkpoints, and measure gradient-noise alignment with the loss Hessian.
Presets cover 20M, 50M, and 90M parameters, plus packed decentralized training.

## Setup

Use Python 3.12+, UV, and an NVIDIA GPU with BF16 support for training.
The locked PyTorch build uses CUDA 13 and requires a compatible driver.
Run commands from the repository root.

```bash
uv sync --locked
```

## Training

Prepare the data once using the largest preset; smaller models reuse prefixes
of the same cache. Preparation needs network access and also caches the official
English C4 validation split. Training and evaluation use the local cache.

```bash
uv run tiny-llm prepare --config configs/90m.yaml
uv run tiny-llm train --config configs/20m.yaml
```

Use `configs/50m.yaml` or `configs/90m.yaml` for larger models. Override fields
with repeated `--set dotted.key=value` arguments; unknown fields are rejected.
Give separate experiments their own output directories:

```bash
uv run tiny-llm train --config configs/50m.yaml \
  --set optimizer.lr=0.0003 --set runtime.output_dir=runs/50m-low-lr
```

Each run saves `resolved.yaml`, logs (`run.log`, `metrics.jsonl`), final state
(`final.pt`), best subset statistics (`best.json`), and final metrics (`result.json`).
`training.checkpoint_policy` defaults to `final`: no periodic, epoch, or interruption
checkpoints are written. Interrupted training must restart unless `final.pt` exists.
To retain epoch weights and rolling recovery states, train with
`--set training.checkpoint_policy=all`. Resume those runs with the saved configuration:

```bash
uv run tiny-llm train --config runs/20m/resolved.yaml \
  --resume runs/20m/latest.pt
```

With policy `all`, `training.save_epoch_training_state` defaults to `true`, retaining
complete `epoch-NNN.pt` snapshots as well. Set it to `false` to keep only epoch
weight exports and rolling recovery states. Packed epoch snapshots store each
worker's weights and AdamW moments in `node-NNN/epoch-NNN.pt`; the root file holds
shared state and references to those files. Keep the complete set together.
To branch from an earlier epoch, use its root checkpoint and a new output directory:

```bash
uv run tiny-llm train --config runs/packed-20m/resolved.yaml \
  --resume runs/packed-20m/epoch-010.pt \
  --set runtime.output_dir=runs/packed-20m-from-epoch010
```

Resume allows device, output-directory, and prefetch changes; keep the training
recipe and data settings unchanged. See [checkpoint compatibility](doc/training_details.md#checkpoints-and-artifacts).

For a short pipeline check:

```bash
uv run tiny-llm prepare --config configs/smoke.yaml
uv run tiny-llm train --config configs/smoke.yaml
```

The smoke configuration truncates C4, so its evaluation is marked incomplete.

### Packed training

Packed presets run four independent local models together on one GPU. To use
eight workers while preserving the global batch:

```bash
uv run tiny-llm train --config configs/packed-20m.yaml
uv run tiny-llm train --config configs/packed-20m.yaml \
  --set decentralized.num_models=8 --set training.micro_batch_size=4 \
  --set runtime.output_dir=runs/packed-20m-n8
```

The global batch must equal `num_models * micro_batch_size * context_length`,
and epoch boundaries must divide evenly across workers. With checkpoint policy
`all`, root epoch checkpoints contain averaged weights and worker weights live
under `node-NNN/`. The default `final.pt` retains all local training states.
[Packed implementation details](doc/training_details.md#packed-decentralized-training)
cover topologies and optimizer behavior.

Optional adaptive consensus weakens mixing as the learning rate falls. The sample
config uses four workers, `start_frac=0.5`, and `p=1.0`:

```bash
uv run tiny-llm train --config configs/packed-20m-adaptive.yaml
```

Activation uses the ceiling of `start_frac * total_steps`, counting shortened
epoch-ending updates. See [adaptive consensus](doc/training_details.md#adaptive-consensus)
for the LR normalization and resume semantics. It is supported by training only;
packed throughput benchmarks reject adaptive-consensus configurations.

To keep the tied embedding/LM head on the configured topology's normal mixing
while applying adaptive scaling to all other parameters:

```bash
uv run tiny-llm train --config configs/packed-20m-adaptive.yaml \
  --set decentralized.adaptive_consensus.exclude_embeddings=true \
  --set runtime.output_dir=runs/packed-20m-adaptive-exclude-embeddings
```

`exclude_embeddings` defaults to `false`, which applies gamma to every parameter.

## Evaluation

Epoch-weight examples below require training with `--set training.checkpoint_policy=all`.

Evaluate a saved checkpoint using the run's resolved configuration. `--full`
selects the full cached validation split; omit it to use the fixed subset.

```bash
uv run tiny-llm evaluate --config runs/20m/resolved.yaml \
  --checkpoint runs/20m/final.pt --full
uv run tiny-llm evaluate --config runs/20m/resolved.yaml \
  --checkpoint runs/20m/epoch-020.safetensors
```

Both `.pt` and `.safetensors` checkpoints are supported. Packed training states
are evaluated at their averaged weights. Loss is token-weighted cross-entropy
in nats. Training already evaluates the subset every epoch and the full split
at completion. For less GPU memory, pass `--set evaluation.batch_size=32`
(the default is 128 sequences).

## Analysis

For epoch-by-epoch analysis, train with `--set training.checkpoint_policy=all`.
Analyze all checkpoints or select filenames relative to the run directory.
Use a prepared cache with enough data for both seen and unseen cases; the tool
checks capacity before computing. Packed runs use root, averaged checkpoints.

Packed runs also analyze every worker's consensus error `x_i - x_bar`, using
the Hessian at the averaged model. Worker-level scalars are saved, and aggregate
alignment and norm curves join the existing figures. Epoch exports require all
matching `node-NNN/` weight files; packed `.pt` states already contain them.
Use `--no-consensus` with `analyze` or `submit-analysis` to disable this diagnostic.

```bash
uv run tiny-llm analyze --run runs/20m --checkpoints all \
  --output runs/20m/analysis
# Analyze a selection with the measured BF16/HVP batch settings:
uv run tiny-llm analyze --run runs/20m \
  --checkpoints epoch-001.safetensors epoch-020.safetensors final.pt \
  --amp --hvp-batch-size 64 --output runs/20m/analysis-bf16
```

Defaults are both data cases, 32 noise samples, 32 random directions, seed 42,
FP32 without AMP, TF32 enabled on CUDA, and compiled Pearlmutter HVPs with
reference attention. Gradient microbatch size always comes from `resolved.yaml`;
HVP batch size defaults to that size. Useful options:

- `--data-case seen|unseen|both` selects data cases.
- `--noise-samples N|all` and `--random-samples N` control sampling.
- `--amp --hvp-batch-size 64` enables BF16 forward AMP with FP32 parameters.
- `--device cpu --dtype float64` runs eager derivative checks on CPU.
- `--no-tf32`, `--no-compile-hvp`, and `--no-plots` disable those features.

Results include per-sample JSON, a manifest, `summary.csv`, and three PDF/PNG
figures: normalized alignment, unnormalized noise alignment, and gradient/noise
norms. Seen and unseen curves share axes against training tokens. Repeat the
same command to reuse completed compatible results; use a new output directory
when changing source or numerical settings. Regenerate figures with:

```bash
uv run tiny-llm plot-analysis --output runs/20m/analysis
```

Add `--training-norms` to include logged training norms. See the
[analysis reference](doc/analysis.md) for formulas, data selection, and precision,
and [measurements](doc/analysis_performance.md) for BF16 accuracy and runtime.

## Slurm jobs

Submit from the repository root. Create `runs/` first because Slurm opens logs
before the launcher starts:

```bash
mkdir -p runs
sbatch scripts/slurm.sh train --config configs/20m.yaml
sbatch scripts/slurm.sh train --config configs/packed-20m.yaml
sbatch scripts/slurm.sh evaluate --config runs/20m/resolved.yaml \
  --checkpoint runs/20m/final.pt --full
```

The launcher defaults to account `naiss2026-3-205-gpu`, partition `gpu`, one GPU,
one task, and **two hours**. Slurm allocates CPUs automatically. Combined output
goes to `runs/slurm-<job-id>.log`. The launcher sources `~/.bashrc` to select the
compute node's architecture-specific UV environment, synchronizes locked
dependencies, and runs the command with `srun`.

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

See [benchmarking and sweeps](doc/benchmarking.md) for tuning, profiling,
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
[frozen source entrypoint](doc/analysis.md#execution-and-provenance).

Each shard writes statistics to its own directory; the watcher publishes the
combined results only after all shards succeed. Compiler caches use a unique
temporary directory under `SLURM_TMPDIR`, `TMPDIR`, or `/tmp`, with cleanup on
normal exit, failure, or catchable termination.

## Further documentation

- [Training recipe and evidence](doc/training_recipe.md)
- [Loader, packed models, and checkpoint formats](doc/training_details.md)
- [Analysis definitions and API](doc/analysis.md)
- [Benchmarking, profiling, and sweeps](doc/benchmarking.md)
- Measurements: [training](doc/gh200_performance.md),
  [packed training](doc/packed_benchmarks.md), [analysis](doc/analysis_performance.md)
- [Three-seed 20M recipe benchmark](doc/recipe_sweep_20m.md): synchronous and
  packed training results across 324 runs.
- [Campaign results](doc/campaign_results.md) and
  [implementation validation](doc/implementation_validation.md)

For local code checks, run `uv run pytest -q`, `uv run ruff check .`, and
`uv run ruff format --check .`. GPU tests require a CUDA allocation.
