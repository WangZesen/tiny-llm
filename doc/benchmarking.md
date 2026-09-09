# Benchmarking, profiling, and sweeps

Submit jobs from the repository root using the [Slurm launcher](../README.md#slurm-jobs).

## GH200 execution and profiling

Use the [SLURM launcher](../README.md#slurm-jobs) on Arrhenius.
The 20M, 50M, and 90M default recipes use the measured optimized settings:
microbatch 32, compilation in `default` mode, automatic SDPA, and eight CPU threads.
Validation uses a separate batch size of 128 sequences (131,072 targets at context
1024), with no gradient accumulation. Override `evaluation.batch_size` for devices
with less memory. Loss remains weighted by valid tokens; batch size can cause small
floating-point differences.
For example, train 20M with:

```bash
mkdir -p runs
sbatch scripts/slurm.sh train --config configs/20m.yaml
```

To repeat the tuning campaign:

```bash
mkdir -p runs
sbatch scripts/slurm.sh benchmark --config configs/20m.yaml \
  --gh200 --budget-minutes 75 --output runs/gh200-tuning
```

The staged tuner measures real C4 updates, then varies microbatch size,
compilation, SDPA kernel selection, and CPU threads. Every candidate runs in a
separate process with 20 warmup updates and three 100-update timing windows.
`selected.yaml` records the fastest successful candidate. Compilation and
warmup are reported separately. Synthetic benchmarks remain available with
`--data-mode synthetic`; they exclude loading and transfers.

To profile a specific configuration, run this inside an allocation or pass the
same arguments to the SLURM launcher:

```bash
.venv-aarch64/bin/python -m tiny_llm benchmark-worker \
  --config runs/gh200-tuning/selected.yaml --data-mode real --profile \
  --output runs/gh200-profile.json
```

The worker writes repeated measurements, hardware/software metadata, and actual
attention dispatch. `--profile` additionally writes a Chrome CPU/CUDA trace and
an operator table after timing finishes. Benchmark windows use full optimizer
batches without validation or checkpoint writes; actual training also handles
short virtual-epoch boundary updates. Use training results to assess elapsed
throughput including validation and checkpoints.

`runtime.compile_mode` accepts `default`, `reduce-overhead`, or `max-autotune`;
`runtime.compile` still controls whether compilation is enabled. Full microbatches
are compiled; short epoch-ending microbatches run eagerly to avoid recompilation.
`runtime.sdpa_backend` accepts `auto`, `flash`, or `cudnn`. Forced unsupported
kernels fail explicitly. The reference path remains available for second-order
analysis. Default-valued new settings preserve existing checkpoint identities;
changing execution settings for a resumed run still requires a compatible recipe.

Training metrics report `tokens_per_second` excluding startup, validation, and
checkpoint writes, and `elapsed_tokens_per_second` including validation and
checkpoint overhead since the first training update. Compilation is included
in the first training window. Final results also record aggregate training and
elapsed seconds; `seconds_this_session` additionally includes validation-subset
preparation and final full validation, but excludes earlier model/cache setup.

See [GH200 measurements and analysis](gh200_performance.md).

## Run the complete tuning campaign

The completed search in `runs/campaign` selected LR 0.001 and weight decay 0.1
for all sizes, with beta2 0.95 for 20M and 0.99 for 50M/90M. These settings are
now in the default presets. See [campaign results](campaign_results.md) for
the comparisons and the limits of the 90M selection.

```bash
uv run tiny-llm sweep --config configs/20m.yaml \
  --set training.checkpoint_policy=all \
  --benchmarks runs/benchmarks --output runs/campaign-buffered --gpus 0,1
uv run tiny-llm report --runs runs/campaign-buffered
```

The sweep runs one independent training process per GPU, never distributed
training. The 12-run search promotes recipes in three stages as described in the
[recipe](training_recipe.md). Rerun the same command after interruption to
resume. Completed runs are skipped; incompatible campaign settings are rejected.

For unattended use, the process can be launched with a terminal multiplexer or
`nohup`, with its log redirected to a local file. Send SIGTERM to the sweep PID
to request checkpoints and graceful interruption of its training children.

## Packed benchmarks

Use the same [SLURM launcher and setup](../README.md#slurm-jobs) for packed training on the
aarch64 compute nodes:

```bash
mkdir -p runs
sbatch scripts/slurm.sh train --config configs/packed-20m.yaml
sbatch scripts/slurm.sh benchmark-packed --config configs/20m.yaml \
  --num-models 4 8 --output runs/packed-benchmarks
```

The benchmark compares packed and sequential ordinary workers in
isolated processes with the same global batch, initialization, and updates.
It reports throughput, speedup, memory, component timings, and profiler traces.
Compilation is enabled by default; use `--set runtime.compile=false` for an eager
comparison. Full validation and data loading are excluded from synthetic benchmark timings.
See [GH200 validation and benchmark results](packed_benchmarks.md) for the
measured N=4/8 comparisons and memory costs.
