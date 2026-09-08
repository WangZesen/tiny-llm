# tiny-llm

Small Llama-style models trained from scratch on English C4, with a fast CUDA
training path and a twice-differentiable PyTorch reference path for research.

| Preset | Unique trainable parameters | Layers / width / heads / FFN |
|---|---:|---|
| 20M | 20,403,520 | 8 / 320 / 5 / 896 |
| 50M | 48,507,392 | 10 / 512 / 8 / 1408 |
| 90M | 91,605,120 | 14 / 640 / 10 / 1792 |

All use a shared 32K TinyLlama tokenizer, tied embeddings, RoPE, RMSNorm, SwiGLU,
zero dropout, and a 1024-token context. The default budget is 20 prediction targets
per unique trainable parameter, split into 40 virtual epochs. Seeds are saved;
`deterministic=false` is the default. See [recipe and papers](docs/training_recipe.md).

## Setup and verification

Python 3.12+, UV, and an NVIDIA GPU with BF16 support are required for training.
The locked PyTorch build uses CUDA 13; the checked local driver is 595.71.05.
CPU fixtures and second-order checks do not require a GPU.

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## Smoke run

This explicitly truncated C4 configuration is for checking the pipeline. Its
final evaluation is labelled incomplete and is not a full-C4 validation result.

```bash
uv run tiny-llm prepare --config configs/smoke.yaml
uv run tiny-llm train --config configs/smoke.yaml
uv run tiny-llm evaluate --config runs/smoke/resolved.yaml \
  --checkpoint runs/smoke/final.pt --full
```

## Prepare and train

Prepare once for the largest model; the smaller runs reuse prefixes of this
cache. The command streams only the training data needed, and all official
English C4 validation data. Model/tokenizer revisions are resolved and recorded
in the manifest. Network access is only needed during preparation and setup.

```bash
uv run tiny-llm prepare --config configs/90m.yaml
uv run tiny-llm benchmark --config configs/20m.yaml \
  --all-presets --output runs/benchmarks
uv run tiny-llm train --config runs/benchmarks/20m/selected.yaml \
  --set runtime.output_dir=runs/baseline
```

Configuration is strict Pydantic + YAML. Override any field with repeated
`--set dotted.key=value` arguments; unknown fields are errors. For example:

```bash
uv run tiny-llm train --config configs/50m.yaml \
  --set optimizer.lr=0.0003 --set runtime.device=cuda:1 \
  --set runtime.deterministic=false --set runtime.output_dir=runs/50m-low-lr
```

Training evaluates a fixed sample of 1,048,576 validation targets each epoch and
the full validation split after the last epoch. Reported loss is token-weighted
cross-entropy in nats. Padding in the last validation block does not contribute.

## Sequential buffered loading

Training partitions the selected cache prefix into sequence-aligned ranges. Each
range is read sequentially across binary shards into a compact `uint16` buffer,
then drained using shuffled sequence indices. Only the requested microbatch is
converted to `int64`; batches and virtual epochs can cross buffer boundaries.
The original sequence boundaries and next-token targets are preserved.

`data.buffer_size_mib: 64` controls the active token buffer. With the default
`data.prefetch: true`, one background reader loads the next range, using about
128 MiB of token storage in total (plus two lookahead tokens, row indices, the
validation subset, and microbatch tensors). Set prefetch to false for one-buffer
loading. `data.shuffle_buffer` remains the separate preparation-time document
shuffle setting. No token-cache regeneration is needed.

`runtime.seed` independently determines range order and each range's row order;
thread timing does not affect samples. Full validation streams in original order.
The fixed epoch-validation subset is collected during one sequential validation
scan at startup and retained in RAM for subsequent epochs (about 2 MiB by default).
Startup also retains the existing cache checksum verification.

## GH200 execution and profiling

On Arrhenius, submit one GPU using the ARM environment already installed in
`.venv-aarch64`. The launcher preserves SLURM's device visibility and CPU allocation.
The 20M, 50M, and 90M default recipes use the measured optimized settings:
microbatch 32, compilation in `default` mode, automatic SDPA, and eight CPU threads.
Validation uses a separate batch size of 128 sequences (131,072 targets at context
1024), with no gradient accumulation. Override `evaluation.batch_size` for devices
with less memory. Loss remains weighted by valid tokens; batch size can cause small
floating-point differences.
For example, train 20M with:

```bash
sbatch scripts/slurm-gh200.sh train --config configs/20m.yaml
```

To repeat the tuning campaign:

```bash
sbatch scripts/slurm-gh200.sh benchmark --config configs/20m.yaml \
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

See [GH200 measurements and analysis](docs/gh200_performance.md).

## Run the complete tuning campaign

The completed search in `runs/campaign` selected LR 0.001 and weight decay 0.1
for all sizes, with beta2 0.95 for 20M and 0.99 for 50M/90M. These settings are
now in the default presets. See [campaign results](docs/campaign_results.md) for
the comparisons and the limits of the 90M selection.

```bash
uv run tiny-llm sweep --config configs/20m.yaml \
  --benchmarks runs/benchmarks --output runs/campaign-buffered --gpus 0,1
uv run tiny-llm report --runs runs/campaign-buffered
```

The sweep runs one independent training process per GPU, never distributed
training. The 12-run search promotes recipes in three stages as described in the
[recipe](docs/training_recipe.md). Rerun the same command after interruption to
resume. Completed runs are skipped; incompatible campaign settings are rejected.

For unattended use, the process can be launched with a terminal multiplexer or
`nohup`, with its log redirected to a local file. Send SIGTERM to the sweep PID
to request checkpoints and graceful interruption of its training children.

## Packed decentralized training

`PackedLlama` executes independent local models together on one GPU. The same
seed produces exactly the ordinary Llama initialization in every worker. Each
step uses one local microbatch, one parameter-mixing event, and one local AdamW
update; decentralized training does not accumulate gradients.

Packed presets use four workers with local microbatch 8. They inherit compilation
in `default` mode, automatic SDPA, and global evaluation batch size 128. Cached
RoPE is shared across workers; shortened local batches run eagerly.

```bash
uv run tiny-llm train --config configs/packed-20m.yaml
# Eight workers at the same global batch size:
uv run tiny-llm train --config configs/packed-20m.yaml \
  --set decentralized.num_models=8 --set training.micro_batch_size=4 \
  --set runtime.output_dir=runs/packed-20m-n8
```

The global batch must equal `num_models * micro_batch_size * context_length`.
Every realized epoch boundary must divide evenly across workers. The packed
presets round each ordinary total budget down to a multiple of `8 * context_length`,
so their prepared caches remain sufficient. They use 40 epochs with a shorter
final epoch (408,068,096 total targets for packed 20M). The ordinary presets retain
their existing token-budget calculation.
The buffered loader's shuffled sample stream is partitioned without overlap:
worker i receives every Nth sample, including across buffer and epoch boundaries.

Each worker computes a mean loss over its own valid tokens. The sum of these
local means supplies independent gradients, with no division by the worker
count. Gradients are clipped locally; parameters are then mixed before local
AdamW updates. Moments remain local. Available `decentralized.topology` values:

- `complete`: globally average parameters each step.
- `one_peer_ring`: half self, half left/right neighbor, alternating each step.
- `one_peer_exponential`: half self, half incoming neighbor at cyclic
  power-of-two offsets modulo N.

Before validation, a separate ordinary Llama receives the global parameter
average. It evaluates with the global `evaluation.batch_size`; training weights,
moments, and topology phase stay unchanged. Validation metrics and `best.json`
refer to this averaged model. Root `epoch-NNN.safetensors` files contain averaged
weights; `node-000/epoch-NNN.safetensors`, etc. contain local weights. Resume
checkpoints retain all local parameters and optimizer states.

Parameters and AdamW's `m` and `v` occupy three contiguous `[N, D_padded]` arenas
with identical offsets and 64-element alignment. Padding is zero and excluded
from optimization and exports. Place the model on its final device/dtype before
constructing the optimizer:

```python
from tiny_llm.packed import PackedLlama, local_mean_losses
from tiny_llm.packed_optimizer import PackedAdamW

model = PackedLlama(config.model, 4).to(device)
optimizer = PackedAdamW(model, config)
parameters = model.parameter_storage
m, v = optimizer.first_moment_storage, optimizer.second_moment_storage
layout = model.layout  # name, local shape, start, numel, padded_numel
```

Local optimizer parameter/moment tensors directly alias these arenas. Arena
edits should occur under `torch.no_grad()` between updates. Use
`model.local_state_dict(i)` for ordinary Llama-compatible weights, and the
training resume checkpoints to preserve arena bindings and optimizer counters.

### SLURM on aarch64

The target platform is an aarch64 GPU compute node. Submit from the repository;
the script sources `~/.bashrc` before selecting the architecture-specific UV
environment and running training:

```bash
sbatch scripts/slurm-packed.sh train --config configs/packed-20m.yaml
sbatch scripts/slurm-packed.sh benchmark-packed --config configs/20m.yaml \
  --num-models 4 8 --output runs/packed-benchmarks
```

The script requests account `naiss2026-3-205-gpu` and `--gpus 1` on the GPU
partition. The benchmark compares packed and sequential ordinary workers in
isolated processes with the same global batch, initialization, and updates.
It reports throughput, speedup, memory, component timings, and profiler traces.
Compilation is enabled by default; use `--set runtime.compile=false` for an eager
comparison. Full validation and data loading are excluded from synthetic benchmark timings.
See [GH200 validation and benchmark results](docs/packed_benchmarks.md) for the
measured N=4/8 comparisons and memory costs.

## Checkpoints and artifacts

Each run contains:

- `resolved.yaml`, `environment.json`, `run.log`, and `metrics.jsonl`.
- `epoch-NNN.safetensors`: canonical FP32 model weights after every epoch.
- `best.json`: best intermediate checkpoint by subset-validation loss.
- `latest.pt`: model, AdamW state, RNG states, data cursor, and schedule progress.
- `final.pt` and `result.json`: final training state and full-validation metrics.

Resume using the saved configuration. Device and output-directory changes are
allowed, as is changing `data.prefetch`. Recipe, precision, data identity, batch,
seed, and buffer-size changes are rejected. Ordinary checkpoint format v2 and
packed format v3 record loader format v1 and a committed sample cursor. Resume
reconstructs the current range and row position directly, without replaying earlier training ranges or saving
buffer contents. The normal startup cache checksum scan still runs.

Old global-permutation checkpoints (ordinary v1 and packed v2) cannot resume
training under this loader; start a fresh run. Their model weights remain usable
for evaluation and analysis.
The previous campaign is preserved under `runs/campaign`; fresh buffered runs use
`runs/campaign-buffered` and reuse the existing model computation benchmarks.

```bash
uv run tiny-llm train --config runs/baseline/resolved.yaml \
  --resume runs/baseline/latest.pt
```

Resume `.pt` files are trusted local pickle artifacts. For exchanging model
weights, use the `.safetensors` files. Model/data artifacts live in gitignored
`runs/` and `data/`; source, configurations, the UV lockfile, and documentation
are tracked in Git.

## Analysis API

```python
import torch
from safetensors.torch import load_file
from tiny_llm.config import load_config
from tiny_llm.model import Llama, token_losses

config = load_config("runs/baseline/resolved.yaml")
model = Llama(config.model, attention_backend="reference").double()
model.load_state_dict(load_file("runs/baseline/epoch-040.safetensors"), strict=True)
inputs = torch.tensor([[1, 4, 8]])
targets = torch.tensor([[4, 8, 2]])
loss = token_losses(model(inputs), targets).mean()
parameters = tuple(model.parameters())
gradient = torch.autograd.grad(loss, parameters, create_graph=True)
direction = tuple(torch.randn_like(p) for p in parameters)
directional_gradient = sum((g * v).sum() for g, v in zip(gradient, direction))
hvp = torch.autograd.grad(directional_gradient, parameters)
```

`torch.func.functional_call` is supported for stateless parameter experiments.
The entire reference computation preserves double precision. Gradient-noise
estimators and Hessian eigensolvers are not implemented.
