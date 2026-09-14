---
title: "Training implementation and artifacts"
---

# Training implementation and artifacts

Run instructions are in the [README](../guides/training.md#training).

## Models

| Preset | Unique trainable parameters | Layers / width / heads / FFN |
|---|---:|---|
| 20M | 20,403,520 | 8 / 320 / 5 / 896 |
| 50M | 48,507,392 | 10 / 512 / 8 / 1408 |
| 90M | 91,605,120 | 14 / 640 / 10 / 1792 |

All use a shared 32K TinyLlama tokenizer, tied embeddings, RoPE, RMSNorm, SwiGLU,
zero dropout, and a 1024-token context. The default budget is 20 prediction targets
per unique trainable parameter, split into 40 virtual epochs. Seeds are saved;
`deterministic=false` is the default. See [recipe and papers](recipe.md).

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

## Packed decentralized training

`PackedLlama` executes independent local models together on one GPU. The same
seed produces exactly the ordinary Llama initialization in every worker. Each
step uses one local microbatch, one parameter-mixing event, and one local AdamW
update; decentralized training does not accumulate gradients.

The tuned 20M AWC and ATC presets use four workers with local microbatch 32
and a global batch of 131,072 tokens. They inherit compilation in `default` mode,
automatic SDPA, and global evaluation batch size 128. Cached
RoPE is shared across workers; each update uses a complete global batch.

```bash
uv run tiny-llm train --config configs/packed4-20m-awc.yaml
# Eight workers at the same global batch size:
uv run tiny-llm train --config configs/packed4-20m-awc.yaml \
  --set decentralized.num_models=8 --set training.micro_batch_size=16 \
  --set runtime.output_dir=runs/packed8-20m-awc
```

The global batch must equal `num_models * micro_batch_size * context_length`.
Every realized epoch boundary must divide evenly across workers. The packed
presets use the same nominal ratios as ordinary models. Epoch boundaries round
the cumulative nominal token count to the nearest complete global batch, with
halfway ties upward. The difference between consecutive boundaries determines
that epoch's batch count. Earlier boundaries are independent of the total budget;
see the [allocation rule](../guides/training.md#token-budget-and-epochs).


Each worker computes a mean loss over its own valid tokens. The sum of these
local means supplies independent gradients, with no division by the worker
count. Gradients are clipped locally before either update scheme, selected with
`decentralized.scheme`:

- `awc` (default, adapt-while-combine): mix parameters, then apply local AdamW updates.
- `atc` (adapt-then-combine): apply complete local AdamW updates, including weight
  decay, then mix the updated parameters.

Use `--set decentralized.scheme=atc` with training or packed benchmarks. Gradients
and optimizer moments remain local in both schemes. The scheme is saved in run
configuration, metadata, summaries, and benchmark results. Existing configurations
and checkpoints retain AWC behavior; resuming with a different scheme is rejected.

Set `--set optimizer.grad_clip=null` to disable gradient clipping. Gradient norms
are still logged and nonfinite gradients still stop training. The default is
`1.0`; positive thresholds retain local clipping. This setting also applies to
ordinary training and packed/sequential benchmarks. Changing it changes the
recipe identity, so resuming requires the same clipping setting.

Available `decentralized.topology` values:

- `complete`: globally average parameters each step.
- `one_peer_ring`: half self, half left/right neighbor, alternating each step.
- `one_peer_exponential`: half self, half incoming neighbor at cyclic
  power-of-two offsets modulo N.

Before validation, a separate ordinary Llama receives the global parameter
average. It evaluates with the global `evaluation.batch_size`; training weights,
moments, and topology phase stay unchanged. Validation metrics and `best.json`
refer to this averaged model. At selected checkpoint epochs, root
`epoch-NNN.safetensors` files contain averaged
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

### Adaptive consensus

Add this optional section to a packed configuration; both fields are required:

```yaml
decentralized:
  num_models: 4
  topology: one_peer_exponential
  adaptive_consensus:
    start_frac: 0.5
    p: 1.0
```

For zero-based update `r`, activation begins at
`ceil(start_frac * total_steps)`. Count actual optimizer updates across all epochs,
using complete global batches. Before activation, `gamma = 1`.
After activation, `gamma = (lr / lr_max)**p`, where `lr` is the LR assigned to that
update and `lr_max` is the largest LR among all active updates in the original run.
The maximum is computed from the selected LR schedule, including its fixed
`warmup_steps`, so activation during warmup is supported.

Mixing uses `W' = gamma * W + (1 - gamma) * I`. Smaller gamma weakens averaging;
gamma zero leaves parameters unchanged at the mixing event. Backward and local
clipping precede both operations. AWC mixes before each worker's AdamW update;
ATC mixes after it. Both use the same gamma and topology step schedule. Gradients
and moments stay local. Evaluation always uses the full global average.
The Python API also accepts a constant `model.mix_(topology, step, gamma=...)`.

`start_frac` must be finite and in `[0, 1]`; `p` must be finite and nonnegative.
Omitting the section or setting `p=0` retains gamma one. If the ceiling places
activation beyond the last update (including `start_frac=1`), gamma stays one.
An active window whose maximum LR is zero is rejected for positive `p`.
`mixing_gamma` is recorded in packed training metrics. Run metadata includes
`adaptive_consensus.total_steps`, `start_step`, `lr_max` (null for an empty window),
and `p`.

Resume reconstructs the schedule from the complete original recipe and restored
step, including when loading an epoch snapshot. Adaptive settings must match;
legacy packed checkpoints remain compatible when the feature is absent.
Existing presets are unchanged. Packed throughput benchmarks reject adaptive
configurations because they use a constant LR rather than the training schedule.

## Checkpoints and artifacts

For disk-saving sweeps, set `--set training.checkpoint_policy=none` to disable
all checkpoint and weight files, including the final checkpoint. Training still
saves configuration, logs, and full-validation results; interrupted runs restart
from scratch.

The default `training.checkpoint_policy: final` saves only `final.pt`.
`interval` plus `checkpoint_epochs: K` saves every K epochs; the default K=1
saves every epoch. `explicit` plus `checkpoint_epochs: [1, 10, 20]` selects epochs.
Inputs normalize to a positive integer list; both scheduled policies also save
`final.pt`. There are no rolling or interruption checkpoint writes.

Each run contains configuration, environment metadata, logs, epoch validation,
best-result statistics, and final metrics. Selected epochs retain root
`epoch-NNN.safetensors` and, for packed models, worker weight exports.
`save_epoch_training_state: true` additionally retains resumable `epoch-NNN.pt`
snapshots. `best.json` references weights only if its best epoch was retained.

Resume with a retained epoch checkpoint or `final.pt` and the saved configuration:

```bash
uv run tiny-llm train --config runs/baseline/resolved.yaml \
  --resume runs/baseline/epoch-010.pt \
  --set runtime.output_dir=runs/baseline-branch
```

Device, output directory, prefetch, and checkpoint retention may change on resume.
Training recipe, total budget, precision, data identity, batch, seed, and buffer
size must match. The loader reconstructs its range and row position from the
committed cursor; the normal startup checksum scan still runs. This config
revision is a clean break: old field names and historical identities are not
translated. Use the archived source for archived runs.

Learning-rate settings now live in `lr_schedule`, selected by `name: cosine` or
`name: wsd`, with an integer `warmup_steps` defaulting to 1000. The former
`warmup_fraction` and optimizer-level schedule fields are rejected. The
schedule is included in the recipe identity, so checkpoints from before this
schema change require the previous source version. See the
[schedule configuration guide](../guides/training.md#learning-rate-schedules).

Resume `.pt` files are trusted local pickle artifacts. Use `.safetensors` when
exchanging model weights. The experiment runner resumes incomplete runs from
`final.pt` when present, otherwise the highest retained root epoch checkpoint.
Runs without a retained state require a fresh output directory.

### Full epoch state and local worker inspection

Snapshots are captured after epoch validation, best-result updates, and advancement
of the completed-epoch counter. They contain parameters, both AdamW moment states,
optimizer counters and groups, Python/NumPy/PyTorch/CUDA RNG states, committed
loader position, global step, configuration/recipe identity, and best-validation
metadata. The token cursor restores learning-rate progress; the global step
restores decentralized mixing phase. Gradients are cleared before the next update,
and compiler caches are not algorithmic training state.

Ordinary epoch files use the existing full checkpoint format v2. Packed epoch
roots use version 4 (`kind: packed_epoch`) and contain shared state plus ordered
worker references and SHA-256 checksums. Each `node-NNN/epoch-NNN.pt` uses version 1
(`kind: packed_worker`) and contains that worker's model configuration, index,
worker count, recipe identity, epoch/step/cursor, named parameters, and named AdamW
state. Parameter groups reference names rather than opaque parameter indices.
All worker tensors are compact CPU clones with their original shapes; arena
padding and other workers' backing storage are excluded.

```python
import torch

local = torch.load(
    "runs/packed4-20m-awc/node-000/epoch-010.pt", map_location="cpu", weights_only=False
)
name = "embedding.weight"
weights = local["model"][name]
first_moment = local["optimizer"]["state"][name]["exp_avg"]
second_moment = local["optimizer"]["state"][name]["exp_avg_sq"]
optimizer_step = local["optimizer"]["state"][name]["step"]
```

A local file is independently inspectable but cannot resume the whole decentralized
run. Pass the root `epoch-NNN.pt` to `--resume`. The shared reader validates the
complete worker set, checksums, names/shapes, groups, counters, and training
position before returning reconstructed packed state. Training, evaluation, and
consensus analysis all use this reader. Final packed state uses the combined
version 3 representation.

Worker files are written atomically and the root is published last. An interrupted
snapshot without a root can be rewritten. Committed epoch snapshots are immutable;
branch from an older epoch into a new output directory. Move/copy a packed root
together with its referenced worker files, preserving their relative paths.

The root stores no duplicate parameter or moment arrays. Each full epoch archive
adds approximately three FP32 model copies per worker (weights plus two moments),
in addition to existing weight exports and rolling/final checkpoints. Snapshots
are retained per epoch, not per periodic save interval.

The new retention field is excluded from recipe identity, so older configurations
and checkpoints remain compatible and retention can change on resume. Complete
state restoration supports bitwise continuation with deterministic execution under
matching conditions. Nondeterministic CUDA kernels and hardware/software changes
retain their existing numerical reproducibility limits.
