# Training implementation and artifacts

Run instructions are in the [README](../README.md#training).

## Models

| Preset | Unique trainable parameters | Layers / width / heads / FFN |
|---|---:|---|
| 20M | 20,403,520 | 8 / 320 / 5 / 896 |
| 50M | 48,507,392 | 10 / 512 / 8 / 1408 |
| 90M | 91,605,120 | 14 / 640 / 10 / 1792 |

All use a shared 32K TinyLlama tokenizer, tied embeddings, RoPE, RMSNorm, SwiGLU,
zero dropout, and a 1024-token context. The default budget is 20 prediction targets
per unique trainable parameter, split into 40 virtual epochs. Seeds are saved;
`deterministic=false` is the default. See [recipe and papers](training_recipe.md).

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
refer to this averaged model. With checkpoint policy `all`, root
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
including shortened epoch-ending updates. Before activation, `gamma = 1`.
After activation, `gamma = (lr / lr_max)**p`, where `lr` is the LR assigned to that
update and `lr_max` is the largest LR among all active updates in the original run.
The maximum is computed from the existing token-based LR schedule, so activation
during warmup is supported. Step fractions and token fractions need not coincide.

Mixing uses `W' = gamma * W + (1 - gamma) * I`. Smaller gamma weakens averaging;
gamma zero leaves parameters unchanged at the mixing event. Backward and local
clipping still precede mixing, and each worker's AdamW update follows it. Gradients
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

By default, `training.checkpoint_policy: final` saves only the final training
checkpoint. No periodic, epoch, worker-weight, or interruption checkpoints are
written. Set `training.checkpoint_policy: all` to enable the previous behavior.
With policy `all`, `training.save_epoch_training_state: true` (the default)
also retains complete epoch snapshots. Set the new field to `false` to disable
these archives while preserving weight exports and rolling recovery saves.
The field has no effect under policy `final`.

Each run contains:

- `resolved.yaml`, `environment.json`, `run.log`, and `metrics.jsonl`.
- `epoch-NNN.safetensors` (policy `all`): FP32 weights after every epoch.
- `epoch-NNN.pt` (policy `all`, epoch states enabled): complete ordinary state,
  or shared packed state with references to separate worker files.
- `best.json`: best epoch and subset loss; `weights` is null under policy `final`.
- `latest.pt` (policy `all`): model, AdamW, RNG states, data cursor, and schedule progress.
- `final.pt` and `result.json`: final training state and full-validation metrics.

Resume using the saved configuration. Device and output-directory changes are
allowed, as are changes to `data.prefetch` and checkpoint retention policy. Recipe, precision, data identity, batch,
seed, and buffer-size changes are rejected. Ordinary checkpoint format v2 and
packed format v3 record loader format v1 and a committed sample cursor. Resume
reconstructs the current range and row position directly, without replaying earlier training ranges or saving
buffer contents. The normal startup cache checksum scan still runs.

Old global-permutation checkpoints (ordinary v1 and packed v2) cannot resume
training under this loader; start a fresh run. Their model weights remain usable
for evaluation and analysis.
The previous campaign is preserved under `runs/campaign`; fresh buffered runs use
`runs/campaign-buffered` and reuse the existing model computation benchmarks.

For a run trained with policy `all`:

```bash
uv run tiny-llm train --config runs/baseline/resolved.yaml \
  --resume runs/baseline/latest.pt
```

Resume `.pt` files are trusted local pickle artifacts. For exchanging model
weights, use the `.safetensors` files produced with policy `all`. Model/data artifacts live in gitignored
`runs/` and `data/`; source, configurations, the UV lockfile, and documentation
are tracked in Git.

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

local = torch.load("runs/packed-20m/node-000/epoch-010.pt", map_location="cpu", weights_only=False)
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
consensus analysis all use this reader. Existing combined packed v3 `latest.pt`
and `final.pt` remain unchanged.

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
