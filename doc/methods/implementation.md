---
archive: true
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

## Cache preparation

Preparation streams C4 in the parent process and shuffles training documents with
`data.shuffle_buffer=100000` and `data.shuffle_seed`. The buffer counts documents,
so its token coverage depends on document lengths and has no enforced token floor.
Validation remains in source order. Shards default to 33,554,432 `uint16` tokens
(64 MiB per full shard).

`data.prepare_workers=8` controls a spawned tokenizer process pool; setting it to
1 uses serial tokenization. Each worker loads the saved, resolved tokenizer once,
without network access or tokenizer-internal parallelism. The queue holds at most
two batches per worker, with 256 documents per batch by default. Results reach one
shard writer in submission order, preserving EOS placement and identical bytes
across worker counts.
Pending tasks are canceled and active tasks drained on early completion or failure.
Both splits and their manifest are published atomically only after success.

Worker count affects neither cache nor training recipe identity. Shuffle settings
remain part of cache compatibility: use a new cache path to change them, or retain
the matching settings to reuse an existing immutable cache.

## Sequential buffered loading

Training visits consecutive groups of cache shards in file order. Each group is
read into a compact `uint16` buffer, then drained using shuffled sequence indices.
Only the requested microbatch is converted to `int64`; batches and virtual epochs
can cross group boundaries. A sequence belongs to the group containing its first
token. Reads align to the original sequence boundaries and include next-token
lookahead, preserving targets even across shard boundaries.

CUDA batches are assembled directly into pinned `int64` host tensors through
NumPy views, avoiding an extra copy from ordinary host arrays into pinned memory.
The owning tensors are then copied to the target device with `non_blocking=True`.
Batch preparation runs on the training thread and transfers use its current CUDA
stream; only shard reads are prefetched. CPU batches retain the NumPy path.

`data.shuffle_group_size: 2` sets the positive integer number of shards per group.
With the default shard size this is about 128 MiB per active buffer. With
`data.prefetch: true`, one background reader loads the next group, using about
256 MiB of token storage (plus sequence alignment/lookahead, row indices, the
validation subset, and microbatch tensors). Set prefetch to false for one-buffer
loading. `data.shuffle_buffer` remains the separate preparation-time document
shuffle setting. The former `data.buffer_size_mib` field is rejected.

The first group is loaded synchronously. At later boundaries the loader waits
if the prefetched group is not ready; filesystem reads have no timeout. With
default sizes, a group supplies 512 updates at 131,072 targets per update, giving
the background reader tens of seconds to fetch the next 128 MiB. See the
[GH200 loading measurements](../performance/training.md#data-loading-on-gh200)
for observed read times and batch preparation overhead.

`runtime.seed` and the physical group index determine each group's row permutation.
The entire final group is shuffled even when training stops partway through it.
For the same cache contents and shard layout, context length, group size, and seed,
extending the training budget preserves the entire earlier sample stream. Batch
size and prefetch timing do not change that stream; virtual epochs never reset it.
Different seeds change the order within groups, while group order stays fixed.
Mixing is limited to each group; larger groups trade more memory for wider mixing.

Preparation rounds the requested training capacity up through the final full group,
including sequence alignment and lookahead. Existing caches can be reused when
these complete groups fit; insufficient caches require preparation at a new path.
Loader version 2 records group size and the committed cursor in checkpoints.
Historical loader order and identities require their original source version.

Full validation streams in original order, including its partial final group.
The fixed epoch-validation subset is collected during one sequential validation
scan at startup and retained in RAM for subsequent epochs (about 2 MiB by default).
Training startup validates the manifest, shard sizes, and configuration without
hashing token contents. Prepared caches are assumed immutable; rerun `prepare`
with matching settings to explicitly verify every shard checksum.

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
- `one_peer_ring`: half self, half peer in reciprocal adjacent pairs. Zero-based
  even steps pair `(0,1), (2,3), ...`; odd steps pair
  `(1,2), (3,4), ..., (N-1,0)`. Requires even N.
- `one_peer_exponential`: half self, half peer at
  `i XOR (1 << (step % log2(N)))`. Requires power-of-two N. With full mixing and
  no local updates, one cycle of `log2(N)` steps gives the exact global average
  in exact arithmetic.

Both one-peer schedules send to and receive from the same peer at each step.
N=1 is allowed for every topology and mixing is a no-op. The saved training step
determines the pairing phase on resume. Decentralized recipe identities include
`mixing_version=2`; checkpoints from the previous directed schedules cannot resume
under the new identity, and there is no legacy mixing mode.

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
Training recipe, total budget, precision, data identity, batch, seed, and shard-group
size must match. The loader reconstructs its group and row position from the
committed cursor, without a full-cache checksum scan at startup. This config
revision is a clean break: old field names and historical identities are not
translated. Use the archived source for archived runs.

Learning-rate settings now live in `lr_schedule`, selected by `name: cosine` or
`name: wsd`, with an integer `warmup_steps` defaulting to 312. The former
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
