---
title: "Training and evaluation"
---

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

Preparation uses `data.prepare_workers=8` tokenizer processes, each loading the
saved tokenizer locally. Set `--set data.prepare_workers=1` for serial preparation,
or another positive integer to change the process count. Worker count does not
change cache contents or checkpoint compatibility. Results are written in source
order, with at most two batches per worker in flight; `data.tokenize_batch_size`
defaults to 256 documents.

The preparation-time training shuffle uses `data.shuffle_seed` and
`data.shuffle_buffer`, which defaults to 100,000 documents. This larger buffer
provides approximate token coverage; document lengths determine whether it holds
more tokens than one shard. Validation keeps its original order.

Completed caches are immutable. A cache prepared with the former 10,000-document
shuffle requires `--set data.shuffle_buffer=10000` to reuse it. To prepare with the
new defaults, select a new path for both preparation and training:

```bash
uv run tiny-llm prepare --config configs/90m.yaml --set data.cache_dir=data/c4-large
uv run tiny-llm train --config configs/20m.yaml --set data.cache_dir=data/c4-large
```

Training reads groups of consecutive cache shards in file order and shuffles
sequences within each group. Set `--set data.shuffle_group_size=4` on preparation
and training to use four shards per group; the default is 2 (67,108,864 tokens,
about 128 MiB with the default 33,554,432-token shards). Prefetching a second group
uses about 256 MiB of token storage. This positive integer replaces
`data.buffer_size_mib`, which is no longer accepted. Larger groups provide wider
mixing and use more memory.
A shorter run uses the exact sample prefix of a longer run under the same seed
and data settings, even when it stops partway through a group.
See [loader details](../methods/implementation.md#sequential-buffered-loading).

Use `configs/50m.yaml` or `configs/90m.yaml` for larger models. Repeat `--config`
to merge YAML files in order: later values override earlier values, mappings
merge recursively, and lists are replaced. Then override fields
with repeated `--set dotted.key=value` arguments; unknown fields are rejected.
Give separate experiments their own output directories:

```bash
uv run tiny-llm train --config configs/50m.yaml \
  --set optimizer.lr=0.0003 --set runtime.output_dir=runs/50m-low-lr
```

By default, each run saves `resolved.yaml`, logs (`run.log`, `metrics.jsonl`), final state
(`final.pt`), best subset statistics (`best.json`), and final metrics (`result.json`).
For disk-saving sweeps, set `--set training.checkpoint_policy=none` to disable
all checkpoint and weight files, including the final checkpoint. Training still
saves configuration, logs, and full-validation results; interrupted runs restart
from scratch.

`training.checkpoint_policy` defaults to `final`. Use `interval` to save every
K epochs, or `explicit` to select epoch numbers:

```bash
uv run tiny-llm train --config configs/20m.yaml \
  --set training.checkpoint_policy=interval --set training.checkpoint_epochs=5
uv run tiny-llm train --config configs/20m.yaml \
  --set training.checkpoint_policy=explicit --set 'training.checkpoint_epochs=[1,10,20,40]'
```

The default interval is 1 (every epoch). Scalar inputs normalize to lists in
`resolved.yaml`; explicit lists are sorted and deduplicated. Epoch numbers must
be positive and explicit selections must lie within the run. Both scheduled
policies also save `final.pt`. Validation and best-loss tracking run every epoch;
`best.json` has a null weight reference when its best epoch was not saved.

There are no step-based or interruption checkpoints. Interrupted runs resume from
a retained `epoch-NNN.pt` or `final.pt`; without one, start in a new output directory.
`training.save_epoch_training_state` defaults to `true`; set it to `false` to retain
only weights at selected epochs, plus final training state.

Packed epoch snapshots store each
worker's weights and AdamW moments in `node-NNN/epoch-NNN.pt`; the root file holds
shared state and references to those files. Keep the complete set together.
To branch from an earlier epoch, use its root checkpoint and a new output directory:

```bash
uv run tiny-llm train --config runs/packed4-20m-awc/resolved.yaml \
  --resume runs/packed4-20m-awc/epoch-010.pt \
  --set runtime.output_dir=runs/packed4-20m-awc-from-epoch010
```

Resume allows device, output-directory, and prefetch changes; keep the training
recipe and data settings unchanged. See [checkpoint compatibility](../methods/implementation.md#checkpoints-and-artifacts).

For a short pipeline check:

```bash
uv run tiny-llm prepare --config configs/smoke.yaml
uv run tiny-llm train --config configs/smoke.yaml
```

The smoke configuration truncates C4, so its evaluation is marked incomplete.

### Learning-rate schedules

Model recipes under `configs/` omit scheduler settings. Select a scheduler by
adding `configs/cosine.yaml` or `configs/wsd.yaml` after the model recipe:

```bash
uv run tiny-llm train --config configs/20m.yaml --config configs/cosine.yaml
uv run tiny-llm train --config configs/20m.yaml --config configs/wsd.yaml \
  --set lr_schedule.warmup_steps=312 --set lr_schedule.decay_fraction=0.2 \
  --set runtime.output_dir=runs/20m-wsd
```

`lr_schedule.name` selects a separate, strict config class for `cosine` or `wsd`.
An omitted `lr_schedule` section defaults to cosine; an explicit section requires
`name`. Both schedules use `optimizer.lr` as the base learning rate and default
to `lr_schedule.warmup_steps=312`. This must be a nonnegative integer; zero
disables warmup. Warmup is linear over optimizer updates: update 1 uses
`optimizer.lr / warmup_steps`, and update `warmup_steps` reaches the base rate.
Microbatches during accumulation and individual packed workers do not count as
additional updates. Changing the total budget does not change the warmup length,
and epoch boundaries or resumes do not reset it.

The scheduler tracks consumed tokens, with one optimizer update consuming
`training.batch_tokens` targets. Thus the warmup token count is
`warmup_steps * training.batch_tokens`; decay uses the realized total budget.
If a run ends during warmup, it keeps its linear warmup rate through the final
update. The configured warmup length is never capped to the run length.

Cosine decays immediately after warmup, reaching
`optimizer.lr * lr_schedule.min_lr_ratio` at the end if the run extends beyond
warmup. The minimum ratio defaults to `0.1` and must be in `[0, 1]`.

WSD holds the base learning rate after warmup, then decays over the final
`lr_schedule.decay_fraction` of the budget (default `0.1`). For consumed tokens
$T$, total tokens $N$, warmup tokens $W$, base learning rate $\eta$, and decay
fraction $d$, decay starts at $S=\max(W,(1-d)N)$ and follows

$$
\mathrm{lr}(T)=\eta(1-\sqrt{q}),\qquad
q=\operatorname{clamp}\left(\frac{T-S}{N-S},0,1\right).
$$

The decay fraction must be in `[0, 1]`. When warmup reaches into the requested
decay window, there is no stable phase and decay is shortened to the remaining
updates after warmup. If warmup covers the whole run, decay is never entered.
A zero decay fraction holds the base rate after warmup, including at the end of
training. Positive decay reaches zero at the total budget when the run extends
beyond warmup. WSD has no `min_lr_ratio` field.

Changing `lr_schedule.name` in a later config file or a `--set` override resets
the scheduler section, so fields from the previous variant do not carry over.
Partial sections and overrides otherwise retain earlier fields. Config files
are merged before any `--set` overrides, and validation occurs only after all
inputs are combined. The saved `resolved.yaml` contains the complete config and
can be loaded on its own.

The Python API also accepts multiple paths:

```python
from tiny_llm.config import load_config

config = load_config(
    ["configs/20m.yaml", "configs/wsd.yaml"],
    ["lr_schedule.warmup_steps=500"],
)
```

The old `lr_schedule.warmup_fraction`, `optimizer.warmup_fraction`, and
`optimizer.min_lr_ratio` fields are rejected. Old configs and checkpoint
identities are not translated; use the
previous source version for existing runs. New runs require the same schedule
and phase settings on resume. Adaptive consensus uses the selected schedule;
any nonempty active window must contain a positive learning rate when `p > 0`.

### Token budget and epochs

`training.tokens_per_parameters` controls the total nominal token budget, and
`training.epoch_tokens_per_parameters` controls one nominal epoch. The defaults
are 20 and 0.5, giving 40 epochs. The total ratio must be divisible by the epoch
ratio; fractional final epochs are rejected.

For a single-model parameter count P, epoch ratio e, and global batch size B,
the cumulative batch count after epoch k is `floor(k * e * P / B + 0.5)`.
Each epoch receives the difference from the previous cumulative count. Halfway
ties round up, and empty epochs are rejected. Every update uses a complete global
batch, including the final update; realized tokens may be above or below the
nominal total by at most B/2.

Increasing only the total ratio preserves earlier epoch batch counts and samples
with the same cache contents, shard layout, context length, group size, and seed.
Learning rates and adaptive consensus still depend on the total budget.
Changing that budget requires a new run rather than resuming an existing recipe.
Preparation rounds the realized budget up to a complete shuffle group, plus
sequence alignment and lookahead; changing global batch size can change the
required capacity. `data.prepare_train_tokens` is also rounded up this way. Old token fields and checkpoint policy `all`
are rejected by the new schema. Archived experiments retain their original
configuration and token positions.

### Packed training

The tuned packed recipes run four or eight local models together on one GPU.
They use a global batch of 131,072 tokens, with 32 sequences per model for four
workers and 16 for eight. The number after `packed` denotes the local model count.

| Workers | Scheme | Recipe | Learning rate | Beta1 | Beta2 |
|---:|---|---|---:|---:|---:|
| 4 | AWC | [packed4-20m-awc.yaml](../../configs/packed4-20m-awc.yaml) | 0.008 | 0.95 | 0.99 |
| 4 | ATC | [packed4-20m-atc.yaml](../../configs/packed4-20m-atc.yaml) | 0.0056 | 0.9 | 0.99 |
| 8 | AWC | [packed8-20m-awc.yaml](../../configs/packed8-20m-awc.yaml) | 0.0056 | 0.95 | 0.999 |

These are the best tested settings in the [four-worker AWC](../results/optimizer-tuning.md),
[four-worker ATC](../results/optimizer-tuning.md), and
[eight-worker AWC](../results/ablations.md) studies.
Each recipe explicitly sets its scheme and uses a separate output directory.

```bash
uv run tiny-llm train --config configs/packed4-20m-awc.yaml
uv run tiny-llm train --config configs/packed4-20m-atc.yaml
uv run tiny-llm train --config configs/packed8-20m-awc.yaml
```

The global batch must equal `num_models * micro_batch_size * context_length`,
and epoch boundaries must divide evenly across workers. With a scheduled checkpoint policy, root epoch weight exports contain averaged
weights and worker weights live
under `node-NNN/`. The default `final.pt` retains all local training states.
[Packed implementation details](../methods/implementation.md#packed-decentralized-training)
cover topologies and optimizer behavior.

Packed training defaults to adapt-while-combine (AWC): local gradients are computed
and clipped, then parameters are mixed before the AdamW update. Set
`--set decentralized.scheme=atc` for adapt-then-combine (ATC), which applies
the AdamW update before mixing. Optimizer moments remain local in both schemes.

Set `--set optimizer.grad_clip=null` to disable gradient clipping. Gradient norms
are still logged and nonfinite gradients still stop training. The default is
`1.0`; positive thresholds retain local clipping. This setting also applies to
ordinary training and packed/sequential benchmarks. Changing it changes the
recipe identity, so resuming requires the same clipping setting.

Optional adaptive consensus weakens mixing as the learning rate falls. The sample
config retains its separate 32,768-token recipe and uses four workers,
`start_frac=0.5`, and `p=1.0`:

```bash
uv run tiny-llm train --config configs/packed-20m-adaptive.yaml
```

Activation uses the ceiling of `start_frac * total_steps`, counting complete
global batches. See [adaptive consensus](../methods/implementation.md#adaptive-consensus)
for the LR normalization and resume semantics. It is supported by training only;
packed throughput benchmarks reject adaptive-consensus configurations.


## Evaluation

Epoch-weight examples below require training with `--set training.checkpoint_policy=interval`.

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
