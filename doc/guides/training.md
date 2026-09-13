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

Use `configs/50m.yaml` or `configs/90m.yaml` for larger models. Override fields
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
and epoch boundaries must divide evenly across workers. With checkpoint policy
`all`, root epoch checkpoints contain averaged weights and worker weights live
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

Activation uses the ceiling of `start_frac * total_steps`, counting shortened
epoch-ending updates. See [adaptive consensus](../methods/implementation.md#adaptive-consensus)
for the LR normalization and resume semantics. It is supported by training only;
packed throughput benchmarks reject adaptive-consensus configurations.


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
