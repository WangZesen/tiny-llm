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

## Run the complete tuning campaign

```bash
uv run tiny-llm sweep --config configs/20m.yaml \
  --benchmarks runs/benchmarks --output runs/campaign --gpus 0,1
uv run tiny-llm report --runs runs/campaign
```

The sweep runs one independent training process per GPU, never distributed
training. The 12-run search promotes recipes in three stages as described in the
[recipe](docs/training_recipe.md). Rerun the same command after interruption to
resume. Completed runs are skipped; incompatible campaign settings are rejected.

For unattended use, the process can be launched with a terminal multiplexer or
`nohup`, with its log redirected to a local file. Send SIGTERM to the sweep PID
to request checkpoints and graceful interruption of its training children.

## Checkpoints and artifacts

Each run contains:

- `resolved.yaml`, `environment.json`, `run.log`, and `metrics.jsonl`.
- `epoch-NNN.safetensors`: canonical FP32 model weights after every epoch.
- `best.json`: best intermediate checkpoint by subset-validation loss.
- `latest.pt`: model, AdamW state, RNG states, data cursor, and schedule progress.
- `final.pt` and `result.json`: final training state and full-validation metrics.

Resume using the saved configuration. Device and output-directory changes are
allowed; recipe, precision, data identity, and batch changes are rejected.

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
The entire reference computation preserves double precision. Stage one does not
implement gradient-noise estimators, Hessian eigensolvers, or decentralized
training algorithms.
