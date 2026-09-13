# tiny-llm

Train small Llama-style language models from scratch on English C4, inspect
decentralized training, and measure gradient-noise alignment with the loss Hessian.
Presets cover 20M, 50M, and 90M parameters, plus packed local models on one GPU.

**[Research website and interactive results](https://wangzesen.github.io/tiny-llm/)**
· [Results](doc/results/overview.md) · [Training guide](doc/guides/training.md)

## Quick start

Training requires Python 3.12+, UV, and an NVIDIA GPU with BF16 support. The locked
PyTorch build uses CUDA 13 and needs a compatible driver. Run from the repository root:

```bash
uv sync --locked
uv run tiny-llm prepare --config configs/90m.yaml
uv run tiny-llm train --config configs/20m.yaml
```

Preparation downloads and caches C4 once; smaller models reuse prefixes of the
same cache. To run the tuned four-worker AWC recipe:

```bash
uv run tiny-llm train --config configs/packed4-20m-awc.yaml
```

The default checkpoint policy saves final training state only. Use
`--set training.checkpoint_policy=all` when you need epoch checkpoints for analysis.
See the [training and evaluation guide](doc/guides/training.md) for smoke runs,
resume behavior, alternate recipes, clipping, and checkpoint retention.

## Documentation

- Results: [overview](doc/results/overview.md), [optimizer tuning](doc/results/optimizer-tuning.md), [clipping and worker count](doc/results/ablations.md).
- Methods: [protocol](doc/methods/protocol.md), [equations](doc/methods/algorithms.md), [implementation](doc/methods/implementation.md), [analysis](doc/methods/analysis.md), [recipe evidence](doc/methods/recipe.md).
- Guides: [training/evaluation](doc/guides/training.md), [analysis](doc/guides/analysis.md), [Slurm](doc/guides/slurm.md), [benchmarking](doc/guides/benchmarking.md).
- Performance: [training](doc/performance/training.md), [checkpoint analysis](doc/performance/checkpoint-analysis.md).
- [Adaptive-consensus presentation and supplement](doc/methods/adaptive-consensus.md).
- [Historical reports](doc/archive/) and [experimental data](doc/data/).

## Development

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

GPU tests require a CUDA allocation. The website uses an independent Node 24+
environment and builds entirely from committed Markdown and data:

```bash
npm --prefix site ci
npm --prefix site run build
npm --prefix site run dev
```

See the [website maintenance guide](doc/guides/website.md) for data refresh,
browser checks, and GitHub Pages publication.
