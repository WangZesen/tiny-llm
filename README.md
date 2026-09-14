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

Preparation downloads and caches C4 once, using 8 tokenizer processes by default;
set `--set data.prepare_workers=1` for serial preparation. The training document
shuffle buffer defaults to 100,000 documents; its token coverage depends on document
lengths. Smaller models reuse prefixes of the same cache. Training visits shard
groups in file order and shuffles within each group. Each shard holds 33,554,432
tokens. `data.shuffle_group_size` defaults to 2 shards (about 128 MiB); it replaces
`data.buffer_size_mib`. Keeping the same seed and data settings preserves earlier
samples when increasing the training budget. Preparation includes the whole final
group and lookahead.

Existing caches require matching preprocessing settings. To use the new shuffle
default with an older cache present, set `--set data.cache_dir=data/c4-large` on
both preparation and training; completed caches are never rewritten.

To run the tuned four-worker AWC recipe:

```bash
uv run tiny-llm train --config configs/packed4-20m-awc.yaml
```

The default checkpoint policy saves final training state only. Use
`--set training.checkpoint_policy=interval` when you need epoch checkpoints for analysis.
Model recipes omit scheduler settings and default to cosine. Add
`--config configs/wsd.yaml` after the recipe's `--config` to use warmup-stable-decay;
`configs/cosine.yaml` selects cosine explicitly. Both schedules default to
312 warmup updates; override with `--set lr_schedule.warmup_steps=500`.
Repeated config files merge in order, followed by `--set` overrides.
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
uv run pyright
scripts/test-cpu.sh
uv run ruff check .
uv run ruff format --check .
```

Type checks cover `src`, `tests`, and `scripts`. Ruff leaves archived research
sources under `doc/archive`, `doc/data`, and `doc/adaptive-consensus-investigation`
unchanged.

The default CPU suite targets less than 60 seconds. It uses one PyTorch CPU thread
and fixed machine metadata; benchmark tests cover actual profiling. Numerical,
resume, corruption, and orchestration checks remain in the default run. Third-party
pytest plugins require explicit loading with `-p`.

The CPU launcher copies locked dependencies into a reusable node-local environment
and compiles their Python bytecode. This avoids slow dependency imports from shared
filesystems, including PyTorch's first-optimizer initialization. The first invocation
installs dependencies; later invocations reuse them and sync any lockfile changes.
Each checkout gets its own environment under `SLURM_TMPDIR`, `TMPDIR`, or `/tmp`,
in that order. Set `TINY_LLM_TEST_ROOT` to choose another local directory.

Additional arguments pass through to pytest, for example:

```bash
scripts/test-cpu.sh tests/test_adaptive_consensus.py -k weighted_mixing
```

CPU compilation and figure rendering are optional:

```bash
scripts/test-cpu.sh --run-slow -m slow
```

The multiprocessing cache comparison is skipped before loading its tokenizer
fixture; run it separately when changing preparation or worker code:

```bash
scripts/test-cpu.sh --run-integration -m integration
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
