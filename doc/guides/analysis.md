---
title: "Running checkpoint analysis"
---

## Analysis

For epoch-by-epoch analysis, train with `--set training.checkpoint_policy=interval`.
Analyze all checkpoints or select filenames relative to the run directory.
Use a prepared cache with enough data for both seen and unseen cases; the tool
checks capacity before computing. Packed runs use root, averaged checkpoints.

Packed runs also analyze every worker's consensus error `x_i - x_bar`, using
the Hessian at the averaged model. Worker-level scalars are saved, and aggregate
alignment and norm curves join the existing figures. Epoch exports require all
matching `node-NNN/` weight files; packed `.pt` states already contain them.
Use `--no-consensus` with `analyze` or `submit-analysis` to disable this diagnostic.

```bash
uv run tiny-llm analyze --run runs/20m --checkpoints all \
  --output runs/20m/analysis
# Analyze a selection with the measured BF16/HVP batch settings:
uv run tiny-llm analyze --run runs/20m \
  --checkpoints epoch-001.safetensors epoch-020.safetensors final.pt \
  --amp --hvp-batch-size 64 --output runs/20m/analysis-bf16
```

Defaults are both data cases, 32 noise samples, 32 random directions, seed 42,
FP32 without AMP, TF32 enabled on CUDA, and compiled Pearlmutter HVPs with
reference attention. Gradient microbatch size always comes from `resolved.yaml`;
HVP batch size defaults to that size. Useful options:

- `--data-case seen|unseen|both` selects data cases.
- `--noise-samples N|all` and `--random-samples N` control sampling.
- `--amp --hvp-batch-size 64` enables BF16 forward AMP with FP32 parameters.
- `--device cpu --dtype float64` runs eager derivative checks on CPU.
- `--no-tf32`, `--no-compile-hvp`, and `--no-plots` disable those features.

Results include per-sample JSON, a manifest, `summary.csv`, and three PDF/PNG
figures: normalized alignment, unnormalized noise alignment, and gradient/noise
norms. Seen and unseen curves share axes against training tokens. Repeat the
same command to reuse completed compatible results; use a new output directory
when changing source or numerical settings. Regenerate figures with:

```bash
uv run tiny-llm plot-analysis --output runs/20m/analysis
```

Add `--training-norms` to include logged training norms. See the
[analysis reference](../methods/analysis.md) for formulas, data selection, and precision,
and [measurements](../archive/analysis_performance.md) for BF16 accuracy and runtime.
