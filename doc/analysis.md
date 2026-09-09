# Gradient-noise and Hessian analysis

For run and submission commands, see [analysis](../README.md#analysis) and
[analysis jobs](../README.md#analysis-jobs) in the README.

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
The entire reference computation preserves double precision. Hessian eigensolvers
are not implemented. Offline gradient-noise measurements are available below.

### Gradient noise and curvature

```bash
uv run tiny-llm analyze --run runs/campaign/20m-lr0.001-wd0.1 \
  --checkpoints all --output runs/campaign/20m-lr0.001-wd0.1/analysis/gradient-noise-hessian
# Checkpoint names are relative to the run directory; absolute root-checkpoint paths also work.
uv run tiny-llm analyze --run runs/20m \
  --checkpoints epoch-001.safetensors epoch-020.safetensors final.pt \
  --noise-samples 32 --random-samples 32 --output runs/20m/analysis
uv run tiny-llm plot-analysis --output runs/20m/analysis
```

Analysis reads `training.micro_batch_size` from `resolved.yaml` for mean and sampled
gradients, preserving the effective optimizer batch and accumulation, including
shortened boundary batches. `--hvp-batch-size N` independently increases curvature
batches without changing the noise samples or full-epoch Hessian. It defaults to
the training microbatch size; neither size is silently reduced on allocation failure. Packed runs use the
averaged model and reconstruct each worker's interleaved local batches. Only root
checkpoints are accepted. Identical parameter states are hashed and measured once;
the manifest and CSV retain each checkpoint filename and training position.

The default analyzes both `seen` (exact replay of epoch 1) and `unseen` (one nominal
epoch immediately after the entire configured training budget). These are virtual
epochs in a single token stream, not repeated dataset sweeps. Unseen blocks lie
outside the physical training prefix and are shuffled independently; extending
the original shuffled loader would change its permutation and is deliberately
avoided. Adjacent blocks share the usual one-token autoregressive lookahead, but
their target positions are disjoint. Insufficient caches produce a preparation
instruction rather than downloading or altering data automatically.

For each case and fixed checkpoint, the token-weighted full-epoch gradient is the
empirical estimate `g_bar` of the true gradient. The Hessian `H` is that same
full-epoch token-mean cross-entropy Hessian. A sampled effective batch supplies
`g_i`, and its gradient noise is `v_i = g_i - g_bar`. Sample batches uniformly
without replacement; defaults are 32 noise samples and 32 independent Rademacher
directions. `--noise-samples all` selects every batch, and small epochs cap the
noise count at the number available. Data and direction seeds are reused across
checkpoints. `D` counts unique trainable parameters, including the tied embedding
only once. No clipping, weight decay, or optimizer transformation enters the loss.

Saved statistics are `sum(v_i^T H v_i)/S_v`,
`sum(v_i^T H v_i)/sum(||v_i||^2)`, `mean(||v_i||)`, `mean(||g_i||)`, `||g_bar||`,
and `sum(s_i^T H s_i)/(D*S_r)`. Both average norms use the same sampled batches.
The norm of the full-epoch mean gradient is a separate quantity. Signed curvature
is retained; zero-noise normalized alignment is saved as JSON null / a CSV blank.

All passes use reference attention (explicit matmul/softmax), which supports
second derivatives; training's fused SDPA settings are not inherited. CUDA analysis
compiles the complete scalar curvature calculation by default, including both
directional derivatives. One graph serves all directions and checkpoints; the last
curvature batch is padded with ignored targets. Mean and sampled gradients remain
eager. `--no-compile-hvp` selects the original double-backward implementation.
The compiled implementation uses Pearlmutter’s `jvp(grad(loss))`. CPU analysis
remains eager.
FP32 CUDA matmuls enable TF32 by default through PyTorch's `fp32_precision` control.
AMP is disabled by default. `--amp` enables BF16 autocast in model forward calls,
while keeping parameters and gradient accumulation in FP32. Reference attention
scores, normalization reductions, and cross-entropy stay in FP32. Backward runs
outside autocast, and the compiled graph includes the explicitly traced casts.
AMP requires `--dtype float32`; precision modes have distinct saved identities.
Use `--no-tf32` for IEEE FP32 comparisons, or
`--dtype float64` for derivative checks. CPU runs (`--device cpu`) and FP64 do not
use TF32. Scalar dot products and statistical reductions use float64. The saved
manifest records actual precision, backend, software, hardware, data provenance,
and batch sizes. Runtime settings are restored after analysis.

Each noise/random direction requires a **full-epoch HVP pass**: defaults require
64 passes per case, plus the mean gradient and sampled gradients. Microbatch
graphs are released promptly; the Hessian and the collection of parameter-sized
directions are never materialized. Completed checkpoint/case results are written
atomically and reused on an identical rerun. Changes to source, settings, or
execution identity require a new output directory. Interrupted cases restart;
completed cases are retained. A lock prevents simultaneous writers.

Outputs include per-sample scalar JSON, `manifest.json`, `summary.csv`, and PDF/PNG
figures for normalized alignment, unnormalized noise alignment, and all three
gradient/noise norms. Seen and unseen share each plot's axes (solid/dashed lines),
with training tokens on the x-axis. `plot-analysis --training-norms` optionally
adds pre-clipping training log norms; packed logs are labeled as the maximum
worker norm, not the gradient at averaged weights.

## Execution and provenance

Verified identical parameter states stay together, with every filename retained
as an alias. Unique states are divided round-robin among `--jobs N` independent
single-GPU jobs. Both data cases for a checkpoint run in the same job. Each shard
writes atomic statistics with plotting disabled. The login-side process waits
for all jobs, validates their results, and creates the combined manifest, CSV,
and three PDF/PNG figures.

The built-in timing profile covers the measured ordinary 20M architecture,
context length 1024, training batch/microbatch 32, HVP batch 64, and the default
BF16/TF32 compilation settings. It requests a GH200 explicitly. Conservative
per-stage timings scale with the actual full-epoch microbatch counts and sample
counts. Each allocation is twice estimated work plus 30 minutes, rounded up to
an hour: ten checkpoints with the default samples receive **25 hours**. Other
architectures, packed runs, or numerical/batch settings require an explicit
`--walltime-hours N`; all requests must fit the partition limit, at most 72 hours.
The account is `naiss2026-3-205-gpu` and the partition is `gpu`.

Compiler caches are created on the compute node under `SLURM_TMPDIR`, `TMPDIR`,
or `/tmp` in that order. Each analysis job owns a unique disposable directory;
normal exits, failures, and catchable termination clean it up. SIGKILL/node failure
may leave temporary files for the node's usual cleanup. Analysis results and
source snapshots remain under the requested output directory.

`OUTPUT/submission.json` records assignments, timing assumptions, resources,
source snapshot, job IDs, attempts, and log paths. The watcher polls `squeue` and
`sacct` every 60 seconds; an absent accounting record remains unknown. Interrupting
the watcher leaves submitted jobs running. Repeat the same arguments with
`--resume` to reattach; terminal failed jobs are retried from completed compatible
checkpoint/case results. Other jobs finish even if a shard fails. A submission
interrupted before its job ID was saved must be reconciled with Slurm before retrying.

Existing campaigns continue with their original frozen source and watcher.
Pre-cleanup receipts must be resumed with their saved source entrypoint and
recorded arguments; they are not migrated or rewritten. The standalone
`tiny-llm plot-analysis --output OUTPUT` command regenerates saved figures.

See [GH200 analysis measurements](analysis_performance.md) for the retained
BF16 accuracy comparisons, throughput measurements, and timing assumptions.
