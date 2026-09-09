# GH200 checkpoint-analysis performance

Measured on `20m-lr0.001-wd0.1`, epoch 1, FP32 with TF32 and reference attention,
context length 1024. Gradient microbatches remain 32. The checkpoint has 20,403,520
unique parameters. Benchmark Slurm job: `2190245` (two-hour allocation).

Each candidate uses the same checkpoint and Rademacher direction. Its reference
is the original double-backward calculation accumulated in training microbatches.
Timing is the average of five warm calls, excluding tracing and compilation.
Memory is peak allocated GPU memory across startup and timed calls, in decimal GB.

| Method | HVP batch | Seconds/batch | Tokens/second | Peak GB |
|---|---:|---:|---:|---:|
| eager | 32 | 0.3661 | 89,510 | 61.7 |
| forward-forward | 32 | 0.1747 | 187,615 | 21.3 |
| forward-forward | 64 | 0.3417 | 191,791 | 42.3 |
| forward-forward | 128 | 0.6762 | 193,847 | 84.3 |
| forward-reverse | 32 | 0.1654 | 198,131 | 35.8 |
| forward-reverse | 64 | 0.3233 | 202,737 | 71.2 |

The selected setting is compiled **forward-over-reverse at batch 64**, giving
**2.26×** the eager throughput. This is Pearlmutter's HVP formulation:
`jvp(grad(loss), (parameters,), (direction,))[1]`, followed by the scalar dot
product. Both derivatives and the reduction are compiled in one graph. The
forward-over-forward alternative directly computes the second directional
derivative of the loss. No production method constructs a dense Hessian.

Tracing uses fake tensors to avoid allocating the unfused derivative intermediates.
The traced graph materializes broadcast zero tangents before generated copy
operations, then compiles the ordinary ATen operations with Inductor. Parameters,
directions, and input batches remain graph inputs. The final batch is padded with
ignored targets; its contribution is weighted by the actual full-epoch token count.

The scalar method also supports batch 128, but gives little extra throughput over
64. Batch 256 failed compilation. Selection excludes failed candidates and those
exceeding 90% of GPU memory. The selected candidate's scalar differed from its
TF32 eager reference by 0.0020%.

These are historical tuning measurements; the one-off tuning scripts and the
alternative forward-forward implementation have been removed. The retained
implementation uses `--compile-hvp --hvp-batch-size 64` for this run.
These measurements are specific to this model, context length, and GPU.

## Paired BF16 AMP check

The separate experiment keeps FP32 weights and gradient accumulation, and enables
BF16 autocast only inside model forward calls. Reference attention still evaluates
scores and softmax in FP32, and cross-entropy and scalar reductions retain the
analysis policy. Autocast weight caching is disabled for functional derivatives.
Direct analysis still allows FP32 execution; the submitter uses the validated BF16 settings.

The one-off comparison and summary scripts have been removed after validation.
Their results remain under
`runs/campaign/20m-lr0.001-wd0.1/analysis/amp-comparison`, including the saved
experiment source and scalar/vector diagnostics.

Each checkpoint/case computes full-epoch means independently in both precisions,
then uses matched batch selections and random directions. In addition to the six
analysis statistics, it measures gradient/noise vector errors and applies the AMP
Hessian to FP32 noise vectors to isolate curvature error from direction error.
JSON results include per-direction timings and peak memory. The summary writes
CSV comparisons and relative-error, sign, cosine, and performance diagnostics.

## Full-epoch FP32+TF32 pilot

Job `2190308` completed with a two-hour allocation. All 130 repository tests
passed on its GH200 before timing. Each data case contains 10,202,112 tokens.

| Case | Mean gradient (s) | Noise HVP (s) | Random HVP (s) |
|---|---:|---:|---:|
| seen | 32.15 | 50.76 | 50.72 |
| unseen | 32.16 | 50.93 | 50.78 |

All three norm statistics matched the original eager pilot exactly. Curvature
statistics differed by less than 0.0025%.

## BF16 AMP comparison results

Job `2190365` completed successfully in 35 minutes 27 seconds. Coverage was
epochs 1 and 40, seen and unseen data, with two matched noise samples and two
matched random directions per case. Every gradient mean and HVP used the full
10,202,112-token epoch. The user-selected alignment tolerance was 5%.

| Statistic | Largest relative difference across four cases |
|---|---:|
| noise alignment | 0.638% |
| normalized noise alignment | 0.675% |
| average noise norm | 0.050% |
| average batch gradient norm | 0.216% |
| mean gradient norm | 0.955% |
| normalized random alignment | 0.039% |

The largest individual quadratic-form difference, including the fixed-FP32-noise
checks, was 1.044%; no tested curvature sign changed.
The largest noise-vector relative L2 error was 1.73%, with minimum
cosine 0.999851. Mean-gradient vector relative L2 error reached
4.86%, although its norm differed by less than 1%.

Median full-epoch HVP time was 32.73 seconds with AMP versus
50.99 seconds with FP32+TF32: **1.56× faster**. Peak HVP memory was
51.67 GB versus 71.41 GB. The requested scalar
statistics pass the 5% criterion. Production uses `--amp --compile-hvp --hvp-batch-size 64`, with training microbatches 32.
The integrated CLI was validated and timed separately before the recorded production submission.

## Independent single-GPU campaign

The checkpoint-sharded launcher is `scripts/submit_analysis.py --jobs N`. Each
allocation computes its own checkpoint groups with `--no-plots`; a detached
login-node process waits for Slurm success, validates coverage and numerical
identities, and creates the combined tables and figures. Identical parameter
states are grouped before round-robin assignment, preserving filename aliases.

The corrected independent pilots **2197314** and **2197315** completed in
10m03s and 10m02s, respectively, each with a two-hour allocation. Each GPU passed
all **151 tests** before analyzing epoch 1 or epoch 40 with both data cases,
one noise sample, and one random direction. The same source snapshot passed
135 CPU tests with 16 CUDA skips. All 26 comparisons with saved serial BF16
references passed; the maximum relative difference was **4.76e-11**. The first
pilot attempt exposed a snapshot-relative launcher-path bug, now fixed and
covered by regression tests; its failed jobs and outputs remain archived.

Compilation took 53.3–55.1 seconds. Seen-data mean-gradient passes took
28.7–28.9 seconds and full-epoch HVPs took 32.0–32.7 seconds. The production
planner uses the maximum observed stage timings across both checkpoints and
both data cases, with 32 noise and 32 random samples per case. It estimates
**11.8074 hours per shard**, including startup, or **47.2298 total GPU-hours**.
Twice that per-shard estimate plus 30 minutes, rounded up, gives **25 hours**.

Production jobs **2198414, 2198415, 2198416, and 2198417** were submitted on
2026-09-09, each with one GPU and 25 hours, account `naiss2026-3-205-gpu`,
partition `gpu`, and no dependencies. Each receives ten unique checkpoint
states; epoch 40, `final.pt`, and `latest.pt` remain aliases in the fourth shard.
All 42 checkpoint filenames are retained. Numerical settings remain BF16
forward AMP, FP32 parameters and gradient accumulation, TF32, reference attention,
compiled forward-over-reverse HVPs, gradient microbatch 32, and HVP batch 64.
Submission and scheduler resource requests were verified. The initial scheduler
state was pending; subsequent progress is recorded in the orchestration receipt.

Receipts and logs live below
`runs/campaign/20m-lr0.001-wd0.1/analysis/`:

- `independent-execution.json`: pilot, archive, validation, and watcher records.
- `independent-pilot-validation.json`: serial-reference comparisons.
- `pilot-independent/`: completed pilot statistics, CSV, PDF/PNG figures, and GPU test logs.
- `gradient-noise-hessian/submission.json`: production assignments, attempts, settings, and estimates.
- `production-submission-verification.json`: scheduler resource verification.
- `production-independent-orchestrator.log`: the persistent login-side watcher.

The former sequential chain 2195785 → 2195786 was canceled and archived under
`gradient-noise-hessian-single-gpu-cancelled-2195785`. No legacy statistics were
migrated into the replacement campaign.

## Direct submission timing profile

New submissions require no pilot files or performance-measurement stage. For the
measured ordinary 20M/GH200 configuration the submitter uses rounded-up constants:

| Stage | Baseline seconds |
|---|---:|
| Compilation/startup per job | 60 |
| Full-epoch mean gradient | 30 |
| Full-epoch HVP per direction | 33 |
| Sampled effective-batch gradient | 0.15 |

The reference epoch has 9,963 blocks: 312 gradient microbatches of at most 32 and
156 HVP batches of at most 64. Full-pass costs scale by actual microbatch counts;
noise counts are capped at available effective batches. Both data cases with
32 + 32 samples cost an estimated 4,293.6 seconds per checkpoint. Ten checkpoints
plus startup estimate 11.9433 hours and request **25 hours** after the required
safety margin and rounding. These conservative constants supersede the earlier
pilot-derived 11.8074-hour estimate for new submissions. Unmeasured recipes require
an explicit wall time rather than extrapolation across model sizes or precision.

The package is now `tiny_llm.analysis`, with core, Slurm, and plotting modules.
Saved source fingerprints include subpackages recursively. Existing production
snapshots, receipts, active compiler caches, and watchers are preserved; new jobs
use disposable node-local compiler caches.

## Cleanup validation

The analysis implementation now resides in the `tiny_llm.analysis` subpackage.
The test suite was reduced from 151 to 126 collected cases, removing obsolete
experiment coverage and repeated compiler/packed-worker combinations. Scalar
fixtures replace repeated analysis in scheduler failure tests; data-preparation
tests stub their upstream modules before importing them, and plotting is tested
once in the end-to-end collection check.

The observed CPU run passed 115 tests with 11 CUDA skips in 15.03 seconds,
compared with 139.61 seconds for the pre-cleanup baseline on the same login host.
These are observed suite durations, including dependency/cache effects.
CUDA validation job **2205721** passed the remaining 11 tests in 125.93 seconds
on a GH200. Package-wheel contents, recursive source snapshots, disposable-cache
cleanup, and pilot-free submission were also checked. The four production jobs
and their existing snapshots and watcher were preserved.
