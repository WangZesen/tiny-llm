# GH200 training performance

Measured on Arrhenius with one NVIDIA GH200, PyTorch 2.14.0 / CUDA 13.0,
cuDNN 9.24, driver 580.159.04, and ARM Python 3.12.13. The allocation used
account `naiss2026-3-205-gpu`, partition `gpu`, and `--gpus 1`. SLURM provided
72 local CPU cores; the recommended configuration uses eight PyTorch threads.
The model remains 20,403,520 parameters, BF16 AMP with FP32 weights, 1024-token
context, and 32,768 targets per AdamW update.

## Reproduced cause and measured changes

The saved RTX result used compilation and microbatch 16, whereas the reported
GH200 run used eager execution and microbatch 8. These settings account for a
large performance difference on GH200 itself. The historical RTX synthetic
result is 352,068 targets/s; the user-reported RTX training result is approximately
343,000 targets/s. Neither is a new RTX measurement in this investigation.

All rates below use real C4 loading, CPU conversion, pinned transfers, forward
and backward computation, gradient clipping, finite checks, and fused AdamW.
Each trial warms up for 20 updates and measures three 100-update windows; the
reported rate is their median. Profiling runs after timing. Windows use full
optimizer batches without validation/checkpoints; the matched training runs
also exercise shortened updates at virtual-epoch boundaries.

| Execution | Microbatch | Targets/s | Peak allocated GiB |
|---|---:|---:|---:|
| Original computation, eager | 8 | 245,035 | 4.53 |
| Original computation, compiled | 8 | 713,010 | 1.70 |
| Original computation, compiled | 16 | 1,194,163 | 2.94 |
| Compiled, uncached RoPE | 32 | 1,267,517 | 5.38 |
| Final implementation, compiled with cached RoPE | 32 | 1,636,281 | 5.39 |

Compilation alone gives approximately 2.9× at microbatch 8. Increasing the
microbatch reduces accumulation and launch overhead while preserving the
optimizer batch. The final measured update rate is 6.68× the reproduced eager
baseline. Peak allocated memory is the PyTorch allocator measurement, not all
memory reserved by CUDA libraries.

The traces contain approximately 5,693 GPU kernels/update for eager microbatch
8, 2,329 for compiled microbatch 8, and 1,139 for compiled microbatch 16.
Compilation reduces launch count and intermediate tensor traffic. The profiled
cross-entropy kernels take about 25.5 ms across three eager updates versus
8.4 ms across three compiled updates at microbatch 8.

RoPE was recomputing angles, sine, and cosine for every query/key transformation
in every layer. Even compiled execution spent about 18.3 ms across three
microbatch-32 updates in fused kernels containing this trigonometry. Caching the
FP32 tables outside compiled forward graphs reduced summed kernel time from
68.5 to 50.9 ms across three profiled updates. Profiler instrumentation affects
wall time; throughput comes from the separate unprofiled windows.

An uncached/cached/cached/uncached comparison measured respectively 1.268,
1.628, 1.621, and 1.270 million targets/s: approximately 28% improvement from
the cache. Both ablation arms used the same experimental synchronization
changes, which were removed for the final measurement. Buffers are nonpersistent
and rebuilt on device/dtype changes. The
reference and FP64 paths retain the original angle computation, and canonical
checkpoint keys and trainable parameters are unchanged.

## Matched training and validation

All runs train on 100,000,768 targets (100 million rounded up to complete
1024-token blocks), with identical seed, sample order, effective batch,
optimizer/schedule, and ten virtual-epoch validation boundaries. Configuration
comparison verifies that only execution settings differ. These are shortened
training budgets, not full-budget convergence experiments.

| Measurement | Original eager code | Final optimized code |
|---|---:|---:|
| Median logged training targets/s | 254,483 | 1,599,463 |
| Training phase including epoch validation/checkpoints | 423.84 s¹ | 91.10 s |
| Full validation over 197,411,295 targets | 326.44 s | 278.83 s |
| Session time including full validation | 750.27 s | 369.98 s |
| Full validation loss | 3.978252 | 3.977021 |

¹ The original code does not record this phase directly; this is session time
minus full-validation time and includes the initial subset-preparation scan.
Both session timers exclude earlier model initialization and cache verification.
The final run records 65.22 seconds in training-update windows, including its
initial compilation, for an aggregate 1.533 million training targets/s.

The approximately 6.29× steady training speedup becomes approximately 2.03×
for the complete shortened session because full validation remains a large cost.
Evaluation uses the eager model. The loss difference is −0.001231 nats; one seed
and one shortened budget do not establish a quality improvement or full
convergence equivalence.

The first optimized run exposed additional compilation at short epoch-ending
batches. A diagnostic probe confirmed size guards failing at batches 11 and 25.
Production now compiles full configured microbatches and runs partial microbatches
eagerly. This preserves their token weighting and update boundaries. In repeated
100-million-target training, this change reduced training-update time from
103.00 to 65.22 seconds and the training phase from 129.46 to 91.10 seconds.
Steady throughput remained approximately 1.60 million targets/s. Both variants
completed full validation; the earlier compiled-partial variant had loss 3.978585.

## Decisions from tuning

- Automatic, forced FlashAttention, and forced cuDNN attention differed by less
  than 1% at the best batch sizes. Automatic dispatch remains recommended.
- `reduce-overhead` measured 1.224 million targets/s, slightly below default
  compilation before the RoPE change.
- `max-autotune` and thread-count changes reached 1.279 million targets/s, only
  about 1% above default compilation, while the first autotuning trial spent
  130 seconds warming up. These changes did not meet the 5% retention threshold.
- Combining finite checks and moving logging-loss accumulation onto the GPU
  improved the controlled update benchmark by approximately 1.2%. That experiment
  was reverted; production keeps the original pre-update failure checks.
- Data preparation/transfers were a small part of the traces. No reusable pinned
  buffer implementation or additional kernel dependency was introduced.

The staged tuner's `selected.yaml` records the numerically fastest measured
candidate, including marginal gains. The default recipes in `configs/20m.yaml`,
`configs/50m.yaml`, and `configs/90m.yaml` use default compilation, microbatch 32,
automatic SDPA, and eight CPU threads, avoiding those marginal changes and their
additional startup cost.

## Larger models, startup, and verification

The same real-data protocol found no regressions with the original eager,
microbatch-8 settings. The compiled columns use microbatch 32 and the final
implementation; these execution settings are now the defaults for all three
model sizes.

| Preset | Original eager targets/s | Final eager targets/s | Final compiled targets/s | Compiled peak GiB |
|---|---:|---:|---:|---:|
| 50M | 211,900 | 219,431 | 871,826 | 8.78 |
| 90M | 146,443 | 159,619 | 508,930 | 13.97 |

With fresh Inductor and Triton cache directories, final 20M throughput was
1,635,536 targets/s. The first update took 17.04 seconds and all 20 warmup updates
took 17.45 seconds. Conservatively charging all 17 seconds as additional startup
and ignoring eager startup gives a training-compute break-even of approximately
5 million targets against the measured 245k baseline. This is an estimate from
these measurements, not a guarantee across environments. The GPU-resident
synthetic benchmark measured 1,610,008 targets/s; its random token distribution
differs from C4, so the rate difference does not isolate loader cost.

The full suite passed on GH200: **45 passed**. The CPU-only run had **36 passed,
9 CUDA-only tests skipped**. Checks cover BF16 backend/compile-mode parity,
FP64 gradcheck/gradgradcheck and HVPs, nonfinite update rejection, partial-batch
dispatch, compiled uneven-epoch resume, cache dtype/device transitions, canonical
state dicts, timing exclusions, benchmark identity, and worker timeout cleanup.
Ruff lint/format and shell syntax checks also pass.

On actual C4 batches at context 1024, the trained final 20M weights loaded strictly
into the reference model. Compiled/reference loss was 4.318610 / 4.319227 and
aggregate relative gradient error was 1.45%, within the existing 5% tolerance.
Fresh 50M and 90M models also passed, with gradient errors of 0.44% and 0.54%.
Their checks establish execution parity, not convergence results.

The dedicated allocation was released after **3,992 seconds (1.11 GPU hours)**,
within the two-hour limit. All benchmark and training processes ran sequentially
on its single GPU.

## Reproduction and artifacts

From the repository root:

```bash
mkdir -p runs
sbatch scripts/slurm.sh train --config configs/20m.yaml
sbatch scripts/slurm.sh benchmark --gh200 --config configs/20m.yaml \
  --budget-minutes 75 --output runs/gh200-retune
sbatch scripts/slurm.sh benchmark-worker --config configs/20m.yaml \
  --data-mode real --profile --output runs/gh200-profile.json
```

Raw measurements, traces, telemetry, source snapshots used for ablations, and
training outputs are under `runs/gh200/`. Compact exact results are also included
as [gh200_results.json](gh200_results.json). `initial-*.json` records the initial
comparison; `tuning/summary.json` contains the execution grid; `nocache-*.json`
and `rope-*.json` contain the cache ablation; `final-20m*.json` records final
full-batch measurements. `matched-optimized-partial/` contains the final training
run with eager partial batches; `matched-optimized/` is the preceding ablation. Chrome traces have the corresponding `.trace.json` suffix.
`telemetry.csv` samples clocks, power, memory, and utilization once per second.

Training now excludes the initial validation-subset scan and periodic checkpoint
writes from the training-rate windows. Elapsed throughput is reported separately
and includes epoch evaluation and checkpoint writes. The first training window
still includes compilation; final full-validation time is separately available
in the result. Benchmark reuse is keyed by source contents, configuration,
measurement protocol, cache identity for real data, and hardware/software metadata.
