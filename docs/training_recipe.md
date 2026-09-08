# Training recipe and evidence

This repository targets validation loss for 20M–92M Llama-style models on English
C4, trained from scratch for 20 prediction targets per **unique trainable
parameter**. The default presets adopt the best recipes from a bounded,
single-seed search; see [campaign results](campaign_results.md). These selections
do not establish optimality outside the tested settings.

## Published precedents

| Source | Experiment and reported settings | How we use it |
|---|---|---|
| [GaLore (ICML 2024), Appendix C.1](https://arxiv.org/html/2403.03507v2#A3.SS1) | Llama-60M on C4 for 1.3B tokens; context 256; 131K-token batches; 10% warmup; cosine decay to 10% of peak LR. Full-rank baselines generally use LR ≤ 0.001. | Closest model/data/budget precedent. Supports the cosine endpoint and searching around 0.001. GaLore's 0.01 learning rate includes projection scaling and is not an AdamW default. |
| [Pythia (ICML 2023)](https://arxiv.org/abs/2304.01373), [official 70M configuration](https://github.com/EleutherAI/pythia/blob/main/models/70M/pythia-70m.yml) | LR 0.001; betas (0.9, 0.95); epsilon 1e-8; weight decay 0.1; clipping 1.0; zero dropout. A model suite intended for analysis across training. | Optimizer starting values and checkpoint-oriented design. Pythia uses the Pile, different architecture/tokenization, much larger batches, and a much longer training budget. |
| [Small Batch Size Training for Language Models (2025), §4.3 and Appendix A](https://arxiv.org/html/2507.07101v2) | Includes a C4 transformer with 19M non-embedding / 52M total trainable parameters. Scaling the second-moment timescale improves small-batch Adam. Pretraining uses 5% warmup. | Use 5% warmup and test beta2=0.99 against 0.95 at 50M. The 0.99 candidate is our approximation for a longer timescale, not a copied optimum. |
| [Is your batch size the problem? (2025), §2 and Appendix A](https://arxiv.org/html/2506.12543v1) | Primarily 160M models with RMSNorm, RoPE and GLU; zero decay; beta2=0.95; Adam grids include 0.0003, 0.001, 0.003. | Include zero decay and the higher LR candidate. This is adjacent evidence above the target model sizes. |

The [original Llama paper](https://arxiv.org/abs/2302.13971) motivates pre-RMSNorm,
RoPE, SwiGLU and bias-free projections. Embedding tying, exact small-model
dimensions, context 1024, initialization std 0.02 with residual projection
scaling, and a 32,768-target effective batch are engineering choices. They are
not directly established as optimal by these papers.

## Selected recipe

- BF16 autocast for CUDA operations; FP32 parameters, gradients, optimizer moments,
  normalization reductions, and cross-entropy. No FP16 gradient scaler.
- AdamW: LR 0.001, beta1 0.9, beta2 0.95 for 20M and 0.99 for 50M/90M,
  epsilon 1e-8, weight decay 0.1.
  Matrix weights, including tied embeddings, receive decay; norm scales do not.
- Global raw-gradient clipping at norm 1.0, after accumulation and before AdamW.
- Linear token-based warmup for 5% of training, then cosine decay to 10% of peak.
- 32,768 prediction targets per update. Smaller accumulation groups at virtual
  epoch boundaries use their actual target count as the gradient denominator.
- Zero dropout; seed 42; `runtime.deterministic: false`.

The schedule is evaluated at the end-token position of each optimizer update.
Virtual epoch boundaries do not reset the schedule, optimizer, or data order.
The token budget counts tied embeddings once; the report also records the
non-embedding count to avoid confusing different papers' size conventions.

Training order uses loader v1: sequence-aligned 64 MiB ranges shuffled by seed,
then independently shuffled sequence indices within each resident range. One
prefetched range overlaps I/O with model computation. The training prefix and
virtual epoch boundaries are unchanged; the sequence order differs from the
previous global permutation. The audited campaign now in `runs/campaign` used
loader v1 for all twelve runs.

## Twelve complete runs

1. **20M, six runs:** LR {0.0003, 0.001, 0.003} × decay {0, 0.1}; beta2=0.95.
2. **50M, four runs:** best two LR/decay pairs × beta2 {0.95, 0.99}.
3. **90M, two runs:** best two complete recipes from the 50M stage.

Every completed run trains for 20 targets per parameter and evaluates the full
official validation split at the end. Promote by final-checkpoint full-validation
loss, breaking exact ties by lower LR, then decay, then beta2. Intermediate
subset-validation best checkpoints are recorded separately. Architecture, data,
seed, effective batch, schedule, clipping, and beta1 remain fixed in the sweep.

This is a single-seed comparison. It does not measure seed robustness, establish
batch-size optimality, or provide an unbiased test-set performance estimate.
Validation data is explicitly used for recipe selection.

## Precision, reproducibility, and performance

The 20M, 50M, and 90M recipes default to microbatch 32, `runtime.compile=true`,
`runtime.compile_mode=default`, automatic SDPA, and eight CPU threads, as measured
on GH200. The effective batch remains 32,768 targets per optimizer update.

Validation defaults to 128 sequences per forward pass for all three sizes, without
gradient accumulation. Both subset and full validation sum losses over valid tokens
and divide by their total count, including partial final batches and excluding
padding. Changing the batch size preserves this weighting but can introduce small
floating-point differences. This larger validation batch is a subsequent default
change; the recorded GH200 performance measurements used evaluation batch size 8.

`deterministic=false` retains seeded initialization, seeded block ordering, and
saved RNG states while allowing fast kernels. It does not promise bitwise replay.
`deterministic=true` enables strict PyTorch deterministic algorithms, configures
cuBLAS before CUDA initialization, disables cuDNN benchmarking and compilation,
uses reference attention, and disables fused/foreach AdamW. Unsupported operations
raise errors. Cross-hardware or cross-version bitwise equality is not guaranteed.

The benchmark runs each candidate in a separate process, using full optimizer
updates at the same effective batch size. It measures compiled/uncompiled paths
and microbatches 4, 8, 16, and 32, rejecting candidates above 90% GPU memory usage.
Candidates warm up for 20 updates and measure three 100-update windows. The
selected configuration is saved rather than recomputed during training. Real-C4
benchmarks and a bounded GH200 tuner are also available; see
[GH200 measurements](gh200_performance.md).
Synthetic throughput estimates exclude data loading, validation, compilation,
and checkpoint overhead; training JSONL records observed throughput.

The fast path uses PyTorch SDPA with automatic kernel selection (including
FlashAttention and cuDNN), or an explicitly requested backend. The
reference path uses explicit matmul, softmax, and masking and retains FP64 for
second-order analysis. Copy weights with strict `load_state_dict`; disable AMP
and compilation for derivative calculations. These tests validate derivatives
of model losses, not differentiation through the optimizer's training history.

## Packed decentralized simulation

The opt-in decentralized path keeps the global token budget based on one local
model's parameter count. It executes N workers together and assigns every Nth
sample from the buffered loader's shuffled stream to each worker. This preserves
disjoint worker partitions across buffer and epoch boundaries, without separate
per-worker file reads. The same seed initializes every worker exactly
as the ordinary Llama path, without consuming additional initialization RNG.

There is no gradient accumulation: `batch_tokens = N * micro_batch_size *
context_length`. For worker i, the gradient is the derivative of its local
microbatch's mean valid-token loss. Backpropagating the sum of worker means
preserves this normalization. Each step clips gradients independently, mixes
parameters, and applies local AdamW updates using gradients evaluated at the
pre-mixing parameters. Weight decay acts on mixed weights; first/second moments
and counters stay local. Shortened epoch-ending steps use equal smaller local
batches and their actual token counts. Incompatible epoch boundaries are rejected.

Parameters, first moments, and second moments each occupy an aligned contiguous
`[N, D_padded]` arena. Local AdamW tensors alias these arenas, and checkpoints
serialize the three blocks once plus counters and layout metadata. Alignment
padding is excluded from parameters and optimization. The moment arenas are
exposed for future experiments but are not mixed by the trainer.
Packed buffered checkpoints use format v3 and preserve loader v1's committed
cursor; ordinary buffered checkpoints retain upstream format v2. Prefetch may
change on resume, but buffer size may not. Earlier packed v2 checkpoints remain
evaluable but cannot resume under the new sample ordering.

Every training step performs one topology event: complete averaging, alternating
left/right one-peer ring averaging, or one-peer exponential averaging with
offsets 1, 2, 4, ... below N. Peers are incoming, so row i receives row
`(i - offset) % N`. One-peer weights are half self and half peer. For exponential
graphs, powers of two admit exact cycle averaging; arbitrary N remains supported
without that guarantee. See the
[exponential-graph paper](https://proceedings.neurips.cc/paper/2021/file/74e1ed8b55ea44fd7dbb685c412568a4-Paper.pdf).

Evaluation first snapshots the global parameter mean into an ordinary Llama.
The snapshot uses FP32 averaging and the global evaluation batch size. It never
changes local training states. Selection and reported validation losses concern
this averaged model, whose ordinary-format weights are exported each epoch.

See the [SLURM launcher setup](../README.md#slurm-jobs) for submission on the
aarch64 GPU nodes, automatic CPU allocation, and console logs. `benchmark-packed`
compares N=4/8 packed and sequential workers with matched global batches,
reports end-to-end and separate component timings, and saves profiler traces.
