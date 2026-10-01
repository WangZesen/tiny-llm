---
title: Small-Batch Training with AdamW
description: Batch size, Adam moment memory, and gradient clipping in synchronous and four-worker decentralized language-model training.
---

Smaller batches can improve validation loss at a fixed training-target budget, but
the smallest tested batch is not consistently best. The strongest observed
synchronous and clipped decentralized recipes use global batches **64 and 32**,
respectively. This investigation combines completed tuning and clipping experiments
through 1 October 2026.

[13-slide presentation](../small-batchsize-investigation/slides.pdf) ·
[Report and technical appendix — 28 pages](../small-batchsize-investigation/report.pdf)

[![Selected synchronous and clipped four-worker validation loss versus global batch; error bars show sample SD over seeds 42–44](../small-batchsize-investigation/assets/quality.svg)](../small-batchsize-investigation/assets/quality.pdf)

The figure compares separately selected recipes. Lower full-validation loss is
better; error bars are sample standard deviations, not confidence intervals.

## Findings

- **Batch size and Adam memory should be tuned together.** Smaller batches favor
  β₁ closer to one. Near-one β₂ values help in the decentralized small-batch search,
  but the preferred β₂ is not strictly monotonic in batch size.
- **The tuned quality gap is small.** Clipped four-worker training is
  0.004224–0.008296 nats per target above synchronous training at matching batches,
  equivalent to approximately 0.42–0.83% higher perplexity. This does not establish
  statistical equivalence.
- **Clipping frequency and usefulness differ.** The selected decentralized
  batch-16 recipe clips about 84.3% of worker updates. Nevertheless, clipping
  benefits more configurations at larger batches in the matched seed-42 grid.
  Its effect depends on learning rate; disabling clipping favors lower tuned LRs
  at batches 32–128.

## Experimental scope

The model has 20,403,520 parameters and context 512. Runs use the same C4 cache,
408,092,672 global training targets, and 197,411,295 final validation targets.
Four logical decentralized workers share one GH200, retain independent Adam
moments, and use reciprocal one-peer exponential parameter mixing. The experiment
does not measure four-GPU scaling.

The clipped decentralized study contains 840 screening configurations and 40
replicated configurations; the no-clipping study contains 480 screening
configurations and 20 replicated configurations. Replication uses seeds 43–44
after seed-42 screening. These seeds contribute to final tuning and are not fresh
confirmation. Synchronous tuning follows a different conditional procedure, so
cross-method comparisons remain exploratory.

The report includes all 64 final replicated configurations across the three
methods, individual seed losses, matched clipping comparisons, moment half-lives,
throughput and memory definitions, and provenance. The
[document package and build instructions](../small-batchsize-investigation/README.md)
provide editable Typst sources and bundled data for reproducing both PDFs without
running training.
