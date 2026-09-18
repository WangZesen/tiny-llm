---
title: From data to measurements
description: Prepare a cache, train locally or through Slurm, then inspect the model's Hessian and gradient noise.
---

Run commands from the repository root. Training uses Python 3.12+, UV, and an
NVIDIA GPU with BF16 support; the locked PyTorch build uses CUDA 13.

## 1. Prepare data

[Prepare the C4 cache](data.md) once with capacity for the intended model and
training horizon. Add enough data for unseen analysis when needed.

## 2. Train locally

[Launch synchronous or packed training](training.md), select cosine-to-zero or
WSD, and set the published hyperparameters explicitly. Give each recipe and
seed a separate output directory. Retain checkpoints if you plan to analyze them.

## 3. Submit Slurm jobs

[Use the Slurm launcher](slurm.md) to submit preparation or training, override
resources, and monitor jobs. The same guide covers distributed checkpoint-analysis jobs.

## 4. Analyze Hessian and gradient noise

[Run checkpoint analysis](analysis.md) on seen and unseen data, choose numerical
settings, and inspect the resulting alignment and norm figures.

The published cosine sweeps saved no checkpoints. Their retained logs support
result and performance analysis; Hessian analysis needs weights from a new run
configured to save them.
