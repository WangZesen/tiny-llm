#!/usr/bin/env bash
#SBATCH --account=naiss2026-3-205-gpu
#SBATCH --partition=gpu
#SBATCH --gpus 1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --job-name=tiny-llm
#SBATCH --output=./runs/slurm-%j.log

# Submit from the repository after mkdir -p runs; SLURM opens the log first.
# ~/.bashrc selects the architecture-specific UV/Python environment.
source ~/.bashrc
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the tiny-llm repository}"
export TOKENIZERS_PARALLELISM=false
uv sync --locked
# Preserve SLURM's GPU visibility and automatically allocated CPUs.
exec srun --ntasks=1 uv run --no-sync tiny-llm "$@"
