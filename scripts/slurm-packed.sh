#!/usr/bin/env bash
#SBATCH --account=naiss2026-3-205-gpu
#SBATCH --partition=gpu
#SBATCH --gpus 1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=tiny-llm-packed

# Run from the repository: sbatch scripts/slurm-packed.sh train --config ...
# ~/.bashrc selects the aarch64 UV/Python environment on compute nodes.
source ~/.bashrc
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the tiny-llm repository}"
uv sync --locked
exec uv run tiny-llm "$@"
