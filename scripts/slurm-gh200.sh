#!/bin/bash
#SBATCH --account=naiss2026-3-205-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --job-name=tiny-llm-gh200
#SBATCH --output=slurm-gh200-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ $(uname -m) != aarch64 ]]; then
    echo 'This launcher requires an ARM GH200 compute node.' >&2
    exit 1
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
nvidia-smi
nvidia-smi topo -m
# SLURM supplies CUDA_VISIBLE_DEVICES and the CPU allocation; do not override them.
srun --ntasks=1 .venv-aarch64/bin/python -m tiny_llm "$@"
