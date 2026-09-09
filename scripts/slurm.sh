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
# Analysis owns a disposable cache per job. Resolve node-local paths here,
# before Python/Triton import, rather than on the submission/login node.
if [[ "${1:-}" == "analyze" ]]; then
    analysis_cache_dir=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/tiny-llm-analysis.XXXXXXXX")
    export TORCHINDUCTOR_CACHE_DIR="$analysis_cache_dir/inductor"
    export TRITON_CACHE_DIR="$analysis_cache_dir/triton"
    trap 'rm -rf -- "$analysis_cache_dir"' EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    # Do not exec: the shell must remain to run its cleanup trap.
    srun --ntasks=1 uv run --no-sync tiny-llm "$@"
else
    # Preserve training's GPU visibility and automatically allocated CPUs.
    exec srun --ntasks=1 uv run --no-sync tiny-llm "$@"
fi
