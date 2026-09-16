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
printf '%(%Y-%m-%d %H:%M:%S)T | INFO    | Startup Slurm batch entry\n' -1 >&2
startup_shell_seconds=$SECONDS
source ~/.bashrc
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the tiny-llm repository}"
export TOKENIZERS_PARALLELISM=false
printf '%(%Y-%m-%d %H:%M:%S)T | INFO    | Startup shell setup: %ds\n' \
    -1 "$((SECONDS - startup_shell_seconds))" >&2
startup_sync_seconds=$SECONDS
uv sync --locked
printf '%(%Y-%m-%d %H:%M:%S)T | INFO    | Startup dependency sync: %ds\n' \
    -1 "$((SECONDS - startup_sync_seconds))" >&2
printf '%(%Y-%m-%d %H:%M:%S)T | INFO    | Startup launching srun\n' -1 >&2
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
