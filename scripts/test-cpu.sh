#!/usr/bin/env bash
# Keep dependency imports off the shared filesystem; reuse one environment per checkout.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo_root"
repo_key=$(printf '%s' "$repo_root" | sha256sum)
repo_key=${repo_key:0:16}
test_root=${TINY_LLM_TEST_ROOT:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/tiny-llm-tests-$(id -u)}
export UV_PROJECT_ENVIRONMENT="$test_root/$repo_key"

# Copy instead of symlinking packages back to the shared UV cache. Syncing also
# picks up lockfile changes while retaining the environment between invocations.
uv sync --locked --group dev --link-mode copy --compile-bytecode

export CUDA_VISIBLE_DEVICES=''
exec "$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest -q -m 'not cuda' --durations=10 "$@"
