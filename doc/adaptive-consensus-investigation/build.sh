#!/usr/bin/env bash
set -euo pipefail

presentation_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd -- "$presentation_dir/../.." && pwd)"
cd -- "$presentation_dir"

if [[ -n "${TINY_LLM_DOC_PYTHON:-}" ]]; then
  presentation_python="$TINY_LLM_DOC_PYTHON"
elif [[ -x "$repository_dir/.venv-x86_64/bin/python" ]]; then
  presentation_python="$repository_dir/.venv-x86_64/bin/python"
else
  presentation_python="python3"
fi

"$presentation_python" build_figures.py "$@"
"$presentation_python" build_review_figure.py "$@"
typst compile --root . slides.typ slides.pdf
typst compile --root . supplement.typ supplement.pdf
"$presentation_python" verify.py
