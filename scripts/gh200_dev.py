"""Submit a reproducible, isolated GH200 development check (not qualification)."""

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/gh200-integration-20261004")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    destination = args.root.resolve() / "development" / uuid.uuid4().hex[:10]
    destination.mkdir(parents=True)
    for name in ("src", "tests", "scripts", "configs"):
        shutil.copytree(ROOT / name, destination / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, destination / name)
    hashes = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in destination.rglob("*") if p.is_file()}
    (destination / "source-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    command = args.args or ["-m", "pytest", "-xq", "tests/test_gh200.py", "-m", "cuda"]
    if command[0] == "--":
        command = command[1:]
    script = "\n".join([
        "#!/bin/bash", "set -euo pipefail", "cd " + shlex.quote(str(destination)),
        "export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false",
        "export PYTHONPATH=" + shlex.quote(str(destination / "src")),
        'dev_cache=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/gh200-dev.XXXXXXXX")',
        'export TORCHINDUCTOR_CACHE_DIR="$dev_cache/inductor" TRITON_CACHE_DIR="$dev_cache/triton"',
        "trap 'rm -rf -- \"$dev_cache\"' EXIT",
        shlex.join([str(ROOT / ".venv-aarch64/bin/python"), *command]), "",
    ])
    (destination / "job.sh").write_text(script)
    job = subprocess.check_output([
        "sbatch", "--parsable", "--account=naiss2026-3-205-gpu", "--partition=gpu",
        "--gpus=nvidia_gh200_120gb:1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G",
        "--time=00:30:00", "--job-name=gh200-dev", f"--output={destination}/slurm.log",
        str(destination / "job.sh"),
    ], text=True).strip()
    receipt = dict(job=job, snapshot=str(destination), command=command)
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
