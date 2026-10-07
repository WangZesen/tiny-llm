#!/usr/bin/env python3
"""Prepare and operate the checkpoint-free 20M/C512 batch-size study.

Run with the repository's Python environment. Start with ``prepare``, then
``submit --stage screen --dry-run``. Preparation never submits jobs. All generated
files live under ignored runs/. ``self-test`` needs neither C4 nor a GPU/Slurm.
The generated README documents the complete staged workflow and report semantics.
``collect`` uses a shared ±0.05 heatmap loss scale, annotating saturated cells.
Use ``collect --heatmap-limit 0.02`` or ``--heatmap-full-range`` to change it.
After sensitivity completes, ``promote --stage beta1-replicate`` freezes each
group's best seed-42 beta1 recipe, and ``submit --stage beta1-replicate`` adds
missing seeds 43–44. Collection exports this optional comparison separately.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import fcntl
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import traceback
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

PARAMETERS = 20_403_520
TOKENS = 408_092_672
WARMUP_TOKENS = 20_447_232
VALIDATION_TOKENS = 197_411_295
BATCHES = (128, 64, 32, 16)
LRS = (0.002, 0.003, 0.004, 0.006, 0.008, 0.012)
BETA1S = (0.0, 0.7, 0.84, 0.9, 0.92, 0.96, 0.98, 0.99)
DECAYS = (0.1, 0.0)
STAGES = ("screen", "replicate", "confirm", "sensitivity")
BETA1_REPLICATION = "beta1-replicate"
PACKAGES = ("torch", "triton", "numpy", "pydantic", "datasets", "transformers")
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
DEFAULT_ROOT = "runs/sync-20m-c512-small-batch"
DEFAULT_BASELINE = "runs/sync-20m-cosine-tpp20-b65536-c512-m128-seeds42-44"
DEFAULT_HEATMAP_LIMIT = 0.05
BASELINE_SOURCE = "4b1dcac5b320fa1620b9529144199581efe550b7037ed1611123b6c407e4963d"
# Audited sync changes: clipping counters only. Unknown revisions require a new audit.
COMPATIBLE_SOURCE = "f5c3286f2653a811c300725fa01e46daca2e4b7402b0e2fdcab1325567887f26"
UPSTREAM = "3558304a0a3e97c616ec3dfedae155fb6b937ba7"
TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def package_hash(path):
    result = hashlib.sha256()
    for file in sorted(Path(path).rglob("*.py")):
        result.update(file.relative_to(path).as_posix().encode())
        result.update(file.read_bytes())
    return result.hexdigest()


def repository():
    for path in Path(__file__).resolve().parents:
        if (path / "pyproject.toml").is_file() and (path / ".git").exists():
            return path
    raise ValueError("Run this script inside its tiny-llm checkout")


def command(args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, text=True, **kwargs).stdout


def git_state(repo):
    return {
        "head": command(["git", "rev-parse", "HEAD"], cwd=repo).strip(),
        "status": command(["git", "status", "--short"], cwd=repo),
        "diff": command(["git", "diff", "--binary", "HEAD"], cwd=repo),
    }


def activate_source(source):
    source = Path(source).resolve()
    if "tiny_llm" in sys.modules:
        filename = sys.modules["tiny_llm"].__file__
        if filename is None:
            raise ValueError("tiny_llm must be loaded from the frozen source files")
        loaded = Path(filename).resolve()
        require(loaded.is_relative_to(source), f"Already imported another tiny_llm: {loaded}")
    sys.path.insert(0, str(source))


@contextmanager
def campaign_lock(root):
    with (root / ".campaign.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def check_no_checkpoints(root):
    forbidden = [str(p) for p in Path(root).rglob("*") if p.suffix in CHECKPOINT_SUFFIXES]
    require(not forbidden, f"Checkpoint-free campaign contains weights/state: {forbidden[:5]}")


def group_key(batch, decay):
    return f"b{batch}-wd{decay:g}"


def recipe(batch, lr, beta2, decay, beta1=0.9):
    return dict(
        batch=int(batch),
        lr=float(lr),
        beta1=float(beta1),
        beta2=float(beta2),
        weight_decay=float(decay),
    )


def recipe_key(value):
    return digest({k: value[k] for k in ("batch", "lr", "beta1", "beta2", "weight_decay")})[:20]


def beta_policies(batch, decay):
    values = {
        "transferred": 0.98 ** (batch / 128),
        "half-life-10m": 2 ** (-512 * batch / 10_000_000),
    }
    if decay == 0.1:
        values["fixed-0.98"] = 0.98
    return values


def half_life(batch, beta2):
    return -batch * 512 * math.log(2) / math.log(beta2)


def screen_recipes():
    unique = {}
    for batch, decay, lr in itertools.product(BATCHES, DECAYS, LRS):
        for policy, beta2 in beta_policies(batch, decay).items():
            value = recipe(batch, lr, beta2, decay)
            key = recipe_key(value)
            unique.setdefault(key, value | {"policies": []})["policies"].append(policy)
    return list(unique.values())


def build_config(base, value, seed, output):
    from tiny_llm.config import Config
    from tiny_llm.data import training_boundaries
    from tiny_llm.train import learning_rate

    raw = copy.deepcopy(base)
    batch = value["batch"]
    raw["model"]["context_length"] = 512
    raw["decentralized"] = None
    raw["optimizer"].update({k: value[k] for k in ("lr", "beta1", "beta2", "weight_decay")})
    raw["optimizer"].update(eps=1e-8, grad_clip=1.0)
    raw["training"].update(
        tokens_per_parameters=20 if batch == 128 else 20.001092,
        epoch_tokens_per_parameters=0.5 if batch == 128 else 0.5000273,
        batch_tokens=batch * 512,
        micro_batch_size=batch,
        checkpoint_policy="none",
        save_epoch_training_state=False,
        log_every=20 * 128 // batch,
    )
    raw["lr_schedule"] = dict(name="cosine", warmup_steps=312 * 128 // batch, min_lr_ratio=0)
    raw["runtime"].update(seed=seed, output_dir=str(output))
    cfg = Config.model_validate(raw)
    boundaries = training_boundaries(cfg, cfg.model.parameter_count)
    require(cfg.model.parameter_count == PARAMETERS, "Model parameter count changed")
    require(len(boundaries) == 40 and boundaries[-1] * 512 == TOKENS, "Token budget mismatch")
    require(TOKENS % cfg.training.batch_tokens == 0, "Partial final update")
    require(
        cfg.lr_schedule.warmup_steps * cfg.training.batch_tokens == WARMUP_TOKENS,
        "Warmup token budget mismatch",
    )
    require(learning_rate(cfg, WARMUP_TOKENS, TOKENS) == value["lr"], "Warmup peak mismatch")
    require(learning_rate(cfg, TOKENS, TOKENS) == 0, "Final learning rate is not zero")
    require(not cfg.training.saved_epochs(), "Checkpoints enabled")
    return cfg


def artifact_paths(output):
    return [
        Path(output) / name
        for name in ("resolved.yaml", "result.json", "environment.json", "metrics.jsonl", "run.log")
    ]


def steady_rate(events, threshold=WARMUP_TOKENS):
    targets, seconds, previous_step, previous_tokens = 0, 0.0, 0, 0
    for event in events:
        if event["event"] != "train":
            continue
        step, tokens, duration = event["step"], event["tokens"], event["training_seconds"]
        require(step > previous_step and tokens > previous_tokens, "Nonmonotonic training log")
        require(math.isfinite(duration) and duration > 0, "Invalid training-window duration")
        if previous_tokens >= threshold:
            targets += tokens - previous_tokens
            seconds += duration
        previous_step, previous_tokens = step, tokens
    return targets / seconds if seconds else None


def validate_result(output, config, identity, source_hash, cache_identity, versions, expected=None):
    """Validate original artifacts, not cached success markers or report summaries."""
    import yaml

    expected = expected or dict(
        parameters=PARAMETERS,
        tokens=TOKENS,
        step=TOKENS // config["training"]["batch_tokens"],
        epochs=40,
        validation_tokens=VALIDATION_TOKENS,
    )
    output = Path(output)
    check_no_checkpoints(output)
    result, env = read_json(output / "result.json"), read_json(output / "environment.json")
    resolved = yaml.safe_load((output / "resolved.yaml").read_text())
    left, right = copy.deepcopy(resolved), copy.deepcopy(config)
    left["runtime"].pop("output_dir")
    right["runtime"].pop("output_dir")
    require(left == right, f"Resolved configuration mismatch: {output}")
    require(resolved["training"]["checkpoint_policy"] == "none", "Checkpoints enabled")
    require(resolved.get("decentralized") is None, "Expected synchronous training")
    require(result["status"] == "complete", f"Incomplete result: {output}")
    for key in ("parameters", "tokens", "step", "epochs"):
        require(result[key] == expected[key], f"Wrong {key}: {output}")
    require(
        result["recipe_identity"] == env["recipe_identity"] == identity,
        f"Recipe identity mismatch: {output}",
    )
    require(
        env["source_hash"] == source_hash and env["cache_identity"] == cache_identity,
        f"Source/cache mismatch: {output}",
    )
    require(env["versions"] == versions, f"Dependency versions changed: {output}")
    require(
        "GH200" in env["gpu"] and env["compiled"] and env["attention"] == "sdpa",
        f"Unexpected execution environment: {output}",
    )
    validation = result["final_validation"]
    require(
        validation["split"] == "full" and validation["validation_complete"],
        "Final validation is incomplete",
    )
    require(validation["tokens"] == expected["validation_tokens"], "Validation tokens changed")
    require(
        all(math.isfinite(validation[k]) for k in ("loss", "perplexity", "seconds")),
        "Nonfinite final validation",
    )
    events = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    training = [e for e in events if e["event"] == "train"]
    require(bool(training), "No training events")
    require(
        (training[-1]["tokens"], training[-1]["step"], training[-1]["lr"])
        == (expected["tokens"], expected["step"], 0),
        "Final training event mismatch",
    )
    require(
        all(math.isfinite(e[k]) for e in training for k in ("loss", "grad_norm", "lr")),
        "Nonfinite training observations",
    )
    subsets = [e for e in events if e["event"] == "validation"]
    finals = [e for e in events if e["event"] == "final_validation"]
    require(len(subsets) == expected["epochs"] and len(finals) == 1, "Evaluation count mismatch")
    require(finals[0]["loss"] == validation["loss"], "Final evaluation log mismatch")
    for key in ("training_seconds", "training_elapsed_seconds", "seconds_this_session"):
        require(math.isfinite(result[key]) and result[key] > 0, f"Invalid {key}")
    steady = steady_rate(events)
    clipped = [e.get("grad_clip_count") for e in subsets]
    stratum = digest(
        {
            "source": source_hash,
            "versions": versions,
            "gpu": env["gpu"],
            "runtime": {
                k: config["runtime"][k]
                for k in (
                    "amp",
                    "compile",
                    "compile_mode",
                    "sdpa_backend",
                    "fused_optimizer",
                    "attention_backend",
                    "cpu_threads",
                )
            },
        }
    )[:12]
    return dict(
        loss=validation["loss"],
        tokens=result["tokens"],
        steps=result["step"],
        steady_targets_per_second=steady,
        training_targets_per_second=result["tokens"] / result["training_seconds"],
        elapsed_targets_per_second=result["tokens"] / result["training_elapsed_seconds"],
        session_targets_per_second=result["tokens"] / result["seconds_this_session"],
        session_seconds=result["seconds_this_session"],
        full_validation_seconds=validation["seconds"],
        intermediate_validation_seconds=sum(e["seconds"] for e in subsets),
        peak_memory_gib=result["peak_memory_bytes"] / 2**30,
        clipping_frequency=sum(clipped) / result["step"]
        if all(c is not None for c in clipped)
        else None,
        source_hash=source_hash,
        environment_stratum=stratum,
        gpu=env["gpu"],
        output=str(output),
        recipe_identity=identity,
    )


def audit_baseline(repo, baseline):
    """Authenticate the specific supplied campaign and its compatible sync source."""
    from tiny_llm.config import Config
    from tiny_llm.data import TokenCache
    from tiny_llm.train import recipe_identity

    historical = read_json(baseline / "manifest.json")
    require(historical["source_hash"] == BASELINE_SOURCE, "Unexpected historical source")
    current_hash = package_hash(repo / "src/tiny_llm")
    require(
        current_hash in (BASELINE_SOURCE, COMPATIBLE_SOURCE),
        "Current source is not the audited sync revision; re-audit before historical reuse",
    )
    for path, expected_hash in historical["source_files"].items():
        require(
            sha(baseline / "source" / path) == expected_hash, f"Historical source changed: {path}"
        )
    require(
        package_hash(baseline / "source/src/tiny_llm") == BASELINE_SOURCE,
        "Historical package contains changed/extra sources",
    )
    for filename in ("model.py", "data.py", "runtime.py", "state.py", "checkpoints.py"):
        require(
            sha(repo / "src/tiny_llm" / filename)
            == sha(baseline / "source/src/tiny_llm" / filename),
            f"Training dependency changed: {filename}",
        )

    def functions(path):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef)
        }

    old, new = (
        functions(p / "train.py") for p in (baseline / "source/src/tiny_llm", repo / "src/tiny_llm")
    )
    for name in (
        "make_optimizer",
        "optimizer_update",
        "learning_rate",
        "evaluate",
        "loss_function",
    ):
        require(old[name] == new[name], f"Historical sync function changed: {name}")
    # recipe_identity added a decentralized-only version field. Verify its
    # synchronous output against every original recipe below instead of its AST.
    cache = TokenCache(repo / "data/c4")
    require(
        cache.manifest == read_json(baseline / "provenance/cache-manifest.json"),
        "Historical cache manifest differs",
    )
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    require(versions == historical["worker_versions"], "Historical dependencies differ")
    records = []
    for spec in historical["runs"]:
        cfg = Config.model_validate(spec["config"])
        cache.validate_config(cfg)
        require(
            recipe_identity(cfg, cache) == spec["recipe_identity"], "Historical recipe mismatch"
        )
        require(
            sha(baseline / spec["config_file"]) == spec["config_sha256"],
            "Historical config file changed",
        )
        require(
            cfg.training.batch_tokens == 65536 and cfg.model.context_length == 512,
            "Wrong historical batch/context",
        )
        attempts = [
            p.parent
            for p in (baseline / "experiments" / spec["slug"]).glob("attempt-*/success.json")
        ]
        require(len(attempts) == 1, f"Expected one historical successful attempt: {spec['slug']}")
        value = recipe(
            128,
            cfg.optimizer.lr,
            cfg.optimizer.beta2,
            cfg.optimizer.weight_decay,
            cfg.optimizer.beta1,
        )
        result = validate_result(
            attempts[0],
            spec["config"],
            spec["recipe_identity"],
            BASELINE_SOURCE,
            cache.manifest["identity"],
            versions,
        )
        records.append(
            value
            | result
            | dict(
                seed=cfg.runtime.seed,
                recipe=recipe_key(value),
                stage="historical",
                kind="historical",
                role="candidate",
                status="complete",
                config=spec["config"],
                artifact_hashes={p.name: sha(p) for p in artifact_paths(attempts[0])},
            )
        )
    require(len(records) == 96, "Expected 96 historical runs")
    require(len({(r["recipe"], r["seed"]) for r in records}) == 96, "Duplicate historical runs")
    return records, cache, versions, current_hash


def add_run(manifest, root, value, seed, stage, *, reference=False):
    """Stable configuration IDs; roles distinguish the deliberate reference repeat."""
    from tiny_llm.config import save_config
    from tiny_llm.data import TokenCache
    from tiny_llm.train import recipe_identity

    key = recipe_key(value)
    run_id = f"{key}-s{seed}" + ("-reference" if reference else "")
    if not reference:
        if any(r["recipe"] == key and r["seed"] == seed for r in manifest["historical"]):
            return None
        if any(r["id"] == run_id for r in manifest["runs"]):
            return None
    require(not any(r["id"] == run_id for r in manifest["runs"]), "Duplicate run ID")
    output = Path(manifest["root"]) / "experiments" / run_id / "attempt-001"
    cfg = build_config(manifest["base_config"], value, seed, output)
    cache = TokenCache(cfg.data.cache_dir)
    cache.validate_config(cfg)
    path = root / "configs" / f"{run_id}.yaml"
    save_config(cfg, path)
    record = {k: value[k] for k in ("batch", "lr", "beta1", "beta2", "weight_decay")}
    record.update(
        id=run_id,
        index=len(manifest["runs"]),
        recipe=key,
        seed=seed,
        stage=stage,
        role="reference" if reference else "candidate",
        policies=value.get("policies", []),
        config=cfg.model_dump(mode="json"),
        config_file=f"configs/{run_id}.yaml",
        config_sha256=sha(path),
        recipe_identity=recipe_identity(cfg, cache),
    )
    manifest["runs"].append(record)
    return run_id


def verify_campaign(root, manifest, *, configs=True):
    require(
        manifest["version"] == 1 and Path(manifest["root"]) == root, "Campaign location changed"
    )
    for name, expected in manifest["source_files"].items():
        require(sha(root / "source" / name) == expected, f"Frozen source changed: {name}")
    if followup := manifest.get("beta1_replication"):
        require(
            sha(root / followup["driver_file"]) == followup["driver_sha256"],
            "Frozen beta1 follow-up driver changed",
        )
    require(
        package_hash(root / "source/src/tiny_llm") == manifest["source_hash"],
        "Frozen package contains changed/extra sources",
    )
    require(
        read_json(Path(manifest["base_config"]["data"]["cache_dir"]) / "manifest.json")
        == read_json(root / "provenance/cache-manifest.json"),
        "Data cache metadata changed",
    )
    if configs:
        for spec in manifest["runs"]:
            require(
                sha(root / spec["config_file"]) == spec["config_sha256"],
                f"Config file changed: {spec['id']}",
            )


def load_campaign(root):
    manifest = read_json(root / "manifest.json")
    verify_campaign(root, manifest)
    return manifest


def job_script(manifest):
    repo = Path(manifest["repository"])
    python = shlex.quote(manifest["worker_python"])
    return f"""#!/usr/bin/env bash
#SBATCH --account=naiss2026-3-205-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=nvidia_gh200_120gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
campaign_root=$1
selection_file=$2
cd {shlex.quote(str(repo))}
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_COMPILE_THREADS=4
campaign_cache=$(mktemp -d "${{SLURM_TMPDIR:-${{TMPDIR:-/tmp}}}}/batch-study-${{SLURM_JOB_ID}}.XXXXXXXX")
export TORCHINDUCTOR_CACHE_DIR="$campaign_cache/inductor"
export TRITON_CACHE_DIR="$campaign_cache/triton"
trap 'rm -rf -- "$campaign_cache"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
srun --ntasks=1 {python} -B -u "$campaign_root/source/batch_size_study.py" \\
  _worker --root "$campaign_root" --selection "$selection_file" --task "$SLURM_ARRAY_TASK_ID"
"""


def campaign_readme(manifest):
    root = manifest["root"]
    script = shlex.quote(f"{root}/source/batch_size_study.py")
    prefix = f".venv-x86_64/bin/python {script}"
    return f"""# 20M/C512 synchronous small-batch study

Prepared {manifest["created_utc"]}. No jobs are submitted by prepare.
One GH200, eight CPUs, 32 GiB RAM, two hours per run; default concurrency **48**.
Every run disables checkpoint and weight saving. Retries start from scratch.

## Workflow

Use the frozen driver so later edits cannot change this experiment:

```bash
{prefix} submit --root {shlex.quote(root)} --stage screen --dry-run
{prefix} submit --root {shlex.quote(root)} --stage screen
{prefix} status --root {shlex.quote(root)} --scheduler
# First submission runs only the fresh B128 reference. Once it finishes:
{prefix} submit --root {shlex.quote(root)} --stage screen
# When screening finishes, this may first materialize one LR-boundary extension:
{prefix} promote --root {shlex.quote(root)} --stage replicate
# If an extension was created, submit screen again, wait, then repeat promote.
{prefix} submit --root {shlex.quote(root)} --stage replicate
# Once replication finishes, freeze the selected recipes:
{prefix} promote --root {shlex.quote(root)} --stage confirm
{prefix} submit --root {shlex.quote(root)} --stage confirm
# After confirmation jobs finish (arrays are serialized to enforce the cap):
{prefix} promote --root {shlex.quote(root)} --stage sensitivity
{prefix} submit --root {shlex.quote(root)} --stage sensitivity
{prefix} collect --root {shlex.quote(root)}
```

`--dry-run` never invokes Slurm or writes submission files. `--concurrency N`
overrides the array cap. Only one campaign array can be active at a time, so
overlapping stage submissions cannot exceed the cap. Status is offline unless
`--scheduler` is specified. Collection can use `--allow-incomplete` for a clearly
marked partial report. It never fabricates missing measurements.

Explicit infrastructure retries use `submit --stage STAGE --retry --reason TEXT`;
optionally limit them with `--ids ID,ID`. Failed attempts are retained. Numerical
divergence is an experimental failure, excluded from rankings, not retried.
Ambiguous submission receipts block further submissions. Inspect Slurm and use
`submit --adopt-job JOB_ID --intent TOKEN` to reconcile an accepted submission;
`--reject-intent TOKEN --reason TEXT` records a manually verified non-submission.

## Protocol

Model: 20,403,520 parameters, C4, TinyLlama tokenizer, context 512.
Batches 128/64/32/16 use one microbatch per update, exactly 408,092,672 targets,
40 virtual epochs, and 20,447,232 warmup targets before cosine decay to zero.
Smaller batches use nominal ratios 20.001092/0.5000273 to match the historical
realized budget. Intermediate evaluation boundaries differ slightly from rounding.
Full final validation contains 197,411,295 targets; evaluation batch is 128.
FP32 parameters/moments, BF16 AMP, clipping 1, epsilon 1e-8 are retained.

Screen LRs {list(LRS)}, beta1 .9, decay .1 and zero, with beta2 transferred from
.98 at B128 or a fixed 10M-target half-life; decay .1 also has a fixed-.98 control.
Seed42 screens; top two recipes per batch/decay receive seeds43–44. Historical
B128 three-seed recipes remain eligible. Extend a winning LR endpoint once by
a factor of two. Flag remaining boundary winners. Freeze selections before
fresh paired seeds45–47; decay .1 is primary, decay zero secondary at the same
selected smaller batch. Sensitivity varies beta1 over {list(BETA1S)} on seed42
with selected LR/beta2 fixed; it does not alter the primary beta1=.9 comparison.

Optional follow-up: `promote --stage beta1-replicate`, then
`submit --stage beta1-replicate --dry-run` and `submit --stage beta1-replicate`.
This freezes each group's best seed42 beta1 and fills missing seeds43–44.
Existing successful runs are reused. The separate beta1-tuned comparison is
conditional on earlier LR/beta2 selection, not joint hyperparameter tuning or
independent confirmation. No new screening or boundary extension is implicit.

## Sources and interpretation

Paper: https://arxiv.org/pdf/2507.07101v4 (15 December 2025).
Official implementation: https://github.com/martin-marek/batch-size/tree/{UPSTREAM}
(checked 29 September 2026; code revision postdates paper v4).
Adopt token-half-life scaling and per-batch LR tuning. Preserve the local fixed
data split and shifted targets instead of the upstream microbatch-dependent split
and masked final target. The headline upstream experiment uses zero decay,
no clipping, FP32 activations and a different model; its C4 transfer experiment
retains decay .1. This study tests transfer to the local recipe, not exact numeric
replication. Stochastic rounding is unnecessary with FP32 master weights.

The historical sync source differs only by current clipping counters in its
training path; source hashes and unchanged optimizer/model/data functions are
audited. The fresh reference's loss difference is reported, without demanding
bitwise equality from nondeterministic CUDA. All tuning uses validation, not an
independent test set. Three confirmation pairs give limited precision. Report
all deltas and a descriptive paired t interval (df=2); crossing zero is inconclusive.

## Outputs and timing

Reports contain Figure4-style conditional seed42 LR/beta1/beta2/half-life
heatmaps, Figure1(a)-style LR/beta2-tuned means ± sample SD across seeds42–44,
separate fresh-seed confirmation, throughput figures, and raw CSV/JSON tables.
Heatmaps subtract the selected recipe's seed42 reference loss at each batch and
decay, share a zero-centered color scale, and leave untested cells blank.

Steady targets/s sums target increments / summed training-window seconds for
complete windows starting at or after 20,447,232 targets. Crossing windows are
excluded, and individual rates are never averaged. Recorded training rate
includes compilation; elapsed training rate includes intermediate validation.
Session rate also includes subset collection and final validation, but excludes
earlier imports/model setup, launcher overhead, and queue time. Peak memory
includes evaluation. Compare current-source screening distributions by batch,
decay, and environment, with selected-recipe results separate; historical timings
stay labeled. Relative speeds require the same environment/source stratum.
Figures are PDF/SVG/PNG; all inputs are retained as CSV/JSON.
"""


def prepare(args):
    repo, root, baseline = repository(), args.root.resolve(), args.baseline.resolve()
    require(
        root.is_relative_to(repo / "runs") and root != repo / "runs",
        "Campaign must be inside this checkout's ignored runs/ directory",
    )
    require(not root.exists(), f"Campaign already exists: {root}; choose a new --root")
    require(
        subprocess.run(["git", "check-ignore", "-q", str(root)], cwd=repo).returncode == 0,
        "Campaign directory must be git-ignored",
    )
    activate_source(repo / "src")
    history, cache, versions, current_hash = audit_baseline(repo, baseline)
    base = copy.deepcopy(history[0]["config"])
    base["data"]["cache_dir"] = str(repo / "data/c4")
    require(
        base["evaluation"] == dict(subset_blocks=1024, batch_size=128, seed=12345),
        "Unexpected baseline evaluation settings",
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".batch-study-prepare-", dir=root.parent) as directory:
        temp = Path(directory) / "campaign"
        for folder in ("configs", "source", "provenance", "logs", "reports", "submissions"):
            (temp / folder).mkdir(parents=True, exist_ok=True)
        sources = list((repo / "src/tiny_llm").rglob("*.py"))
        sources += [
            repo / name
            for name in ("pyproject.toml", "uv.lock", "configs/20m.yaml", "configs/cosine.yaml")
        ]
        for source in sources:
            destination = temp / "source" / source.relative_to(repo)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        shutil.copy2(Path(__file__), temp / "source/batch_size_study.py")
        state = git_state(repo)
        (temp / "provenance/repository.patch").write_text(state.pop("diff"))
        atomic_json(temp / "provenance/repository.json", state)
        atomic_json(temp / "provenance/cache-manifest.json", cache.manifest)
        atomic_json(
            temp / "provenance/historical-manifest.json", read_json(baseline / "manifest.json")
        )
        for record in history:
            origin = Path(record["output"])
            relative = Path("historical") / f"{record['recipe']}-s{record['seed']}"
            destination = temp / relative
            destination.mkdir(parents=True)
            for path in artifact_paths(origin):
                shutil.copy2(path, destination / path.name)
            record["original_output"] = str(origin)
            record["output"] = str(root / relative)
            record["artifact_dir"] = str(relative)
        manifest: dict = dict(
            version=1,
            created_utc=now(),
            root=str(root),
            repository=str(repo),
            baseline=str(baseline),
            base_config=base,
            historical=history,
            runs=[],
            source_hash=current_hash,
            cache_identity=cache.manifest["identity"],
            versions=versions,
            source_files={
                str(p.relative_to(temp / "source")): sha(p)
                for p in sorted((temp / "source").rglob("*"))
                if p.is_file()
            },
            concurrency=48,
            worker_python=str(repo / ".venv-aarch64/bin/python"),
            phases={"screen": {"created_utc": now()}},
            extension_checked=False,
            expected=dict(
                parameters=PARAMETERS, tokens=TOKENS, validation_tokens=VALIDATION_TOKENS
            ),
        )
        add_run(manifest, temp, recipe(128, 0.008, 0.98, 0.1), 42, "screen", reference=True)
        for value in screen_recipes():
            add_run(manifest, temp, value, 42, "screen")
        require(len(manifest["runs"]) == 109, "Unexpected initial run count")
        (temp / "job.sh").write_text(job_script(manifest))
        manifest["job_sha256"] = sha(temp / "job.sh")
        (temp / "README.md").write_text(campaign_readme(manifest))
        atomic_json(temp / "manifest.json", manifest)
        atomic_json(
            temp / "provenance/preflight.json",
            dict(
                checked_utc=now(),
                historical_runs=96,
                new_runs=109,
                no_checkpoints=True,
                exact_tokens=TOKENS,
                warmup_tokens=WARMUP_TOKENS,
                concurrency=48,
                source_compatibility="Audited revisions; unchanged model/data/optimizer/schedule/evaluation",
            ),
        )
        check_no_checkpoints(temp)
        temp.rename(root)
    print(f"Prepared {root}: 109 new runs, 96 validated historical runs; no jobs submitted.")


def local_records(root, manifest):
    records = []
    for spec in manifest["runs"]:
        directory = root / "experiments" / spec["id"]
        attempts = sorted(directory.glob("attempt-*")) if directory.exists() else []
        successes = [p for p in attempts if (p / "success.json").exists()]
        require(len(successes) <= 1, f"Multiple successes for {spec['id']}")
        row = {
            k: spec[k]
            for k in (
                "id",
                "recipe",
                "batch",
                "lr",
                "beta1",
                "beta2",
                "weight_decay",
                "seed",
                "stage",
                "role",
                "index",
            )
        }
        row.update(kind="current", status="pending")
        if successes:
            output = successes[0]
            require(not (output / "failure.json").exists(), f"Conflicting result markers: {output}")
            result = validate_result(
                output,
                spec["config"],
                spec["recipe_identity"],
                manifest["source_hash"],
                manifest["cache_identity"],
                manifest["versions"],
            )
            success = read_json(output / "success.json")
            require(success["loss"] == result["loss"], "Success marker disagrees with raw result")
            for path in artifact_paths(output):
                require(
                    sha(path) == success["artifact_hashes"][path.name],
                    f"Completed artifact changed: {path}",
                )
            row.update(result, status="complete", attempt=int(output.name.split("-")[-1]))
        elif attempts:
            output = attempts[-1]
            row.update(
                output=str(output), attempt=int(output.name.split("-")[-1]), status="started"
            )
            if (output / "failure.json").exists():
                failure = read_json(output / "failure.json")
                row.update(status=failure["status"], error=failure["error"])
            elif (output / "status.json").exists():
                state = read_json(output / "status.json")
                if state["status"] in ("failed", "interrupted"):
                    row.update(status="failed", error=state.get("error", state["status"]))
        records.append(row)
    return records


def all_records(root, manifest):
    history = []
    for record in manifest["historical"]:
        output = root / record["artifact_dir"]
        for path in artifact_paths(output):
            require(
                sha(path) == record["artifact_hashes"][path.name],
                f"Historical artifact changed: {path}",
            )
        result = validate_result(
            output,
            record["config"],
            record["recipe_identity"],
            record["source_hash"],
            manifest["cache_identity"],
            manifest["versions"],
        )
        history.append(
            {k: v for k, v in record.items() if k not in ("config", "artifact_hashes")} | result
        )
    return history + local_records(root, manifest)


def candidate_rows(records, seeds):
    selected = {}
    for row in records:
        if row["status"] != "complete" or row["role"] == "reference" or row["seed"] not in seeds:
            continue
        require(math.isfinite(row["loss"]), "Nonfinite loss cannot be ranked")
        key = row["recipe"], row["seed"]
        require(key not in selected, f"Duplicate candidate observation: {key}")
        selected[key] = row
    return list(selected.values())


def ranked_groups(records, seeds=(42, 43, 44)):
    grouped = defaultdict(list)
    for row in candidate_rows(records, seeds):
        if row["beta1"] == 0.9 and row["stage"] not in (
            "confirm",
            "sensitivity",
            BETA1_REPLICATION,
        ):
            grouped[row["recipe"]].append(row)
    groups = defaultdict(list)
    for key, rows in grouped.items():
        if {r["seed"] for r in rows} != set(seeds):
            continue
        value = {k: rows[0][k] for k in ("batch", "lr", "beta1", "beta2", "weight_decay")}
        losses = [next(r["loss"] for r in rows if r["seed"] == seed) for seed in seeds]
        value.update(
            recipe=key,
            seeds=list(seeds),
            losses=losses,
            mean_loss=statistics.mean(losses),
            std_loss=statistics.stdev(losses) if len(losses) > 1 else None,
        )
        groups[group_key(value["batch"], value["weight_decay"])].append(value)
    for values in groups.values():
        values.sort(key=lambda v: (v["mean_loss"], v["lr"], v["beta2"], v["recipe"]))
    return groups


def require_stage_ready(manifest, records, stage):
    require(stage in manifest["phases"], f"Stage {stage} is not prepared")
    rows = [r for r in records if r["kind"] == "current" and r["stage"] == stage]
    blockers = [r["id"] for r in rows if r["status"] not in ("complete", "numerical_failure")]
    require(not blockers, f"{stage} has incomplete/infrastructure-failed runs: {blockers[:8]}")
    references = [r for r in rows if r["role"] == "reference"]
    require(all(r["status"] == "complete" for r in references), "Fresh reference must succeed")


def freeze_selection(manifest, records):
    require_stage_ready(manifest, records, "screen")
    require_stage_ready(manifest, records, "replicate")
    require(manifest["extension_checked"], "LR boundary check has not run")
    if "selected" not in manifest:
        ranked = ranked_groups(records)
        require(
            all(group_key(b, d) in ranked for b, d in itertools.product(BATCHES, DECAYS)),
            "Every batch/decay group needs a complete three-seed recipe",
        )
        manifest["selected"] = {k: values[0] for k, values in ranked.items()}
        manifest["selection_frozen_utc"] = now()
    return manifest["selected"]


def promote(args):
    root = args.root.resolve()
    activate_source(root / "source/src")
    with campaign_lock(root):
        manifest = load_campaign(root)
        require(args.stage != "screen", "Screening is materialized by prepare")
        if args.stage in manifest["phases"]:
            print(f"{args.stage} is already prepared; its selections are immutable.")
            return
        records = all_records(root, manifest)
        added = []
        if args.stage == BETA1_REPLICATION:
            require_stage_ready(manifest, records, "sensitivity")
            selections = beta1_replication_selection(manifest, records)
            followup = root / "followups" / BETA1_REPLICATION
            followup.mkdir(parents=True, exist_ok=True)
            # Preserve the completed campaign and freeze the follow-up implementation.
            require(
                not (followup / "manifest-before.json").exists(),
                "Follow-up preparation already started; inspect its snapshot",
            )
            atomic_json(followup / "manifest-before.json", manifest)
            driver = followup / "batch_size_study.py"
            shutil.copy2(Path(__file__).resolve(), driver)
            manifest["beta1_replication"] = dict(
                frozen_utc=now(),
                selection_seed=42,
                seeds=[42, 43, 44],
                recipes=selections,
                driver_file=str(driver.relative_to(root)),
                driver_sha256=sha(driver),
            )
            for value, seed in itertools.product(selections.values(), (43, 44)):
                run = add_run(manifest, root, value, seed, BETA1_REPLICATION)
                if run:
                    added.append(run)
        elif args.stage == "replicate":
            require_stage_ready(manifest, records, "screen")
            ranked = ranked_groups(records, (42,))
            require(
                all(
                    len(ranked.get(group_key(b, d), [])) >= 2
                    for b, d in itertools.product(BATCHES, DECAYS)
                ),
                "Every batch/decay group needs at least two finite candidates",
            )
            if not manifest["extension_checked"]:
                extensions = []
                for key, values in ranked.items():
                    best = values[0]
                    # Include failed numerical trials when determining tested grid endpoints.
                    grid = [
                        r
                        for r in records
                        if r["seed"] == 42
                        and r["role"] == "candidate"
                        and r["beta1"] == 0.9
                        and r["batch"] == best["batch"]
                        and r["weight_decay"] == best["weight_decay"]
                    ]
                    low, high = min(r["lr"] for r in grid), max(r["lr"] for r in grid)
                    new_lr = (
                        low / 2 if best["lr"] == low else high * 2 if best["lr"] == high else None
                    )
                    if new_lr is not None:
                        betas = sorted({r["beta2"] for r in grid})
                        for beta2 in betas:
                            value = recipe(best["batch"], new_lr, beta2, best["weight_decay"])
                            run = add_run(manifest, root, value, 42, "screen")
                            if run:
                                added.append(run)
                        extensions.append(dict(group=key, lr=new_lr, beta2s=betas))
                manifest.update(extension_checked=True, extensions=extensions)
                if added:
                    atomic_json(root / "manifest.json", manifest)
                    print(
                        f"Created {len(added)} one-round LR extensions. Submit screen, then promote again."
                    )
                    return
            flags, promotions = {}, {}
            for key, values in ranked.items():
                best = values[0]
                grid = [
                    r["lr"]
                    for r in records
                    if r["seed"] == 42
                    and r["beta1"] == 0.9
                    and r["batch"] == best["batch"]
                    and r["weight_decay"] == best["weight_decay"]
                ]
                flags[key] = best["lr"] in (min(grid), max(grid))
                promotions[key] = values[:2]
                for value, seed in itertools.product(values[:2], (43, 44)):
                    run = add_run(manifest, root, value, seed, "replicate")
                    if run:
                        added.append(run)
            manifest["boundary_flags"] = flags
            manifest["promoted_recipes"] = promotions
        else:
            selected = freeze_selection(manifest, records)
            if args.stage == "confirm":
                best_small = min(
                    (selected[group_key(b, 0.1)] for b in BATCHES if b < 128),
                    key=lambda r: (r["mean_loss"], r["batch"]),
                )
                batch = best_small["batch"]
                comparisons = {}
                for decay in DECAYS:
                    smaller, control = (
                        selected[group_key(batch, decay)],
                        selected[group_key(128, decay)],
                    )
                    comparisons[str(decay)] = dict(
                        smaller=smaller, control=control, primary=decay == 0.1
                    )
                    for value, seed in itertools.product((smaller, control), (45, 46, 47)):
                        run = add_run(manifest, root, value, seed, "confirm")
                        if run:
                            added.append(run)
                manifest["confirmation"] = comparisons
            else:
                for value, beta1 in itertools.product(selected.values(), BETA1S):
                    run = add_run(manifest, root, value | {"beta1": beta1}, 42, "sensitivity")
                    if run:
                        added.append(run)
        manifest["phases"][args.stage] = dict(created_utc=now(), added_runs=added)
        atomic_json(root / "manifest.json", manifest)
        print(f"Prepared {args.stage}: {len(added)} new runs; no jobs submitted.")


def receipts(root):
    return [
        read_json(p)
        for p in sorted((root / "submissions").glob("*.json"))
        if not p.name.endswith(".selection.json")
    ]


def scheduler_states(items):
    ids = sorted({r["job_id"] for r in items if r.get("status") == "submitted"})
    if not ids:
        return {}
    output = command(
        ["sacct", "--array", "-n", "-P", "-j", ",".join(ids), "--format=JobID%60,State%30,ExitCode"]
    )
    result = {}
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) >= 3 and "." not in parts[0]:
            result[parts[0]] = dict(state=parts[1].split()[0].rstrip("+"), exit_code=parts[2])
    return result


def submitted_assignments(items):
    result = defaultdict(list)
    for receipt in items:
        if receipt["status"] == "submitted":
            for task in receipt["tasks"]:
                result[task["id"]].append(task | {"job_id": receipt["job_id"]})
    return result


def submission_tasks(manifest, records, items, stage, *, retry=False, ids=None):
    require(stage in manifest["phases"], f"Stage {stage} has not been prepared")
    require(
        not [r for r in items if r["status"] == "intent"],
        "Unresolved submission intent; reconcile with --adopt-job or --reject-intent",
    )
    assigned = submitted_assignments(items)
    by_id = {r["id"]: r for r in records if r["kind"] == "current"}
    specs = [s for s in manifest["runs"] if s["stage"] == stage]
    if stage == "screen":
        reference = next(s for s in specs if s["role"] == "reference")
        if by_id[reference["id"]]["status"] != "complete":
            specs = [reference]
    if ids:
        require(
            set(ids) <= {s["id"] for s in specs}, "Requested IDs are outside the eligible stage"
        )
        specs = [s for s in specs if s["id"] in ids]
    tasks = []
    for spec in specs:
        row, previous = by_id[spec["id"]], assigned[spec["id"]]
        if row["status"] in ("complete", "numerical_failure"):
            continue
        if retry:
            if not previous and row["status"] == "pending":
                continue
            attempt = max([r["attempt"] for r in previous] + [row.get("attempt", 0)]) + 1
        else:
            if previous or row["status"] != "pending":
                continue
            attempt = 1
        tasks.append(dict(id=spec["id"], index=spec["index"], attempt=attempt))
    return tasks


def sbatch_command(root, manifest, selection, tasks, concurrency, token):
    require(concurrency > 0, "Concurrency must be positive")
    return [
        "sbatch",
        "--parsable",
        f"--array=0-{len(tasks) - 1}%{concurrency}",
        f"--job-name=bs-{token[:12]}",
        f"--chdir={manifest['repository']}",
        f"--output={root}/logs/{token}-%A_%a.log",
        str(root / "job.sh"),
        str(root),
        str(selection),
    ]


def require_no_active_arrays(items):
    ids = sorted({r["job_id"] for r in items if r.get("status") == "submitted"})
    if not ids:
        return
    # Accounting includes pending/running tasks and works after completed jobs
    # disappear from squeue. Missing accounting also blocks overlapping arrays.
    states = scheduler_states(items)
    for item in items:
        if item["status"] != "submitted":
            continue
        for task_index in range(len(item["tasks"])):
            key = f"{item['job_id']}_{task_index}"
            require(
                states.get(key, {}).get("state") in TERMINAL_STATES,
                f"Scheduler accounting not terminal/available for {key}; retry later",
            )


def reconcile(args, root):
    token = args.intent or args.reject_intent
    require(
        token and len(token) == 32 and all(c in "0123456789abcdef" for c in token),
        "Use the 32-character token from the unresolved intent filename",
    )
    path = root / "submissions" / f"{token}.json"
    receipt = read_json(path)
    require(receipt["status"] == "intent", "Only unresolved intents can be reconciled")
    if args.adopt_job:
        require(args.adopt_job.isdigit(), "Job ID must be numeric")
        output = command(
            ["sacct", "-n", "-P", "-j", args.adopt_job, "--format=JobID%60,JobName%80"]
        )
        require(
            any(
                line.split("|")[1] == f"bs-{token[:12]}"
                for line in output.splitlines()
                if len(line.split("|")) >= 2
            ),
            "Job name does not match this intent",
        )
        receipt.update(status="submitted", job_id=args.adopt_job, reconciled_utc=now())
    else:
        require(bool(args.reason), "--reject-intent requires the evidence in --reason")
        receipt.update(status="rejected", reason=args.reason, reconciled_utc=now())
    atomic_json(path, receipt)
    print(json.dumps(receipt, indent=2))


def submit(args):
    root = args.root.resolve()
    manifest = load_campaign(root)
    require(sha(root / "job.sh") == manifest["job_sha256"], "Job script changed")
    if args.adopt_job or args.reject_intent:
        require(not args.dry_run, "Reconciliation is separate from submission dry-run")
        with campaign_lock(root):
            reconcile(args, root)
        return
    require(args.stage is not None, "submit requires --stage")
    require(
        not args.retry or bool(args.reason),
        "Retries require --reason describing infrastructure failure",
    )

    def preview():
        current = load_campaign(root)
        items = receipts(root)
        records = local_records(root, current)
        tasks = submission_tasks(
            current,
            records,
            items,
            args.stage,
            retry=args.retry,
            ids=args.ids.split(",") if args.ids else None,
        )
        return current, items, tasks

    if args.dry_run:
        current, _, tasks = preview()
        if not tasks:
            print(
                "No eligible unsubmitted runs (submitted runs require status/retry as appropriate)."
            )
            return
        cmd = sbatch_command(
            root,
            current,
            root / "submissions/DRY-RUN.selection.json",
            tasks,
            args.concurrency or current["concurrency"],
            "DRY-RUN",
        )
        print(
            json.dumps(
                dict(
                    dry_run=True,
                    stage=args.stage,
                    tasks=tasks,
                    command=cmd,
                    note="No Slurm calls made; active-array checks happen at submission.",
                ),
                indent=2,
            )
        )
        return
    with campaign_lock(root):
        current, items, tasks = preview()
        if not tasks:
            print("No eligible unsubmitted runs.")
            return
        require_no_active_arrays(items)
        # A killed worker may have produced no marker; only explicitly requested retries
        # after terminal scheduler accounting may restart such an attempt.
        if args.retry:
            states = scheduler_states(items)
            for task in tasks:
                for receipt in items:
                    for index, old in enumerate(receipt.get("tasks", [])):
                        if receipt["status"] == "submitted" and old["id"] == task["id"]:
                            state = states.get(f"{receipt['job_id']}_{index}", {}).get("state")
                            require(
                                state in TERMINAL_STATES, "Retry has no terminal scheduler evidence"
                            )
        token = uuid.uuid4().hex
        selection = root / "submissions" / f"{token}.selection.json"
        atomic_json(selection, dict(tasks=tasks, token=token, stage=args.stage))
        cmd = sbatch_command(
            root, current, selection, tasks, args.concurrency or current["concurrency"], token
        )
        receipt = dict(
            token=token,
            stage=args.stage,
            tasks=tasks,
            created_utc=now(),
            status="intent",
            command=cmd,
            reason=args.reason,
            selection_sha256=sha(selection),
        )
        path = root / "submissions" / f"{token}.json"
        atomic_json(path, receipt)
        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode:
            receipt.update(status="rejected", stderr=process.stderr, stdout=process.stdout)
            atomic_json(path, receipt)
            raise RuntimeError(f"sbatch rejected submission: {process.stderr}")
        job_id = process.stdout.strip().split(";")[0]
        require(job_id.isdigit(), f"Ambiguous sbatch response; intent {token} needs reconciliation")
        receipt.update(status="submitted", job_id=job_id)
        atomic_json(path, receipt)
        print(json.dumps(receipt, indent=2))


def worker(args):
    root = args.root.resolve()
    manifest = load_campaign(root)
    activate_source(root / "source/src")
    selection_path = args.selection.resolve()
    require(selection_path.parent == root / "submissions", "Selection must belong to this campaign")
    selection = read_json(selection_path)
    receipt = read_json(root / "submissions" / f"{selection['token']}.json")
    require(sha(selection_path) == receipt["selection_sha256"], "Selection changed")
    require(receipt["status"] in ("intent", "submitted"), "Submission was rejected")
    require(0 <= args.task < len(selection["tasks"]), "Invalid array task index")
    task = selection["tasks"][args.task]
    spec = manifest["runs"][task["index"]]
    require(spec["id"] == task["id"], "Task/run ID mismatch")
    output = root / "experiments" / spec["id"] / f"attempt-{task['attempt']:03d}"
    output.mkdir(parents=True, exist_ok=False)
    launch = dict(
        started_utc=now(),
        run_id=spec["id"],
        attempt=task["attempt"],
        job_id=os.environ.get("SLURM_JOB_ID"),
        array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
    )
    atomic_json(output / "launch.json", launch)
    try:
        from tiny_llm.config import load_config
        from tiny_llm.train import train

        require(
            {name: importlib.metadata.version(name) for name in PACKAGES} == manifest["versions"],
            "Worker dependency versions differ from prepared campaign",
        )
        config = load_config(root / spec["config_file"])
        require(
            config.training.checkpoint_policy == "none"
            and not config.training.save_epoch_training_state,
            "All worker checkpoints must be disabled",
        )
        config.runtime.output_dir = output
        result = train(config)
        require(
            result["status"] == "complete", f"Training ended {result['status']}; retry from scratch"
        )
        validated = validate_result(
            output,
            spec["config"],
            spec["recipe_identity"],
            manifest["source_hash"],
            manifest["cache_identity"],
            manifest["versions"],
        )
        atomic_json(
            output / "success.json",
            launch
            | dict(
                finished_utc=now(),
                loss=validated["loss"],
                artifact_hashes={p.name: sha(p) for p in artifact_paths(output)},
            ),
        )
        print(f"Validated {spec['id']}: loss={validated['loss']:.9f}", flush=True)
    except BaseException as error:
        numerical = isinstance(error, FloatingPointError) or any(
            word in str(error).lower() for word in ("nonfinite", "non-finite")
        )
        atomic_json(
            output / "failure.json",
            launch
            | dict(
                finished_utc=now(),
                status="numerical_failure" if numerical else "failed",
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            ),
        )
        raise


def status(args):
    root = args.root.resolve()
    manifest = load_campaign(root)
    records, items = local_records(root, manifest), receipts(root)
    assigned = submitted_assignments(items)
    stages = {}
    for stage in manifest["phases"]:
        stages[stage] = dict(
            Counter(
                "submitted" if r["status"] == "pending" and assigned[r["id"]] else r["status"]
                for r in records
                if r["stage"] == stage
            )
        )
    result: dict = dict(
        stages=stages,
        historical_runs=len(manifest["historical"]),
        unresolved_intents=[r["token"] for r in items if r["status"] == "intent"],
        failures=[
            {k: r.get(k) for k in ("id", "status", "error", "output")}
            for r in records
            if r["status"] in ("failed", "numerical_failure")
        ],
    )
    if args.scheduler:
        result["scheduler"] = scheduler_states(items)
    print(json.dumps(result, indent=2))


def write_tables(directory, name, rows):
    atomic_json(directory / f"{name}.json", rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (directory / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def confirmation_results(manifest, records):
    observations = {(r["recipe"], r["seed"]): r for r in candidate_rows(records, (45, 46, 47))}
    results = []
    for decay, comparison in manifest.get("confirmation", {}).items():
        rows, deltas = [], []
        for seed in (45, 46, 47):
            smaller = observations.get((comparison["smaller"]["recipe"], seed))
            control = observations.get((comparison["control"]["recipe"], seed))
            rows.append(
                dict(
                    seed=seed,
                    smaller_loss=smaller["loss"] if smaller else None,
                    control_loss=control["loss"] if control else None,
                    delta=smaller["loss"] - control["loss"] if smaller and control else None,
                )
            )
            if smaller and control:
                deltas.append(smaller["loss"] - control["loss"])
        result: dict = dict(
            weight_decay=float(decay),
            primary=comparison["primary"],
            smaller_batch=comparison["smaller"]["batch"],
            pairs=rows,
            complete=len(deltas) == 3,
        )
        if len(deltas) == 3:
            mean = statistics.mean(deltas)
            margin = 4.302652729911275 * statistics.stdev(deltas) / math.sqrt(3)
            low, high = mean - margin, mean + margin
            result.update(
                mean_delta=mean,
                ci95_low=low,
                ci95_high=high,
                interpretation="lower loss"
                if high < 0
                else "higher loss"
                if low > 0
                else "inconclusive",
            )
        results.append(result)
    return results


def heatmap_cells(selected, records):
    observations = candidate_rows(records, (42,))
    cells = []
    for reference in selected.values():
        baseline = next((r for r in observations if r["recipe"] == reference["recipe"]), None)
        if baseline is None:
            raise ValueError("Selected recipe has no seed42 heatmap reference")
        for row in observations:
            if (row["batch"], row["weight_decay"]) != (
                reference["batch"],
                reference["weight_decay"],
            ):
                continue
            for panel, varying in (
                ("lr", "lr"),
                ("beta1", "beta1"),
                ("beta2", "beta2"),
                ("half_life", "beta2"),
            ):
                if any(row[k] != reference[k] for k in ("lr", "beta1", "beta2") if k != varying):
                    continue
                value = (
                    half_life(row["batch"], row["beta2"]) if panel == "half_life" else row[varying]
                )
                cells.append(
                    dict(
                        panel=panel,
                        batch=row["batch"],
                        weight_decay=row["weight_decay"],
                        value=value,
                        loss=row["loss"],
                        reference_loss=baseline["loss"],
                        delta=row["loss"] - baseline["loss"],
                        selected=row["recipe"] == reference["recipe"],
                        seed=42,
                        recipe=row["recipe"],
                        source=row["kind"],
                    )
                )
    return cells


def heatmap_matrix(cells, panel, decay):
    import numpy as np

    rows = [r for r in cells if r["panel"] == panel and r["weight_decay"] == decay]

    # Inverting beta2 introduces ~1e-14 relative roundoff. Coalesce mathematically
    # identical token half-lives for display only; CSV and optimizer values stay exact.
    def coordinate(value):
        return float(f"{value:.10g}") if panel == "half_life" else value

    values = sorted({coordinate(r["value"]) for r in rows})
    batches = sorted(BATCHES)
    matrix = np.full((len(values), len(batches)), np.nan)
    stars = []
    for row in rows:
        y, x = values.index(coordinate(row["value"])), batches.index(row["batch"])
        require(math.isnan(matrix[y, x]), "Duplicate heatmap cell")
        matrix[y, x] = row["delta"]
        if row["selected"]:
            stars.append((x, y))
    return values, batches, matrix, stars


def heatmap_settings(cells, limit: float | None = DEFAULT_HEATMAP_LIMIT) -> dict:
    """Limit display colors only; preserve every measured loss and missing cell."""
    require(limit is None or (math.isfinite(limit) and limit > 0), "Invalid heatmap limit")
    deltas = [r["delta"] for r in cells]
    require(all(math.isfinite(v) for v in deltas), "Nonfinite heatmap difference")
    extent = limit if limit is not None else max([abs(v) for v in deltas] + [1e-6])
    figures = {}
    for decay in DECAYS:
        values = [r["delta"] for r in cells if r["weight_decay"] == decay]
        below, above = sum(v < -extent for v in values), sum(v > extent for v in values)
        figures[f"wd{decay:g}"] = dict(
            cells=len(values),
            below_scale=below,
            above_scale=above,
            colorbar_extend="both"
            if below and above
            else "min"
            if below
            else "max"
            if above
            else "neither",
        )
    return dict(
        mode="full-range" if limit is None else "fixed-limit",
        limit=extent,
        vmin=-extent,
        vcenter=0.0,
        vmax=extent,
        raw_delta_min=min(deltas, default=None),
        raw_delta_max=max(deltas, default=None),
        figures=figures,
        semantics="Display colors saturate; annotated values and exported data are unchanged. "
        "Counts refer to plotted cells; beta2 and half-life repeat the same measurements.",
    )


def beta1_diagnostics(selected, records):
    """Post-selection seed-42 diagnostics, never inputs to primary selection."""
    observations = candidate_rows(records, (42,))
    summaries = []
    for reference in sorted(selected.values(), key=lambda r: (r["weight_decay"], r["batch"])):
        rows = [
            r
            for r in observations
            if r["stage"] != "confirm"
            and r["beta1"] in BETA1S
            and all(r[k] == reference[k] for k in ("batch", "weight_decay", "lr", "beta2"))
        ]
        baseline = next((r for r in rows if r["recipe"] == reference["recipe"]), None)
        if baseline is None:
            raise ValueError("Selected recipe has no seed42 beta1 reference")
        best = min(rows, key=lambda r: (r["loss"], r["beta1"], r["recipe"]))
        tested = sorted({r["beta1"] for r in rows})
        summaries.append(
            dict(
                batch=reference["batch"],
                weight_decay=reference["weight_decay"],
                lr=reference["lr"],
                beta2=reference["beta2"],
                seed=42,
                seed_count=1,
                tested_beta1=tested,
                sweep_complete=tested == sorted(BETA1S),
                reference_recipe=reference["recipe"],
                reference_beta1=reference["beta1"],
                reference_loss=baseline["loss"],
                reference_source=baseline["kind"],
                best_recipe=best["recipe"],
                best_source=best["kind"],
                best_beta1=best["beta1"],
                best_loss=best["loss"],
                delta=best["loss"] - baseline["loss"],
                first_moment_half_life_targets=half_life(best["batch"], best["beta1"])
                if best["beta1"] > 0
                else 0.0,
                beta1_boundary=best["beta1"] in (min(BETA1S), max(BETA1S)),
            )
        )
    return summaries


def beta1_replication_selection(manifest, records):
    """Freeze one conditional beta1 winner per group without using replication seeds."""
    diagnostics = beta1_diagnostics(manifest.get("selected", {}), records)
    require(
        {group_key(r["batch"], r["weight_decay"]) for r in diagnostics}
        == {group_key(b, d) for b, d in itertools.product(BATCHES, DECAYS)},
        "Every batch/decay group needs a frozen primary recipe",
    )
    require(all(r["sweep_complete"] for r in diagnostics), "All beta1 sweeps must be complete")
    selections = {}
    for row in diagnostics:
        key = group_key(row["batch"], row["weight_decay"])
        value = recipe(
            row["batch"], row["lr"], row["beta2"], row["weight_decay"], row["best_beta1"]
        )
        require(recipe_key(value) == row["best_recipe"], "Diagnostic recipe mismatch")
        selections[key] = value | dict(
            recipe=row["best_recipe"],
            seed42_loss=row["best_loss"],
            reference_recipe=row["reference_recipe"],
            beta1_boundary=row["beta1_boundary"],
            lr_boundary=manifest.get("boundary_flags", {}).get(key, False),
        )
    return selections


def beta1_replication_results(manifest, records):
    frozen = manifest.get("beta1_replication", {}).get("recipes", {})
    observations = candidate_rows(records, (42, 43, 44))
    results = []
    for value in sorted(frozen.values(), key=lambda r: (r["weight_decay"], r["batch"])):
        rows = sorted(
            (r for r in observations if r["recipe"] == value["recipe"]), key=lambda r: r["seed"]
        )
        baseline = next((r for r in rows if r["seed"] == 42), None)
        require(
            baseline is not None and baseline["loss"] == value["seed42_loss"],
            "Frozen seed42 result changed",
        )
        seeds, losses = [r["seed"] for r in rows], [r["loss"] for r in rows]
        complete = seeds == [42, 43, 44]
        results.append(
            value
            | dict(
                seeds=seeds,
                losses=losses,
                sources=[r["kind"] for r in rows],
                complete=complete,
                mean_loss=statistics.mean(losses) if complete else None,
                std_loss=statistics.stdev(losses) if complete else None,
                replication_mean_loss=statistics.mean(losses[1:]) if complete else None,
            )
        )
    return results


def performance_tables(records, selected):
    import numpy as np

    measures = [r for r in records if r["status"] == "complete"]
    grouped = defaultdict(list)
    for row in measures:
        if row["kind"] == "current" and row["stage"] == "screen":
            grouped[(row["batch"], row["weight_decay"], row["environment_stratum"])].append(row)
    summaries = []
    fields = (
        "steady_targets_per_second",
        "training_targets_per_second",
        "elapsed_targets_per_second",
        "session_targets_per_second",
        "session_seconds",
        "full_validation_seconds",
        "intermediate_validation_seconds",
        "peak_memory_gib",
    )
    for (batch, decay, stratum), rows in sorted(grouped.items()):
        item: dict = dict(
            batch=batch, weight_decay=decay, environment_stratum=stratum, runs=len(rows)
        )
        for field in fields:
            values = [r[field] for r in rows if r.get(field) is not None]
            item.update(
                {
                    field + "_median": float(np.median(values)) if values else None,
                    field + "_q25": float(np.quantile(values, 0.25)) if values else None,
                    field + "_q75": float(np.quantile(values, 0.75)) if values else None,
                }
            )
        summaries.append(item)
    for row in summaries:
        reference = next(
            (
                r
                for r in summaries
                if r["batch"] == 128
                and r["weight_decay"] == row["weight_decay"]
                and r["environment_stratum"] == row["environment_stratum"]
            ),
            None,
        )
        for field in ("steady_targets_per_second", "session_targets_per_second"):
            value = row[field + "_median"]
            base = reference[field + "_median"] if reference else None
            row[field + "_relative_to_b128"] = value / base if value is not None and base else None
    selected_keys = {value["recipe"] for value in selected.values()}
    selected_rows = [
        r
        for r in measures
        if r["recipe"] in selected_keys and r["seed"] in (42, 43, 44) and r["role"] != "reference"
    ]
    return measures, summaries, selected_rows


def plot_reports(
    directory,
    selected,
    records,
    cells,
    confirmations,
    performance,
    *,
    partial=False,
    heatmap_limit: float | None = DEFAULT_HEATMAP_LIMIT,
    beta1_replicated=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    suffix = " — PARTIAL" if partial else ""

    def save(fig, name):
        for extension in ("pdf", "svg", "png"):
            fig.savefig(directory / f"{name}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    settings = heatmap_settings(cells, heatmap_limit)
    scale = settings["limit"]
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e5e7eb")
    labels = dict(lr="Learning rate", beta1="β₁", beta2="β₂", half_life="Second-moment half-life")
    for decay in DECAYS:
        fig, axes = plt.subplots(1, 4, figsize=(14, 5.4), layout="constrained")
        mapped = None
        for ax, panel in zip(axes, labels, strict=True):
            values, batches, matrix, stars = heatmap_matrix(cells, panel, decay)
            if values:
                mapped = ax.imshow(
                    matrix,
                    origin="lower",
                    aspect="auto",
                    cmap=cmap,
                    norm=norm,
                    interpolation="none",
                )
                ticks = [f"{v / 1e6:.4g}M" if panel == "half_life" else f"{v:.8g}" for v in values]
                ax.set_yticks(range(len(values)), ticks, fontsize=8)
                for x, y in stars:
                    ax.plot(x, y, marker="*", color="black", markersize=9)
                for y, row in enumerate(matrix):
                    for x, delta in enumerate(row):
                        if abs(delta) > scale:
                            ax.text(
                                x,
                                y,
                                f"{delta:+.3g}",
                                color="white",
                                fontsize=8,
                                ha="center",
                                va="center",
                            )
            else:
                ax.text(0.5, 0.5, "No selected recipe yet", ha="center", transform=ax.transAxes)
                ax.set_yticks([])
            ax.set_xticks(range(len(batches)), [str(b) for b in batches])
            ax.set_xlabel("Batch size")
            ax.set_title(labels[panel])
        if mapped is not None:
            fig.colorbar(
                mapped,
                ax=axes,
                shrink=0.8,
                label="Seed-42 loss − selected reference loss",
                extend=settings["figures"][f"wd{decay:g}"]["colorbar_extend"],
            )
        fig.suptitle(
            f"Conditional hyperparameter sensitivity · AdamW, decay {decay:g}{suffix}\n"
            "Other hyperparameters fixed; gray = untested; ★ = selected reference\n"
            f"Shared color limits ±{scale:.3g}; labels give actual Δ for saturated cells",
            fontsize=12,
        )
        save(fig, f"hyperparameter-heatmaps-wd{decay:g}")

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for decay, color in zip(DECAYS, ("#2563eb", "#c2410c"), strict=True):
        values = sorted(
            (v for v in selected.values() if v["weight_decay"] == decay), key=lambda v: v["batch"]
        )
        if not values:
            continue
        ax.errorbar(
            [v["batch"] for v in values],
            [v["mean_loss"] for v in values],
            yerr=[v["std_loss"] for v in values],
            marker="o",
            capsize=4,
            color=color,
            label=f"AdamW, decay {decay:g}",
        )
        for value in values:
            ax.scatter([value["batch"]] * 3, value["losses"], color=color, alpha=0.4, s=18)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(BATCHES), [str(b) for b in sorted(BATCHES)])
    ax.set(
        xlabel="Batch size (sequences)",
        ylabel="Final full-C4 validation loss",
        title=f"LR/β₂-tuned AdamW, β₁=0.9{suffix}\nSeeds 42–44: mean ± sample SD",
    )
    if selected:
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "Awaiting completed tuning and frozen selection",
            ha="center",
            transform=ax.transAxes,
        )
    ax.grid(alpha=0.2)
    save(fig, "tuned-loss-versus-batch")

    if beta1_replicated:
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        for decay, color in zip(DECAYS, ("#2563eb", "#c2410c"), strict=True):
            values = sorted(
                (r for r in beta1_replicated if r["weight_decay"] == decay),
                key=lambda r: r["batch"],
            )
            # NaNs break lines through groups with missing seeds; no partial means.
            ax.errorbar(
                [r["batch"] for r in values],
                [r["mean_loss"] if r["complete"] else math.nan for r in values],
                yerr=[r["std_loss"] if r["complete"] else math.nan for r in values],
                marker="o",
                capsize=4,
                color=color,
                label=f"AdamW, decay {decay:g}",
            )
            for row in values:
                ax.scatter(
                    [row["batch"]] * len(row["losses"]), row["losses"], color=color, alpha=0.4, s=18
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(BATCHES), [str(b) for b in sorted(BATCHES)])
        ax.set(
            xlabel="Batch size (sequences)",
            ylabel="Final full-C4 validation loss",
            title="Conditional β₁ tuning after LR/β₂ tuning"
            + suffix
            + "\nSeeds 42–44: mean ± sample SD; seed 42 selected β₁",
        )
        ax.legend()
        ax.grid(alpha=0.2)
        save(fig, "beta1-tuned-loss-versus-batch")

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    complete = [r for r in confirmations if r["complete"]]
    for index, result in enumerate(complete):
        points = [r["delta"] for r in result["pairs"]]
        ax.scatter([index] * 3, points, alpha=0.6, s=30)
        ax.errorbar(
            index,
            result["mean_delta"],
            yerr=[
                [result["mean_delta"] - result["ci95_low"]],
                [result["ci95_high"] - result["mean_delta"]],
            ],
            fmt="D",
            capsize=6,
            color="black",
        )
    ax.set_xticks(
        range(len(complete)),
        [f"Decay {r['weight_decay']:g}\nB{r['smaller_batch']} − B128" for r in complete],
    )
    ax.axhline(0, color="gray", linestyle="--")
    ax.set(
        ylabel="Paired loss difference (negative = improvement)",
        title=f"Independent confirmation, seeds 45–47{suffix}\nPaired differences and descriptive 95% t interval",
    )
    if not complete:
        ax.text(
            0.5, 0.5, "Awaiting all three fresh-seed pairs", ha="center", transform=ax.transAxes
        )
    save(fig, "fresh-seed-confirmation")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, field, title in zip(
        axes,
        ("steady_targets_per_second", "session_targets_per_second"),
        ("Steady training", "Session including validation"),
        strict=True,
    ):
        groups = sorted({(r["weight_decay"], r["environment_stratum"]) for r in performance})
        for decay, stratum in groups:
            rows = sorted(
                (
                    r
                    for r in performance
                    if r["weight_decay"] == decay
                    and r["environment_stratum"] == stratum
                    and r[field + "_median"] is not None
                ),
                key=lambda r: r["batch"],
            )
            if not rows:
                continue
            values = [r[field + "_median"] / 1e6 for r in rows]
            errors = [
                [(r[field + "_median"] - r[field + "_q25"]) / 1e6 for r in rows],
                [(r[field + "_q75"] - r[field + "_median"]) / 1e6 for r in rows],
            ]
            line = ax.errorbar(
                [r["batch"] for r in rows],
                values,
                yerr=errors,
                marker="o",
                capsize=4,
                label=f"Decay {decay:g}, {stratum[:6]}",
            )
            samples = [
                r
                for r in records
                if r["status"] == "complete"
                and r["kind"] == "current"
                and r["stage"] == "screen"
                and r["weight_decay"] == decay
                and r["environment_stratum"] == stratum
                and r.get(field) is not None
            ]
            ax.scatter(
                [r["batch"] for r in samples],
                [r[field] / 1e6 for r in samples],
                color=line[0].get_color(),
                alpha=0.2,
                s=10,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(BATCHES), [str(b) for b in sorted(BATCHES)])
        ax.set(xlabel="Batch size", ylabel="Million training targets/s", title=title)
        ax.grid(alpha=0.2)
        if performance:
            ax.legend(fontsize=8)
    fig.suptitle(f"Current-source screening throughput: median and IQR{suffix}")
    save(fig, "throughput-versus-batch")


def report_markdown(
    manifest,
    records,
    selected,
    confirmations,
    performance,
    complete,
    settings,
    diagnostics,
    reporter,
    beta1_replicated=None,
):
    counts = Counter(r["status"] for r in records if r["kind"] == "current")
    lines = [
        "# Small-batch AdamW study",
        "",
        "**Complete**" if complete else "**PARTIAL — campaign is not complete**",
        "",
        f"Generated {now()}. Current-run statuses: `{dict(counts)}`.",
        "",
        "Context 512; 408,092,672 training targets; 197,411,295 final validation targets. "
        "No weights or checkpoints are saved. Smaller loss is a hypothesis, not a required outcome.",
        "",
        "## Tuned results",
        "",
        "LR/β₂ tuning with β₁=0.9. Seeds 42–44 were used for selection; "
        "error bars show sample SD, not uncertainty corrected for selection.",
        "",
        "| Batch | Decay | LR | β₂ | Half-life (targets) | Mean loss | Sample SD | Seed losses |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for value in sorted(selected.values(), key=lambda r: (r["weight_decay"], r["batch"])):
        lines.append(
            f"| {value['batch']} | {value['weight_decay']:g} | {value['lr']:g} | "
            f"{value['beta2']:.10g} | {half_life(value['batch'], value['beta2']):.0f} | "
            f"{value['mean_loss']:.6f} | {value['std_loss']:.6f} | "
            + ", ".join(f"{loss:.6f}" for loss in value["losses"])
            + " |"
        )
    if not selected:
        lines += ["", "Selections have not been frozen; no final tuned comparison is available."]
    lines += [
        "",
        "![Tuned loss](tuned-loss-versus-batch.png)",
        "",
        "## Independent confirmation",
        "",
        "Fresh seeds 45–47; decay 0.1 is primary. Differences are smaller batch minus B128. "
        "The paired 95% t interval has two degrees of freedom and limited precision. "
        "These are validation results, not an independent test-set estimate.",
    ]
    for result in confirmations:
        if result["complete"]:
            lines += [
                "",
                f"Decay {result['weight_decay']:g}, B{result['smaller_batch']}: "
                f"mean Δ={result['mean_delta']:+.6f}, 95% interval "
                f"[{result['ci95_low']:+.6f}, {result['ci95_high']:+.6f}]: "
                f"**{result['interpretation']}**.",
            ]
        else:
            lines += ["", f"Decay {result['weight_decay']:g}: incomplete; no inference."]
        lines += [
            "",
            "| Seed | Smaller-batch loss | B128 loss | Difference |",
            "| ---: | ---: | ---: | ---: |",
        ]
        for pair in result["pairs"]:
            values = [
                "unavailable" if pair[k] is None else f"{pair[k]:.6f}"
                for k in ("smaller_loss", "control_loss", "delta")
            ]
            lines.append(f"| {pair['seed']} | " + " | ".join(values) + " |")
    lines += [
        "",
        "![Fresh-seed confirmation](fresh-seed-confirmation.png)",
        "",
        "## First-moment sensitivity: diagnostic only",
        "",
        "The primary comparison fixed β₁=0.9 and tuned LR/β₂. This table instead shows the "
        "best tested β₁ at each selected recipe's fixed LR and β₂, using only seed 42. "
        "These minima are selected on one seed and are not replicated estimates; "
        "they do not replace the frozen recipes, tuned curves, or fresh-seed confirmation.",
        "",
        "| Batch | Decay | β₁=0.9 loss | Best tested β₁ | Diagnostic loss | Δ versus reference | "
        "First-moment half-life (targets) | Sweep status |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in diagnostics:
        status = "complete" if row["sweep_complete"] else "incomplete"
        if row["beta1_boundary"]:
            status += "; best at β₁ endpoint"
        lines.append(
            f"| {row['batch']} | {row['weight_decay']:g} | {row['reference_loss']:.6f} | "
            f"{row['best_beta1']:g} | {row['best_loss']:.6f} | {row['delta']:+.6f} | "
            f"{row['first_moment_half_life_targets']:.0f} | {status} |"
        )
    lines += [
        "",
        "For β₁>0, the first-moment half-life in targets is −512·batch·ln(2)/ln(β₁); "
        "β₁=0 retains no first-moment history. Fixing β₁ while reducing batch shortens this "
        "token horizon. Holding the B128/β₁=0.9 horizon constant would give "
        "β₁≈0.94868, 0.97400, 0.98692 at batches 64, 32, 16. "
        "This is an interpretation of the diagnostic sweep, not a verified scaling rule. "
        "A follow-up would freeze the promising β₁ recipes and compare them with a B128 "
        "control on fresh paired seeds, keeping these exploratory results separate.",
    ]
    if beta1_replicated:
        lines += [
            "",
            "## Replicated conditional β₁ tuning",
            "",
            "One best seed-42 β₁ per batch/decay was frozen before replication on seeds 43–44. "
            "LR/β₂ stay at the original selected values. Existing valid observations are reused, "
            "including the historical B128 decay-0.1 control. This is sequential conditional "
            "tuning, not a joint grid search or fresh-seed confirmation. No recipe is reselected "
            "after seeing these replications. Means require all three seeds; error bars are sample SD. "
            "The 43–44 mean excludes the β₁ selection seed but uses seeds involved in earlier LR/β₂ tuning.",
            "",
            "| Batch | Decay | LR | β₁ | β₂ | Mean 42–44 | Sample SD | Mean 43–44 | Seed losses |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in beta1_replicated:
            stats = [
                "pending" if row[k] is None else f"{row[k]:.6f}"
                for k in ("mean_loss", "std_loss", "replication_mean_loss")
            ]
            points = ", ".join(
                f"{seed}: {loss:.6f}"
                for seed, loss in zip(row["seeds"], row["losses"], strict=True)
            )
            lines.append(
                f"| {row['batch']} | {row['weight_decay']:g} | {row['lr']:g} | {row['beta1']:g} | "
                f"{row['beta2']:.10g} | " + " | ".join(stats) + f" | {points} |"
            )
        lines += [
            "",
            "![Replicated beta1 tuning](beta1-tuned-loss-versus-batch.png)",
            "",
            "`beta1-replicated.csv/json` records the frozen recipes, individual seed losses, "
            "source labels, and remaining LR/β₁ endpoint flags. Missing or failed seeds "
            "leave means unavailable. The original β₁=0.9 comparison and confirmation stay separate.",
        ]
    lines += [
        "",
        "## Hyperparameter heatmaps",
        "",
        "Conditional seed-42 measurements, fixing other settings at each selected recipe. "
        "Colors subtract that recipe's seed-42 loss; negative values improve it. "
        "The shared scale is centered at zero. Gray cells are untested, with no interpolation. "
        "β₂ and half-life show the same measurements in different coordinates. "
        "The β₁ sweep is diagnostic and does not change the primary tuning selection.",
        "",
        f"All panels and both decay figures share color limits ±{settings['limit']:.6g} "
        f"({settings['mode']}). Values outside those limits retain their measured losses "
        "in CSV/JSON, use saturated colors, and carry signed numerical Δ labels "
        "(three significant digits). Colorbar extensions indicate values beyond the scale. "
        "This display choice reveals small differences without removing large-loss runs. "
        "`heatmap-settings.json` records the scale, raw range, and saturation counts. "
        "Use `collect --heatmap-limit VALUE` to change the range or "
        "`collect --heatmap-full-range` to cover all measured differences.",
        "",
        "![Decay 0.1](hyperparameter-heatmaps-wd0.1.png)",
        "",
        "![Decay zero](hyperparameter-heatmaps-wd0.png)",
        "",
        "## Throughput",
        "",
        "Current-source screening runs, separated by decay and environment/source stratum. "
        "Values are medians; CSV/JSON also contains individual runs and IQRs. "
        "Relative rates use only a B128 reference within the same stratum.",
        "",
        "| Batch | Decay | Stratum | Runs | Steady targets/s | Session targets/s | "
        "Steady/B128 | Session seconds | Full validation seconds | Peak GiB |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in performance:
        names = (
            "steady_targets_per_second_median",
            "session_targets_per_second_median",
            "steady_targets_per_second_relative_to_b128",
            "session_seconds_median",
            "full_validation_seconds_median",
            "peak_memory_gib_median",
        )
        values = ["unavailable" if row[k] is None else f"{row[k]:.3f}" for k in names]
        lines.append(
            f"| {row['batch']} | {row['weight_decay']:g} | {row['environment_stratum']} | "
            f"{row['runs']} | " + " | ".join(values) + " |"
        )
    lines += [
        "",
        "![Throughput](throughput-versus-batch.png)",
        "",
        "Steady rate = total targets / total seconds for complete logging windows starting "
        "at or after 20,447,232 targets; crossing windows are excluded. Recorded training rate "
        "includes compilation. Elapsed training rate includes intermediate validation. "
        "Session rate additionally includes subset collection and final validation, but excludes "
        "earlier model setup, launcher overhead, and Slurm queue time. Peak memory includes "
        "evaluation. Historical and selected-recipe measurements are separate in the data tables.",
        "",
        "## Diagnostics and provenance",
        "",
        f"Frozen source hash: `{manifest['source_hash']}`. Cache: `{manifest['cache_identity']}`.",
        "",
        f"Reporting script SHA256: `{reporter['script_sha256']}`. "
        "The reporting script may differ from the frozen training driver. "
        "`validation.json` records its identity, plotting versions, and reproduction command. "
        "Re-rendering does not modify training artifacts or selections.",
        "",
        "[Paper v4](https://arxiv.org/pdf/2507.07101v4) · "
        f"[Official code revision](https://github.com/martin-marek/batch-size/tree/{UPSTREAM}). "
        "This tests transfer to the local model. The headline upstream experiment differs in "
        "architecture, data, precision and clipping; the official C4 experiment retains decay 0.1. "
        "See the campaign README for the adoption audit and exact timing boundaries.",
    ]
    reference = next(
        (r for r in records if r["role"] == "reference" and r["status"] == "complete"), None
    )
    if reference:
        old = next(
            r
            for r in records
            if r["kind"] == "historical" and r["recipe"] == reference["recipe"] and r["seed"] == 42
        )
        lines += [
            "",
            f"Fresh B128 seed-42 reference loss: {reference['loss']:.9f}; "
            f"historical: {old['loss']:.9f}; difference: {reference['loss'] - old['loss']:+.9f}. "
            "CUDA execution is nondeterministic; bitwise equality is not required. "
            "A single repeat cannot estimate run-to-run variability. Both arms of fresh-seed "
            "confirmation use current-source runs; neither reuses historical losses.",
        ]
    flags = [key for key, flag in manifest.get("boundary_flags", {}).items() if flag]
    lines += [
        "",
        f"Remaining LR-boundary winners: {', '.join(flags) if flags else 'none recorded'}.",
        "",
        "`runs.csv/json` retains failures and missing runs. `throughput-runs` contains "
        "individual valid measurements and clipping frequency (unavailable in historical logs). "
        "`throughput-selected` identifies the selected recipes' seeds 42–44. "
        "`heatmap-cells`, `selected-recipes`, and `confirmation` are the exact plotted data.",
        "`beta1-diagnostics` retains each diagnostic minimum, its reference, source, "
        "tested values, and endpoint flags. It is not a new selection stage.",
    ]
    return "\n".join(lines) + "\n"


def collect(args):
    root = args.root.resolve()
    with campaign_lock(root):
        manifest = load_campaign(root)
        check_no_checkpoints(root)
        records = all_records(root, manifest)
        selected = manifest.get("selected", {})
        confirmations = confirmation_results(manifest, records)
        beta1_replicated = beta1_replication_results(manifest, records)
        complete = all(stage in manifest["phases"] for stage in STAGES)
        complete = complete and all(
            r["status"] in ("complete", "numerical_failure")
            for r in records
            if r["kind"] == "current"
        )
        complete = (
            complete and len(confirmations) == 2 and all(r["complete"] for r in confirmations)
        )
        complete = complete and all(r["complete"] for r in beta1_replicated)
        require(
            complete or args.allow_incomplete,
            "Campaign incomplete; use --allow-incomplete for an explicitly partial report",
        )
        cells = heatmap_cells(selected, records)
        settings = heatmap_settings(cells, args.heatmap_limit)
        diagnostics = beta1_diagnostics(selected, records)
        reporter = dict(
            script_sha256=sha(__file__),
            python=sys.version,
            versions={name: importlib.metadata.version(name) for name in ("matplotlib", "numpy")},
            command=shlex.join(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "collect",
                    "--root",
                    str(root),
                    *(["--allow-incomplete"] if args.allow_incomplete else []),
                    *(
                        ["--heatmap-full-range"]
                        if args.heatmap_limit is None
                        else [
                            "--heatmap-limit",
                            str(args.heatmap_limit),
                        ]
                    ),
                ]
            ),
        )
        measures, performance, selected_measures = performance_tables(records, selected)
        destination = root / "reports"
        # Publish the entire report directory together, so a failed renderer leaves
        # the previous report intact. Old reports are retained under ignored runs/.
        temporary = Path(tempfile.mkdtemp(prefix=".report-", dir=root))
        try:
            for name, rows in (
                ("runs", records),
                ("selected-recipes", list(selected.values())),
                ("confirmation", confirmations),
                ("heatmap-cells", cells),
                ("beta1-diagnostics", diagnostics),
                ("beta1-replicated", beta1_replicated),
                ("throughput-runs", measures),
                ("throughput-summary", performance),
                ("throughput-selected", selected_measures),
            ):
                write_tables(temporary, name, rows)
            plot_reports(
                temporary,
                selected,
                records,
                cells,
                confirmations,
                performance,
                partial=not complete,
                heatmap_limit=args.heatmap_limit,
                beta1_replicated=beta1_replicated,
            )
            atomic_json(temporary / "heatmap-settings.json", settings)
            (temporary / "report.md").write_text(
                report_markdown(
                    manifest,
                    records,
                    selected,
                    confirmations,
                    performance,
                    complete,
                    settings,
                    diagnostics,
                    reporter,
                    beta1_replicated,
                )
            )
            atomic_json(
                temporary / "validation.json",
                dict(
                    generated_utc=now(),
                    complete=complete,
                    no_checkpoints=True,
                    counts=dict(Counter(r["status"] for r in records)),
                    source_hash=manifest["source_hash"],
                    cache_identity=manifest["cache_identity"],
                    reporter=reporter,
                    heatmap_settings=settings,
                    configuration_hashes={r["id"]: r["config_sha256"] for r in manifest["runs"]},
                ),
            )
            if destination.exists():
                destination.rename(root / f"reports-previous-{uuid.uuid4().hex[:8]}")
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        print(f"Wrote {'complete' if complete else 'PARTIAL'} report: {destination / 'report.md'}")


def self_test(args):
    """Embedded tests keep the script the only task-related trackable change."""
    import signal
    import unittest
    from unittest.mock import patch

    import numpy as np

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    activate_source(repository() / "src")
    import torch

    from tiny_llm.config import Config, ModelConfig, TrainingConfig, load_config
    from tiny_llm.data import BufferedTokenLoader, TokenCache, TokenWriter, fingerprint
    from tiny_llm.train import learning_rate

    torch.set_num_threads(1)
    test_root = Path(tempfile.mkdtemp(prefix="tiny-llm-batch-study-self-test-"))
    module = sys.modules[__name__]
    training_module = importlib.import_module("tiny_llm.train")
    base = load_config([repository() / "configs/20m.yaml", repository() / "configs/cosine.yaml"])
    base_raw = base.model_dump(mode="json")

    def observation(value, seed=42, stage="screen", loss=3.5, kind="current"):
        key = recipe_key(value)
        return value | dict(
            id=f"{key}-s{seed}",
            recipe=key,
            seed=seed,
            stage=stage,
            role="candidate",
            kind=kind,
            status="complete",
            loss=loss,
            environment_stratum="test",
            steady_targets_per_second=1e6,
            training_targets_per_second=9e5,
            elapsed_targets_per_second=8e5,
            session_targets_per_second=7e5,
            session_seconds=500,
            full_validation_seconds=100,
            intermediate_validation_seconds=20,
            peak_memory_gib=10,
            clipping_frequency=0.05,
        )

    def tiny_cache(directory):
        config = Config(
            model=ModelConfig(
                vocab_size=17, layers=1, width=8, heads=2, ffn_width=16, context_length=4
            )
        )
        config.runtime.training_backend = "native"
        config.runtime.device = "cpu"
        config.runtime.amp = config.runtime.compile = False
        config.runtime.cpu_threads = 1
        config.runtime.output_dir = directory / "run"
        config.data.cache_dir = directory / "cache"
        config.data.shard_tokens = 11
        config.data.prefetch = False
        config.lr_schedule.warmup_steps = 0
        config.training = TrainingConfig(
            tokens_per_parameters=0.08,
            epoch_tokens_per_parameters=0.04,
            batch_tokens=16,
            micro_batch_size=4,
            log_every=1,
            checkpoint_policy="none",
            save_epoch_training_state=False,
        )
        config.evaluation.subset_blocks, config.evaluation.batch_size = 3, 2
        config.data.cache_dir.mkdir(parents=True)
        manifest: dict = dict(
            version=1,
            dataset=config.data.dataset,
            dataset_revision="fixture-dataset",
            tokenizer=config.data.tokenizer,
            tokenizer_revision="fixture-tokenizer",
            tokenizer_fingerprint="fixture",
            vocab_size=17,
            eos_token_id=2,
            dtype="<u2",
            preprocessing=dict(
                shuffle_seed=config.data.shuffle_seed,
                shuffle_buffer=config.data.shuffle_buffer,
                append_eos=True,
                add_special_tokens=False,
            ),
            validation_complete=True,
            splits={},
        )
        for split, count in (("train", 133), ("validation", 30)):
            writer = TokenWriter(config.data.cache_dir, split, shard_tokens=11)
            writer.write(np.arange(count) % 17)
            manifest["splits"][split] = writer.finish()
        atomic_json(
            config.data.cache_dir / "manifest.json", manifest | dict(identity=fingerprint(manifest))
        )
        return config

    class StudyTests(unittest.TestCase):
        def test_exact_protocol_and_stable_ids(self):
            for batch, updates, warmup in zip(
                BATCHES, (6227, 12454, 24908, 49816), (312, 624, 1248, 2496), strict=True
            ):
                value = recipe(batch, 0.004, 0.98 ** (batch / 128), 0.1)
                cfg = build_config(base_raw, value, 42, test_root / "never-trained")
                self.assertEqual(TOKENS // cfg.training.batch_tokens, updates)
                self.assertEqual(cfg.lr_schedule.warmup_steps, warmup)
                self.assertEqual(cfg.training.micro_batch_size, batch)
                self.assertEqual(cfg.training.checkpoint_policy, "none")
                self.assertFalse(cfg.training.save_epoch_training_state)
                self.assertAlmostEqual(
                    half_life(batch, value["beta2"]), half_life(128, 0.98), places=6
                )
                self.assertAlmostEqual(
                    half_life(batch, beta_policies(batch, 0.1)["half-life-10m"]), 1e7, places=5
                )
                self.assertAlmostEqual(learning_rate(cfg, WARMUP_TOKENS // 2, TOKENS), 0.002)
                self.assertEqual(learning_rate(cfg, TOKENS, TOKENS), 0)
                self.assertEqual(recipe_key(value), recipe_key(dict(reversed(list(value.items())))))
            configs = screen_recipes()
            self.assertEqual(len(configs), 114)
            historical = {recipe_key(recipe(128, lr, 0.98, 0.1)) for lr in LRS}
            self.assertEqual(len([c for c in configs if recipe_key(c) not in historical]) + 1, 109)

        def test_batch_independent_sample_order(self):
            config = tiny_cache(test_root / "order")
            cache = TokenCache(config.data.cache_dir)
            streams = []
            for batch in (1, 2, 4, 8):
                loader = BufferedTokenLoader(cache, "train", 4, 24, 2, seed=42, prefetch=False)
                try:
                    targets = [
                        loader.next_batch(batch, torch.device("cpu"))[1] for _ in range(24 // batch)
                    ]
                    streams.append(torch.cat(targets))
                finally:
                    loader.close()
            for stream in streams[1:]:
                torch.testing.assert_close(stream, streams[0], atol=0, rtol=0)

        def test_checkpoint_free_completion_and_interruption(self):
            config = tiny_cache(test_root / "cpu")
            metadata = dict(source_hash="fixture", versions={"torch": torch.__version__}, gpu=None)
            forbidden = AssertionError("Checkpoint/weights writer was called")
            with (
                patch.object(training_module, "environment", return_value=metadata),
                patch.object(training_module, "atomic_checkpoint", side_effect=forbidden),
                patch.object(training_module, "save_weights", side_effect=forbidden),
                patch.object(training_module, "save_file", side_effect=forbidden),
            ):
                result = training_module.train(config)
                self.assertEqual(result["status"], "complete")
                self.assertTrue(result["final_validation"]["validation_complete"])
                self.assertTrue(math.isfinite(result["final_validation"]["loss"]))
                self.assertIsNone(read_json(config.runtime.output_dir / "best.json")["weights"])
                self.assertTrue((config.runtime.output_dir / "result.json").exists())
                check_no_checkpoints(config.runtime.output_dir)
                config.runtime.output_dir = test_root / "cpu-interrupted"
                original = training_module.evaluate

                def interrupt(*a, **kw):
                    os.kill(os.getpid(), signal.SIGTERM)
                    return original(*a, **kw)

                with patch.object(training_module, "evaluate", side_effect=interrupt):
                    interrupted = training_module.train(config)
                self.assertEqual(interrupted["status"], "interrupted")
                check_no_checkpoints(config.runtime.output_dir)

        def test_weighted_timing_and_invalid_windows(self):
            events = [
                dict(event="train", step=1, tokens=8, training_seconds=5),
                dict(event="train", step=2, tokens=16, training_seconds=10),
                dict(event="validation", tokens=16, seconds=100),
                dict(event="train", step=3, tokens=24, training_seconds=2),
                dict(event="train", step=4, tokens=40, training_seconds=8),
            ]
            self.assertEqual(steady_rate(events, threshold=10), 24 / 10)
            self.assertEqual(steady_rate(events, threshold=16), 24 / 10)
            self.assertIsNone(steady_rate(events, threshold=41))
            for altered in (
                [events[0], events[0]],
                [events[0] | {"training_seconds": 0}],
                [events[0] | {"training_seconds": float("nan")}],
            ):
                with self.assertRaises(ValueError):
                    steady_rate(altered)

        def test_rank_complete_groups_and_reject_nonfinite(self):
            value = recipe(16, 0.004, 0.999, 0.1)
            rows = [observation(value, seed=s, loss=3.5 + (s - 42) * 0.01) for s in (42, 43, 44)]
            self.assertFalse(ranked_groups(rows[:2]))
            self.assertEqual(len(ranked_groups(rows)), 1)
            self.assertAlmostEqual(ranked_groups(rows)["b16-wd0.1"][0]["mean_loss"], 3.51)
            with self.assertRaises(ValueError):
                ranked_groups(rows + [rows[0]])
            with self.assertRaises(ValueError):
                ranked_groups([rows[0] | {"loss": float("nan")}])
            self.assertFalse(ranked_groups([r | {"status": "failed"} for r in rows]))

        def test_staged_promotions_and_fresh_seed_isolation(self):
            directory = test_root / "stages"
            directory.mkdir()
            manifest: dict = dict(
                runs=[], historical=[], phases={"screen": {}}, extension_checked=False
            )
            rows = []
            for batch, decay, lr in itertools.product(BATCHES, DECAYS, (0.002, 0.004, 0.006)):
                row = observation(
                    recipe(batch, lr, 0.98 ** (batch / 128), decay),
                    loss=3.5 + abs(lr - 0.004) + batch / 100000,
                )
                rows.append(row)
                manifest["runs"].append(row)

            def fake_add(m, root, value, seed, stage, **kw):
                key = recipe_key(value)
                if any(r["recipe"] == key and r["seed"] == seed for r in m["runs"]):
                    return None
                row = observation(
                    recipe(
                        value["batch"],
                        value["lr"],
                        value["beta2"],
                        value["weight_decay"],
                        value["beta1"],
                    ),
                    seed=seed,
                    stage=stage,
                    loss=3.5 + abs(value["lr"] - 0.004) + value["batch"] / 100000,
                )
                m["runs"].append(row)
                rows.append(row)
                return row["id"]

            with (
                patch.object(module, "activate_source"),
                patch.object(module, "load_campaign", return_value=manifest),
                patch.object(module, "all_records", return_value=rows),
                patch.object(module, "add_run", side_effect=fake_add),
            ):
                promote(argparse.Namespace(root=directory, stage="replicate"))
                self.assertEqual(sum(r["stage"] == "replicate" for r in rows), 32)
                promote(argparse.Namespace(root=directory, stage="confirm"))
                frozen = copy.deepcopy(manifest["selected"])
                self.assertEqual(sum(r["stage"] == "confirm" for r in rows), 12)
                self.assertEqual({r["seed"] for r in rows if r["stage"] == "confirm"}, {45, 46, 47})
                promote(argparse.Namespace(root=directory, stage="sensitivity"))
                self.assertEqual(sum(r["stage"] == "sensitivity" for r in rows), 56)
                self.assertEqual(manifest["selected"], frozen)
                self.assertEqual({r["beta1"] for r in frozen.values()}, {0.9})
                previous = len(rows)
                promote(argparse.Namespace(root=directory, stage="confirm"))
                self.assertEqual(len(rows), previous)
                for row in rows:
                    if row["stage"] == "sensitivity":
                        improved = row["beta1"] == 0.96 and (row["batch"], row["weight_decay"]) != (
                            128,
                            0.1,
                        )
                        row["loss"] += -0.02 if improved else 0.02
                with self.assertRaises(ValueError):
                    beta1_replication_selection(manifest, [r for r in rows if r["beta1"] != 0.7])
                promote(
                    parser().parse_args(
                        ["promote", "--root", str(directory), "--stage", BETA1_REPLICATION]
                    )
                )
                self.assertEqual(sum(r["stage"] == BETA1_REPLICATION for r in rows), 14)
                self.assertEqual(manifest["selected"], frozen)
                selections = copy.deepcopy(manifest["beta1_replication"])
                promote(argparse.Namespace(root=directory, stage=BETA1_REPLICATION))
                self.assertEqual(sum(r["stage"] == BETA1_REPLICATION for r in rows), 14)
                self.assertEqual(manifest["beta1_replication"], selections)
                replicated = beta1_replication_results(manifest, rows)
                self.assertEqual(len(replicated), 8)
                self.assertTrue(
                    all(r["complete"] and r["seeds"] == [42, 43, 44] for r in replicated)
                )
                for row in replicated:
                    self.assertAlmostEqual(row["std_loss"], statistics.stdev(row["losses"]))
                    self.assertAlmostEqual(row["replication_mean_loss"], sum(row["losses"][1:]) / 2)
                missing = beta1_replication_results(manifest, [r for r in rows if r["seed"] != 44])
                self.assertTrue(all(not r["complete"] and r["mean_loss"] is None for r in missing))
                corrupted = [
                    r | dict(loss=math.nan) if r["stage"] == BETA1_REPLICATION else r for r in rows
                ]
                with self.assertRaises(ValueError):
                    beta1_replication_results(manifest, corrupted)
                submission = parser().parse_args(
                    ["submit", "--stage", BETA1_REPLICATION, "--dry-run"]
                )
                self.assertEqual(submission.concurrency, 48)
            results = confirmation_results(manifest, rows)
            self.assertTrue(all(r["complete"] for r in results))
            self.assertTrue(all(r["interpretation"] == "lower loss" for r in results))
            self.assertFalse(
                confirmation_results(manifest, [r for r in rows if r["seed"] != 47])[0]["complete"]
            )

        def test_boundary_extension_once(self):
            directory = test_root / "boundary"
            directory.mkdir()
            rows = [
                observation(recipe(b, lr, 0.98 ** (b / 128), d), loss=3.5 + lr)
                for b, d, lr in itertools.product(BATCHES, DECAYS, (0.002, 0.004))
            ]
            manifest: dict = dict(phases={"screen": {}}, extension_checked=False)
            additions = []
            with (
                patch.object(module, "activate_source"),
                patch.object(module, "load_campaign", return_value=manifest),
                patch.object(module, "all_records", return_value=rows),
                patch.object(
                    module,
                    "add_run",
                    side_effect=lambda m, p, v, s, t: additions.append(v) or recipe_key(v),
                ),
            ):
                promote(argparse.Namespace(root=directory, stage="replicate"))
            self.assertTrue(manifest["extension_checked"])
            self.assertEqual(len(additions), 8)
            self.assertEqual({r["lr"] for r in additions}, {0.001})
            self.assertNotIn("replicate", manifest["phases"])

        def test_submission_default_dry_run_and_retry_dedup(self):
            directory = test_root / "dry-run"
            directory.mkdir()
            (directory / "job.sh").write_text("fixture")
            spec = observation(recipe(128, 0.008, 0.98, 0.1)) | dict(index=0, role="reference")
            row = spec | dict(status="pending")
            manifest: dict = dict(
                runs=[spec],
                phases={"screen": {}},
                concurrency=48,
                repository=str(repository()),
                job_sha256=sha(directory / "job.sh"),
            )
            args = parser().parse_args(
                ["submit", "--root", str(directory), "--stage", "screen", "--dry-run"]
            )
            self.assertEqual(args.concurrency, 48)
            with (
                patch.object(module, "load_campaign", return_value=manifest),
                patch.object(module, "local_records", return_value=[row]),
                patch.object(module, "receipts", return_value=[]),
                patch.object(
                    subprocess, "run", side_effect=AssertionError("dry-run called subprocess")
                ),
            ):
                submit(args)
            self.assertEqual(sorted(p.name for p in directory.iterdir()), ["job.sh"])
            tasks = submission_tasks(manifest, [row], [], "screen")
            cmd = sbatch_command(
                directory, manifest, directory / "selection.json", tasks, 48, "fixture"
            )
            self.assertIn("--array=0-0%48", cmd)
            item: dict = dict(status="submitted", tasks=tasks, job_id="123")
            self.assertFalse(submission_tasks(manifest, [row], [item], "screen"))
            retry = submission_tasks(manifest, [row], [item], "screen", retry=True)
            self.assertEqual(retry[0]["attempt"], 2)
            self.assertFalse(
                submission_tasks(
                    manifest, [row | {"status": "numerical_failure"}], [item], "screen", retry=True
                )
            )
            with self.assertRaises(ValueError):
                submission_tasks(manifest, [row], [dict(status="intent")], "screen")

        def test_raw_result_validation_rejects_corruption(self):
            import yaml

            output = test_root / "validation"
            output.mkdir()
            cfg = build_config(base_raw, recipe(128, 0.008, 0.98, 0.1), 42, output).model_dump(
                mode="json"
            )
            (output / "resolved.yaml").write_text(yaml.safe_dump(cfg))
            (output / "run.log").write_text("synthetic validation fixture")
            result: dict = dict(
                status="complete",
                parameters=PARAMETERS,
                tokens=TOKENS,
                step=6227,
                epochs=40,
                recipe_identity="test",
                training_seconds=100,
                training_elapsed_seconds=120,
                seconds_this_session=220,
                peak_memory_bytes=1024,
                final_validation=dict(
                    split="full",
                    validation_complete=True,
                    tokens=VALIDATION_TOKENS,
                    loss=3.5,
                    perplexity=math.exp(3.5),
                    seconds=100,
                ),
            )
            atomic_json(output / "result.json", result)
            atomic_json(
                output / "environment.json",
                dict(
                    recipe_identity="test",
                    source_hash="test",
                    cache_identity="test",
                    versions={},
                    gpu="NVIDIA GH200",
                    compiled=True,
                    attention="sdpa",
                ),
            )
            events = [
                dict(
                    event="train",
                    tokens=WARMUP_TOKENS,
                    step=312,
                    lr=0.008,
                    loss=4.0,
                    grad_norm=1.0,
                    training_seconds=20,
                ),
                dict(
                    event="train",
                    tokens=TOKENS,
                    step=6227,
                    lr=0.0,
                    loss=3.5,
                    grad_norm=0.5,
                    training_seconds=80,
                ),
            ]
            events += [dict(event="validation", seconds=0.5, loss=3.5) for _ in range(40)]
            events += [dict(event="final_validation", loss=3.5)]
            (output / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in events))
            self.assertEqual(validate_result(output, cfg, "test", "test", "test", {})["loss"], 3.5)
            atomic_json(output / "result.json", result | dict(status="interrupted"))
            with self.assertRaises(ValueError):
                validate_result(output, cfg, "test", "test", "test", {})
            atomic_json(output / "result.json", result)
            (output / "bad.pt").touch()
            with self.assertRaises(ValueError):
                validate_result(output, cfg, "test", "test", "test", {})
            (output / "bad.pt").unlink()
            wrong = copy.deepcopy(cfg)
            wrong["optimizer"]["beta2"] = 0.95
            with self.assertRaises(ValueError):
                validate_result(output, wrong, "test", "test", "test", {})

        def test_heatmap_display_scale_preserves_data(self):
            cells = [
                dict(weight_decay=decay, delta=value)
                for decay, value in ((0.1, -0.2), (0.1, 2.0), (0, 0.05), (0, 0.001))
            ]
            original = copy.deepcopy(cells)
            settings = heatmap_settings(cells)
            self.assertEqual(
                (settings["vmin"], settings["vcenter"], settings["vmax"]), (-0.05, 0, 0.05)
            )
            self.assertEqual(settings["figures"]["wd0.1"]["colorbar_extend"], "both")
            self.assertEqual(settings["figures"]["wd0"]["colorbar_extend"], "neither")
            self.assertEqual(settings["figures"]["wd0.1"]["above_scale"], 1)
            self.assertEqual(settings["raw_delta_max"], 2)
            self.assertEqual(
                heatmap_settings(cells, 0.3)["figures"]["wd0.1"]["colorbar_extend"], "max"
            )
            self.assertEqual(
                heatmap_settings(cells[:1])["figures"]["wd0.1"]["colorbar_extend"], "min"
            )
            full = heatmap_settings(cells, None)
            self.assertEqual(full["limit"], 2)
            self.assertEqual(full["figures"]["wd0.1"]["colorbar_extend"], "neither")
            self.assertEqual(cells, original)
            self.assertGreater(heatmap_settings([], None)["limit"], 0)
            self.assertIsNone(heatmap_settings([])["raw_delta_min"])
            for invalid in (0, -1, math.nan, math.inf):
                with self.assertRaises(ValueError):
                    heatmap_settings(cells, invalid)
                with self.assertRaises(argparse.ArgumentTypeError):
                    positive_float(str(invalid))
            with self.assertRaises(ValueError):
                heatmap_settings([dict(weight_decay=0.1, delta=math.nan)])
            self.assertEqual(parser().parse_args(["collect"]).heatmap_limit, 0.05)
            self.assertIsNone(
                parser().parse_args(["collect", "--heatmap-full-range"]).heatmap_limit
            )
            self.assertEqual(
                parser().parse_args(["collect", "--heatmap-limit", "0.02"]).heatmap_limit, 0.02
            )

        def test_beta1_diagnostics_do_not_reselect_or_mix_seeds(self):
            value = recipe(32, 0.002, 0.998, 0.1)
            rows = [observation(value, seed=seed, loss=3.6) for seed in (42, 43, 44)]
            original_ranking = ranked_groups(rows)
            selected = {group_key(32, 0.1): original_ranking[group_key(32, 0.1)][0]}
            original_selection = copy.deepcopy(selected)
            rows += [
                observation(value | dict(beta1=0.98), stage="sensitivity", loss=3.4),
                observation(value | dict(beta1=0.7), seed=43, stage="sensitivity", loss=0.1),
                observation(value | dict(beta1=0.7), seed=45, stage="confirm", loss=0.01),
                observation(value | dict(beta1=0.7, lr=0.008), loss=0.1),
                observation(value | dict(beta1=0.7), loss=0.1) | dict(status="numerical_failure"),
                observation(value | dict(beta1=0.7), loss=0.1) | dict(role="reference"),
            ]
            result = beta1_diagnostics(selected, rows)[0]
            self.assertEqual(
                (result["best_beta1"], result["best_loss"], result["seed_count"]), (0.98, 3.4, 1)
            )
            self.assertAlmostEqual(result["delta"], -0.2)
            self.assertAlmostEqual(result["first_moment_half_life_targets"], half_life(32, 0.98))
            self.assertFalse(result["sweep_complete"])
            self.assertFalse(result["beta1_boundary"])
            self.assertEqual(selected, original_selection)
            self.assertEqual(ranked_groups(rows), original_ranking)
            rows.append(observation(value | dict(beta1=0), stage="sensitivity", loss=3.3))
            result = beta1_diagnostics(selected, rows)[0]
            self.assertEqual(result["first_moment_half_life_targets"], 0)
            self.assertTrue(result["beta1_boundary"])
            self.assertEqual(beta1_diagnostics({}, rows), [])

        def test_plot_data_and_all_exports(self):
            selected, rows = {}, []
            for batch, decay in itertools.product(BATCHES, DECAYS):
                value = recipe(batch, 0.004, 0.98 ** (batch / 128), decay)
                group_rows = [
                    observation(value, seed=s, loss=3.5 + s * 0.0001) for s in (42, 43, 44)
                ]
                rows += group_rows
                best = ranked_groups(group_rows)[group_key(batch, decay)][0]
                selected[group_key(batch, decay)] = best
                rows += [
                    observation(value | dict(lr=0.002), loss=5.52),
                    observation(value | dict(beta1=0.7), stage="sensitivity", loss=3.39),
                    observation(
                        value | dict(beta2=beta_policies(batch, decay)["half-life-10m"]), loss=3.51
                    ),
                ]
            cells = heatmap_cells(selected, rows)
            self.assertTrue(all(r["delta"] == 0 for r in cells if r["selected"]))
            self.assertTrue(any(r["delta"] < 0 for r in cells if r["panel"] == "beta1"))
            values, batches, matrix, stars = heatmap_matrix(cells, "beta2", 0.1)
            self.assertEqual(batches, [16, 32, 64, 128])
            self.assertTrue(np.isnan(matrix).any())
            self.assertEqual(len(stars), 4)
            self.assertEqual(matrix.shape, (len(values), 4))
            half_values, _, half_matrix, _ = heatmap_matrix(cells, "half_life", 0.1)
            self.assertEqual(len(half_values), 2)
            self.assertFalse(np.isnan(half_matrix).any())
            measures, summary, selected_rows = performance_tables(rows, selected)
            self.assertEqual(len(selected_rows), 24)
            self.assertTrue(
                all(r["steady_targets_per_second_relative_to_b128"] == 1 for r in summary)
            )
            fixture = test_root / "SYNTHETIC-plot-fixtures"
            fixture.mkdir()
            (fixture / "README.txt").write_text(
                "SYNTHETIC SELF-TEST DATA. Not experimental results.\n"
            )
            plot_reports(fixture, selected, rows, cells, [], summary, partial=True)
            self.assertEqual(len(list(fixture.glob("*.png"))), 5)
            self.assertEqual(len(list(fixture.glob("*.pdf"))), 5)
            self.assertEqual(len(list(fixture.glob("*.svg"))), 5)
            svg = (fixture / "hyperparameter-heatmaps-wd0.1.svg").read_text()
            self.assertIn("+2.02", svg)
            self.assertIn("-0.114", svg)
            self.assertIn("Shared color limits", svg)
            frozen = {
                key: value | dict(seed42_loss=value["losses"][0]) for key, value in selected.items()
            }
            replicated = beta1_replication_results(
                dict(beta1_replication=dict(recipes=frozen)), rows
            )
            plot_reports(fixture, selected, rows, cells, [], summary, beta1_replicated=replicated)
            self.assertTrue((fixture / "beta1-tuned-loss-versus-batch.pdf").is_file())
            self.assertIn(
                "Conditional β₁ tuning", (fixture / "beta1-tuned-loss-versus-batch.svg").read_text()
            )
            partial = beta1_replication_results(
                dict(beta1_replication=dict(recipes=frozen)), [r for r in rows if r["seed"] == 42]
            )
            plot_reports(
                fixture, selected, rows, cells, [], summary, partial=True, beta1_replicated=partial
            )
            self.assertIn("PARTIAL", (fixture / "beta1-tuned-loss-versus-batch.svg").read_text())
            write_tables(fixture, "cells", cells)
            self.assertEqual(read_json(fixture / "cells.json"), cells)
            self.assertGreater(len(measures), 24)

    try:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(StudyTests)
        )
        require(result.wasSuccessful(), "Embedded self-tests failed")
        print(
            "Self-tests passed: protocol, CPU training/interruption, promotion, submission, and report exports."
        )
    finally:
        if args.keep_artifacts:
            print(f"Synthetic self-test artifacts retained at {test_root}")
        else:
            shutil.rmtree(test_root)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def parser():
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = cli.add_subparsers(dest="command", required=True)
    descriptions = dict(
        prepare="Validate the baseline, freeze source, and generate screening configs; no jobs.",
        submit="Submit an eligible stage (first screen submission is the reference only).",
        status="Validate local run status; optionally query Slurm.",
        promote="Materialize the next stage after its prerequisites finish; no jobs.",
        collect="Validate results and export all figures, data tables, and report.",
        **{"self-test": "Run embedded tests with temporary synthetic data, without GPU or Slurm."},
        _worker="Internal frozen-source Slurm worker.",
    )
    for name, description in descriptions.items():
        sub = commands.add_parser(name, help=description, description=description)
        sub.add_argument(
            "--root",
            type=Path,
            default=repository() / DEFAULT_ROOT,
            help="Ignored campaign directory (default: %(default)s)",
        )
        if name == "prepare":
            sub.add_argument("--baseline", type=Path, default=repository() / DEFAULT_BASELINE)
        elif name == "submit":
            sub.add_argument("--stage", choices=(*STAGES, BETA1_REPLICATION))
            sub.add_argument(
                "--concurrency",
                type=positive_int,
                default=48,
                help="Maximum running array tasks; default 48; arrays cannot overlap.",
            )
            sub.add_argument(
                "--dry-run", action="store_true", help="No Slurm calls or submission writes."
            )
            sub.add_argument(
                "--retry",
                action="store_true",
                help="Fresh attempts for terminal infrastructure failures.",
            )
            sub.add_argument(
                "--reason", help="Evidence for an infrastructure retry or rejected intent."
            )
            sub.add_argument("--ids", help="Optional comma-separated eligible run IDs.")
            sub.add_argument(
                "--adopt-job", help="Reconcile an unresolved intent with an existing Slurm job ID."
            )
            sub.add_argument("--intent", help="32-character submission token for --adopt-job.")
            sub.add_argument(
                "--reject-intent", help="Token manually verified not to have submitted a job."
            )
        elif name == "status":
            sub.add_argument("--scheduler", action="store_true")
        elif name == "promote":
            sub.add_argument("--stage", required=True, choices=(*STAGES[1:], BETA1_REPLICATION))
        elif name == "collect":
            sub.add_argument("--allow-incomplete", action="store_true")
            scale = sub.add_mutually_exclusive_group()
            scale.add_argument(
                "--heatmap-limit",
                type=positive_float,
                default=DEFAULT_HEATMAP_LIMIT,
                help="Shared symmetric loss-difference color limit (default: %(default)s). "
                "Out-of-range cells are labeled; exported measurements are unchanged.",
            )
            scale.add_argument(
                "--heatmap-full-range",
                dest="heatmap_limit",
                action="store_const",
                const=None,
                help="Set the shared symmetric color range to the largest absolute difference.",
            )
        elif name == "self-test":
            sub.add_argument(
                "--keep-artifacts",
                action="store_true",
                help="Retain labeled synthetic fixtures in /tmp.",
            )
        elif name == "_worker":
            sub.add_argument("--selection", type=Path, required=True)
            sub.add_argument("--task", type=int, required=True)
    return cli


def main():
    args = parser().parse_args()
    actions = dict(
        prepare=prepare,
        submit=submit,
        status=status,
        promote=promote,
        collect=collect,
        **{"self-test": self_test, "_worker": worker},
    )
    try:
        actions[args.command](args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
