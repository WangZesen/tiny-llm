#!/usr/bin/env python3
"""Frozen GH200 migration workloads, submissions, and qualification gates.

The baseline source is copied before any production changes. Workers import only
that snapshot; edits to the checkout cannot alter a submitted native run.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import itertools
import json
import math
import shlex
import shutil
import statistics
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "runs/gh200-integration-20261004"
PROTOCOL = {
    "version": 1,
    "repeats": 3,
    "warmup": 20,
    "windows": 5,
    "minimum_window_seconds": 1.0,
    "local_updates": 5,
    "convergence_seeds": [45, 46, 47],
    "loss_atol": 0.002,
    "loss_rtol": 0.002,
    "gradient_relative_l2": 0.05,
    "parameter_relative_l2": 0.02,
    "update_relative_l2": 0.10,
    "small_batch_speedup": 1.05,
    "production_minimum_speedup": 0.98,
    "convergence_margin": 0.005,
    "convergence_t_multiplier": 2.920,
    "timing": "production loader, transfers, scheduler, finite checks, consensus, logging",
    "primary_reference": "current compiled BF16 native implementation",
}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def command(argv, **kwargs):
    return subprocess.run(argv, check=True, capture_output=True, text=True, **kwargs).stdout


def activate(source):
    source = Path(source).resolve()
    if "tiny_llm" in sys.modules:
        filename = sys.modules["tiny_llm"].__file__
        if filename is None:
            raise RuntimeError("tiny_llm must be loaded from an explicit source directory")
        loaded = Path(filename).resolve()
        if not loaded.is_relative_to(source):
            raise RuntimeError(f"already loaded non-frozen tiny_llm: {loaded}")
    sys.path.insert(0, str(source))


def workloads(repo):
    from tiny_llm.config import Config, load_config
    from tiny_llm.data import training_boundaries

    cache = read(repo / "data/c4/manifest.json")

    def normalized(config):
        raw = config.model_dump(mode="json") if isinstance(config, Config) else copy.deepcopy(config)
        raw["data"].update(
            cache_dir=str((repo / "data/c4").resolve()),
            revision=cache["dataset_revision"],
            tokenizer_revision=cache["tokenizer_revision"],
        )
        raw["runtime"].update(
            training_backend="native", device="cuda:0", cpu_threads=8, compile=True, seed=45
        )
        raw["training"].update(checkpoint_policy="final", save_epoch_training_state=True)
        return raw

    cases = []

    def add(name, raw, category, **labels):
        cfg = Config.model_validate(raw)
        cases.append(dict(id=name, category=category, config=cfg.model_dump(mode="json"), **labels))

    for preset, n, batch, length, clip in itertools.product(
        ("20m", "50m", "90m"), (1, 4, 8), (4, 8, 16), (512, 1024), (1.0, None)
    ):
        raw = normalized(load_config(repo / f"configs/{preset}.yaml"))
        raw["model"]["context_length"] = length
        raw["training"].update(batch_tokens=n * batch * length, micro_batch_size=batch)
        raw["optimizer"]["grad_clip"] = clip
        raw["decentralized"] = (
            dict(num_models=n, topology="one_peer_exponential", scheme="awc") if n > 1 else None
        )
        mode = "sync" if n == 1 else f"packed{n}"
        add(f"{preset}-{mode}-b{batch}-c{length}-clip{'1' if clip else 'off'}", raw,
            "small_batch", model=preset, mode=mode)

    production = {}
    for preset, overlay, name in (
        ("20m", None, "20m-sync-production"),
        ("50m", None, "50m-sync-production"),
        ("90m", None, "90m-sync-production"),
        ("20m", "packed4-20m-awc", "20m-packed4-production"),
        ("20m", "packed8-20m-awc", "20m-packed8-production"),
        ("20m", "packed4-20m-atc", "20m-atc-production"),
        ("20m", "packed-20m-adaptive", "20m-adaptive-production"),
    ):
        paths = [repo / f"configs/{preset}.yaml"]
        if overlay:
            paths.append(repo / f"configs/{overlay}.yaml")
        raw = normalized(load_config(paths))
        production[name] = raw
        add(name, raw, "production")
    for global_batch, micro in ((32, 8), (11, 4)):
        raw = copy.deepcopy(production["20m-sync-production"])
        raw["model"]["context_length"] = 512
        raw["training"].update(batch_tokens=global_batch * 512, micro_batch_size=micro)
        add(f"20m-accumulate-b{global_batch}-m{micro}", raw, "production")

    convergence = []
    recipes = {name: production[name] for name in (
        "20m-sync-production", "20m-packed4-production", "20m-packed8-production"
    )}
    selection_dir = repo / "doc/small-batchsize-investigation/data"
    for name, selection_file, batch, old_manifest in (
        ("20m-sync-small", "synchronous-selected.json", 16, "sync-manifest.json"),
        ("20m-packed4-small", "clipped-selected-recipes.json", 32,
         "clipped-extension-manifest.json"),
    ):
        selected = next(row for row in read(selection_dir / selection_file) if row["batch"] == batch)
        old = read(selection_dir / old_manifest)
        candidates = [r for r in old["runs"] if r.get("recipe") == selected["recipe"]]
        if not candidates:
            raise ValueError(f"selected recipe missing in source campaign: {selected['recipe']}")
        recipes[name] = normalized(candidates[0]["config"])
    for name, raw in recipes.items():
        for seed in PROTOCOL["convergence_seeds"]:
            cfg = copy.deepcopy(raw)
            cfg["runtime"]["seed"] = seed
            # Preserve the selected studies' exact realized budget and warmup.
            cfg["training"].update(checkpoint_policy="explicit", checkpoint_epochs=[20, 40],
                                    save_epoch_training_state=True)
            validated = Config.model_validate(cfg)
            convergence.append(dict(
                id=f"{name}-s{seed}", recipe=name, seed=seed,
                realized_tokens=training_boundaries(validated, validated.model.parameter_count)[-1]
                * validated.model.context_length, config=validated.model_dump(mode="json"),
            ))
    return cases, convergence, cache


def freeze(root, repo=ROOT):
    if (root / "manifest.json").exists():
        raise ValueError("campaign already frozen")
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "native"
    snapshot.mkdir()
    tracked = command(["git", "ls-files", "-z", "src", "configs", "tests", "scripts",
                       "pyproject.toml", "uv.lock", "README.md"], cwd=repo).split("\0")
    files = {}
    for name in filter(None, tracked):
        source = repo / name
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[name] = sha(target)
    (snapshot / "working-tree.patch").write_text(command(
        ["git", "diff", "--binary", "HEAD"], cwd=repo))
    activate(snapshot / "src")
    cases, convergence, cache = workloads(repo)
    write(root / "cache-manifest.json", cache)
    manifest = dict(
        version=1, created_utc=now(), repository=str(repo.resolve()),
        native_source=str((snapshot / "src").resolve()),
        git_head=command(["git", "rev-parse", "HEAD"], cwd=repo).strip(),
        git_status=command(["git", "status", "--short"], cwd=repo),
        source_files=files, patch_sha256=sha(snapshot / "working-tree.patch"),
        protocol=PROTOCOL, cache_identity=cache["identity"],
        cache_manifest_sha256=sha(root / "cache-manifest.json"),
        cases=cases, convergence=convergence,
    )
    manifest["identity"] = digest(manifest)
    write(root / "manifest.json", manifest)
    (root / "manifest.sha256").write_text(sha(root / "manifest.json") + "\n")
    print(json.dumps(dict(root=str(root), identity=manifest["identity"],
                          workloads=len(cases), convergence=len(convergence))))


def verify(root):
    manifest = read(root / "manifest.json")
    if sha(root / "manifest.json") != (root / "manifest.sha256").read_text().strip():
        raise ValueError("frozen manifest changed")
    for name, expected in manifest["source_files"].items():
        if sha(root / "native" / name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    if sha(root / "cache-manifest.json") != manifest["cache_manifest_sha256"]:
        raise ValueError("frozen cache manifest changed")
    return manifest


def convergence_gate(native, candidate, protocol=PROTOCOL):
    if set(native) != set(protocol["convergence_seeds"]) or set(candidate) != set(native):
        return dict(status="incomplete")
    differences = [candidate[s] - native[s] for s in sorted(native)]
    if not all(math.isfinite(v) for v in differences):
        return dict(status="failed", reason="nonfinite loss")
    mean = statistics.mean(differences)
    error = statistics.stdev(differences) / math.sqrt(len(differences))
    upper = mean + protocol["convergence_t_multiplier"] * error
    return dict(status="pass" if upper < protocol["convergence_margin"] else "not_qualified",
                differences=differences, mean_difference=mean, standard_error=error,
                upper_95=upper, margin=protocol["convergence_margin"])


def performance_gate(pairs, required_ids, protocol=PROTOCOL):
    """Require complete paired allocations; never qualify a partial result set.

    Each pair names a case, category/mode, repeat and actual allocation. Primary
    statistics are per-mode medians across workloads in each repeat. A threshold
    crossed in opposite directions between allocations remains inconclusive.
    """
    grouped = {}
    for pair in pairs:
        if pair["id"] not in required_ids:
            raise ValueError("unexpected performance workload")
        if not all(math.isfinite(pair[k]) and pair[k] > 0 for k in ("native", "candidate")):
            return dict(status="failed", reason="invalid throughput")
        rows = grouped.setdefault(pair["id"], {})
        if pair["repeat"] in rows:
            raise ValueError("duplicate performance repeat")
        rows[pair["repeat"]] = pair
    if set(grouped) != set(required_ids) or any(
        set(rows) != set(range(protocol["repeats"])) for rows in grouped.values()
    ):
        return dict(status="incomplete")
    cases, primary = {}, {}
    production_pass = True
    for name, rows in grouped.items():
        if len({row["allocation"] for row in rows.values()}) != protocol["repeats"]:
            raise ValueError("performance repeats must use independent allocations")
        sample = next(iter(rows.values()))
        ratios = [row["candidate"] / row["native"] for row in rows.values()]
        cases[name] = dict(median=statistics.median(ratios), minimum=min(ratios), maximum=max(ratios))
        if sample["category"] == "small_batch":
            by_repeat = primary.setdefault(sample["mode"], {r: [] for r in rows})
            for repeat, row in rows.items():
                by_repeat[repeat].append(row["candidate"] / row["native"])
        else:
            # Conservatively reject unresolved >2% regressions as well as clear ones.
            production_pass &= min(ratios) >= protocol["production_minimum_speedup"]
    primary_pass = set(primary) == {"sync", "packed4", "packed8"}
    categories = {}
    for mode, repeats in primary.items():
        medians = [statistics.median(values) for values in repeats.values()]
        passed = min(medians) >= protocol["small_batch_speedup"]
        primary_pass &= passed
        categories[mode] = dict(allocation_medians=medians, median=statistics.median(medians),
                                passed=passed)
    return dict(status="pass" if primary_pass and production_pass else "not_qualified",
                categories=categories, cases=cases, production_pass=production_pass)


def submit(root, kind, ids=None, repeats=None, dry_run=False, remaining=False,
           retry_failed=False, serialized=False):
    manifest = verify(root)
    available = manifest["convergence"] if kind == "convergence" else manifest["cases"]
    chosen = [r for r in available if ids is None or r["id"] in ids]
    if not chosen or (ids is not None and set(ids) != {r["id"] for r in chosen}):
        raise ValueError("unknown or empty workload selection")
    repeat_ids = range(PROTOCOL["repeats"]) if kind == "throughput" else [0]
    if repeats is not None:
        repeat_ids = repeats
    tasks = [dict(id=row["id"], repeat=rep, kind=kind) for rep in repeat_ids for row in chosen]
    if serialized:
        if kind != "numerical":
            raise ValueError("serialized controls apply only to numerical baseline tasks")
        tasks = [task | {"serialized": True} for task in tasks]
    if dry_run:
        print(json.dumps(tasks, indent=2))
        return
    with (root / ".submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = [read(p) for p in (root / "submissions").glob("*/receipt.json")]
        claimed = {(t["id"], t["repeat"], t["kind"]) for r in existing for t in r["tasks"]}
        overlap = [t for t in tasks if (t["id"], t["repeat"], t["kind"]) in claimed]
        if retry_failed and remaining:
            raise ValueError("choose either remaining tasks or explicit failed retries")
        if overlap and not remaining and not retry_failed:
            raise ValueError(f"already submitted; reconcile receipts before retrying: {overlap[:3]}")
        if remaining:
            tasks = [t for t in tasks if (t["id"], t["repeat"], t["kind"]) not in claimed]
            if not tasks:
                print("All requested tasks already have submission receipts.")
                return
        submission = root / "submissions" / f"{kind}-{uuid.uuid4().hex[:10]}"
        submission.mkdir(parents=True)
        previous_attempts = []
        if retry_failed:
            # A failed artifact and terminal scheduler state are both necessary;
            # ambiguous submissions are never retried automatically.
            for task in tasks:
                prior = root / "baseline" / kind / f"{task['id']}-r{task['repeat']}"
                result = read(prior / "result.json")
                if result.get("status") != "failed":
                    raise ValueError(f"only confirmed failed tasks can be retried: {task}")
                accounting = command(["sacct", "-j", result["slurm_job_id"], "-X", "-n", "-P",
                                      "-o", "JobIDRaw,State"], timeout=30)
                terminal = {"FAILED", "COMPLETED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                            "NODE_FAIL", "PREEMPTED"}
                rows = [line.split("|") for line in accounting.splitlines() if line.strip()]
                if not rows or any(row[1].split()[0].rstrip("+") not in terminal for row in rows):
                    raise ValueError(f"prior allocation is not confirmed terminal: {accounting}")
                previous_attempts.append(dict(task=task, path=str(prior), accounting=accounting,
                                              result_sha256=sha(prior / "result.json")))
            for previous in previous_attempts:
                prior = Path(previous["path"])
                archived = root / "baseline-attempts" / submission.name / kind / prior.name
                archived.parent.mkdir(parents=True, exist_ok=True)
                prior.rename(archived)
                previous["archived"] = str(archived)
            write(submission / "retry-provenance.json", previous_attempts)
        driver = submission / "gh200_campaign.py"
        worker = submission / "gh200_worker.py"
        shutil.copy2(Path(__file__), driver)
        shutil.copy2(Path(__file__).with_name("gh200_worker.py"), worker)
        # Different repeats never share an allocation. Isolated child processes
        # share its compiler cache, amortizing setup across related shapes.
        jobs = []
        chunk = 1 if kind == "convergence" else 6
        for rep in repeat_ids:
            repeat_tasks = [t for t in tasks if t["repeat"] == rep]
            jobs.extend(repeat_tasks[i:i + chunk] for i in range(0, len(repeat_tasks), chunk))
        write(submission / "tasks.json", jobs)
        repo = Path(manifest["repository"])
        argv = [str(repo / ".venv-aarch64/bin/python"), str(worker), "--root", str(root),
                "--tasks", str(submission / "tasks.json")]
        # Each group gets a distinct allocation and a fresh compiler cache.
        script = "\n".join([
            "#!/bin/bash", "set -euo pipefail", f"cd {shlex.quote(str(repo))}",
            "export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1",
            'qualification_cache=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/gh200-qualification.XXXXXXXX")',
            'export TORCHINDUCTOR_CACHE_DIR="$qualification_cache/inductor"',
            'export TRITON_CACHE_DIR="$qualification_cache/triton"',
            "trap 'rm -rf -- \"$qualification_cache\"' EXIT",
            shlex.join(argv) + ' --index "$SLURM_ARRAY_TASK_ID"', "",
        ])
        (submission / "job.sh").write_text(script)
        receipt = dict(created_utc=now(), status="submitting", tasks=tasks,
                       driver_sha256=sha(driver), worker_sha256=sha(worker),
                       manifest_identity=manifest["identity"], previous_attempts=previous_attempts)
        write(submission / "receipt.json", receipt)
        argv = ["sbatch", "--parsable", "--account=naiss2026-3-205-gpu", "--partition=gpu",
                "--gpus=nvidia_gh200_120gb:1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G",
                "--time=01:00:00" if kind == "convergence" else "--time=00:20:00",
                f"--array=0-{len(jobs)-1}%4", f"--job-name=gh200-{kind}",
                f"--output={submission}/slurm-%A_%a.log", str(submission / "job.sh")]
        try:
            answer = command(argv, timeout=60).strip()
            receipt.update(status="submitted", job_id=answer.split(";")[0])
        except BaseException as exc:
            receipt.update(status="uncertain", error=repr(exc))
            write(submission / "receipt.json", receipt)
            raise
        write(submission / "receipt.json", receipt)
        print(json.dumps(dict(submission=str(submission), **receipt)))


def collect(root):
    manifest = verify(root)
    records = {}
    invalid = []
    source = Path(manifest["native_source"]) / "tiny_llm"
    source_hash = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        source_hash.update(path.relative_to(source).as_posix().encode())
        source_hash.update(path.read_bytes())
    native_hash = source_hash.hexdigest()
    for kind in ("throughput", "numerical", "convergence"):
        paths = sorted((root / "baseline" / kind).glob("*/result.json"))
        records[kind] = [dict(path=str(p.relative_to(root)), **read(p)) for p in paths]
        expected_rows = {r["id"]: r for r in manifest["convergence" if kind == "convergence" else "cases"]}
        seen = set()
        for record in records[kind]:
            if record.get("status") != "ok":
                continue
            try:
                task = record["task"]
                key = (task["id"], task["repeat"])
                if task["kind"] != kind or key in seen:
                    raise ValueError("duplicate or mismatched task")
                seen.add(key)
                row = expected_rows[task["id"]]
                if task["repeat"] not in (range(3) if kind == "throughput" else [0]):
                    raise ValueError("unexpected repeat")
                if record["manifest_identity"] != manifest["identity"]:
                    raise ValueError("wrong frozen manifest")
                if record["config_identity"] != digest(row["config"]):
                    raise ValueError("configuration changed")
                directory = root / record["path"]
                metadata = record.get("environment") or read(directory.parent / "environment.json")
                if metadata["source_hash"] != native_hash or "GH200" not in metadata["gpu"]:
                    raise ValueError("wrong source or hardware")
                if kind == "throughput":
                    windows = record["windows"]
                    if len(windows) != manifest["protocol"]["windows"] or any(
                        w["seconds"] < manifest["protocol"]["minimum_window_seconds"]
                        or not math.isfinite(w["tokens_per_second"]) or w["tokens_per_second"] <= 0
                        for w in windows
                    ):
                        raise ValueError("incomplete/invalid timing windows")
                elif kind == "numerical":
                    if record["updates_sha256"] != sha(directory.parent / "updates.json"):
                        raise ValueError("numerical evidence checksum mismatch")
                    if record["steps"] != manifest["protocol"]["local_updates"]:
                        raise ValueError("incomplete numerical control")
                elif record["training"]["tokens"] != row["realized_tokens"]:
                    raise ValueError("convergence token budget mismatch")
            except (KeyError, ValueError, TypeError, OSError) as exc:
                record["status"] = "invalid"
                invalid.append(dict(path=record["path"], reason=str(exc)))
    expected = dict(throughput=3 * len(manifest["cases"]), numerical=len(manifest["cases"]),
                    convergence=len(manifest["convergence"]))
    summary = dict(manifest_identity=manifest["identity"], expected=expected,
                   complete={k: sum(r.get("status") == "ok" for r in rows)
                             for k, rows in records.items()}, records=records, invalid=invalid)
    summary["baseline_complete"] = summary["complete"] == expected
    write(root / "baseline-summary.json", summary)
    rates = {}
    for record in records["throughput"]:
        if record["status"] == "ok":
            rates.setdefault(record["task"]["id"], []).append(record)
    performance = {}
    for name, rows in rates.items():
        values = [row["tokens_per_second"] for row in rows]
        independent = len({r["slurm_job_id"] for r in rows}) == len(rows)
        performance[name] = dict(
            repeats=len(rows), independent=independent, median_tokens_per_second=statistics.median(values),
            minimum=min(values), maximum=max(values), complete=len(rows) == 3 and independent,
        )
    losses = {}
    recipes = {row["id"]: row["recipe"] for row in manifest["convergence"]}
    for record in records["convergence"]:
        if record["status"] == "ok":
            name = recipes[record["task"]["id"]]
            losses.setdefault(name, []).append(record["training"]["final_validation"]["loss"])
    convergence = {name: dict(n=len(values), losses=values, mean=statistics.mean(values),
                              sample_sd=statistics.stdev(values) if len(values) > 1 else None)
                   for name, values in losses.items()}
    write(root / "baseline-report.json", dict(
        manifest_identity=manifest["identity"], performance=performance, convergence=convergence,
        baseline_complete=summary["baseline_complete"], invalid=invalid,
    ))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "submit", "collect"))
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--kind", choices=("throughput", "numerical", "convergence"))
    parser.add_argument("--ids", help="comma-separated frozen workload IDs")
    parser.add_argument("--repeats", help="comma-separated zero-based repeats")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remaining", action="store_true", help="skip tasks already claimed by receipts")
    parser.add_argument("--retry-failed", action="store_true", help="archive and retry confirmed terminal failures")
    parser.add_argument("--serialized", action="store_true", help="offload independent numerical control states")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "freeze":
        freeze(root)
    elif args.action == "verify":
        print(verify(root)["identity"])
    elif args.action == "submit":
        if not args.kind:
            parser.error("submit requires --kind")
        submit(root, args.kind, args.ids.split(",") if args.ids else None,
               [int(v) for v in args.repeats.split(",")] if args.repeats else None,
               args.dry_run, args.remaining, args.retry_failed, args.serialized)
    else:
        collect(root)


if __name__ == "__main__":
    main()
