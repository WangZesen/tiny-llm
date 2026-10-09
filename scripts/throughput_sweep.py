#!/usr/bin/env python3
"""Resume independent, single-GH200 batch sweeps through actual CUDA OOM.

The login-side watcher owns submissions. GPU jobs never submit more jobs.
Source, configuration, receipts, and every failed attempt remain in the campaign.
"""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = ("20m", "90m", "250m", "500m")
CONTEXTS = (512, 1024, 2048, 4096)
RESOURCES = dict(account="naiss2026-3-205-gpu", partition="gpu",
                 gpus="nvidia_gh200_120gb:1", cpus_per_task=8, mem="64G", time="00:14:00")
PROTOCOL = dict(warmup=20, windows=5, window_seconds=10.0, deadline_seconds=780)
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED",
          "RESIZING", "SIGNALING", "STAGE_OUT"}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_digest(package):
    result = hashlib.sha256()
    for path in sorted(Path(package).rglob("*.py")):
        result.update(path.relative_to(package).as_posix().encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def command(argv):
    return subprocess.run(argv, check=True, text=True, capture_output=True).stdout.strip()


def now():
    return datetime.now(timezone.utc).isoformat()


def point_config(base, length, batch, directory):
    from tiny_llm.config import Config

    raw = copy.deepcopy(base)
    raw["model"]["context_length"] = length
    raw["training"].update(micro_batch_size=batch, batch_tokens=batch * length,
                           checkpoint_policy="none", save_epoch_training_state=False)
    raw["runtime"]["output_dir"] = str(directory)
    return Config.model_validate(raw).model_dump(mode="json")


def optimizer_grid(source):
    """Freeze the successful contiguous grids of a completed memory sweep."""
    original = validate_manifest(source)
    report = read(source / "results.json")
    if not report["complete"] or report["campaign_identity"] != original["identity"]:
        raise ValueError("optimizer timing requires a completed throughput campaign")
    limits = {}
    for summary in report["summaries"]:
        key = f"{summary['model']}-c{summary['context_length']}"
        rows = [row for row in report["points"] if row["status"] == "ok"
                and row["model"] == summary["model"]
                and row["context_length"] == summary["context_length"]]
        largest = summary["largest_batch"]
        if sorted({row["batch_size"] for row in rows}) != [1 << i for i in range(largest.bit_length())]:
            raise ValueError(f"noncontiguous original curve: {key}")
        for row in rows:
            directory = source / row["directory"]
            validated_result(directory, original, read(directory / "receipt.json"))
        limits[key] = largest
    if set(limits) != {f"{model}-c{length}" for model in MODELS for length in original["contexts"]}:
        raise ValueError("original campaign must cover every declared model/context curve")
    return original, limits


def selected_contexts(contexts=None, original=None):
    lengths = tuple(contexts) if contexts is not None else tuple(original["contexts"]) if original else CONTEXTS
    if not lengths or len(set(lengths)) != len(lengths) or any(x not in CONTEXTS for x in lengths):
        raise ValueError(f"contexts must be a nonempty, unique subset of {CONTEXTS}")
    if original and set(lengths) != set(original["contexts"]):
        raise ValueError("optimizer timing must use the original campaign's contexts")
    return tuple(sorted(lengths))


def prepare(root, optimizer_timing_from=None, contexts=None, parallel_optimizer_points=False):
    from tiny_llm.config import load_config

    if (root / "manifest.json").exists():
        raise ValueError("campaign already exists; use --resume")
    if any(root.iterdir()):
        # The live watcher lock is the only file allowed before preparation.
        if any(p.name != ".lock" for p in root.iterdir()):
            raise ValueError("new campaign directory must be empty")
    original, limits = optimizer_grid(optimizer_timing_from) if optimizer_timing_from else (None, None)
    if parallel_optimizer_points and not original:
        raise ValueError("parallel optimizer points require a completed throughput campaign")
    contexts = selected_contexts(contexts, original)
    source = root / "source"
    shutil.copytree(REPO / "src/tiny_llm", source / "tiny_llm",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, source / "throughput_sweep.py")
    shutil.copy2(REPO / "uv.lock", source / "uv.lock")
    bases = {}
    for model in MODELS:
        config = load_config(REPO / f"configs/{model}.yaml")
        raw = config.model_dump(mode="json")
        raw["decentralized"] = None
        raw["optimizer"].update(lr=0.001, beta1=0.9, beta2=0.99, eps=1e-8,
                                weight_decay=0.1, grad_clip=1.0)
        raw["runtime"].update(training_backend="gh200", seed=42, cpu_threads=8,
                              device="cuda:0", amp=True, compile=True, compile_mode="default",
                              sdpa_backend="auto", attention_backend="sdpa", deterministic=False,
                              fused_optimizer=True)
        bases[model] = raw
    if original:
        bases = copy.deepcopy(original["models"])
    manifest: dict = dict(version=1, campaign_id=uuid.uuid4().hex[:10], created=now(),
                    repo=str(REPO), source_hash=source_digest(source / "tiny_llm"),
                    driver_sha256=hashlib.sha256((source / "throughput_sweep.py").read_bytes()).hexdigest(),
                    resources=RESOURCES, protocol=PROTOCOL, models=bases, contexts=list(contexts),
                    concurrency=12, git_revision=command(["git", "-C", str(REPO), "rev-parse", "HEAD"]))
    (root / "source" / "working-tree.patch").write_text(
        command(["git", "-C", str(REPO), "diff", "HEAD"]))
    if original:
        manifest.update(mode="optimizer_timing", batch_limits=limits,
                        original_campaign=dict(path=str(optimizer_timing_from),
                                               identity=original["identity"]))
        manifest["protocol"] = dict(PROTOCOL, optimizer_timing=True, paired_baseline=True)
        manifest["parallel_optimizer_points"] = parallel_optimizer_points
    manifest["identity"] = digest(manifest)
    write(root / "manifest.json", manifest)
    state: dict = dict(curves=[dict(id=f"{model}-c{length}", model=model, context_length=length,
                             next_batch=1, complete=False, blocked=None, current=None, attempts=[])
                        for model in MODELS for length in contexts])
    if parallel_optimizer_points:
        state["work_items"] = optimizer_work_items(state["curves"], limits)
    save_state(root, state)
    (root / "job.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'root=$1\nattempt=$2\n'
        'cache=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/tiny-throughput.XXXXXXXX")\n'
        'export TORCHINDUCTOR_CACHE_DIR="$cache/inductor"\n'
        'export TRITON_CACHE_DIR="$cache/triton"\n'
        'export TOKENIZERS_PARALLELISM=false\nexport OMP_NUM_THREADS=8\n'
        'export PYTHONPATH="$root/source"\n'
        "trap 'rm -rf -- \"$cache\"' EXIT\ntrap 'exit 143' TERM\ntrap 'exit 130' INT\n"
        f"cd {shlex.quote(str(REPO))}\n"
        f"srun --ntasks=1 {shlex.quote(str(REPO / '.venv-aarch64/bin/python'))} -B "
        '"$root/source/throughput_sweep.py" _execute --output "$root" --attempt "$attempt"\n'
    )
    return manifest, state


def optimizer_work_items(curves, limits):
    """Known fitting points are independent; memory-discovery sweeps stay sequential."""
    return [dict(copy.deepcopy(curve), next_batch=1 << i, fixed_batch=1 << i)
            for i in range(max(limits.values()).bit_length()) for curve in curves
            if (1 << i) <= limits[curve["id"]]]


def save_state(root, state):
    if "work_items" in state:
        for curve in state["curves"]:
            items = [item for item in state["work_items"] if item["id"] == curve["id"]]
            active = [item["current"] for item in items if item["current"]]
            curve.update(complete=all(item["complete"] for item in items),
                         blocked=next((item["blocked"] for item in items if item["blocked"]), None),
                         current=active[0] if active else None, active_points=active,
                         attempts=[path for item in items for path in item["attempts"]],
                         next_batch=min((item["fixed_batch"] for item in items if not item["complete"]),
                                        default=2 * max(item["fixed_batch"] for item in items)))
    write(root / "state.json", state)


def validate_manifest(root):
    manifest = read(root / "manifest.json")
    if digest({key: value for key, value in manifest.items() if key != "identity"}) != manifest["identity"]:
        raise ValueError("manifest identity changed")
    if source_digest(root / "source/tiny_llm") != manifest["source_hash"]:
        raise ValueError("frozen source changed")
    if hashlib.sha256((root / "source/throughput_sweep.py").read_bytes()).hexdigest() != manifest["driver_sha256"]:
        raise ValueError("frozen driver changed")
    return manifest


def submit(root, manifest, curve):
    batch = curve["next_batch"]
    number = 1 + sum(read(root / p / "receipt.json")["batch_size"] == batch for p in curve["attempts"])
    relative = f"points/{curve['id']}/b{batch}/attempt-{number}"
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=False)
    config = point_config(manifest["models"][curve["model"]], curve["context_length"], batch, directory)
    write(directory / "config.json", config)
    name = f"tp-{manifest['campaign_id']}-{curve['id']}-b{batch}-a{number}"
    argv = ["sbatch", "--parsable", *[
        f"--{key.replace('_', '-')}={value}" for key, value in manifest["resources"].items()
    ], "--ntasks=1", f"--job-name={name}", f"--chdir={manifest['repo']}",
        f"--output={directory}/slurm.log", str(root / "job.sh"), str(root), relative]
    receipt = dict(job_id=None, job_name=name, submitted=now(), command=argv,
                   batch_size=batch, model=curve["model"], context_length=curve["context_length"],
                   config_identity=digest(config), campaign_identity=manifest["identity"],
                   status="submission_intent", attempt=number)
    write(directory / "receipt.json", receipt)
    # Persist the intent before contacting Slurm, including its index in state.json.
    curve["current"] = relative
    curve["attempts"].append(relative)
    return receipt, directory


def launch(directory, receipt):
    try:
        answer = command(receipt["command"])
    except subprocess.CalledProcessError as error:
        receipt.update(status="submission_rejected", error=error.stderr)
    else:
        job_id = answer.split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"unrecognized sbatch receipt: {answer!r}; reconcile before retry")
        receipt.update(job_id=job_id, status="submitted")
    write(directory / "receipt.json", receipt)
    print(f"{now()} {receipt['job_name']}: {receipt['status']} {receipt['job_id']}", flush=True)


def reconcile_intent(receipt):
    """Recover a submission interrupted between sbatch and writing its job ID."""
    name = receipt["job_name"]
    queued = command(["squeue", "--me", "--name", name, "--noheader", "--format=%i|%j"])
    accounted = command(["sacct", "-X", "--name", name, "--starttime", receipt["submitted"][:10],
                         "--noheader", "--parsable2", "--format=JobIDRaw,JobName%128"])
    ids = {line.split("|")[0] for text in (queued, accounted) for line in text.splitlines()
           if len(line.split("|")) >= 2 and line.split("|")[1] == name}
    if len(ids) == 1:
        receipt.update(job_id=ids.pop(), status="submitted")
    else:
        raise RuntimeError(f"unresolved submission intent for {name}: matching jobs={sorted(ids)}")


def scheduler_rows(job_ids):
    if not job_ids:
        return {}
    joined = ",".join(job_ids)
    rows = {}
    output = command(["sacct", "-X", "-j", joined, "--noheader", "--parsable2",
                      "--format=JobIDRaw,State,ExitCode,Elapsed,Timelimit,NodeList,Reason"])
    for line in output.splitlines():
        parts = line.split("|")
        if parts[0] in job_ids and len(parts) >= 7:
            rows[parts[0]] = dict(zip(
                ("job_id", "state", "exit_code", "elapsed", "timelimit", "nodes", "reason"),
                parts[:7], strict=True))
    output = command(["squeue", "--me", "--noheader", "--format=%i|%T|%R"])
    for line in output.splitlines():
        job_id, state, reason = line.split("|", 2)
        if job_id in job_ids:
            rows[job_id] = dict(job_id=job_id, state=state, reason=reason)
    return rows


def validated_result(directory, manifest, receipt, filename="result.json"):
    result = read(directory / filename)
    if digest(result.get("config")) != receipt["config_identity"]:
        raise ValueError("result configuration differs from submitted configuration")
    if result.get("environment", {}).get("source_hash") != manifest["source_hash"]:
        raise ValueError("result source differs from frozen source")
    for key in ("warmup", "windows", "window_seconds"):
        if result.get("protocol", {}).get(key) != manifest["protocol"][key]:
            raise ValueError(f"result protocol differs: {key}")
    if result["status"] == "ok":
        rows = result["measurements"]
        windows = manifest["protocol"]["windows"]
        expected = manifest["protocol"]["warmup"] + 5 + sum(row["steps"] for row in rows)
        if len(rows) not in (windows, 2 * windows) or result["successful_updates"] != expected:
            raise ValueError("incomplete measurement windows or optimizer update count")
        if any(row["steps"] < 20 or not math.isfinite(row["seconds"])
               or row["seconds"] <= 0 for row in rows):
            raise ValueError("invalid timing window")
        rate = statistics.median(row["steps"] * receipt["batch_size"] * receipt["context_length"]
                                 / row["seconds"] for row in rows)
        if not abs(rate / result["tokens_per_second"] - 1) < 1e-12:
            raise ValueError("throughput arithmetic mismatch")
        if manifest.get("mode") == "optimizer_timing" and filename == "result.json":
            validate_optimizer_result(directory, manifest, receipt, result)
    return result


def paired_optimizer_summary(result, baseline):
    baseline_ms = baseline["milliseconds_per_update"]
    change = result["milliseconds_per_update"] / baseline_ms - 1
    return dict(
        baseline_milliseconds_per_update=baseline_ms,
        baseline_coefficient_of_variation=baseline["coefficient_of_variation"],
        optimizer_fraction=result["optimizer_milliseconds_per_update"] / baseline_ms,
        instrumented_wall_time_change=change,
        instrumentation_warning=abs(change) > 0.05,
        stable=result["stable"] and baseline["stable"],
    )


def validate_optimizer_result(directory, manifest, receipt, result):
    from tiny_llm.throughput import measurement_summary, optimizer_summary, optimizer_window

    if not result["protocol"].get("optimizer_timing"):
        raise ValueError("missing optimizer instrumentation")
    for row in result["measurements"]:
        samples = row["optimizer_samples"]
        if not samples or sum(s["steps"] for s in samples) != row["steps"]:
            raise ValueError("incomplete optimizer samples")
        if any(s["steps"] < 1 or not (0 < s["optimizer_ms"] <= s["graph_ms"])
               or not math.isfinite(s["graph_ms"]) or not math.isfinite(s["seconds"])
               or s["seconds"] <= 0 for s in samples):
            raise ValueError("invalid optimizer samples")
        if not math.isclose(sum(s["seconds"] for s in samples), row["seconds"], rel_tol=1e-12):
            raise ValueError("burst timing mismatch")
        aggregate = optimizer_window(samples)
        for key in ("optimizer_ms", "graph_ms"):
            if not math.isclose(row[key], aggregate[key], rel_tol=1e-12):
                raise ValueError("optimizer window arithmetic mismatch")
    summary = optimizer_summary(result["measurements"])
    for key, value in summary.items():
        if not math.isclose(result[key], value, rel_tol=1e-12):
            raise ValueError(f"optimizer summary mismatch: {key}")
    baseline = validated_result(directory, manifest, receipt, "baseline.json")
    if baseline["status"] != "ok" or baseline["protocol"].get("optimizer_timing"):
        raise ValueError("invalid paired baseline")
    instrumented = dict(result, stable=summary["optimizer_stable"] and measurement_summary(
        result["measurements"], receipt["batch_size"], receipt["context_length"])["stable"])
    for key, value in paired_optimizer_summary(instrumented, baseline).items():
        if not math.isclose(result[key], value, rel_tol=1e-12):
            raise ValueError(f"paired baseline arithmetic mismatch: {key}")


def finish_point(root, manifest, curve, receipt, accounting):
    directory = root / curve["current"]
    receipt["accounting"] = accounting
    state = accounting["state"].split()[0].rstrip("+")
    receipt["finished"] = now()
    if state == "COMPLETED" and accounting.get("exit_code") == "0:0":
        try:
            result = validated_result(directory, manifest, receipt)
            receipt["status"] = result["status"]
        except (ValueError, KeyError, OSError) as error:
            receipt.update(status="invalid_result", error=str(error))
    else:
        receipt["status"] = {"TIMEOUT": "timeout", "OUT_OF_MEMORY": "host_oom",
                             "NODE_FAIL": "infrastructure_failure"}.get(state, "job_failed")
        if (directory / "result.json").exists():
            receipt["worker_status"] = read(directory / "result.json").get("status")
            try:
                result = validated_result(directory, manifest, receipt)
                if result["status"] in ("nonfinite", "compiler_error", "error", "timeout"):
                    receipt["status"] = result["status"]
            except (ValueError, KeyError, OSError):
                pass
    if receipt["status"] == "ok":
        curve["next_batch"] *= 2
        if manifest.get("mode") == "optimizer_timing":
            curve["complete"] = ("fixed_batch" in curve
                                 or curve["next_batch"] > manifest["batch_limits"][curve["id"]])
    elif receipt["status"] == "cuda_oom":
        if manifest.get("mode") == "optimizer_timing":
            curve["blocked"] = "cuda_oom"
        else:
            curve["complete"] = True
    else:
        curve["blocked"] = receipt["status"]
    curve["current"] = None
    write(directory / "receipt.json", receipt)
    print(f"{now()} {curve['id']} b{receipt['batch_size']}: {receipt['status']}", flush=True)


def run(root, *, resume=False, retry_failed=False, poll_seconds=20, optimizer_timing_from=None,
        contexts=None, parallel_optimizer_points=False):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            manifest, state = validate_manifest(root), read(root / "state.json")
            if contexts is not None and selected_contexts(contexts) != tuple(manifest["contexts"]):
                raise ValueError("resume cannot change frozen contexts")
            if parallel_optimizer_points and not manifest.get("parallel_optimizer_points"):
                raise ValueError("resume cannot change frozen scheduling")
        else:
            manifest, state = (prepare(root, optimizer_timing_from, contexts, parallel_optimizer_points)
                               if optimizer_timing_from or contexts or parallel_optimizer_points else prepare(root))
        work: list[dict] = state.get("work_items", state["curves"])
        if retry_failed:
            for curve in work:
                curve["retrying"] = bool(curve["blocked"])
                curve["blocked"] = None
        while True:
            pending = []
            for curve in work:
                if curve["current"]:
                    directory = root / curve["current"]
                    receipt = read(directory / "receipt.json")
                    if receipt["status"] == "submission_intent":
                        reconcile_intent(receipt)
                        write(directory / "receipt.json", receipt)
                    if receipt["status"] == "submission_rejected":
                        curve.update(blocked="submission_rejected", current=None)
                    else:
                        pending.append((curve, receipt))
            accounting = scheduler_rows([receipt["job_id"] for _, receipt in pending])
            for curve, receipt in pending:
                row = accounting.get(receipt["job_id"])
                if row and row["state"].split()[0].rstrip("+") not in ACTIVE:
                    finish_point(root, manifest, curve, receipt, row)
            save_state(root, state)
            blocked = {item["id"] for item in work if item["blocked"]}
            retrying = {item["id"] for item in work if item.get("retrying") and not item["complete"]}
            available = min(12, manifest.get("concurrency", 12)) - sum(
                curve["current"] is not None for curve in work)
            for curve in work:
                if available <= 0:
                    break
                if curve["id"] in retrying and not curve.get("retrying"):
                    continue
                if not curve["complete"] and curve["id"] not in blocked and not curve["current"]:
                    receipt, directory = submit(root, manifest, curve)
                    save_state(root, state)
                    launch(directory, receipt)
                    available -= 1
            if all(not curve["current"] and (curve["complete"] or curve["id"] in blocked) for curve in work):
                collect(root)
                return all(curve["complete"] for curve in work)
            time.sleep(poll_seconds)


def execute(root, relative):
    """Lightweight parent: timeout covers worker imports, setup, capture and timing."""
    manifest = read(root / "manifest.json")
    directory = root / relative
    protocol = manifest["protocol"]
    began = time.monotonic()
    paired = manifest.get("mode") == "optimizer_timing"
    for filename in (("baseline.json", "result.json") if paired else ("result.json",)):
        argv = [sys.executable, "-B", "-m", "tiny_llm", "benchmark-throughput-worker",
                "--config", str(directory / "config.json"), "--output", str(directory / filename),
                "--warmup", str(protocol["warmup"]), "--windows", str(protocol["windows"]),
                "--window-seconds", str(protocol["window_seconds"])]
        if paired and filename == "result.json":
            argv.append("--optimizer-timing")
        process = subprocess.Popen(argv, start_new_session=True)
        try:
            code = process.wait(timeout=max(0.001, protocol["deadline_seconds"]
                                            - (time.monotonic() - began)))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            progress = (directory / filename).with_suffix(".progress.json")
            result = read(progress) if progress.exists() else dict(stage="worker_import")
            result.update(status="timeout", error="13-minute worker deadline exceeded",
                          worker_phase=filename)
            write(directory / "result.json", result)
            return 1
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if not (directory / filename).exists():
            write(directory / "result.json", dict(status="error", stage="worker_exit",
                                                  worker_phase=filename, exit_code=code))
            return 1
        result = read(directory / filename)
        if code or result["status"] != "ok":
            if filename != "result.json":
                result["worker_phase"] = filename
                write(directory / "result.json", result)
            return code
    if paired:
        baseline = read(directory / "baseline.json")
        result = read(directory / "result.json")
        result.update(paired_optimizer_summary(result, baseline))
        result["paired_elapsed_seconds"] = time.monotonic() - began
        write(directory / "result.json", result)
    return 0


def collect(root):
    manifest, state = validate_manifest(root), read(root / "state.json")
    points, summaries = [], []
    for curve in state["curves"]:
        successful, oom = {}, None
        for relative in curve["attempts"]:
            directory = root / relative
            receipt = read(directory / "receipt.json")
            row = {key: receipt.get(key) for key in
                   ("model", "context_length", "batch_size", "attempt", "job_id", "status")}
            row["directory"] = relative
            if receipt["status"] in ("ok", "cuda_oom"):
                result = validated_result(directory, manifest, receipt)
                row.update({key: result.get(key) for key in (
                    "parameters", "tokens_per_second", "sequences_per_second",
                    "milliseconds_per_update", "coefficient_of_variation", "stable",
                    "setup_peak_allocated_bytes", "setup_peak_reserved_bytes",
                    "measurement_peak_allocated_bytes", "measurement_peak_reserved_bytes",
                    "failure_peak_allocated_bytes", "failure_peak_reserved_bytes",
                    "initial_free_memory_bytes", "total_memory_bytes", "preparation_seconds",
                    "setup_seconds", "stage")})
                if manifest.get("mode") == "optimizer_timing":
                    row.update({key: result.get(key) for key in (
                        "optimizer_milliseconds_per_update", "optimizer_fraction",
                        "optimizer_fraction_instrumented", "optimizer_fraction_graph",
                        "graph_milliseconds_per_update", "optimizer_coefficient_of_variation",
                        "optimizer_stable", "baseline_milliseconds_per_update",
                        "baseline_coefficient_of_variation", "instrumented_wall_time_change",
                        "instrumentation_warning")})
                if row["status"] == "ok":
                    successful[row["batch_size"]] = row
                else:
                    oom = row["batch_size"]
            points.append(row)
        stable = [row for row in successful.values() if row["stable"]]
        best = max(stable, key=lambda row: row["tokens_per_second"]) if stable else None
        largest = max(successful, default=None)
        expected = list(1 << i for i in range(largest.bit_length())) if largest else []
        complete = (curve["complete"] and sorted(successful) == expected
                    and oom == (2 * largest if largest else 1))
        if manifest.get("mode") == "optimizer_timing":
            complete = (curve["complete"] and sorted(successful) == expected
                        and largest == manifest["batch_limits"][curve["id"]])
        summaries.append(dict(model=curve["model"], context_length=curve["context_length"],
                              complete=complete, largest_batch=largest, first_oom_batch=oom,
                              best_stable_batch=best["batch_size"] if best else None,
                              best_tokens_per_second=best["tokens_per_second"] if best else None,
                              unstable_batches=[b for b, row in successful.items() if not row["stable"]],
                              blocked=curve["blocked"]))
    report = dict(complete=all(row["complete"] for row in summaries), summaries=summaries,
                  points=points, campaign_identity=manifest["identity"], collected=now())
    write(root / "results.json", report)
    fields = list(dict.fromkeys(key for row in points for key in row))
    with (root / "results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(points)
    render(root, report, manifest)
    return report


def render(root, report, manifest):
    if manifest.get("mode") == "optimizer_timing":
        render_optimizer(root, report, manifest)
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contexts = manifest["contexts"]
    for metric, label, name, scale in (
        ("tokens_per_second", "Target tokens / second", "throughput.png", 1),
        ("measurement_peak_reserved_bytes", "Peak reserved GPU memory (GiB)", "memory.png", 2**30),
    ):
        fig, axes = plt.subplots(1, len(contexts), figsize=(5.3 * len(contexts), 4.5),
                                 constrained_layout=True, squeeze=False)
        for length, axis in zip(contexts, axes[0], strict=True):
            for model in MODELS:
                rows = [row for row in report["points"] if row["status"] == "ok"
                        and row["model"] == model and row["context_length"] == length]
                rows = sorted({row["batch_size"]: row for row in rows}.values(),
                              key=lambda row: row["batch_size"])
                if rows:
                    axis.plot([row["batch_size"] for row in rows],
                              [row[metric] / scale for row in rows], marker="o", label=model)
                    for row in rows:
                        if not row["stable"]:
                            axis.scatter(row["batch_size"], row[metric] / scale, marker="x",
                                         color="red", s=70, zorder=5)
            axis.set(xscale="log", xlabel="Batch size (sequences)", ylabel=label,
                     title=f"Context {length}")
            axis.set_xscale("log", base=2)
            axis.grid(alpha=0.25)
            if axis.lines:
                axis.legend()
        fig.savefig(root / name, dpi=160)
        plt.close(fig)
    capacities = sorted({row["total_memory_bytes"] / 2**30 for row in report["points"]
                         if row.get("total_memory_bytes") is not None})
    capacity = (f"{capacities[0]:.2f} GiB" if len(capacities) == 1 else
                f"{capacities[0]:.2f}–{capacities[-1]:.2f} GiB" if capacities else "not measured yet")
    lines = ["# Single-GH200 training throughput", "",
             f"Campaign complete: **{report['complete']}**. One GH200 per measurement; 14-minute allocations.",
             "", f"CUDA-reported device capacity: **{capacity}**. The requested Slurm GPU type is "
             f"`{manifest['resources']['gpus']}`; memory boundaries apply to the measured capacity.",
             "", "BF16 autocast, FP32 weights/moments, compiled SDPA and complete-update CUDA graphs. "
             "GPU-resident synthetic inputs and targets; no gradient accumulation. "
             "Timing includes the production device copies, forward, loss, backward, clipping and AdamW. "
             "It excludes compilation, setup, warmup, inspection, logging, validation and checkpointing.",
             "", "Seed 42; 8 CPU threads; constant LR 0.001; AdamW betas (0.9, 0.99), epsilon 1e-8, "
             "weight decay 0.1, gradient clipping 1.0. New 250m/500m presets have "
             "251,943,360 / 516,814,080 parameters; optimizer recipes are not tuned.",
             "", "20 warmup and 5 calibration updates precede five windows targeting 10 seconds each "
             "(minimum 20 updates). Throughput is the median window rate. If sample CV exceeds 5%, "
             "five more windows are included. Remaining instability is flagged, and red crosses mark "
             "unstable points in plots. Only stable points compete for fastest batch.",
             "", "Batches double from 1 until an actual CUDA allocation failure. Compilation/capture "
             "memory counts toward this limit. Timeouts, host OOM, numerical and compiler failures "
             "do not establish the GPU memory boundary. Every attempt is retained in results.csv.",
             "", "| Model | Context | Fastest stable batch | Tokens/s | Largest fitting batch | First OOM | Unstable batches |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for row in report["summaries"]:
        rate = row["best_tokens_per_second"]
        lines.append(f"| {row['model']} | {row['context_length']} | {row['best_stable_batch']} | "
                     f"{f'{rate:,.0f}' if rate else '—'} | {row['largest_batch']} | "
                     f"{row['first_oom_batch']} | {row['unstable_batches']} |")
    lines += ["", "![Throughput](throughput.png)", "", "![Memory](memory.png)", "",
              "[Detailed CSV](results.csv) · [JSON](results.json) · [Manifest](manifest.json)", "",
              f"Frozen source SHA-256: `{manifest['source_hash']}`. "
              "Per-point directories contain resolved configurations, raw timing windows, hardware/software "
              "metadata, submission receipts, Slurm accounting and logs.", ""]
    if (root / "validation.json").exists():
        lines += ["[CPU and GH200 validation](validation.json)", ""]
    (root / "README.md").write_text("\n".join(lines))


def render_optimizer(root, report, manifest):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contexts = manifest["contexts"]
    rows = [row for row in report["points"] if row["status"] == "ok"]
    # Retain failed attempts in CSV/JSON; plot the latest successful measurement.
    rows = list({(row["model"], row["context_length"], row["batch_size"]): row
                 for row in rows}.values())
    for metric, label, name, factor in (
        ("optimizer_milliseconds_per_update", "Optimizer step (ms)", "optimizer-time.png", 1),
        ("optimizer_fraction", "Optimizer / baseline iteration (%)", "optimizer-share.png", 100),
    ):
        fig, axes = plt.subplots(1, len(contexts), figsize=(5.3 * len(contexts), 4.5),
                                 constrained_layout=True, squeeze=False)
        for length, axis in zip(contexts, axes[0], strict=True):
            for model in MODELS:
                points = sorted((row for row in rows if row["model"] == model
                                 and row["context_length"] == length), key=lambda r: r["batch_size"])
                if points:
                    axis.plot([r["batch_size"] for r in points],
                              [factor * r[metric] for r in points], marker="o", label=model)
                    for row in points:
                        if not row["stable"] or row["instrumentation_warning"]:
                            axis.scatter(row["batch_size"], factor * row[metric], marker="x",
                                         color="red", s=70, zorder=5)
            axis.set(xlabel="Batch size (sequences)", ylabel=label, title=f"Context {length}")
            axis.set_xscale("log", base=2)
            axis.grid(alpha=0.25)
            if axis.lines:
                axis.legend()
        fig.savefig(root / name, dpi=160)
        plt.close(fig)
    original = os.path.relpath(manifest["original_campaign"]["path"], root)
    lines = ["# GH200 optimizer timing", "",
             f"Complete: **{report['complete']}**. **{len(rows)}** successful configurations from "
             f"the [original throughput campaign]({original}/README.md). "
             "One GH200 and 14 minutes per allocation; at most twelve concurrent jobs.", "",
             ("Already validated fitting points are scheduled independently. A failed point "
              "pauses further submissions on its curve until a successful retry."
              if manifest.get("parallel_optimizer_points") else
              "Each curve's fitting batches are measured sequentially."), "",
             "The measured boundary is the production `ArenaAdamW.step`: gradient gathering, "
             "global norm and clipping, finite checks, coefficients, AdamW moment/weight updates, "
             "and metric/counter commit. Forward and backward are outside this boundary. "
             "No gradient accumulation; BF16 compute and FP32 parameters/moments.", "",
             "Three external CUDA timing events are captured inside the full update graph: "
             "graph start, optimizer start, and optimizer end. The optimizer runs after the actual "
             "forward/backward on every replay, preserving its production memory-access context. "
             "Events are read after synchronized bursts targeting 0.1 seconds; they measure the "
             "last replay of each burst. Window means weight these samples by burst update count. "
             "This estimates average per-update cost; it is not a trace of every individual update.", "",
             "Each allocation first measures an uninstrumented baseline, then starts a fresh worker "
             "with event instrumentation using the same configuration, GPU and compiler cache. "
             "Each worker uses 20 warmup updates, 5 calibration updates, and five windows targeting "
             "10 seconds (at least 20 updates). Instrumented window wall time is the sum of burst "
             "times; reading events, inspection and logging are excluded. If throughput or optimizer "
             "window CV exceeds 5%, all ten windows are reported. Both workers share a 13-minute deadline.", "",
             "Optimizer ms is the median of window means. The main percentage divides this by the "
             "paired baseline's median full-iteration wall time, including device copies. CSV/JSON "
             "also report shares of the instrumented iteration and captured GPU graph. "
             "The measured wall-time change includes event/synchronization overhead and any drift "
             "between sequential workers; changes over 5% are flagged. Red crosses mark these "
             "warnings or remaining timing instability.", "",
             "Seed 42; eight CPU threads; LR 0.001; AdamW betas (0.9, 0.99), epsilon 1e-8, "
             "weight decay 0.1 and gradient clipping 1.0. Synthetic inputs/targets are generated "
             "once on GPU. Compilation, graph capture, warmup and data generation are excluded.", "",
             "| Model | Context | Batch | Iteration ms (baseline) | Optimizer ms | Optimizer % | Wall change % | Stable |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for model in MODELS:
        for length in contexts:
            for row in sorted((r for r in rows if r["model"] == model and r["context_length"] == length),
                              key=lambda r: r["batch_size"]):
                lines.append(f"| {model} | {length} | {row['batch_size']} | "
                             f"{row['baseline_milliseconds_per_update']:.3f} | "
                             f"{row['optimizer_milliseconds_per_update']:.3f} | "
                             f"{100 * row['optimizer_fraction']:.2f} | "
                             f"{100 * row['instrumented_wall_time_change']:+.2f} | {row['stable']} |")
    warnings = [f"{r['model']}-c{r['context_length']}/b{r['batch_size']}" for r in rows
                if not r["stable"] or r["instrumentation_warning"]]
    lines += ["", f"Flagged points: {', '.join(warnings) if warnings else 'none'}.", "",
              "![Optimizer time](optimizer-time.png)", "", "![Optimizer share](optimizer-share.png)", "",
              "[CSV](results.csv) · [JSON](results.json) · [Frozen manifest](manifest.json)", "",
              "Per-point `baseline.json` and `result.json` retain raw windows, event samples, "
              "update counts, finite-state checks, GPU memory peaks and environment metadata. "
              "Submission receipts retain Slurm accounting. Original throughput/OOM results are unchanged.", "",
              f"Frozen source SHA-256: `{manifest['source_hash']}`.", ""]
    (root / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "collect", "_execute"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--attempt")
    parser.add_argument("--contexts", type=int, nargs="+", choices=CONTEXTS,
                        help="contexts to sweep (default: all; optimizer timing inherits its source)")
    parser.add_argument("--parallel-optimizer-points", action="store_true",
                        help="time known fitting optimizer points independently, up to twelve at once")
    parser.add_argument("--optimizer-timing-from", type=Path,
                        help="remeasure fitting points from a completed throughput campaign")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.command == "run":
        if args.poll_seconds <= 0 or (args.retry_failed and not args.resume):
            parser.error("positive poll interval and --resume with --retry-failed required")
        if args.parallel_optimizer_points and not (args.optimizer_timing_from or args.resume):
            parser.error("--parallel-optimizer-points requires --optimizer-timing-from")
        if args.dry_run:
            original, limits = (optimizer_grid(args.optimizer_timing_from.resolve())
                                if args.optimizer_timing_from else (None, None))
            contexts = selected_contexts(args.contexts, original)
            resources = RESOURCES
            protocol = dict(PROTOCOL, optimizer_timing=True, paired_baseline=True) if limits else PROTOCOL
            parallel_points = args.parallel_optimizer_points
            if args.resume:
                manifest = validate_manifest(root)
                contexts = tuple(manifest["contexts"])
                resources, protocol, limits = manifest["resources"], manifest["protocol"], manifest.get("batch_limits")
                parallel_points = manifest.get("parallel_optimizer_points", False)
                if args.contexts is not None and selected_contexts(args.contexts) != contexts:
                    parser.error("resume cannot change frozen contexts")
            print(json.dumps(dict(output=str(root), models=MODELS, contexts=contexts,
                                  resources=resources, protocol=protocol, concurrency=12,
                                  parallel_optimizer_points=parallel_points,
                                  optimizer_batch_limits=limits,
                                  sweep="paired baseline and optimizer timing of fitting batches"
                                  if limits else "one job per point; B=1,2,4,... through CUDA OOM"), indent=2))
            return
        if not run(root, resume=args.resume, retry_failed=args.retry_failed,
                   poll_seconds=args.poll_seconds,
                   contexts=args.contexts,
                   parallel_optimizer_points=args.parallel_optimizer_points,
                   optimizer_timing_from=args.optimizer_timing_from.resolve()
                   if args.optimizer_timing_from else None):
            raise SystemExit(1)
    elif args.command == "status":
        state = read(root / "state.json")
        for curve in state["curves"]:
            print(f"{curve['id']}: next_batch={curve['next_batch']} complete={curve['complete']} "
                  f"blocked={curve['blocked']} current={curve.get('active_points', curve['current'])}")
    elif args.command == "collect":
        print(json.dumps(collect(root)["summaries"], indent=2))
    else:
        if not args.attempt:
            parser.error("_execute requires --attempt")
        raise SystemExit(execute(root, args.attempt))


if __name__ == "__main__":
    main()
