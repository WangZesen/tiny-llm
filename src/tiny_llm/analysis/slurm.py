"""Login-node orchestration of independent, checkpoint-sharded Slurm jobs.

No CUDA computation occurs here. Each shard has its own analysis lock and files;
only the orchestrator publishes the combined manifest and figures.
"""

import copy
import fcntl
import hashlib
import json
import logging
import math
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypedDict

import torch

from tiny_llm.analysis.core import (
    EpochData,
    checkpoint_paths,
    load_checkpoint,
    result_key,
    validate_consensus,
)
from tiny_llm.analysis.plot import plot_analysis
from tiny_llm.config import PRESETS, ModelConfig, load_config
from tiny_llm.data import TokenCache
from tiny_llm.runtime import atomic_json, source_digest

LOG = logging.getLogger(__name__)
TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
    "SPECIAL_EXIT",
}


def repository_root():
    """Locate launchers when imported from a checkout or a frozen source snapshot."""
    for directory in (*Path(__file__).resolve().parents, Path.cwd()):
        if (directory / "pyproject.toml").is_file() and (directory / "scripts/slurm.sh").is_file():
            return directory
    raise ValueError("run the analysis submitter from the tiny-llm repository")


def allocation_hours(seconds):
    return math.ceil((2 * seconds + 1800) / 3600)


class TimingProfile(TypedDict):
    profile: str
    checkpoint_seconds: float
    startup_seconds: float


def measured_timing(config, data, options) -> TimingProfile | None:
    """Conservative GH200 timings; see doc/performance/checkpoint-analysis.md for measurements.

    Full passes scale by microbatch counts, including padded boundary batches.
    Other execution recipes require an explicit wall time, not extrapolation.
    """
    if not (
        config.model == ModelConfig(**PRESETS["20m"])
        and config.decentralized is None
        and config.training.micro_batch_size == 32
        and config.training.batch_tokens == 32768
        and options.dtype == "float32"
        and options.amp
        and options.tf32
        and options.compile_hvp
        and options.hvp_batch_size == 64
        and options.device == "cuda"
    ):
        return None
    seconds = 0.0
    for row in data:
        available = row["available_noise_batches"]
        count = (
            available if options.noise_samples == "all" else min(options.noise_samples, available)
        )
        mean = 30 * math.ceil(row["blocks"] / 32) / 312
        hvp = 33 * math.ceil(row["blocks"] / 64) / 156
        seconds += mean + count * (0.15 + hvp) + options.random_samples * hvp
    return TimingProfile(profile="20m-gh200-bf16", checkpoint_seconds=seconds, startup_seconds=60.0)


def plan_shards(
    checkpoints, jobs: int, timing, max_hours: float = 72, walltime_hours: float | None = None
) -> dict[str, Any]:
    groups = {}
    for checkpoint in sorted(checkpoints, key=lambda row: (row["tokens"], row["name"])):
        groups.setdefault(result_key(checkpoint), []).append(checkpoint)
    if not 1 <= jobs <= len(groups):
        raise ValueError(f"jobs must be between 1 and {len(groups)} unique checkpoints")
    if timing is None and walltime_hours is None:
        raise ValueError("unmeasured configuration: provide --walltime-hours")
    if walltime_hours is not None and not 1 <= walltime_hours <= max_hours:
        raise ValueError(f"walltime must be between 1 and {max_hours} hours")
    if timing:
        per_checkpoint, startup = timing["checkpoint_seconds"], timing["startup_seconds"]
        capacity = math.floor((max_hours * 3600 - 1800 - 2 * startup) / (2 * per_checkpoint))
        if capacity < 1:
            raise ValueError("one checkpoint exceeds the partition limit with the required margin")
        minimum = math.ceil(len(groups) / capacity)
        if jobs < minimum:
            raise ValueError(f"request at least --jobs {minimum} to fit the {max_hours}-hour limit")
    shards = []
    for index in range(jobs):
        assigned = list(groups.values())[index::jobs]
        seconds = (
            timing["startup_seconds"] + len(assigned) * timing["checkpoint_seconds"]
            if timing
            else None
        )
        hours = allocation_hours(seconds) if timing else walltime_hours
        if walltime_hours is not None:
            if hours is not None and walltime_hours < hours:
                raise ValueError("walltime override must cover the padded estimate")
            hours = walltime_hours
        shards.append(
            dict(
                index=index,
                checkpoints=[row for group in assigned for row in group],
                unique_checkpoints=len(assigned),
                estimated_seconds=seconds,
                hours=hours,
                attempts=[],
            )
        )
    return dict(
        version=2,
        mode="independent-single-gpu",
        jobs=shards,
        timing=timing,
        checkpoint_files=len(checkpoints),
        unique_checkpoints=len(groups),
        estimated_elapsed_seconds=max(row["estimated_seconds"] for row in shards)
        if timing
        else None,
        estimated_gpu_hours=sum(row["estimated_seconds"] for row in shards) / 3600
        if timing
        else None,
        max_hours=max_hours,
    )


def partition_hours():
    result = subprocess.run(
        ["scontrol", "show", "partition", "gpu", "-o"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    value = next(word.split("=", 1)[1] for word in result.split() if word.startswith("MaxTime="))
    days, clock = value.split("-", 1) if "-" in value else ("0", value)
    hours, _, _ = map(int, clock.split(":"))
    return min(72, int(days) * 24 + hours)


def prepare_plan(run, selected, output, jobs, options, walltime_hours=None):
    config = load_config(run / "resolved.yaml")
    cache = TokenCache(config.data.cache_dir)
    cache.validate_config(config)
    cache.verify()
    # Capacity validation is deliberately before hashing checkpoints or submitting anything.
    cases = ("seen", "unseen") if options.data_case == "both" else (options.data_case,)
    data = [EpochData(cache, config, case).metadata() for case in cases]
    infos = []
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config.runtime.cpu_threads)
        for path in checkpoint_paths(run, selected):
            model, info = load_checkpoint(config, path, consensus=options.consensus)
            infos.append(info)
            del model
    finally:
        torch.set_num_threads(threads)
    timing = measured_timing(config, data, options)
    plan = plan_shards(infos, jobs, timing, partition_hours(), walltime_hours)
    plan.update(
        run=str(run),
        output=str(output),
        selected=list(selected),
        options=asdict(options),
        data=data,
        micro_batch_size=config.training.micro_batch_size,
        source_hash=source_digest(Path(__file__).parents[1]),
        source_snapshot=str(output / "source"),
        config_hash=hashlib.sha256((run / "resolved.yaml").read_bytes()).hexdigest(),
        resources=dict(account="naiss2026-3-205-gpu", partition="gpu", nodes=1, tasks=1, gpus=1),
        status="planned",
    )
    for row in plan["jobs"]:
        row["output"] = str(output / "shards" / f"{row['index']:03d}")
        unique = {result_key(info): info for info in row["checkpoints"]}
        row["consensus_workers"] = {
            key: len(info.get("workers", [])) for key, info in unique.items()
        }
        row["additional_hvp_passes"] = len(cases) * sum(row["consensus_workers"].values())
    return plan


def snapshot_source(plan):
    repository = repository_root()
    snapshot = Path(plan["source_snapshot"])
    shutil.copytree(
        Path(__file__).parents[1],
        snapshot / "tiny_llm",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(repository / "scripts" / "slurm.sh", snapshot / "slurm.sh")
    if source_digest(snapshot / "tiny_llm") != plan["source_hash"]:
        raise ValueError("analysis source changed while preparing submission")


def job_command(plan, shard, attempt):
    snapshot = Path(plan["source_snapshot"])
    launcher = snapshot / "slurm.sh"
    directory = Path(shard["output"])
    log = str(directory / f"slurm-attempt-{attempt}-%j.log")
    command = [
        "sbatch",
        "--parsable",
        f"--time={shard['hours'] // 24}-{shard['hours'] % 24:02d}:00:00",
        "--account=naiss2026-3-205-gpu",
        "--partition=gpu",
        "--nodes=1",
        "--ntasks=1",
        "--gpus=nvidia_gh200_120gb:1" if plan.get("timing") else "--gpus=1",
        f"--job-name=llm-analysis-{shard['index']:03d}-a{attempt}",
        f"--output={log}",
        f"--export=ALL,PYTHONPATH={snapshot}",
        str(launcher),
    ]
    command.append("analyze")
    command.extend(["--run", plan["run"], "--output", str(directory), "--no-plots"])
    command.extend(["--checkpoint-manifest", str(directory / "checkpoints.json")])
    for key, value in plan["options"].items():
        if value is None:
            continue
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            command.append(f"--{flag}" if value else f"--no-{flag}")
        else:
            command.extend([f"--{flag}", str(value)])
    command.extend(["--checkpoints", *[row["name"] for row in shard["checkpoints"]]])
    return command, log


def scheduler_states(job_ids):
    """Account only the allocation rows, never .batch/.extern/step rows."""
    if not job_ids:
        return {}
    ids = ",".join(job_ids)
    queued = subprocess.run(
        ["squeue", "--noheader", "--jobs", ids, "--format=%i|%T"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    accounted = subprocess.run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs",
            ids,
            "--format=JobIDRaw,State%40,ExitCode",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    states = {}
    for line in accounted.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 3 and fields[0] in job_ids:
            states[fields[0]] = dict(state=fields[1].split()[0].rstrip("+"), exit_code=fields[2])
    # An allocation still in squeue is active, including COMPLETING; accounting can lag.
    for line in queued.splitlines():
        fields = line.strip().split("|")
        if len(fields) == 2 and fields[0] in job_ids:
            states[fields[0]] = dict(state=fields[1], exit_code=None)
    return {job_id: states.get(job_id, dict(state="UNKNOWN", exit_code=None)) for job_id in job_ids}


def validated_shard(plan, shard):
    directory = Path(shard["output"])
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest["run"] != plan["run"]
        or manifest["options"] != plan["options"]
        or manifest["data"] != plan["data"]
        or manifest["micro_batch_size"] != plan["micro_batch_size"]
        or manifest["environment"]["source_hash"] != plan["source_hash"]
    ):
        raise ValueError(f"incompatible shard metadata: {directory}")
    expected = {row["name"]: row for row in shard["checkpoints"]}
    actual = {row["name"]: row for row in manifest["checkpoints"]}
    if actual != expected or len(actual) != len(manifest["checkpoints"]):
        raise ValueError(f"checkpoint coverage mismatch: {directory}")
    results = {}
    for row in expected.values():
        for data in plan["data"]:
            name = f"{result_key(row)}-{data['case']}.json"
            if name in results:
                continue
            result = json.loads((directory / name).read_text())
            if (
                result["identity"] != manifest["identity"]
                or result["weights_hash"] != row["weights_hash"]
                or result_key(result) != result_key(row)
                or result["case"] != data["case"]
                or result["data"] != data
            ):
                raise ValueError(f"incompatible result: {directory / name}")
            available = data["available_noise_batches"]
            samples = plan["options"]["noise_samples"]
            count = available if samples == "all" else min(samples, available)
            if (
                len(result["noise"]) != count
                or len(result["random"]) != plan["options"]["random_samples"]
                or result["statistics"]["noise_samples"] != count
                or result["statistics"]["random_samples"] != len(result["random"])
            ):
                raise ValueError(f"incomplete sample counts: {directory / name}")
            validate_consensus(result, row)
            results[name] = result
    return manifest, results


def collect_results(plan):
    manifests, combined = [], {}
    for shard in plan["jobs"]:
        manifest, results = validated_shard(plan, shard)
        if manifests and manifest["identity"] != manifests[0]["identity"]:
            raise ValueError("shards have incompatible numerical identities")
        if combined.keys() & results.keys():
            raise ValueError("checkpoint was assigned to multiple shards")
        manifests.append(manifest)
        combined.update(results)
    merged = copy.deepcopy(manifests[0])
    merged["checkpoints"] = sorted(
        [row for manifest in manifests for row in manifest["checkpoints"]],
        key=lambda row: (row["tokens"], row["name"]),
    )
    merged["workers"] = [
        dict(
            index=shard["index"],
            attempts=shard["attempts"],
            environment=manifest["environment"],
            hvp_warmup_seconds=manifest.get("hvp_warmup_seconds", 0),
        )
        for shard, manifest in zip(plan["jobs"], manifests, strict=True)
    ]
    merged["hvp_warmup_seconds"] = max(row["hvp_warmup_seconds"] for row in merged["workers"])
    output = Path(plan["output"])
    for name, result in combined.items():
        atomic_json(output / name, result)
    atomic_json(output / "manifest.json", merged)
    plot_analysis(output)
    return merged


def submit_pending(plan, receipt, *, resume):
    ids = [row["attempts"][-1]["job_id"] for row in plan["jobs"] if row["attempts"]]
    states = scheduler_states(ids) if resume else {}
    repository = repository_root()
    for shard in plan["jobs"]:
        if shard.get("submission_uncertain"):
            raise RuntimeError(
                f"shard {shard['index']} has an uncertain submission; reconcile its Slurm job ID "
                "in submission.json before resuming to avoid duplicate work"
            )
        if shard["attempts"]:
            state = states[shard["attempts"][-1]["job_id"]]
            if state["state"] not in TERMINAL:
                continue
            if state["state"] == "COMPLETED" and state["exit_code"] == "0:0":
                try:
                    validated_shard(plan, shard)
                    continue
                except (OSError, ValueError, KeyError):
                    LOG.warning("Shard %s has incomplete or invalid outputs", shard["index"])
        Path(shard["output"]).mkdir(parents=True, exist_ok=True)
        atomic_json(Path(shard["output"]) / "checkpoints.json", shard["checkpoints"])
        command, log = job_command(plan, shard, len(shard["attempts"]) + 1)
        # A crash between sbatch accepting the job and receipt persistence must not
        # cause a blind duplicate submission on restart.
        shard["submission_uncertain"] = True
        shard["pending_command"] = command
        atomic_json(receipt, plan)
        try:
            response = subprocess.run(
                command, cwd=repository, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError:
            shard["submission_uncertain"] = False
            atomic_json(receipt, plan)
            raise
        job_id = response.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"unexpected sbatch response: {response.stdout}")
        shard["attempts"].append(
            dict(job_id=job_id, command=command, log=log.replace("%j", job_id))
        )
        shard["submission_uncertain"] = False
        shard.pop("pending_command", None)
        plan["status"] = "submitted"
        atomic_json(receipt, plan)
        LOG.info(
            "Submitted shard %s: job %s, %s checkpoints, %sh, log %s",
            shard["index"],
            job_id,
            shard["unique_checkpoints"],
            shard["hours"],
            shard["attempts"][-1]["log"],
        )


def wait_and_collect(plan, receipt, poll_seconds=60):
    ids = [row["attempts"][-1]["job_id"] for row in plan["jobs"]]
    while True:
        try:
            states = scheduler_states(ids)
        except (subprocess.SubprocessError, OSError) as exc:
            LOG.warning("Scheduler query failed; retaining jobs and retrying: %s", exc)
            time.sleep(poll_seconds)
            continue
        plan["scheduler"] = states
        atomic_json(receipt, plan)
        LOG.info("Scheduler: %s", ", ".join(f"{key}={row['state']}" for key, row in states.items()))
        if all(row["state"] in TERMINAL for row in states.values()):
            break
        time.sleep(poll_seconds)
    failed = {
        key: row
        for key, row in states.items()
        if row["state"] != "COMPLETED" or row["exit_code"] != "0:0"
    }
    try:
        if failed:
            raise RuntimeError(
                f"analysis jobs failed: {failed}; use --resume to retry unfinished work"
            )
        collect_results(plan)
    except Exception as exc:
        plan.update(status="failed", error=str(exc))
        atomic_json(receipt, plan)
        raise
    plan.update(status="complete", completed_at=time.time())
    plan.pop("error", None)
    atomic_json(receipt, plan)
    return plan


def orchestrate(
    run, selected, output, jobs, options, *, dry_run=False, resume=False, walltime_hours=None
):
    run, output = run.resolve(), output.resolve()
    if options.device != "cuda":
        raise ValueError("Slurm analysis requires --device cuda")
    if dry_run:
        return prepare_plan(run, selected, output, jobs, options, walltime_hours)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".orchestrator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = output / "submission.json"
        if resume:
            plan = json.loads(receipt.read_text())
            if plan.get("version") != 2:
                raise ValueError("legacy campaign: resume using its frozen source entrypoint")
            if (
                plan.get("mode") != "independent-single-gpu"
                or plan["run"] != str(run)
                or plan["output"] != str(output)
                or len(plan["jobs"]) != jobs
                or set(plan["selected"]) != set(selected)
                or plan["options"] != asdict(options)
            ):
                raise ValueError("resume arguments differ from the saved campaign")
            if (
                source_digest(Path(plan["source_snapshot"]) / "tiny_llm") != plan["source_hash"]
                or hashlib.sha256((run / "resolved.yaml").read_bytes()).hexdigest()
                != plan["config_hash"]
            ):
                raise ValueError("saved source or resolved.yaml changed; refusing to resume")
        else:
            if receipt.exists() or (output / "manifest.json").exists():
                raise ValueError("output already contains a campaign; use --resume or a new output")
            plan = prepare_plan(run, selected, output, jobs, options, walltime_hours)
            snapshot_source(plan)
            atomic_json(receipt, plan)
        submit_pending(plan, receipt, resume=resume)
        return wait_and_collect(plan, receipt)
