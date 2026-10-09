#!/usr/bin/env python3
"""Audit a completed frozen GH200 campaign, including actual Slurm allocations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(root):
    audit_source = Path(__file__).read_bytes()
    sys.path.insert(0, str(root / "source"))
    spec = importlib.util.spec_from_file_location("frozen_driver", root / "source/throughput_sweep.py")
    if spec is None or spec.loader is None:
        raise ValueError("missing frozen campaign driver")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    manifest = driver.validate_manifest(root)
    report = driver.read(root / "results.json")
    require(report["complete"] and report["campaign_identity"] == manifest["identity"],
            "campaign incomplete or mismatched")
    paired = manifest.get("mode") == "optimizer_timing"
    if paired:
        original = Path(manifest["original_campaign"]["path"])
        source, limits = driver.optimizer_grid(original)
        require(source["identity"] == manifest["original_campaign"]["identity"]
                and limits == manifest["batch_limits"], "original grid changed")
        for path in (root / "source/tiny_llm/gh200").glob("*.py"):
            require(path.read_bytes() == (original / "source/tiny_llm/gh200" / path.name).read_bytes(),
                    "production implementation changed")
    expected_curves = {(m, length) for m in manifest["models"] for length in manifest["contexts"]}
    require({(s["model"], s["context_length"]) for s in report["summaries"]} == expected_curves,
            "missing curve")
    for model, length in expected_curves:
        points = [p for p in report["points"] if p["model"] == model and p["context_length"] == length]
        batches = {p["batch_size"] for p in points if p["status"] == "ok"}
        require(bool(batches), "curve has no successful points")
        largest = max(batches)
        require(batches == {1 << i for i in range(largest.bit_length())}, "noncontiguous batch grid")
        if paired:
            require(largest == manifest["batch_limits"][f"{model}-c{length}"], "missing paired point")
        else:
            require({p["batch_size"] for p in points if p["status"] == "cuda_oom"} == {2 * largest},
                    "no confirmed first OOM boundary")
    ids, results, changes, warnings = [], [], [], []
    for point in report["points"]:
        directory = root / point["directory"]
        receipt = driver.read(directory / "receipt.json")
        ids.append(receipt["job_id"])
        if point["status"] not in ("ok", "cuda_oom"):
            continue
        require(receipt["accounting"]["state"] == "COMPLETED"
                and receipt["accounting"]["exit_code"] == "0:0", "unsuccessful allocation")
        raw = driver.validated_result(directory, manifest, receipt)
        if raw["status"] == "cuda_oom":
            continue
        workers = [raw]
        if paired:
            baseline = driver.validated_result(directory, manifest, receipt, "baseline.json")
            for key in ("hostname", "gpu", "cpu_affinity", "compiler_cache"):
                require(raw["environment"][key] == baseline["environment"][key], "paired environment differs")
            require(raw["environment"]["gpu_status"].splitlines()[1].split(",")[0]
                    == baseline["environment"]["gpu_status"].splitlines()[1].split(",")[0], "paired GPU differs")
            workers.append(baseline)
            changes.append(raw["instrumented_wall_time_change"])
            warnings.append(raw["instrumentation_warning"])
        for result in workers:
            config = result["config"]
            require(config["training"]["micro_batch_size"] == point["batch_size"]
                    and config["training"]["batch_tokens"] == point["batch_size"] * point["context_length"]
                    and config["decentralized"] is None, "accumulation or multiple models")
            require(math.isfinite(result["final_loss"]) and math.isfinite(result["final_grad_norm"]),
                    "nonfinite state")
            require(result["environment"]["cpu_threads"] == 8
                    and result["environment"]["slurm"]["SLURM_JOB_ID"] == receipt["job_id"], "worker identity")
            for phase in ("setup", "measurement"):
                require(0 < result[f"{phase}_peak_allocated_bytes"]
                        <= result[f"{phase}_peak_reserved_bytes"] <= result["total_memory_bytes"], "memory bounds")
            results.append(result)
    require(len(ids) == len(set(ids)), "duplicate allocation")
    accounting = driver.command(["sacct", "-X", "-j", ",".join(ids), "--noheader", "--parsable2",
                                 "--format=JobIDRaw,State,ExitCode,ElapsedRaw,Timelimit,Start,End,AllocTRES%256"])
    events, durations, accounted = [], [], set()
    for line in accounting.splitlines():
        job, state, code, elapsed, limit, start, end, resources = line.split("|")[:8]
        require(job in ids and limit == "00:14:00", "unexpected allocation or time limit")
        resource_map = dict(item.split("=", 1) for item in resources.split(","))
        require(resource_map["cpu"] == "8" and resource_map["gres/gpu"] == "1"
                and resource_map["mem"] == "64G", "allocation resources differ")
        accounted.add(job)
        durations.append(int(elapsed))
        events += [(datetime.fromisoformat(start), 1), (datetime.fromisoformat(end), -1)]
    require(accounted == set(ids), "incomplete scheduler accounting")
    concurrent = maximum = 0
    for _, change in sorted(events):
        concurrent += change
        maximum = max(maximum, concurrent)
    require(concurrent == 0 and maximum <= 12 and max(durations) <= 840, "resource limits exceeded")
    summary = dict(
        passed=True, campaign_identity=manifest["identity"], contexts=manifest["contexts"],
        curves=len(expected_curves), points=dict(Counter(p["status"] for p in report["points"])),
        jobs=len(ids), job_ids=ids, exact_counts_and_arithmetic=True, all_finite=True,
        maximum_timing_cv=max(r["coefficient_of_variation"] for r in results),
        unstable_workers=sum(not r["stable"] for r in results), instrumentation_warnings=sum(warnings),
        maximum_optimizer_cv=max((r.get("optimizer_coefficient_of_variation", 0.) for r in results)),
        maximum_absolute_instrumented_wall_change=max(map(abs, changes), default=0.),
        maximum_concurrent_jobs=maximum, longest_job_seconds=max(durations), total_gpu_seconds=sum(durations),
        source_hash=manifest["source_hash"],
        results_sha256=hashlib.sha256((root / "results.json").read_bytes()).hexdigest(),
        audit_script_sha256=hashlib.sha256(audit_source).hexdigest(),
    )
    (root / "source/throughput_audit.py").write_bytes(audit_source)
    (root / "accounting.txt").write_text(accounting + "\n")
    driver.write(root / "audit.json", summary)
    return summary


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="completed campaign to audit")
    args = parser.parse_args()
    print(json.dumps(audit(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
