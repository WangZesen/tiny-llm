"""Audit the frozen eight-worker AWC sweep and export all measurements."""

import csv
import hashlib
import itertools
import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent.parent
ARCHIVE = REPO / "runs/packed8-awc-20m-batch128k-20260912"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    with (ROOT / name).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    for name, expected in manifest["hashes"].items():
        assert sha(ARCHIVE / name) == expected, name
    package = ARCHIVE / "source/tiny_llm"
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    source_hash = digest.hexdigest()
    job = json.loads((ARCHIVE / "submission.json").read_text())["sweep"]
    accounting = subprocess.check_output(
        ["sacct", "-X", "-j", job, "--noheader", "--parsable2", "--format=JobID,State,ExitCode"],
        text=True,
    )
    jobs = {line.split("|")[0]: line.split("|")[1:3] for line in accounting.splitlines()}
    assert len(jobs) == 240
    assert all(value == ["COMPLETED", "0:0"] for value in jobs.values())
    (ROOT / "slurm-accounting.txt").write_text(accounting)
    rows, groups, environments, artifacts = [], defaultdict(list), [], {}
    for row in manifest["runs"]:
        directory = Path(row["directory"])
        expected = yaml.safe_load(Path(row["config"]).read_text())
        actual = yaml.safe_load((directory / "resolved.yaml").read_text())
        assert actual == expected, row["run_id"]
        assert actual["training"]["checkpoint_policy"] == "none"
        assert actual["training"]["batch_tokens"] == 131072
        assert actual["training"]["micro_batch_size"] == 16
        assert actual["decentralized"] == dict(
            num_models=8, topology="one_peer_exponential", scheme="awc", adaptive_consensus=None
        )
        assert not list(directory.rglob("*.pt"))
        assert not list(directory.rglob("*.safetensors"))
        result = json.loads((directory / "result.json").read_text())
        assert result == json.loads((directory / "status.json").read_text())
        assert result["status"] == "complete"
        for key, value in dict(
            parameters=20403520,
            total_parameters=163228160,
            num_models=8,
            topology="one_peer_exponential",
            scheme="awc",
            tokens=408068096,
            tokens_per_model=51008512,
            step=3119,
            epochs=40,
        ).items():
            assert result[key] == value, (row["run_id"], key)
        val = result["final_validation"]
        assert val["split"] == "full" and val["validation_complete"]
        assert val["tokens"] == 197411295 and math.isfinite(val["loss"])
        env = json.loads((directory / "environment.json").read_text())
        assert env["source_hash"] == source_hash
        assert env["scheme"] == "awc" and env["num_models"] == 8
        assert env["loader"]["cache_identity"] == manifest["cache_identity"]
        assert env["loader"]["seed"] == row["seed"]
        assert jobs[f"{job}_{row['index']}"] == ["COMPLETED", "0:0"]
        environments.append(
            {k: env[k] for k in ["python", "versions", "cuda", "gpu", "cpu_threads"]}
        )
        for name in ["result.json", "resolved.yaml", "environment.json"]:
            artifacts[str((directory / name).relative_to(REPO))] = sha(directory / name)
        output = dict(
            index=row["index"],
            run_id=row["run_id"],
            job_id=f"{job}_{row['index']}",
            lr=row["lr"],
            beta1=row["beta1"],
            beta2=row["beta2"],
            seed=row["seed"],
            loss=val["loss"],
            seconds=result["seconds_this_session"],
            training_seconds=result["training_seconds"],
            peak_memory_bytes=result["peak_memory_bytes"],
            directory=str(directory.relative_to(REPO)),
        )
        rows.append(output)
        groups[(row["lr"], row["beta1"], row["beta2"])].append(output)
    assert all(env == environments[0] for env in environments)
    assert len(rows) == 240 and len(groups) == 80
    assert set(groups) == set(
        itertools.product(manifest["lrs"], manifest["beta1s"], manifest["beta2s"])
    )
    summary = []
    for (lr, beta1, beta2), members in groups.items():
        assert sorted(r["seed"] for r in members) == [42, 43, 44]
        losses = [r["loss"] for r in members]
        summary.append(
            dict(
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                seeds=3,
                mean_loss=statistics.mean(losses),
                std_loss=statistics.stdev(losses),
            )
        )
    summary.sort(key=lambda r: (r["mean_loss"], r["lr"], r["beta1"], r["beta2"]))
    reference_path = REPO / "doc/data/recipe_sweep_packed4_20m_128k/results.json"
    four = json.loads(reference_path.read_text())
    four_groups = {(r["lr"], r["beta1"], r["beta2"]): r for r in four["groups"]}
    matched = []
    for r in summary:
        key = (r["lr"], r["beta1"], r["beta2"])
        if key in four_groups:
            f = four_groups[key]
            matched.append(
                dict(
                    lr=r["lr"],
                    beta1=r["beta1"],
                    beta2=r["beta2"],
                    eight_mean=r["mean_loss"],
                    eight_std=r["std_loss"],
                    four_mean=f["mean_loss"],
                    four_std=f["std_loss"],
                    difference_eight_minus_four=r["mean_loss"] - f["mean_loss"],
                )
            )
    assert len(matched) == 28
    write_csv("runs.csv", rows)
    write_csv("summary.csv", summary)
    write_csv(
        "matched_packed4.csv", sorted(matched, key=lambda r: (r["lr"], r["beta1"], r["beta2"]))
    )
    results = dict(
        schema_version=1,
        measured_date="2026-09-12",
        run_count=240,
        group_count=80,
        selection="Mean final full-validation loss; ties: LR, beta1, beta2",
        winner=summary[0],
        groups=summary,
        runs=rows,
        protocol={k: v for k, v in manifest.items() if k not in ["hashes", "runs"]},
        archive=str(ARCHIVE.relative_to(REPO)),
        source_hash=source_hash,
        source_and_config_hashes=manifest["hashes"],
        artifact_hashes=artifacts,
        environment=environments[0],
        job_id=job,
        audit=dict(successful_jobs=240, complete_three_seed_groups=80, checkpoint_files=0),
        packed4_reference=dict(
            path=str(reference_path.relative_to(REPO)),
            sha256=sha(reference_path),
            winner=four["winner"],
            run_count=four["run_count"],
        ),
        matched_packed4=matched,
    )
    (ROOT / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(dict(winner=summary[0], top5=summary[:5], environment=environments[0]), indent=2)
    )


if __name__ == "__main__":
    main()
