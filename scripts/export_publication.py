"""Retain experiment evidence and regenerate the documentation without training imports.

Import once: uv run python scripts/export_publication.py --import-runs runs
Regenerate: uv run python scripts/export_publication.py
Verify: uv run python scripts/export_publication.py --check
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import re
import shutil
import statistics
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "doc/data/current-training"
SEEDS = [42, 43, 44]
METHODS = {"sync": "Synchronous", "awc4": "Four workers", "awc8": "Eight workers"}
SCHEDULES = {"cosine": "Cosine-to-zero", "wsd": "WSD"}
COLORS = {"sync": "#30658a", "awc4": "#b25c3c", "awc8": "#766193"}
SOURCES = [
    (
        "cosine",
        "sync",
        "sync-cosine-zero-20m-horizon-tuning",
        "combined-20-40-80-lr-expanded-2",
        216,
    ),
    (
        "cosine",
        "awc4",
        "packed4-awc-cosine-zero-20m-horizon-tuning",
        "combined-20-40-80-lr-expanded-2",
        432,
    ),
    (
        "cosine",
        "awc8",
        "packed8-awc-cosine-zero-20m-horizon-tuning",
        "combined-20-40-80-lr-expanded-3",
        540,
    ),
    ("wsd", "sync", "sync-wsd-20m-horizon-tuning", "", 270),
    ("wsd", "awc8", "packed8-awc-wsd-20m-horizon-tuning", "", 324),
]
METRICS = ["steady", "training", "elapsed", "validationSeconds", "sessionSeconds", "peakGiB"]
# Per-worker diagnostics describe the individual packed models. No published result
# reads them and they are 37% of the logged bytes, so the retained copy omits them.
DROPPED_METRIC_FIELDS = ("local_grad_norms", "local_losses")
# Deleting those members is a textual deletion: every other byte of every line, and
# the line order, are preserved. The pattern states that claim so it can be checked.
HF_REPO = "zesen-kth/tiny-llm"
# The untouched per-file bundle is archived here before the repack, so nothing the
# repack deletes is lost even though the original runs/ tree no longer exists.
HF_ORIGINAL_REVISION = "per-file-original"
DROPPED_METRIC_PATTERN = re.compile(rb', "local_(?:losses|grad_norms)": \[[^]]*\]')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data if isinstance(data, bytes) else data.encode())
    temporary.replace(path)


def json_write(path, value):
    write(path, json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")


def container_of(name: str) -> str:
    """Placement is a pure rule, so a misfiled or renamed container is detectable."""
    parts = PurePosixPath(name).parts
    if parts[0] == "logs":
        kind = "metrics" if parts[-1] == "metrics.jsonl" else "records"
        return f"logs/{parts[1]}-{parts[2]}-{parts[3]}.{kind}.jsonl.gz"
    if parts[0] == "shared":
        return "shared/environments.jsonl.gz"
    if parts[0] == "sources":
        return "sources/campaigns.jsonl.gz"
    raise ValueError(f"No container for {name}")


def strip_metrics(data: bytes) -> tuple[bytes, list[str]]:
    """Delete the per-worker arrays; every other byte of every line is preserved."""
    events, dropped = [], set()
    for line in data.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        for field in DROPPED_METRIC_FIELDS:
            if field in event:
                del event[field]
                dropped.add(field)
        events.append(json.dumps(event))
    text = ("\n".join(events) + "\n").encode()
    require(text == DROPPED_METRIC_PATTERN.sub(b"", data), "Metrics rewrite is not a deletion")
    return text, sorted(dropped)


class Containers:
    """Write grouped JSONL archives, one line per retained artifact.

    Metrics archives concatenate one gzip member per line so a single run stays
    directly addressable; every other archive is one stream, which compresses the
    near-identical run records far better.
    """

    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.lines: dict[str, list[bytes]] = defaultdict(list)

    def add(self, name: str, text: bytes, source: dict, dropped: list[str]) -> dict:
        require(text.decode().encode() == text, f"Retained artifact is not UTF-8: {name}")
        container = container_of(name)
        entry = {"path": name, "bytes": len(text), "sha256": digest(text)}
        record = dict(source, container=container)
        if dropped:
            record |= {
                "dropped": dropped,
                "retainedSha256": entry["sha256"],
                "retainedBytes": entry["bytes"],
            }
        line = json.dumps(dict(entry, text=text.decode()), separators=(",", ":")).encode() + b"\n"
        if container.endswith(".metrics.jsonl.gz"):
            member = gzip.compress(line, 6, mtime=0)
            record |= {
                "offset": sum(len(m) for m in self.lines[container]),
                "compressedBytes": len(member),
            }
            self.lines[container].append(member)
        else:
            self.lines[container].append(line)
        return record

    def close(self) -> None:
        for container, lines in self.lines.items():
            body = b"".join(lines)
            write(
                self.destination / container,
                body
                if container.endswith(".metrics.jsonl.gz")
                else gzip.compress(body, 6, mtime=0),
            )


def table(root, name, rows):
    json_write(root / f"{name}.json", rows)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(
        {
            k: json.dumps(v, separators=(",", ":")) if isinstance(v, (dict, list)) else v
            for k, v in row.items()
        }
        for row in rows
    )
    write(root / f"{name}.csv", stream.getvalue())


def group_id(schedule, method, horizon, lr, beta1, beta2):
    return f"{method}-{schedule}-h{horizon}--{lr:g}--{beta1:g}--{beta2:g}"


def import_sources(runs_root, destination):
    """Copy only publication evidence; original artifact paths remain provenance."""
    mapping, rows, campaigns = {}, [], []
    imported = {}
    containers = Containers(destination)

    def resolve(original):
        p = Path(original)
        parts = p.parts
        require("runs" in parts, f"Unrecognized source path: {p}")
        return runs_root.joinpath(*parts[parts.index("runs") + 1 :])

    def retain(path, target):
        original = str(path.resolve())
        if original in imported:
            return imported[original]
        data = path.read_bytes()
        text, dropped = strip_metrics(data) if path.name == "metrics.jsonl" else (data, [])
        # sha256 and bytes stay in the original domain: they are this artifact's identity
        # on the system that wrote it, which retention must never silently redefine.
        mapping[target] = containers.add(
            target,
            text,
            {"originalPath": original, "sha256": digest(data), "bytes": len(data)},
            dropped,
        )
        imported[original] = target
        return target

    for schedule, method, folder, combined, expected in SOURCES:
        source = runs_root / folder
        report = source / combined
        native = read_json(report / "runs.json")
        require(len(native) == expected, f"Incomplete {schedule}/{method}: {len(native)}")
        prefix = f"sources/{schedule}/{method}"
        campaign = {
            "schedule": schedule,
            "method": method,
            "expected": expected,
            "summary": retain(report / "summary.json", f"{prefix}/summary.json"),
            "runs": retain(report / "runs.json", f"{prefix}/runs.json"),
            "definitions": [],
        }
        for name in ["coverage.json", "audit.json"]:
            if (report / name).exists():
                campaign[name[:-5]] = retain(report / name, f"{prefix}/{name}")
        # Combined audits reference the native audits of their constituent campaigns.
        # Retain that small chain too, so verifying it never requires the source tree.
        campaign["supportingAudits"] = []
        pending = [read_json(report / "audit.json")] if (report / "audit.json").exists() else []
        audited = set()
        while pending:
            audit = pending.pop()
            references = list(audit.get("parent_audits", []))
            if audit.get("extension_audit"):
                references.append(audit["extension_audit"])
            for reference in references:
                if reference["path"] in audited:
                    continue
                audited.add(reference["path"])
                path = resolve(reference["path"])
                raw = path.read_bytes()
                require(
                    len(raw) == reference["bytes"] and digest(raw) == reference["sha256"],
                    "Referenced audit checksum mismatch",
                )
                campaign["supportingAudits"].append(
                    retain(path, f"{prefix}/audit-{reference['sha256']}.json")
                )
                pending.append(json.loads(raw))
        if schedule == "wsd":
            definition = read_json(source / "campaign.json")
            require(definition["status"] == "complete", "WSD campaign is incomplete")
            # Use successful native receipts, rather than discovering arbitrary attempt folders.
            native = [
                dict(
                    r,
                    horizon=s["horizon"],
                    directory=str(Path(r["completed"]["result"]["path"]).parent),
                )
                for s in definition["stages"]
                for r in s["runs"]
            ]
        definitions = {}
        for r in native:
            h = r["horizon"]
            gid = group_id(schedule, method, h, r["lr"], r["beta1"], r["beta2"])
            directory = resolve(r["directory"])
            receipt = read_json(directory / "receipt.json")
            target = f"logs/{schedule}/{method}/h{h}/lr-{r['lr']:g}-b1-{r['beta1']:g}-b2-{r['beta2']:g}/s{r['seed']}"
            artifacts = {}
            for name in [
                "metrics.jsonl",
                "result.json",
                "resolved.yaml",
                "receipt.json",
                "request.json",
            ]:
                artifacts[name] = retain(directory / name, f"{target}/{name}")
            environment = (
                directory / "environment.json"
                if schedule == "cosine"
                else resolve(receipt["environment"]["path"])
            )
            artifacts["environment.json"] = retain(
                environment, f"shared/environment-{digest(environment.read_bytes())}.json"
            )
            if schedule == "wsd" and h > 20:
                artifacts["continuation.json"] = retain(
                    directory / "continuation.json", f"{target}/continuation.json"
                )
            ancestor = next(p for p in directory.parents if (p / "campaign.json").exists())
            definition = read_json(ancestor / "campaign.json")
            identity = definition["identity"]
            if identity not in definitions:
                definitions[identity] = retain(
                    ancestor / "campaign.json", f"{prefix}/campaign-{identity}.json"
                )
            rows.append(
                {
                    "id": f"{gid}--s{r['seed']}",
                    "configuration": gid,
                    "schedule": schedule,
                    "method": method,
                    "horizon": h,
                    "lr": r["lr"],
                    "beta1": r["beta1"],
                    "beta2": r["beta2"],
                    "seed": r["seed"],
                    "artifacts": artifacts,
                }
            )
        campaign["definitions"] = list(definitions.values())
        campaigns.append(campaign)
        print(
            f"Retained {len(native)} {SCHEDULES[schedule]} / {METHODS[method]} records", flush=True
        )
    containers.close()
    json_write(destination / "sources.json", mapping)
    json_write(destination / "manifest.json", {"version": 1, "campaigns": campaigns, "runs": rows})


def repack_sources(bundle: Path, destination: Path) -> None:
    """Migrate a per-file bundle into containers. The original runs/ tree is long gone,
    so this transforms the committed evidence rather than re-importing it."""
    legacy = read_json(bundle / "sources.json")
    require(
        all("container" not in source for source in legacy.values()), "Bundle is already packed"
    )
    containers = Containers(destination)
    mapping, renamed, legacy_name = {}, {}, {}
    for name in sorted(legacy):
        source = legacy[name]
        raw = (bundle / name).read_bytes()
        data = gzip.decompress(raw) if source["compression"] == "gzip" else raw
        # The last moment the recorded original digests can be checked against real bytes.
        require(
            len(data) == source["bytes"] and digest(data) == source["sha256"],
            f"Artifact checksum mismatch: {name}",
        )
        # Compression becomes the container's concern, so the artifact keeps its own name.
        logical = name.removesuffix(".gz")
        text, dropped = strip_metrics(data) if logical.endswith("metrics.jsonl") else (data, [])
        mapping[logical] = containers.add(
            logical, text, {k: source[k] for k in ["originalPath", "sha256", "bytes"]}, dropped
        )
        renamed[name], legacy_name[logical] = logical, name
    containers.close()
    json_write(destination / "sources.json", mapping)

    manifest = read_json(bundle / "manifest.json")
    for campaign in manifest["campaigns"]:
        for key in ["summary", "runs", "coverage", "audit"]:
            if key in campaign:
                campaign[key] = renamed[campaign[key]]
        for key in ["definitions", "supportingAudits"]:
            campaign[key] = [renamed[name] for name in campaign.get(key, [])]
    for row in manifest["runs"]:
        row["artifacts"] = {k: renamed[v] for k, v in row["artifacts"].items()}
    json_write(destination / "manifest.json", manifest)

    # Prove the transform against the bytes it replaced, one artifact at a time.
    identical = deleted = 0
    for container in sorted({record["container"] for record in mapping.values()}):
        with gzip.open(destination / container, "rb") as stream:
            for line in stream:
                entry = json.loads(line)
                logical, text = entry["path"], entry["text"].encode()
                name = legacy_name[logical]
                raw = (bundle / name).read_bytes()
                data = gzip.decompress(raw) if legacy[name]["compression"] == "gzip" else raw
                if "dropped" in mapping[logical]:
                    require(
                        text == DROPPED_METRIC_PATTERN.sub(b"", data)
                        and len(text.splitlines()) == len(data.splitlines()),
                        f"Metrics rewrite is not a deletion: {logical}",
                    )
                    deleted += 1
                else:
                    require(text == data, f"Retained bytes differ: {logical}")
                    identical += 1
    require(identical + deleted == len(mapping), "Container entries do not cover the index")

    write(
        destination / "provenance/pre-repack-checksums.json.gz",
        gzip.compress((bundle / "checksums.json").read_bytes(), 6, mtime=0),
    )
    json_write(
        destination / "provenance/repack.json",
        {
            "version": 1,
            "repackedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tool": "scripts/export_publication.py --repack",
            "source": {
                "artifacts": len(legacy),
                "checksumsSha256": digest((bundle / "checksums.json").read_bytes()),
            },
            "retained": {"byteIdentical": identical, "transformed": deleted},
            "dropped": {
                "fields": sorted(DROPPED_METRIC_FIELDS),
                "transform": "deletion of two JSON members from every train event",
            },
            "archive": {"repo": HF_REPO, "revision": HF_ORIGINAL_REVISION},
        },
    )
    print(
        f"Repacked {len(mapping)} artifacts into {len(containers.lines)} containers; "
        f"{identical} byte-identical, {deleted} with per-worker arrays deleted",
        flush=True,
    )


def unpacked_names(value):
    """Retained metrics kept a .gz suffix before the repack; nothing else moved."""
    if isinstance(value, dict):
        return {k: unpacked_names(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpacked_names(v) for v in value]
    if isinstance(value, str) and value.endswith("metrics.jsonl.gz"):
        return value.removesuffix(".gz")
    return value


def artifact_reader(root):
    sources = read_json(root / "sources.json")
    grouped = defaultdict(dict)
    for name, source in sources.items():
        require(
            not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe artifact path"
        )
        require(source["container"] == container_of(name), f"Misplaced artifact: {name}")
        require(
            ("dropped" in source) == ("retainedSha256" in source) == ("retainedBytes" in source),
            f"Inconsistent retention record: {name}",
        )
        require(
            set(source.get("dropped", ())) <= set(DROPPED_METRIC_FIELDS),
            f"Unexpected retention record: {name}",
        )
        grouped[source["container"]][name] = source
    # Check every retained byte, including metadata not used in a particular derived figure.
    # An entry without "dropped" is byte-identical to the artifact named by originalPath.
    for container, wanted in grouped.items():
        seen = set()
        with gzip.open(root / container, "rb") as stream:
            for line in stream:
                entry = json.loads(line)
                name, text = entry["path"], entry["text"].encode()
                require(name in wanted, f"Unindexed container entry: {container}/{name}")
                source = wanted[name]
                require(
                    len(text) == entry["bytes"] == source.get("retainedBytes", source["bytes"])
                    and digest(text)
                    == entry["sha256"]
                    == source.get("retainedSha256", source["sha256"]),
                    f"Artifact checksum mismatch: {name}",
                )
                for field in source.get("dropped", ()):
                    require(
                        field.encode() not in text, f"Retained a field declared dropped: {name}"
                    )
                seen.add(name)
        require(seen == set(wanted), f"Missing container entries: {container}")
    cached: dict[str, dict[str, str]] = {}

    def text_of(name: str) -> str:
        source = sources[name]
        if "offset" in source:  # one gzip member per metrics log, so seek straight to it
            with (root / source["container"]).open("rb") as handle:
                handle.seek(source["offset"])
                return json.loads(gzip.decompress(handle.read(source["compressedBytes"])))["text"]
        if source["container"] not in cached:
            with gzip.open(root / source["container"], "rb") as stream:
                cached[source["container"]] = {
                    entry["path"]: entry["text"] for entry in map(json.loads, stream)
                }
        return cached[source["container"]][name]

    def read(name: str) -> Any:
        require(name in sources, f"Unretained source reference: {name}")
        text = text_of(name)
        return (
            yaml.safe_load(text)
            if name.endswith(".yaml")
            else (
                [json.loads(line) for line in text.splitlines()]
                if name.endswith(".jsonl")
                else json.loads(text)
            )
        )

    return read, sources


def rank_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["configuration"]].append(row)
    groups = []
    for identity, runs in grouped.items():
        runs.sort(key=lambda r: r["seed"])
        require([r["seed"] for r in runs] == SEEDS, f"Incomplete or duplicate seeds: {identity}")
        require(all(math.isfinite(r["loss"]) for r in runs), f"Nonfinite loss: {identity}")
        group = {k: runs[0][k] for k in ["schedule", "method", "horizon", "lr", "beta1", "beta2"]}
        groups.append(
            dict(
                group,
                id=identity,
                mean=statistics.mean(r["loss"] for r in runs),
                sd=statistics.stdev(r["loss"] for r in runs),
                runs=[
                    {k: r[k] for k in ["id", "seed", "loss", "jobId", "artifacts"]} for r in runs
                ],
            )
        )
    groups.sort(
        key=lambda g: (
            g["schedule"],
            g["method"],
            g["horizon"],
            g["mean"],
            g["lr"],
            g["beta1"],
            g["beta2"],
        )
    )
    ranks = defaultdict(int)
    for g in groups:
        key = (g["schedule"], g["method"], g["horizon"])
        ranks[key] += 1
        g["rank"] = ranks[key]
    return groups


def stitch(parent, local, cursor):
    require(all(t > cursor for t, _ in local), "Child observations precede continuation cursor")
    points = [p for p in parent if p[0] <= cursor] + local
    require(all(a[0] < b[0] for a, b in itertools.pairwise(points)), "Nonmonotonic curve")
    require(all(math.isfinite(p[1]) for p in points), "Nonfinite curve")
    return points


def steady_throughput(events, warmup=312):
    tokens, seconds, previous_step, previous_tokens = 0, 0.0, 0, 0
    for e in events:
        if e["event"] != "train":
            continue
        require(
            e["step"] > previous_step and e["tokens"] > previous_tokens, "Nonmonotonic training log"
        )
        require(
            math.isfinite(e["training_seconds"]) and e["training_seconds"] > 0,
            "Invalid timing window",
        )
        if previous_step >= warmup:
            tokens += e["tokens"] - previous_tokens
            seconds += e["training_seconds"]
        previous_step, previous_tokens = e["step"], e["tokens"]
    require(seconds > 0, "No complete post-warmup windows")
    return tokens / seconds


def matched_comparisons(groups):
    comparisons = []
    for left, right in itertools.combinations(groups, 2):
        if any(left[k] != right[k] for k in ["horizon", "lr", "beta1", "beta2"]):
            continue
        kind = (
            "schedule"
            if left["method"] == right["method"] and left["schedule"] != right["schedule"]
            else "workers"
            if left["schedule"] == right["schedule"] and left["method"] != right["method"]
            else None
        )
        if kind is None:
            continue
        if kind == "workers" and list(METHODS).index(left["method"]) > list(METHODS).index(
            right["method"]
        ):
            left, right = right, left
        require(
            [r["seed"] for r in left["runs"]] == [r["seed"] for r in right["runs"]] == SEEDS,
            "Unmatched seeds",
        )
        differences = [
            b["loss"] - a["loss"] for a, b in zip(left["runs"], right["runs"], strict=True)
        ]
        comparisons.append(
            {
                "kind": kind,
                "left": left["id"],
                "right": right["id"],
                "horizon": left["horizon"],
                "lr": left["lr"],
                "beta1": left["beta1"],
                "beta2": left["beta2"],
                "difference": statistics.mean(differences),
                "pairedSD": statistics.stdev(differences),
                "seedDifferences": differences,
            }
        )
    return comparisons


def distribution(values):
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return {"median": statistics.median(values), "q1": q1, "q3": q3}


def collect(root):
    read, sources = artifact_reader(root)
    manifest = read_json(root / "manifest.json")
    runs, histories, curves, performance = [], {}, {}, []
    identities, environments, dates, definitions = set(), {}, [], {}
    for c in manifest["campaigns"]:
        for name in c.get("supportingAudits", []):
            require(read(name)["status"] == "passed", "Failed supporting audit")
        for p in c["definitions"]:
            d = read(p)
            definitions[d["identity"]] = d["definition"]
        if "audit" in c:
            audit = read(c["audit"])
            require(
                audit["status"] == "passed" and audit["runs"] == c["expected"],
                "Failed campaign audit",
            )
            dates.append(audit["audited_at"])
        if "coverage" in c:
            coverage = read(c["coverage"])
            require(
                coverage["status"] == "complete"
                and coverage["accepted"] == coverage["expected"] == c["expected"]
                and not coverage["failures"],
                "Incomplete coverage",
            )
    for r in sorted(manifest["runs"], key=lambda r: (r["horizon"], r["id"])):
        require(r["id"] not in identities, f"Duplicate run: {r['id']}")
        identities.add(r["id"])
        a = r["artifacts"]
        result, cfg, receipt, env, request = (
            read(a[n])
            for n in [
                "result.json",
                "resolved.yaml",
                "receipt.json",
                "environment.json",
                "request.json",
            ]
        )
        definition = definitions[receipt["campaign_identity"]]
        h, schedule, method = r["horizon"], r["schedule"], r["method"]
        workers = {"sync": 1, "awc4": 4, "awc8": 8}[method]
        require(
            result["status"] == "complete"
            and result["recipe_identity"]
            == receipt["recipe_identity"]
            == request["recipe_identity"],
            "Incomplete or inconsistent result",
        )
        require(
            cfg["runtime"]["seed"] == r["seed"] and cfg["training"]["tokens_per_parameters"] == h,
            "Wrong seed or horizon",
        )
        require(
            all(cfg["optimizer"][k] == r[k] for k in ["lr", "beta1", "beta2"]),
            "Wrong hyperparameters",
        )
        require(
            cfg["lr_schedule"]["name"] == schedule and cfg["lr_schedule"]["warmup_steps"] == 312,
            "Wrong schedule",
        )
        require(
            schedule != "cosine" or cfg["lr_schedule"]["min_lr_ratio"] == 0,
            "Cosine does not reach zero",
        )
        require(schedule != "wsd" or cfg["lr_schedule"]["decay_fraction"] == 0.1, "Wrong WSD decay")
        require(
            (cfg.get("decentralized") or {}).get("num_models", 1) == workers, "Wrong worker count"
        )
        require(workers == 1 or cfg["decentralized"]["scheme"] == "awc", "Wrong mixing scheme")
        require(
            result["parameters"] == 20403520
            and cfg["training"]["batch_tokens"] == 131072
            and cfg["model"]["context_length"] == 1024,
            "Wrong model or batch",
        )
        require(
            result["tokens"] == result["step"] * 131072 and result["epochs"] == h * 2,
            "Wrong realized budget",
        )
        bounds = next(b for b in definition["horizons"] if b["horizon"] == h)
        require(result["tokens"] == bounds["tokens"], "Wrong token target")
        require(
            env["source_hash"] == definition["code"]["package"]
            and env["cache_identity"] == definition["cache_identity"],
            "Wrong trainer or cache identity",
        )
        ev = result["final_validation"]
        require(
            ev["split"] == "full"
            and ev["validation_complete"]
            and ev["tokens"] == 197411295
            and math.isfinite(ev["loss"])
            and ev["loss"] == receipt["loss"],
            "Wrong final validation",
        )
        # Compare the retained evidence with the original receipt hashes.
        receipt_artifacts = (
            receipt.get("artifacts", {})
            if schedule == "cosine"
            else {
                "result.json": receipt["result"],
                "environment.json": receipt["environment"],
            }
        )
        require(
            schedule != "cosine" or "metrics.jsonl" in receipt_artifacts,
            f"Receipt omits the metrics attestation: {r['id']}",
        )
        for name, item in receipt_artifacts.items():
            if name in a:
                require(
                    sources[a[name]]["sha256"] == item["sha256"],
                    f"Receipt checksum mismatch: {r['id']}/{name}",
                )
        events = read(a["metrics.jsonl"])
        final = [e for e in events if e["event"] == "final_validation"]
        require(
            len(final) == 1
            and final[0]["loss"] == ev["loss"]
            and final[0]["tokens"] == result["tokens"],
            "Wrong logged endpoint",
        )
        key = (schedule, method, r["lr"], r["beta1"], r["beta2"], r["seed"])
        cursor, parent = 0, None
        if schedule == "wsd" and h > 20:
            lineage = read(a["continuation.json"])
            parent = histories.get(key)
            if parent is None:
                raise ValueError("Missing continuation parent")
            require(
                parent["horizon"] == lineage["source_horizon"] and lineage["target_horizon"] == h,
                "Missing continuation parent",
            )
            cursor = lineage["cursor"] * cfg["model"]["context_length"]
            require(
                cursor == lineage["step"] * 131072
                and receipt["parent"] == lineage["source"]
                and request["parent"] == lineage["source"],
                "Invalid continuation boundary",
            )
            require(
                lineage["previous"]["recipe_identity"] == parent["recipeIdentity"]
                and lineage["updated"]["recipe_identity"] == result["recipe_identity"],
                "Wrong continuation identity",
            )
            prior_bounds = next(
                b for b in definition["horizons"] if b["horizon"] == parent["horizon"]
            )
            require(
                cursor == prior_bounds["checkpoint_tokens"]
                and cursor <= prior_bounds["stable_end"],
                "Continuation crosses parent decay",
            )
        curve = {"seed": r["seed"], "final": ev["loss"]}
        for kind, event, field in [
            ("train", "train", "loss"),
            ("validation", "validation", "loss"),
            ("gradientNorm", "train", "grad_norm"),
        ]:
            local = [[e["tokens"], e[field]] for e in events if e["event"] == event]
            points = stitch(parent[kind] if parent else [], local, cursor)
            require(points[-1][0] == result["tokens"], "Incomplete curve endpoint")
            curve[kind] = points
        require(len(curve["validation"]) == result["epochs"], "Incomplete epoch validation history")
        curve["sources"] = (
            [
                dict(s, endTokens=min(s["endTokens"], cursor))
                for s in parent["sources"]
                if s["startTokens"] < cursor
            ]
            if parent
            else []
        ) + [
            {
                "path": a["metrics.jsonl"],
                "container": sources[a["metrics.jsonl"]]["container"],
                "sha256": sources[a["metrics.jsonl"]]["sha256"],
                "startTokens": cursor,
                "endTokens": result["tokens"],
            }
        ]
        histories[key] = dict(curve, horizon=h, recipeIdentity=result["recipe_identity"])
        curves.setdefault(r["configuration"], {"configuration": r["configuration"], "runs": []})[
            "runs"
        ].append(curve)
        runs.append(
            dict(
                r,
                loss=ev["loss"],
                jobId=receipt["job_id"],
                tokens=result["tokens"],
                steps=result["step"],
                epochs=result["epochs"],
                startTokens=cursor,
            )
        )
        if schedule == "cosine":
            require(env["gpu"] == "NVIDIA GH200 120GB", "Non-GH200 performance record")
            performance.append(
                {
                    "id": r["id"],
                    "method": method,
                    "horizon": h,
                    "steady": steady_throughput(events),
                    "training": result["training_tokens_per_second"],
                    "elapsed": result["elapsed_tokens_per_second"],
                    "validationSeconds": ev["seconds"],
                    "sessionSeconds": result["seconds_this_session"],
                    "peakGiB": result["peak_memory_bytes"] / 2**30,
                    "jobId": receipt["job_id"],
                    "measurementDate": datetime.fromtimestamp(
                        int(env["slurm"]["SLURM_JOB_START_TIME"]), timezone.utc
                    )
                    .date()
                    .isoformat(),
                    "artifacts": a,
                }
            )
        environments[(schedule, method)] = {
            "gpu": env["gpu"],
            "python": env["python"].split()[0],
            "torch": env["versions"]["torch"],
            "cuda": env["cuda"],
            "sourceHash": env["source_hash"],
            "cacheIdentity": env["cache_identity"],
        }
    groups = rank_groups(runs)
    campaigns = []
    for c in manifest["campaigns"]:
        gs = [g for g in groups if (g["schedule"], g["method"]) == (c["schedule"], c["method"])]
        require(sum(len(g["runs"]) for g in gs) == c["expected"], "Wrong publication count")
        reference = {(r["horizon"], r["lr"], r["beta1"], r["beta2"]): r for r in read(c["summary"])}
        require(len(reference) == len(gs), "Wrong configuration count")
        for g in gs:
            old = reference[(g["horizon"], g["lr"], g["beta1"], g["beta2"])]
            require(
                abs(g["mean"] - old["mean"]) < 1e-12
                and abs(g["sd"] - old["sd"]) < 1e-12
                and g["rank"] == old["rank"],
                "Published ranking differs from source",
            )
        grid = {k: sorted({g[k] for g in gs}) for k in ["lr", "beta1", "beta2"]}
        stages = []
        for h in sorted({g["horizon"] for g in gs}):
            stage_groups = [g for g in gs if g["horizon"] == h]
            eligible = sorted({g["lr"] for g in stage_groups})
            require(
                len(stage_groups) == len(eligible) * len(grid["beta1"]) * len(grid["beta2"]),
                "Incomplete hyperparameter grid",
            )
            row = next(r for r in runs if r["configuration"] == stage_groups[0]["id"])
            stages.append(
                {
                    "horizon": h,
                    "tokens": row["tokens"],
                    "steps": row["steps"],
                    "epochs": row["epochs"],
                    "eligibleLRs": eligible,
                    "groups": len(stage_groups),
                    "runs": len(stage_groups) * 3,
                }
            )
        if c["schedule"] == "wsd":
            ceiling = max(grid["lr"])
            for stage in stages:
                require(
                    stage["eligibleLRs"] == [lr for lr in grid["lr"] if lr <= ceiling],
                    "Invalid WSD pruning",
                )
                ceiling = next(
                    g["lr"] for g in gs if g["horizon"] == stage["horizon"] and g["rank"] == 1
                )
        campaigns.append(
            {
                "schedule": c["schedule"],
                "method": c["method"],
                "title": f"{SCHEDULES[c['schedule']]} · {METHODS[c['method']]}",
                "grid": grid,
                "stages": stages,
                "environment": environments[(c["schedule"], c["method"])],
            }
        )
    summary = []
    for method, h in itertools.product(METHODS, [20, 40, 80]):
        rs = [r for r in performance if r["method"] == method and r["horizon"] == h]
        require(
            all(math.isfinite(r[k]) and r[k] > 0 for r in rs for k in METRICS),
            "Invalid performance measurement",
        )
        summary.append(
            {
                "method": method,
                "horizon": h,
                "n": len(rs),
                "measurementDates": {
                    "first": min(r["measurementDate"] for r in rs),
                    "last": max(r["measurementDate"] for r in rs),
                },
                "metrics": {k: distribution([r[k] for r in rs]) for k in METRICS},
            }
        )
    return (
        {
            "version": 1,
            "snapshotDate": datetime.fromtimestamp(max(dates), timezone.utc).date().isoformat(),
            "protocol": {
                "parameters": 20403520,
                "batchTokens": 131072,
                "validationTokens": 197411295,
                "warmupSteps": 312,
                "seeds": SEEDS,
            },
            "campaigns": campaigns,
            "groups": groups,
            "matches": matched_comparisons(groups),
            "performance": summary,
        },
        runs,
        curves,
        performance,
    )


def figures(root, data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#faf6ee",
            "axes.facecolor": "#faf6ee",
            "svg.hashsalt": "tiny-llm-current-training",
        }
    )
    winners = [g for g in data["groups"] if g["rank"] == 1 and g["horizon"] <= 80]

    def save(fig, name):
        fig.tight_layout()
        for extension in ["png", "pdf", "svg"]:
            metadata = (
                {"CreationDate": None, "ModDate": None}
                if extension == "pdf"
                else ({"Date": None} if extension == "svg" else None)
            )
            fig.savefig(root / f"{name}.{extension}", dpi=170, metadata=metadata)
        plt.close(fig)

    def line(ax, groups, label, color, style="-"):
        groups = sorted(groups, key=lambda g: g["horizon"])
        ax.errorbar(
            [g["horizon"] for g in groups],
            [g["mean"] for g in groups],
            yerr=[g["sd"] for g in groups],
            label=label,
            color=color,
            linestyle=style,
            marker="o",
            capsize=4,
            linewidth=1.8,
        )
        ax.set_xticks([20, 40, 80])
        ax.set_xlabel("Global tokens per parameter")
        ax.set_ylabel("Final full-validation loss (nats)")
        ax.grid(axis="y", alpha=0.18)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, method in zip(axes, ["sync", "awc8"], strict=True):
        for schedule, style, color in [("cosine", "-", "#30658a"), ("wsd", "--", "#b25c3c")]:
            line(
                ax,
                [g for g in winners if g["method"] == method and g["schedule"] == schedule],
                SCHEDULES[schedule],
                color,
                style,
            )
        ax.set_title(METHODS[method])
        ax.legend(frameon=False)
    save(fig, "schedule-comparison")
    limits = (
        min(g["mean"] - g["sd"] for g in winners) - 0.01,
        max(g["mean"] + g["sd"] for g in winners) + 0.01,
    )
    for method in ["sync", "awc8"]:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        for schedule, style, color in [("cosine", "-", "#30658a"), ("wsd", "--", "#b25c3c")]:
            line(
                ax,
                [g for g in winners if g["method"] == method and g["schedule"] == schedule],
                SCHEDULES[schedule],
                color,
                style,
            )
        ax.set_title(METHODS[method])
        ax.set_ylim(limits)
        ax.legend(frameon=False, fontsize=9)
        save(fig, f"schedule-comparison-{method}")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for method in METHODS:
        line(
            ax,
            [g for g in winners if g["method"] == method and g["schedule"] == "cosine"],
            METHODS[method],
            COLORS[method],
        )
    ax.legend(frameon=False)
    save(fig, "worker-comparison")
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for method in METHODS:
        line(
            ax,
            [g for g in winners if g["method"] == method and g["schedule"] == "cosine"],
            METHODS[method],
            COLORS[method],
        )
    ax.legend(frameon=False, fontsize=9)
    save(fig, "worker-comparison-mobile")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    rows = [r for r in data["performance"] if r["horizon"] == 80]
    for i, row in enumerate(rows):
        d = row["metrics"]["steady"]
        ax.barh(i, d["median"] / 1e6, color=COLORS[row["method"]], height=0.5)
        ax.errorbar(
            d["median"] / 1e6,
            i,
            xerr=[[max(0, d["median"] - d["q1"]) / 1e6], [max(0, d["q3"] - d["median"]) / 1e6]],
            color="#24221f",
            capsize=4,
        )
    ax.set_yticks(range(3), [METHODS[r["method"]] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Steady training throughput · million global tokens/s")
    ax.grid(axis="x", alpha=0.18)
    save(fig, "gh200-throughput")


def winner_markdown(data):
    lines = [
        "| Schedule | Training mode | Tokens/parameter | LR | β₁ | β₂ | Final loss ± sample SD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in sorted(
        (g for g in data["groups"] if g["rank"] == 1 and g["horizon"] <= 80),
        key=lambda g: (g["horizon"], g["schedule"], list(METHODS).index(g["method"])),
    ):
        lines.append(
            f"| {SCHEDULES[g['schedule']]} | {METHODS[g['method']]} | {g['horizon']} | "
            f"{g['lr']:g} | {g['beta1']:g} | {g['beta2']:g} | {g['mean']:.6f} ± {g['sd']:.6f} |"
        )
    return "\n".join(lines)


def update_markdown(data):
    blocks = {
        ROOT / "README.md": ("results", winner_markdown(data)),
        ROOT / "doc/results/overview.md": ("results", winner_markdown(data)),
        ROOT / "doc/performance/training.md": ("performance", performance_markdown(data)),
    }
    for path, (name, content) in blocks.items():
        if not path.exists():
            continue
        text = path.read_text()
        start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
        if start in text and end in text:
            before, rest = text.split(start, 1)
            _, after = rest.split(end, 1)
            write(path, before + start + "\n\n" + content + "\n\n" + end + after)


def performance_markdown(data):
    first = min(r["measurementDates"]["first"] for r in data["performance"])
    last = max(r["measurementDates"]["last"] for r in data["performance"])
    lines = [
        f"Measurements from jobs started **{first}–{last} (UTC)**, "
        f"as recorded in their execution environments. Snapshot: {data['snapshotDate']}.",
        "",
        "| Mode | Tokens/parameter | Runs | Steady M tokens/s [IQR] | Recorded training M tokens/s | Elapsed M tokens/s | Full validation s | Session s | Peak GiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in data["performance"]:
        m, s = r["metrics"], r["metrics"]["steady"]
        lines.append(
            f"| {METHODS[r['method']]} | {r['horizon']} | {r['n']} | "
            f"{s['median'] / 1e6:.3f} [{s['q1'] / 1e6:.3f}, {s['q3'] / 1e6:.3f}] | "
            f"{m['training']['median'] / 1e6:.3f} | {m['elapsed']['median'] / 1e6:.3f} | "
            f"{m['validationSeconds']['median']:.1f} | {m['sessionSeconds']['median']:.1f} | "
            f"{m['peakGiB']['median']:.2f} |"
        )
    return "\n".join(lines)


def export(root, data, runs, curves, performance):
    json_write(root / "dataset.json", data)
    table(root, "runs", runs)
    table(root, "configurations", data["groups"])
    table(root, "winners", [g for g in data["groups"] if g["rank"] == 1])
    table(root, "matched", data["matches"])
    table(root, "performance-runs", performance)
    table(root, "performance", data["performance"])
    packed = defaultdict(list)
    for name, curve in sorted(curves.items()):
        curve["runs"].sort(key=lambda r: r["seed"])
        packed[name.split("--")[0]].append(
            json.dumps(curve, allow_nan=False, separators=(",", ":"))
        )
    for group, lines in packed.items():
        write(
            root / "curves" / f"{group}.jsonl.gz",
            gzip.compress(("\n".join(lines) + "\n").encode(), 6, mtime=0),
        )
    figures(root, data)
    json_write(
        root / "checksums.json",
        {
            str(p.relative_to(root)): digest(p.read_bytes())
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "checksums.json"
        },
    )


def dataset_card(root: Path) -> str:
    """Generated from the bundle so the card cannot drift from what it describes."""
    data = read_json(root / "dataset.json")
    repack = read_json(root / "provenance/repack.json")
    groups, runs = len(data["groups"]), sum(len(g["runs"]) for g in data["groups"])
    return f"""---
pretty_name: tiny-llm training logs
license: mit
tags:
- language-model
- pretraining
- learning-rate-schedule
- decentralized-training
- training-logs
size_categories:
- 1K<n<10K
---

# tiny-llm retained training evidence

{runs} seed/horizon records across {groups} configurations: cosine-to-zero and WSD
schedules, synchronous and four- and eight-worker decentralized training, at 20 to 160
global tokens per parameter, for 20.4M-parameter models on C4. Snapshot {data["snapshotDate"]}.

This mirrors `doc/data/current-training/` in
[WangZesen/tiny-llm](https://github.com/WangZesen/tiny-llm). The website built from it is
at <https://wangzesen.github.io/tiny-llm/>.

## Layout

Retained artifacts are grouped into gzipped JSONL containers, one line per artifact:

```json
{{"path":"logs/cosine/sync/h20/lr-0.01-b1-0.9-b2-0.99/s42/metrics.jsonl",
  "bytes":71410,"sha256":"...","text":"<the artifact verbatim>"}}
```

`sources.json` is the index: it maps every artifact to its container and records the
digest of the original file. Metrics containers concatenate one gzip member per line, so
`offset` and `compressedBytes` address a single run's log directly; every other container
is a single stream. Placement is a pure rule, so a misfiled container is detectable.

## What is retained, and what is not

{repack["retained"]["byteIdentical"]} artifacts are byte-identical to the files written
during training. The {repack["retained"]["transformed"]} `metrics.jsonl` logs had two
members deleted from every train event: `{"`, `".join(sorted(DROPPED_METRIC_FIELDS))}`.
These are per-worker diagnostics for the packed models; no published figure, table or
curve reads them, and they were 37% of the logged bytes.

The transform is a deletion, not a re-serialization: no other byte of any line, and no
line, was changed, added or reordered. `sources.json.sha256` remains the digest of the
original file and can no longer be recomputed from this copy for those logs — but for the
cosine runs it stays independently attested by the retained `receipt.json`, which records
the trainer's own hash of `metrics.jsonl` and is checked on every verification run. The
complete pre-repack bundle, with the arrays intact, is this repo at revision
`{repack["archive"]["revision"]}`.

## Verify and regenerate

```bash
uv run python scripts/export_publication.py --check --bundle hf://{HF_REPO}
```

No GPU, campaign directory or checkpoint is needed. Verification checks every retained
byte against `sources.json` and every derived file against `checksums.json`.
"""


def hf_snapshot(repo: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download  # imported lazily; never at module load

    return Path(snapshot_download(repo_id=repo, repo_type="dataset", revision=revision))


def resolve_bundle(spec: str) -> Path:
    """hf://repo[@revision] downloads a snapshot; anything else is a local path."""
    if not spec.startswith("hf://"):
        return Path(spec)
    repo, _, revision = spec.removeprefix("hf://").partition("@")
    return hf_snapshot(repo, revision or "main")


def publish_hf(root: Path, repo: str, revision: str, message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    if revision != "main":
        api.create_branch(repo, branch=revision, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(root),
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        commit_message=message,
        delete_patterns=["*"],
        ignore_patterns=["README.md"],
    )
    # The card lives only on the Hub; a README under doc/ would join the website build.
    api.upload_file(
        path_or_fileobj=dataset_card(root).encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        commit_message="Update dataset card",
    )
    print(f"Published {root.name} to {repo}@{revision}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-runs", type=Path)
    parser.add_argument("--bundle", default=str(BUNDLE), help="local path, or hf://repo[@rev]")
    parser.add_argument("--output", type=Path, default=BUNDLE)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--repack", action="store_true", help="migrate a per-file bundle")
    parser.add_argument("--publish-hf", action="store_true")
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--hf-revision", default="main")
    args = parser.parse_args()
    bundle = resolve_bundle(args.bundle)
    if args.check:
        require(not args.import_runs, "--check cannot import sources")
        data, *_ = collect(bundle)
        require(data == read_json(bundle / "dataset.json"), "Derived publication is stale")
        for name, expected in read_json(bundle / "checksums.json").items():
            require(
                digest((bundle / name).read_bytes()) == expected,
                f"Publication checksum mismatch: {name}",
            )
        print(f"Verified {len(data['groups'])} groups and all retained evidence without runs/")
        return
    require(
        not args.output.exists() or (args.output / "manifest.json").is_file(),
        "Existing output directory is not a publication bundle",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".publication-", dir=args.output.parent))
    try:
        if args.import_runs:
            import_sources(args.import_runs.resolve(), stage)
        elif args.repack:
            print("Repacking retained evidence into containers…", flush=True)
            repack_sources(bundle, stage)
        else:
            print("Staging retained documentation evidence…", flush=True)
            shutil.copytree(bundle, stage, dirs_exist_ok=True)
        print("Validating source bytes and reconstructing measurements…", flush=True)
        data, runs, curves, performance = collect(stage)
        if args.repack:
            # Only artifact names may move; every published measurement must be unchanged.
            require(
                data == unpacked_names(read_json(bundle / "dataset.json")),
                "Repacking changed the published measurements",
            )
        print("Writing tables, curves, and publication figures…", flush=True)
        export(stage, data, runs, curves, performance)
        if args.publish_hf:
            publish_hf(stage, args.hf_repo, args.hf_revision, f"Publish {len(runs)} run records")
        backup = args.output.with_name("." + args.output.name + "-previous")
        require(not backup.exists(), f"Previous publication backup needs recovery: {backup}")
        if args.output.exists():
            args.output.replace(backup)
        try:
            stage.replace(args.output)
        except BaseException:
            if backup.exists():
                backup.replace(args.output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        if args.output.resolve() == BUNDLE.resolve():
            update_markdown(data)
        print(
            f"Published {len(runs)} seed/horizon results, {len(data['groups'])} groups; {args.output}"
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == "__main__":
    main()
