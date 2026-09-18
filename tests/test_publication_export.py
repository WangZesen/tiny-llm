"""Publication calculations and source-retention failures need no GPU or campaign data."""

import gzip
import json
import runpy
import sys
from pathlib import Path

import pytest

publication = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/export_publication.py")
)


def rows(method="sync", schedule="cosine", lr=0.01, losses=(1.0, 2.0, 3.0)):
    identity = publication["group_id"](schedule, method, 20, lr, 0.9, 0.99)
    return [
        dict(
            id=f"{identity}--s{seed}",
            configuration=identity,
            schedule=schedule,
            method=method,
            horizon=20,
            lr=lr,
            beta1=0.9,
            beta2=0.99,
            seed=seed,
            loss=loss,
            jobId="job",
            artifacts={},
        )
        for seed, loss in zip([42, 43, 44], losses, strict=True)
    ]


def test_rankings_require_three_distinct_seeds_and_break_ties():
    rank = publication["rank_groups"]
    groups = rank(rows(lr=0.02) + rows(lr=0.01))
    assert [(g["lr"], g["rank"], g["mean"], g["sd"]) for g in groups] == [
        (0.01, 1, 2.0, 1.0),
        (0.02, 2, 2.0, 1.0),
    ]
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        rank(rows()[:2])
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        rank(rows() + rows()[:1])
    with pytest.raises(ValueError, match="Nonfinite"):
        rank(rows(losses=(1.0, float("nan"), 3.0)))


def test_matched_differences_use_seed_pairs_and_matching_hyperparameters():
    groups = publication["rank_groups"](
        rows()
        + rows(method="awc4", losses=(3.0, 3.0, 3.0))
        + rows(schedule="wsd", lr=0.02, losses=(2.0, 3.0, 4.0))
    )
    matches = publication["matched_comparisons"](groups)
    assert len(matches) == 1
    assert matches[0]["kind"] == "workers"
    assert matches[0]["difference"] == 1.0
    assert matches[0]["pairedSD"] == 1.0
    assert matches[0]["seedDifferences"] == [2.0, 1.0, 0.0]


def test_continuation_discards_parent_decay_and_rejects_overlap():
    stitch = publication["stitch"]
    assert stitch([[10, 4], [20, 3], [30, 2]], [[30, 2.5], [40, 2]], 20) == [
        [10, 4],
        [20, 3],
        [30, 2.5],
        [40, 2],
    ]
    with pytest.raises(ValueError, match="precede"):
        stitch([[10, 4]], [[10, 3]], 10)
    with pytest.raises(ValueError, match="Nonmonotonic"):
        stitch([], [[20, 4], [10, 3]], 0)


def test_steady_rate_uses_tokens_over_time_and_excludes_crossing_warmup_window():
    events = [
        dict(event="train", step=300, tokens=3000, training_seconds=100),
        dict(event="train", step=320, tokens=3200, training_seconds=100),
        dict(event="validation", seconds=1000),
        dict(event="train", step=340, tokens=3400, training_seconds=2),
        dict(event="train", step=350, tokens=3500, training_seconds=2),
    ]
    assert publication["steady_throughput"](events) == 75
    with pytest.raises(ValueError, match="No complete"):
        publication["steady_throughput"](events[:2])
    with pytest.raises(ValueError, match="Nonmonotonic"):
        publication["steady_throughput"](events + [events[-1]])


LOGS = "logs/cosine/sync/h20/lr-0.01-b1-0.9-b2-0.99/s42"


def bundle(tmp_path, entries):
    """Write retained artifacts into containers and index them, as an import would."""
    containers = publication["Containers"](tmp_path)
    mapping = {}
    for name, data, dropped in entries:
        text = publication["strip_metrics"](data)[0] if dropped else data
        mapping[name] = containers.add(
            name,
            text,
            {
                "originalPath": "/runs/" + name,
                "sha256": publication["digest"](data),
                "bytes": len(data),
            },
            dropped,
        )
    containers.close()
    publication["json_write"](tmp_path / "sources.json", mapping)
    return mapping


def test_container_placement_is_a_pure_rule():
    container_of = publication["container_of"]
    assert container_of(f"{LOGS}/metrics.jsonl") == "logs/cosine-sync-h20.metrics.jsonl.gz"
    assert container_of(f"{LOGS}/result.json") == "logs/cosine-sync-h20.records.jsonl.gz"
    assert container_of("shared/environment-ab.json") == "shared/environments.jsonl.gz"
    assert container_of("sources/wsd/awc8/runs.json") == "sources/campaigns.jsonl.gz"
    with pytest.raises(ValueError, match="No container"):
        container_of("curves/sync-cosine-h20--0.01--0.9--0.99.json")


def test_strip_metrics_deletes_only_the_per_worker_arrays():
    strip = publication["strip_metrics"]
    plain = b'{"event": "train", "step": 20, "loss": 9.70669174194336}\n'
    assert strip(plain) == (plain, [])
    packed = (
        b'{"event": "train", "step": 20, "loss": 9.5, "local_losses": [7.8, 7.7], '
        b'"local_grad_norms": [1.39, 1.24], "tokens": 2621440}\n'
    )
    text, dropped = strip(packed)
    assert dropped == ["local_grad_norms", "local_losses"]
    assert text == b'{"event": "train", "step": 20, "loss": 9.5, "tokens": 2621440}\n'
    assert json.loads(text)["loss"] == json.loads(packed)["loss"]


def test_container_entries_and_hashes_detect_corrupt_or_missing_sources(tmp_path):
    metrics, result = b'{"event": "train", "tokens": 100}\n', b'{"status":"complete"}'
    mapping = bundle(
        tmp_path, [(f"{LOGS}/metrics.jsonl", metrics, []), (f"{LOGS}/result.json", result, [])]
    )
    read, sources = publication["artifact_reader"](tmp_path)
    assert read(f"{LOGS}/metrics.jsonl") == [{"event": "train", "tokens": 100}]
    assert read(f"{LOGS}/result.json") == {"status": "complete"}
    assert sources[f"{LOGS}/metrics.jsonl"]["container"] == "logs/cosine-sync-h20.metrics.jsonl.gz"
    with pytest.raises(ValueError, match="Unretained source reference"):
        read(f"{LOGS}/absent.json")

    mapping[f"{LOGS}/result.json"]["sha256"] = "0" * 64
    publication["json_write"](tmp_path / "sources.json", mapping)
    with pytest.raises(ValueError, match="checksum mismatch"):
        publication["artifact_reader"](tmp_path)

    bundle(tmp_path, [(f"{LOGS}/metrics.jsonl", metrics, [])])
    (tmp_path / "logs/cosine-sync-h20.records.jsonl.gz").unlink()
    (tmp_path / "logs/cosine-sync-h20.metrics.jsonl.gz").unlink()
    with pytest.raises(FileNotFoundError):
        publication["artifact_reader"](tmp_path)


def test_container_index_and_entries_must_agree(tmp_path):
    metrics = b'{"event": "train", "tokens": 100}\n'
    mapping = bundle(tmp_path, [(f"{LOGS}/metrics.jsonl", metrics, [])])
    orphan = dict(mapping)
    orphan.pop(f"{LOGS}/metrics.jsonl")
    orphan[f"{LOGS}/result.json"] = {
        "originalPath": "/runs/result.json",
        "sha256": "0" * 64,
        "bytes": 1,
        "container": "logs/cosine-sync-h20.metrics.jsonl.gz",
    }
    publication["json_write"](tmp_path / "sources.json", orphan)
    with pytest.raises(ValueError, match="Misplaced artifact"):
        publication["artifact_reader"](tmp_path)


def test_repack_preserves_every_original_byte(tmp_path):
    source, destination = tmp_path / "legacy", tmp_path / "packed"
    metrics = (
        b'{"event": "train", "step": 20, "loss": 9.5, "local_losses": [7.8], "tokens": 100}\n'
        b'{"event": "final_validation", "loss": 3.5, "tokens": 100}\n'
    )
    result = b'{"status":"complete"}'
    files = {f"{LOGS}/metrics.jsonl.gz": metrics, f"{LOGS}/result.json": result}
    legacy = {}
    for name, data in files.items():
        stored = gzip.compress(data, mtime=0) if name.endswith(".gz") else data
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes(stored)
        legacy[name] = {
            "originalPath": "/runs/" + name,
            "sha256": publication["digest"](data),
            "bytes": len(data),
            "compression": "gzip" if name.endswith(".gz") else None,
        }
    publication["json_write"](source / "sources.json", legacy)
    publication["json_write"](
        source / "manifest.json",
        {
            "version": 1,
            "campaigns": [],
            "runs": [{"artifacts": dict(zip(["m", "r"], files, strict=True))}],
        },
    )
    publication["json_write"](source / "checksums.json", {})

    publication["repack_sources"](source, destination)
    read, packed = publication["artifact_reader"](destination)
    assert set(packed) == {f"{LOGS}/metrics.jsonl", f"{LOGS}/result.json"}
    # An untransformed artifact keeps one digest; a stripped log records both.
    assert "dropped" not in packed[f"{LOGS}/result.json"]
    assert read(f"{LOGS}/result.json") == {"status": "complete"}
    stripped = packed[f"{LOGS}/metrics.jsonl"]
    assert stripped["dropped"] == ["local_losses"]
    assert stripped["sha256"] == publication["digest"](metrics) != stripped["retainedSha256"]
    events = read(f"{LOGS}/metrics.jsonl")
    assert [e["loss"] for e in events] == [9.5, 3.5] and "local_losses" not in events[0]
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["runs"][0]["artifacts"]["m"] == f"{LOGS}/metrics.jsonl"
    provenance = json.loads((destination / "provenance/repack.json").read_text())
    assert provenance["retained"] == {"byteIdentical": 1, "transformed": 1}

    (source / f"{LOGS}/result.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        publication["repack_sources"](source, tmp_path / "rejected")


def test_importing_the_exporter_needs_no_network():
    # Regeneration and every test must work offline; the Hub client is imported lazily.
    assert "huggingface_hub" not in sys.modules


def test_failed_regeneration_preserves_previous_publication(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "published"
    source.mkdir()
    output.mkdir()
    (source / "sources.json").write_text(
        json.dumps(
            {
                f"{LOGS}/absent.json": {
                    "sha256": "0" * 64,
                    "bytes": 1,
                    "container": "logs/cosine-sync-h20.records.jsonl.gz",
                }
            }
        )
    )
    (output / "dataset.json").write_text("previous publication")
    (output / "manifest.json").write_text("{}")
    monkeypatch.setattr(
        sys, "argv", ["export_publication.py", "--bundle", str(source), "--output", str(output)]
    )
    with pytest.raises(FileNotFoundError):
        publication["main"]()
    assert (output / "dataset.json").read_text() == "previous publication"
    assert not list(tmp_path.glob(".publication-*"))
