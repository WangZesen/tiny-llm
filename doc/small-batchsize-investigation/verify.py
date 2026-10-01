#!/usr/bin/env python3
"""Portable numerical and document acceptance checks; no training is required."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import platform
import re
import statistics
import subprocess
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
BATCHES = (16, 32, 64, 128)
FIGURES = ("quality", "gap", "beta1", "beta2", "clipping", "ablation",
           "matched-grid", "retuned", "throughput", "replication")
CHECKS: dict[str, object] = {}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label, *, atol=1e-11):
    require(math.isfinite(float(actual)) and math.isclose(
        float(actual), float(expected), rel_tol=2e-12, abs_tol=atol),
        f"{label}: {actual!r} != {expected!r}")


def read(path):
    return json.loads((HERE / path).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def key(row):
    return (int(row["batch"]), float(row["lr"]), float(row["beta1"]), float(row["beta2"]))


def observation_key(row):
    return (row["recipe"], int(row["seed"]))


def verify_provenance(check_sources):
    provenance = read("provenance.json")
    sources = {r["path"]: r for r in provenance["sources"]}
    require(set(r["id"] for r in sources.values()) == {f"S{i}" for i in range(1, 6)},
            "Expected five named campaign evidence groups")
    paths = set()
    for row in provenance["extracts"]:
        path = HERE / row["file"]
        require(path.is_file(), f"Missing extract: {row['file']}")
        require(row["file"] not in paths, f"Duplicate extract: {row['file']}")
        paths.add(row["file"])
        require(digest(path) == row["sha256"], f"Changed extract: {row['file']}")
        for origin in row["sources"]:
            require(origin in sources, f"Unindexed origin: {origin}")
        if row.get("selection") == "complete source file; bytes unchanged":
            require(len(row["sources"]) == 1 and row["sha256"] == sources[row["sources"][0]]["sha256"],
                    f"Unchanged-copy identity mismatch: {row['file']}")
    if check_sources:
        for source in sources.values():
            path = Path(source["path"])
            if not path.is_absolute():
                path = HERE.parents[1] / path
            require(path.is_file(), f"Original source unavailable: {path}")
            require(digest(path) == source["sha256"], f"Original source changed: {path}")
    for row in provenance["derived"] + provenance["figures"]:
        require(digest(HERE / row["file"]) == row["sha256"], f"Changed generated artifact: {row['file']}")
    CHECKS["provenance"] = {"bundled_extracts": len(paths), "indexed_sources": len(sources),
                            "derived_files": len(provenance["derived"]), "figure_files": len(provenance["figures"]),
                            "original_sources_checked": bool(check_sources)}
    return provenance


def verify_grid(rows, clipped):
    lrs = {16: [.00025, .0005, .001, .002, .003, .004],
           32: [.0005, .001, .002, .003, .004, .006],
           64: [.001, .002, .003, .004, .006, .008],
           128: [.002, .003, .004, .006, .008, .012]}
    betas1 = [.9, .95, .98, .99, .995] if clipped else [.95, .98, .99, .995]
    betas2 = [.98, .99, .999, .9999, .99999, .999999] if clipped else [.99, .999, .9999, .99999, .999999]
    expected = set()
    for batch in BATCHES:
        expected.update((batch, lr, b1, b2) for lr, b1, b2 in itertools.product(
            lrs[batch] if clipped else lrs[batch][:-1], betas1,
            betas2 + [2 ** (-batch * 512 / 10_000_000)]))
    require(len(rows) == len(expected) and {key(r) for r in rows} == expected,
            f"Incorrect {'clipped' if clipped else 'unclipped'} Cartesian grid")
    require(all(r["status"] == "complete" and r["seed"] == 42 for r in rows),
            "Screening must contain one successful seed-42 observation per configuration")
    require(all(r["tokens"] == 408_092_672 for r in rows), "Training-target mismatch")
    require(all(r["steps"] * r["batch"] * 512 == r["tokens"] for r in rows), "Step/token mismatch")


def verify_statistics():
    groups = {
        "sync": [r for r in read("data/sync-beta1-replicated.json") if r["weight_decay"] == .1],
        "clipped": read("data/clipped-replicated.json"),
        "unclipped": read("data/unclipped-replicated.json"),
    }
    require([len(groups[k]) for k in groups] == [4, 40, 20], "Expected 64 primary replicated recipes")
    for method, rows in groups.items():
        require(len({(r["batch"], r["recipe"]) for r in rows}) == len(rows), "Duplicate replicated recipe")
        for row in rows:
            label = f"{method}/{row['recipe']}"
            losses = row["losses"]
            require(row["seeds"] == [42, 43, 44] and len(losses) == 3, f"Wrong seeds: {label}")
            close(row["mean_loss"], statistics.mean(losses), label + " mean")
            close(row["std_loss"], statistics.stdev(losses), label + " sample SD")
            close(row["replication_mean_loss"], statistics.mean(losses[1:]), label + " seeds 43–44")
            for seed, loss in zip(row["seeds"], losses):
                if f"seed{seed}" in row:
                    close(row[f"seed{seed}"], loss, label + f" seed {seed}")
            for name in ("beta1", "beta2"):
                half = -math.log(2) / math.log(row[name])
                expected = {f"{name}_half_life_updates": half,
                            f"{name}_global_half_life_targets": half * row["batch"] * 512,
                            f"{name}_per_worker_half_life_targets": half * row["batch"] * 512 / (1 if method == "sync" else 4)}
                for field, value in expected.items():
                    if field in row:
                        close(row[field], value, label + " " + field)
        if method != "sync":
            for batch in BATCHES:
                ranked = sorted((r for r in rows if r["batch"] == batch),
                                key=lambda r: (r["mean_loss"], r["lr"], r["beta1"], r["beta2"]))
                require([r["rank"] for r in ranked] == list(range(1, len(ranked) + 1)),
                        f"Ranking mismatch: {method}/{batch}")
    screen = {method: read(f"data/{method}-screening.json") for method in ("clipped", "unclipped")}
    for method in screen:
        verify_grid(screen[method], method == "clipped")
    matched = read("data/matched-screening.json")
    require(len(matched) == 480 and {key(r) for r in matched} == {key(r) for r in screen["unclipped"]},
            "Matched grid must equal the entire reduced grid")
    by_key = {method: {key(r): r for r in rows} for method, rows in screen.items()}
    for row in matched:
        k = key(row)
        close(row["clipped_loss"], by_key["clipped"][k]["loss"], "Matched clipped loss")
        close(row["unclipped_loss"], by_key["unclipped"][k]["loss"], "Matched unclipped loss")
        close(row["loss_difference"], row["unclipped_loss"] - row["clipped_loss"], "Matched difference")
        close(row["baseline_clipping_frequency"], by_key["clipped"][k]["clipping_frequency"], "Matched clipping frequency")
    CHECKS["statistics"] = {"replicated_recipes": 64, "clipped_screening": 840,
                            "unclipped_screening": 480, "matched_seed42_pairs": 480,
                            "sample_sd_not_standard_error": True}
    return groups, screen, matched


def verify_eligibility(groups, screen):
    clipped_manifests = [read(f"data/{name}-manifest.json") for name in ("clipped-base", "clipped-extension")]
    frozen_clipped = {r["recipe"] for m in clipped_manifests for r in m["promoted_recipes"]}
    require(frozen_clipped == {r["recipe"] for r in groups["clipped"]}, "Clipped ranking changes frozen eligibility")
    manifest = read("data/unclipped-manifest.json")
    frozen = {r["recipe"] for r in manifest["promoted_recipes"]}
    expected = set()
    for batch in BATCHES:
        ranked = sorted((r for r in screen["unclipped"] if r["batch"] == batch),
                        key=lambda r: (r["loss"], r["lr"], r["beta1"], r["beta2"]))
        expected.update(r["recipe"] for r in ranked[:5])
    require(len(frozen) == 20 and frozen == expected, "No-clipping frozen selection differs from seed-42 top five")
    require(frozen == {r["recipe"] for r in groups["unclipped"]}, "Unpromoted no-clipping recipe entered ranking")
    primary = read("data/unclipped-runs.json")
    supplemental = read("data/unclipped-supplemental.json")
    imports = read("data/unclipped-imports.json")
    require(len(primary) == 520 and len(supplemental) == 4 and len(imports) == 12, "Incorrect import/primary counts")
    pkeys = {observation_key(r) for r in primary}
    skeys = {observation_key(r) for r in supplemental}
    require(len(pkeys) == 520 and len(skeys) == 4 and not pkeys & skeys, "Duplicate primary/supplemental observations")
    require(all(r["seed"] == 42 or r["recipe"] in frozen for r in primary), "Unselected replication in primary data")
    require(all(r["seed"] in (43, 44) and r["recipe"] not in frozen for r in supplemental), "Invalid supplemental role")
    imported = {observation_key(e["row"]) for e in imports}
    require(len(imported) == 12 and len(imported & pkeys) == 8 and imported - pkeys == skeys,
            "Imported observations were lost, duplicated, or admitted incorrectly")
    require(not imported & {observation_key(r) for r in manifest["runs"]}, "Imported successes were allocated local jobs")
    for entry in imports:
        require(entry["expected_config"]["optimizer"]["grad_clip"] is None, "Clipped import mislabeled as unclipped")
        require(entry["spec"]["config"]["optimizer"]["grad_clip"] is None, "Imported original configuration has clipping")
    for method in ("clipped", "unclipped"):
        runs = read(f"data/{method}-runs.json")
        require(len({observation_key(r) for r in runs}) == len(runs), "Duplicate logical observations")
        lookup = {observation_key(r): r for r in runs}
        for recipe in groups[method]:
            for seed, loss in zip(recipe["seeds"], recipe["losses"]):
                row = lookup[(recipe["recipe"], seed)]
                require(row["status"] == "complete", "Failed observation entered ranking")
                close(row["loss"], loss, "Replicated loss against observation")
    CHECKS["eligibility"] = {"clipped_frozen_recipes": 40, "unclipped_frozen_recipes": 20,
                             "primary_imports": 8, "supplemental_imports": 4,
                             "seed42_only_promotion": True, "duplicate_observations": 0}


def verify_analysis(groups, screen, matched):
    """Authenticate the normalized numbers actually consumed by Typst and plots."""
    data = read("data/analysis.json")
    winners = {}
    for method, rows in groups.items():
        normalized = data["replicated"][method]
        originals = {r["recipe"]: r for r in rows}
        require(len(normalized) == len(rows) and {r["recipe"] for r in normalized} == set(originals),
                f"Normalized replication eligibility: {method}")
        for row in normalized:
            original = originals[row["recipe"]]
            require(key(row) == key(original) and row["losses"] == original["losses"]
                    and row["seeds"] == [42, 43, 44], "Normalized recipe/seed/loss mismatch")
            for field in ("mean_loss", "std_loss", "replication_mean_loss"):
                close(row[field], original[field], "Normalized " + field)
            for moment in ("beta1", "beta2"):
                half = -math.log(2) / math.log(row[moment])
                close(row[f"{moment}_half_life_updates"], half, "Normalized update half-life")
                close(row[f"{moment}_global_half_life_targets"], half * row["batch"] * 512, "Global half-life")
                close(row[f"{moment}_per_worker_half_life_targets"], half * row["batch"] * 512 / (1 if method == "sync" else 4), "Per-worker half-life")
        winners[method] = {b: min((r for r in rows if r["batch"] == b),
                                  key=lambda r: (r["mean_loss"], r["lr"], r["beta1"], r["beta2"])) for b in BATCHES}
        require(len(data["winners"][method]) == 4, "Expected one winner per batch")
        for row in data["winners"][method]:
            original = winners[method][row["batch"]]
            require(row["recipe"] == original["recipe"], "Wrong normalized winner")
            close(row["mean_loss"], original["mean_loss"], "Winner mean")
    for method, rows in screen.items():
        normalized = data["screening"][method]
        original = {observation_key(r): r for r in rows}
        require(len(normalized) == len(rows), "Normalized screening count")
        require({observation_key(r) for r in normalized} == set(original), "Normalized screening identity")
        for row in normalized:
            close(row["loss"], original[observation_key(row)]["loss"], "Normalized screening loss")
    require(data["matched_screening"] == matched, "Normalized matched grid differs from bundled source")
    for summary in data["matched_summary"]:
        ds = [r["loss_difference"] for r in matched if r["batch"] == summary["batch"]]
        require(summary["n"] == len(ds) == 120, "Matched summary denominator")
        for field, expected in (("mean_delta", statistics.mean(ds)), ("median_delta", statistics.median(ds)),
                                ("minimum_delta", min(ds)), ("maximum_delta", max(ds))):
            close(summary[field], expected, "Matched summary " + field)
        require(summary["unclipped_wins"] == sum(d < 0 for d in ds)
                and summary["clipped_wins"] == sum(d > 0 for d in ds)
                and summary["ties"] == sum(d == 0 for d in ds), "Matched win counts")
    for row in data["gaps"]:
        b = row["batch"]
        sync, clipped, unclipped = (winners[m][b]["mean_loss"] for m in ("sync", "clipped", "unclipped"))
        for field, value in (("clipped_minus_sync", clipped - sync), ("unclipped_minus_clipped", unclipped - clipped),
                             ("unclipped_minus_sync", unclipped - sync),
                             ("perplexity_increase_percent", 100 * math.expm1(clipped - sync))):
            close(row[field], value, "Quality difference " + field)
    for row in data["beta2_profiles"]:
        expected = min((r for r in screen["clipped"] if r["batch"] == row["batch"] and r["beta2"] == row["beta2"]),
                       key=lambda r: (r["loss"], r["lr"], r["beta1"]))
        require(row["recipe"] == expected["recipe"], "Incorrect profiled beta2 recipe")
        close(row["loss"], expected["loss"], "Beta2 profile")
    for row in data["ablation"]:
        differences = [a - b for a, b in zip(row["losses"], row["baseline_losses"])]
        close(row["mean_loss_difference"], statistics.mean(differences), "Fixed-recipe ablation mean")
        close(row["std_loss_difference"], statistics.stdev(differences), "Fixed-recipe ablation SD")
        require(row["successful_seeds"] == [42, 43, 44], "Wrong ablation seeds")
    require(data["matched_replicated"] == read("data/matched-replicated.json"), "Changed matched replication extract")
    replicated = {m: {r["recipe"]: r for r in rows} for m, rows in groups.items()}
    for row in data["matched_replicated"]:
        a, b = (replicated[m][row["recipe"]]["losses"] for m in ("unclipped", "clipped"))
        require(row["unclipped_losses"] == a and row["clipped_losses"] == b, "Matched three-seed identity")
        differences = [x - y for x, y in zip(a, b)]
        close(row["mean_loss_difference"], statistics.mean(differences), "Matched three-seed mean")
        close(row["std_loss_difference"], statistics.stdev(differences), "Matched three-seed SD")
        for actual, expected in zip(row["seed_loss_differences"], differences):
            close(actual, expected, "Matched seed difference")
    for method, rows in data["clipping_selected"].items():
        require(len(rows) == 4, "Expected four clipping entries with explicit missing values")
        for row in rows:
            if method == "sync" and row["batch"] == 128:
                require(row["clipping_frequency"] is None and not row["available_seeds"],
                        "Historical winner clipping unavailable; a fresh repeat is a different observation")
                continue
            require(row["available_seeds"] == [42, 43, 44], "Incomplete selected clipping observations")
            steps = 408_092_672 // (row["batch"] * 512)
            denominator = steps * (1 if method == "sync" else 4)
            require(row["steps_per_seed"] == steps and row["denominator_per_seed"] == denominator,
                    "Clipping denominator must count worker updates")
            rates = [count / denominator for count in row["clip_counts"]]
            close(row["clipping_frequency"], statistics.mean(rates), "Selected clipping fraction")
            for rate, expected in zip(row["seed_clipping_frequencies"], rates):
                close(rate, expected, "Seed clipping fraction")
    CHECKS["normalized_analysis"] = {"replicated_recipes": 64, "winner_recipes": 12,
                                      "clipping_denominators": 7, "historical_sync128_clipping": "missing",
                                      "loss_differences_and_perplexity": True, "beta2_profile_cells": len(data["beta2_profiles"])}


def verify_documents(groups, matched):
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    counts = {}
    for name in ("slides", "report"):
        document = HERE / f"{name}.pdf"
        bbox = ET.fromstring(command("pdftotext", "-bbox", str(document), "-"))
        pages = bbox.findall(".//x:page", ns)
        counts[name] = len(pages)
        require(len(pages) == 13 if name == "slides" else len(pages) > 0, f"Unexpected {name} page count")
        for index, page in enumerate(pages, 1):
            width, height = float(page.attrib["width"]), float(page.attrib["height"])
            if name == "slides":
                require(abs(width / height - 16 / 9) < 1e-4, f"Slide {index} is not 16:9")
            else:
                require(abs(min(width, height) - 595.276) < .1 and abs(max(width, height) - 841.890) < .1,
                        f"Report page {index} is not A4")
            for word in page.findall(".//x:word", ns):
                a = {k: float(v) for k, v in word.attrib.items()}
                require(a["xMin"] >= 0 and a["yMin"] >= 0 and a["xMax"] <= width + .5 and a["yMax"] <= height + .5,
                        f"Text outside page: {name}, page {index}, {word.text!r}")
    pages = command("pdftotext", "-layout", str(HERE / "slides.pdf"), "-").replace("−", "-").split("\f")[:13]
    winners = {method: {batch: min((r for r in rows if r["batch"] == batch), key=lambda r: r["mean_loss"])
                        for batch in BATCHES} for method, rows in groups.items()}
    sync_best = min(r["mean_loss"] for r in winners["sync"].values())
    dec_best = min(r["mean_loss"] for r in winners["clipped"].values())
    gaps = [winners["clipped"][b]["mean_loss"] - winners["sync"][b]["mean_loss"] for b in BATCHES]
    retuned = [winners["unclipped"][b]["mean_loss"] - winners["clipped"][b]["mean_loss"] for b in BATCHES]
    expected = {3: [f"{sync_best:.6f}", f"{dec_best:.6f}",
                    f"{winners['sync'][128]['mean_loss'] - sync_best:.6f}",
                    f"{winners['clipped'][128]['mean_loss'] - dec_best:.6f}"],
                4: [f"{min(gaps):.6f}", f"{max(gaps):.6f}",
                    f"{100 * math.expm1(min(gaps)):.2f}", f"{100 * math.expm1(max(gaps)):.2f}"],
                9: [f"{sum(r['loss_difference'] < 0 for r in matched if r['batch'] == b)}/120" for b in BATCHES],
                10: [f"{retuned[0]:.6f}", f"{retuned[-1]:.6f}"],
                13: [f"{sync_best:.6f}", f"{dec_best:.6f}"]}
    for page, callouts in expected.items():
        text = re.sub(r"\s+", "", pages[page - 1])
        for value in callouts:
            require(value in text, f"Slide {page}: expected numeric callout {value}")
    report_text = re.sub(r"\s+", "", command("pdftotext", "-layout", str(HERE / "report.pdf"), "-"))
    appendix_values = set()
    for rows in groups.values():
        for row in rows:
            for value in row["losses"] + [row["mean_loss"], row["std_loss"], row["replication_mean_loss"]]:
                appendix_values.add(f"{value:.6f}".rstrip("0").rstrip("."))
    for value in appendix_values:
        require(value in report_text, f"Missing replicated numerical value in report: {value}")
    CHECKS["documents"] = {"slide_pages": counts["slides"], "report_pages": counts["report"],
                            "slide_aspect_ratio": "16:9", "report_paper": "A4",
                            "text_within_page_bounds": True,
                            "recomputed_numeric_callouts": sum(map(len, expected.values())),
                            "replicated_report_values": len(appendix_values)}


def verify_figures():
    metadata = read("data/analysis.json")["figure_metadata"]
    clean_axes = 0
    svg_ns = {"s": "http://www.w3.org/2000/svg"}
    for name in FIGURES:
        for extension in ("pdf", "svg"):
            path = HERE / "assets" / f"{name}.{extension}"
            require(path.is_file() and path.stat().st_size > 1000, f"Missing or empty figure: {path.name}")
        require(metadata[name]["vector"] is True, "Figure must retain vector output")
        if metadata[name]["batch_ticks"] is not None:
            require(metadata[name]["batch_ticks"] == list(BATCHES), "Noncanonical batch tick metadata")
            svg = ET.parse(HERE / "assets" / f"{name}.svg")
            axes = [g for g in svg.findall(".//s:g", svg_ns) if re.fullmatch(r"matplotlib\.axis_\d+", g.get("id", ""))]
            found = 0
            for axis in axes:
                tick_groups = [g for g in axis.findall("s:g", svg_ns) if re.fullmatch(r"xtick_\d+", g.get("id", ""))]
                if not tick_groups:
                    continue
                labels = ["".join(t.itertext()).strip() for g in tick_groups for t in g.findall(".//s:text", svg_ns)]
                if set(map(str, BATCHES)) <= set(labels):
                    require(labels == list(map(str, BATCHES)) and len(tick_groups) == 4,
                            f"Unwanted batch-axis ticks in {name}: {labels}")
                    found += 1
            require(found > 0, f"Could not authenticate actual batch tick labels in {name}.svg")
            clean_axes += found
    CHECKS["figures"] = {"figures": len(FIGURES), "vector_formats": ["pdf", "svg"],
                         "batch_axes_with_only_16_32_64_128": clean_axes,
                         "sha256": {f"{n}.{e}": digest(HERE / "assets" / f"{n}.{e}") for n in FIGURES for e in ("pdf", "svg")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sources", action="store_true", help="Also read and authenticate original campaign artifacts")
    parser.add_argument("--output", type=Path, default=HERE / "validation.json",
                        help="Write verification record here (default: package validation.json)")
    args = parser.parse_args()
    verify_provenance(args.check_sources)
    groups, screen, matched = verify_statistics()
    verify_eligibility(groups, screen)
    verify_analysis(groups, screen, matched)
    verify_figures()
    verify_documents(groups, matched)
    software = {"python": platform.python_version(), "platform": platform.platform()}
    version = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True, check=True)
    software["pdftotext"] = (version.stdout + version.stderr).splitlines()[0]
    for name in ("matplotlib", "numpy"):
        try:
            software[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            software[name] = "not installed in verification environment"
    report = {"status": "passed", "checks": CHECKS,
              "software": software,
              "document_sha256": {f"{name}.pdf": digest(HERE / f"{name}.pdf") for name in ("slides", "report")},
              "verifier_sha256": digest(Path(__file__)),
              "provenance_sha256": digest(HERE / "provenance.json"),
              "analysis_sha256": digest(HERE / "data/analysis.json"),
              "scope": "Checks bundled evidence and presentation; does not revalidate every raw training metric or rerun training.",
              "manual_review": "Visual review, including overlap and legibility, is separate from these automated page-bound checks."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified 13 slides, {CHECKS['documents']['report_pages']} report pages, "
          "64 replicated recipes, 1,320 screening configurations, and import/selection eligibility.")


if __name__ == "__main__":
    main()
