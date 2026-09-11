#!/usr/bin/env python3
"""Acceptance checks for the compiled research presentation and source package."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads((HERE / path).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args):
    result = subprocess.run(args, text=True, capture_output=True, check=True)
    return result.stdout


def main():
    numbers = read_json("data/callouts.json")
    slide_pdf = HERE / "slides.pdf"
    supplement_pdf = HERE / "supplement.pdf"
    slide_info = command("pdfinfo", str(slide_pdf))
    supplement_info = command("pdfinfo", str(supplement_pdf))
    slide_count = int(re.search(r"Pages:\s+(\d+)", slide_info).group(1))
    supplement_count = int(re.search(r"Pages:\s+(\d+)", supplement_info).group(1))
    assert slide_count == 13, f"Expected title + 12 content slides, found {slide_count}"
    assert supplement_count > 0
    size = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", slide_info)
    assert abs(float(size[1]) / float(size[2]) - 16 / 9) < 1e-4

    # Check that rounded numbers in the actual PDF agree with bundled measurements.
    pages = command("pdftotext", "-layout", str(slide_pdf), "-").split("\f")[:13]
    expect = {
        4: [f"{numbers['noise'][run][key]:.2f}" for run in ("sync", "packed")
            for key in ("ratio_first", "ratio_last")],
        5: [f"{v:.4f}" for v in numbers["baseline"].values()],
        7: [f"{numbers['alignment_epoch20'][key]:,.0f}" for key in ("sqrt_v_ratio", "v_ratio")],
        8: [f"{numbers['similarity_final'][key]:.4f}" for key in ("v_cosine", "m_cosine", "adam_direction_cosine")],
        10: [f"{v:.4f}" for v in numbers["scaled_full_validation"].values()],
        11: [f"{numbers['diagnosis']['final_coordinate_percent']:.1f}"],
    }
    with (HERE / "data/alignment-adaptive.csv").open(newline="") as stream:
        reference = next(r for r in csv.DictReader(stream) if r["epoch"] == "20" and r["case"] == "unseen")
    ratio = float(reference["normalized_noise_alignment"]) / float(reference["normalized_consensus_alignment"])
    expect[6] = [f"{ratio:.0f}"]
    for content_slide, values in expect.items():
        for value in values:
            assert value in pages[content_slide], f"Slide {content_slide}: expected numeric callout {value}"
    for number, page in enumerate(pages[1:], 1):
        assert re.search(rf"\b{number}\s*/\s*12\b", page), f"Missing content-slide footer {number}"

    # Check actual text bounding boxes, not just the page count.
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    for document in (slide_pdf, supplement_pdf):
        bbox = ET.fromstring(command("pdftotext", "-bbox", str(document), "-"))
        for index, page in enumerate(bbox.findall(".//x:page", ns), 1):
            width, height = float(page.attrib["width"]), float(page.attrib["height"])
            if document == supplement_pdf:
                assert abs(min(width, height) - 595.276) < .1
                assert abs(max(width, height) - 841.890) < .1
            for word in page.findall(".//x:word", ns):
                a = word.attrib
                assert float(a["xMin"]) >= 0 and float(a["yMin"]) >= 0, (document.name, index, word.text)
                assert float(a["xMax"]) <= width + .5 and float(a["yMax"]) <= height + .5, (document.name, index, word.text)

    originals = read_json("assets/originals/sources.json")
    assert len(originals) == 6
    for original in originals:
        assert digest(HERE / original["asset"]) == original["sha256"], original["asset"]
    provenance = read_json("provenance.json")
    source_hashes = {r["path"]: r["sha256"] for r in provenance["sources"]}
    for extract in provenance["extracts"]:
        path = HERE / extract["file"]
        assert path.is_file(), path
        if "sha256" in extract:
            assert digest(path) == extract["sha256"], path
        if extract["selection"] == "complete source file; bytes unchanged":
            assert digest(path) == source_hashes[extract["sources"][0]], path
    for name in provenance["figures"]:
        for extension in ("pdf", "svg"):
            assert (HERE / "assets" / f"{name}.{extension}").stat().st_size > 1000

    # Verify that saved full-validation measurements match the review table.
    with (HERE / "data/validation.csv").open(newline="") as stream:
        full = {r["run"]: float(r["loss"]) for r in csv.DictReader(stream) if r["split"] == "full"}
    with (HERE / "data/run-comparison.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    supplement_rows = {r["run"]: r for r in read_json("supplement-data.json")["run_comparison"]}
    for row in rows:
        assert abs(full[row["run"]] - float(row["validation_loss"])) < 1e-12
        assert abs(float(supplement_rows[row["run"]]["validation_loss"]) - full[row["run"]]) < 1e-12

    report = {
        "status": "passed",
        "slide_pages": slide_count,
        "supplement_pages": supplement_count,
        "slide_aspect_ratio": "16:9",
        "supplement_paper": "A4, portrait and landscape",
        "original_figures_verified": len(originals),
        "review_rows_verified": len(rows),
        "numeric_callouts_verified": sum(map(len, expect.values())),
        "text_within_page_bounds": True,
        "bundled_extract_hashes_verified": True,
        "training_or_hessian_computation_required": False,
        "document_sha256": {"slides.pdf": digest(slide_pdf), "supplement.pdf": digest(supplement_pdf)},
        "manual_review": "Visual page review and historical-source equation audit are separate from these automated checks.",
    }
    (HERE / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {slide_count} slides, {supplement_count} supplement pages, "
          f"{report['numeric_callouts_verified']} numeric callouts, six original figures, and all 12 review rows.")


if __name__ == "__main__":
    main()
