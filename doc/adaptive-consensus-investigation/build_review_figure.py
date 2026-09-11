#!/usr/bin/env python3
"""Rebuild the archived review chart with consensus-error terminology.

The four panels, observations, and frozen-checkpoint curves match the archived
review figure. Only audience-facing wording changes. The default build uses
bundled data; --refresh snapshots the two additional similarity time series.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXTRA = DATA / "review-figure"
SCALED = "packed-ac-full-scale-4-20m-0-0048"
AC = "packed-ac-4-20m-0-0048"
EXCLUDED = "packed-ac-exclude-full-4-20m-0-0048"
REVIEW = "runs/consensus-scaling-review-2026-09-11"
INPUTS = ["validation.csv", "similarity-aggregate.csv", "checkpoint-summary.csv",
          "counterfactual-mixing.csv", "fresh-update-counterfactual.csv",
          "parameter-breakdown.csv"]


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh(repo):
    EXTRA.mkdir(parents=True, exist_ok=True)
    sources, extracts = [], []
    for run in (SCALED, EXCLUDED):
        relative = f"runs/{run}/analysis/similarity/full/epoch_summary.csv"
        source = repo / relative
        rows = [row for row in read(source)
                if row["category"] == "second_moment" and row["level"] == "aggregate"]
        output = EXTRA / f"{run}.csv"
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        sources.append({"path": relative, "sha256": sha256(source), "bytes": source.stat().st_size})
        extracts.append({"path": str(output.relative_to(HERE)), "sha256": sha256(output),
                         "source": relative,
                         "selection": "category=second_moment and level=aggregate; all columns and original decimal strings"})
    original_script = repo / REVIEW / "plot_review.py"
    sources.append({"path": f"{REVIEW}/plot_review.py", "sha256": sha256(original_script),
                    "bytes": original_script.stat().st_size})
    inputs = [{"path": f"data/{name}", "sha256": sha256(DATA / name)} for name in INPUTS]
    provenance = {
        "description": "Terminology-adjusted four-panel derivative of archived scaling review diagnosis; original PDF retained unchanged",
        "sources": sources, "extracts": extracts, "existing_bundled_inputs": inputs,
        "existing_input_provenance": "../../provenance.json",
        "outputs": ["assets/review-diagnosis-consensus-error.svg", "assets/review-diagnosis-consensus-error.pdf"],
        "transformations": {
            "A": "validation events with epoch>=10, raw subset loss; five original runs",
            "B": "second_moment aggregate pairwise cosine; three original runs",
            "C": "original four largest-coordinate energy fractions multiplied by 100",
            "D": "original named-tensor error energy fractions multiplied by 100; Other=100 minus sum of displayed tensors",
        },
        "terminology": "Consensus error means local model minus global average model; labels updated without changing data or original curves.",
    }
    (EXTRA / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=HERE.parent.parent)
    args = parser.parse_args()
    if args.refresh:
        refresh(args.repo_root.resolve())
    provenance = json.loads((EXTRA / "provenance.json").read_text())
    for record in provenance["extracts"] + provenance["existing_bundled_inputs"]:
        assert sha256(HERE / record["path"]) == record["sha256"], record["path"]

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "svg.fonttype": "path", "savefig.dpi": 170})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3), layout="constrained")
    colors = ["#18764d", "#af2944", "#d87e1b", "#306bb0", "#7a53a0"]
    curves = [("packed-full-4-20m-0-0048", r"Full mixing ($\gamma=1$)"),
              (SCALED, r"$\alpha=0.5, p=2$"),
              (SCALED + "-t2", r"$\alpha=0.1, p=2$"),
              (SCALED + "-t5", r"$\alpha=-0.5, p=2$"),
              (SCALED + "-t7", r"$\alpha=0.025, p=1$")]
    validation = read(DATA / "validation.csv")
    for (run, label), color in zip(curves, colors):
        rows = [r for r in validation if r["run"] == run and r["event"] == "validation" and int(r["epoch"]) >= 10]
        axes[0, 0].plot([int(r["epoch"]) for r in rows], [float(r["loss"]) for r in rows],
                        label=label, color=color, lw=1.6)
    axes[0, 0].set(title="A. Validation spikes with positive exponents", xlabel="Epoch",
                   ylabel="Subset validation loss", xlim=(10, 40))
    axes[0, 0].legend(fontsize=8.5)

    for run, label, color in [(SCALED, r"$\alpha=0.5$; complete topology", colors[1]),
                              (AC, "AC; exponential topology", "#555555"),
                              (EXCLUDED, "AC; complete, embedding excluded", colors[0])]:
        path = DATA / "similarity-aggregate.csv" if run == AC else EXTRA / f"{run}.csv"
        rows = [r for r in read(path) if r["category"] == "second_moment" and r["level"] == "aggregate"]
        axes[0, 1].plot([int(r["epoch"]) for r in rows], [float(r["cosine_mean"]) for r in rows],
                        color=color, label=label, lw=1.8)
    axes[0, 1].axvline(10, color="#aaaaaa", ls=":", lw=1)
    axes[0, 1].set(title="B. Local second moments separate after activation", xlabel="Epoch",
                   ylabel="Mean pairwise cosine", ylim=(0.69, 1.01))
    axes[0, 1].legend(fontsize=8.5, loc="lower left")

    actual = [r for r in read(DATA / "checkpoint-summary.csv") if r["run"] == SCALED]
    cf = [r for r in read(DATA / "counterfactual-mixing.csv") if r["run"] == SCALED and r["variant"] == "local_sqrt"]
    fresh = read(DATA / "fresh-update-counterfactual.csv")
    for rows, key, label, color, style in [
        (actual, "largest_coordinate_error_energy_fraction", "Observed consensus error", colors[1], "-"),
        (cf, "largest_coordinate_scaled_energy_fraction", r"Reapply $\sqrt{v}$ to consensus error (frozen)", colors[2], "--"),
        (fresh, "largest_sqrt_shaped_update_coordinate_fraction", r"Shape fresh update with $\sqrt{\bar v}$ (frozen)", colors[3], "--"),
        (fresh, "largest_bounded_shaped_update_coordinate_fraction", "Bounded fresh-update shaping (frozen)", colors[0], ":"),
    ]:
        axes[1, 0].plot([int(r["epoch"]) for r in rows], [100 * float(r[key]) for r in rows],
                        color=color, ls=style, marker="o", ms=4, label=label)
    axes[1, 0].set(title="C. Consensus error concentrates in single coordinates", xlabel="Epoch",
                   ylabel="Largest coordinate share, worst worker (%)", ylim=(-3, 104))
    axes[1, 0].legend(fontsize=8.0, loc="center right")

    details = read(DATA / "parameter-breakdown.csv")
    epochs = [int(r["epoch"]) for r in actual]
    groups = [("embedding.weight", "Embedding / tied head", colors[4]),
              ("blocks.7.ffn.gate_proj.weight", "Last FFN gate", colors[1]),
              ("blocks.7.ffn.up_proj.weight", "Last FFN up", colors[2]),
              ("blocks.7.ffn.down_proj.weight", "Last FFN down", colors[3])]
    bottom = [0.0] * len(epochs)
    for name, label, color in groups:
        values = [100 * sum(float(r["error_energy_fraction"]) for r in details
                           if r["run"] == SCALED and int(r["epoch"]) == e and r["name"] == name)
                  for e in epochs]
        axes[1, 1].bar(range(len(epochs)), values, bottom=bottom, color=color, label=label, width=0.75)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[1, 1].bar(range(len(epochs)), [100 - v for v in bottom], bottom=bottom,
                   color="#d6d8db", label="Other parameters", width=0.75)
    axes[1, 1].set(title=r"D. Consensus-error energy migrates ($\alpha=0.5$)", xlabel="Epoch",
                   ylabel="Share of total squared consensus-error norm (%)",
                   xticks=range(len(epochs)), xticklabels=epochs, ylim=(0, 101))
    axes[1, 1].legend(fontsize=8, ncol=2, loc="lower left")
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle("Moment-scaled adaptive consensus: observed failure and diagnostic alternatives", fontsize=14)
    for suffix in ("pdf", "svg"):
        fig.savefig(HERE / "assets" / f"review-diagnosis-consensus-error.{suffix}")
    plt.close(fig)
    print("Built review diagnosis PDF/SVG with consensus-error terminology from verified bundled data.")


if __name__ == "__main__":
    main()
