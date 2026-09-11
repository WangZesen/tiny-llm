#!/usr/bin/env python3
"""Rebuild the investigation's vector figures from the bundled, unsmoothed data.

Default mode needs only Python, Matplotlib, NumPy and this document directory.
Use --refresh from a checkout with the original runs to update the extracts.
No checkpoints are loaded and no training or Hessian analyses are run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, MaxNLocator, PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ASSETS = HERE / "assets"
REPO = HERE.parent.parent
REVIEW = "runs/consensus-scaling-review-2026-09-11"
SYNC = "20m-0-0022"
PLAIN = "packed-4-20m-0-0048"
AC = "packed-ac-4-20m-0-0048"
FULL = "packed-full-4-20m-0-0048"
SCALED = "packed-ac-full-scale-4-20m-0-0048"
NAVY, BLUE, ORANGE = "#17324D", "#0072B2", "#D55E00"
GREEN, PURPLE, GRAY = "#009E73", "#CC79A7", "#6B7280"
FIGSIZE = (28 / 2.54, 8 / 2.54)


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh(repo):
    """Copy small numerical records and record their exact source identity."""
    source_records = {}
    extract_records = []

    def register(relative):
        path = repo / relative
        source_records[relative] = {
            "path": relative,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        return path

    def copy(relative, target):
        path = register(relative)
        destination = DATA / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        extract_records.append({"file": "data/" + target, "sources": [relative],
                                "selection": "complete source file; bytes unchanged"})

    for run, key in [(SYNC, "sync"), (PLAIN, "plain"), (AC, "adaptive")]:
        copy(f"runs/{run}/analysis/alignment/summary.csv", f"alignment-{key}.csv")
    for run, key in [(AC, "adaptive"), (SCALED, "scaled")]:
        copy(f"runs/{run}/analysis/scaled-alignment/summary.csv", f"scaled-alignment-{key}.csv")

    relative = f"runs/{AC}/analysis/similarity/full/epoch_summary.csv"
    rows = [r for r in read_csv(register(relative)) if r["level"] == "aggregate" and r["group"] == "all"]
    write_csv(DATA / "similarity-aggregate.csv", rows)
    extract_records.append({"file": "data/similarity-aggregate.csv", "sources": [relative],
                            "selection": "level=aggregate and group=all; all categories, epochs and columns; original decimal strings"})

    for source, target in [
        ("checkpoint_summary.csv", "checkpoint-summary.csv"),
        ("counterfactual_mixing.csv", "counterfactual-mixing.csv"),
        ("fresh_update_counterfactual.csv", "fresh-update-counterfactual.csv"),
        ("largest_error_coordinates.csv", "largest-error-coordinates.csv"),
        ("parameter_breakdown.csv", "parameter-breakdown.csv"),
        ("algebra_checks.json", "algebra-checks.json"),
        ("provenance.json", "original-review-provenance.json"),
    ]:
        copy(f"{REVIEW}/{source}", target)

    comparison_relative = f"{REVIEW}/run_comparison.csv"
    comparison = read_csv(register(comparison_relative))
    final_checkpoints = {r["run"]: r for r in read_csv(DATA / "checkpoint-summary.csv") if r["epoch"] == "40"}
    for row in comparison:
        checkpoint = final_checkpoints.get(row["run"], {})
        row["final_second_moment_cosine"] = checkpoint.get("second_moment_cosine", "")
        row["final_largest_coordinate_error_energy_fraction"] = checkpoint.get("largest_coordinate_error_energy_fraction", "")
    write_csv(DATA / "run-comparison.csv", comparison)
    extract_records.append({"file": "data/run-comparison.csv", "sources": [comparison_relative, f"{REVIEW}/checkpoint_summary.csv"],
                            "selection": "all comparison rows joined on run with epoch=40 checkpoint diagnostics; blanks mean unavailable"})

    validation = []
    gamma = []
    run_names = [SYNC] + [r["run"] for r in comparison]
    result_map = {}
    for run in run_names:
        copy(f"runs/{run}/resolved.yaml", f"recipes/{run}.yaml")
        copy(f"runs/{run}/result.json", f"results/{run}.json")
        result = json.loads((DATA / f"results/{run}.json").read_text())
        result_map[run] = result
        relative = f"runs/{run}/metrics.jsonl"
        metrics = read_jsonl(register(relative))
        for record in metrics:
            if record["event"] in ("validation", "final_validation"):
                assert record["split"] == ("full" if record["event"] == "final_validation" else "subset")
                validation.append({"run": run, "event": record["event"], "epoch": record.get("epoch", ""),
                                   "step": record["step"], "tokens": record["tokens"],
                                   "split": record["split"], "loss": record["loss"],
                                   "perplexity": record["perplexity"], "evaluation_tokens": record["evaluation_tokens"]})
            if run == AC and record["event"] == "train":
                gamma.append({"step": record["step"], "epoch": record["epoch"], "tokens": record["tokens"],
                              "lr": record["lr"], "mixing_step": record["mixing_step"], "gamma": record["mixing_gamma"]})
        finals = [r for r in validation if r["run"] == run and r["split"] == "full"]
        assert len(finals) == 1 and finals[0]["loss"] == result["final_validation"]["loss"]
        review_rows = [r for r in comparison if r["run"] == run]
        if review_rows:
            assert float(review_rows[0]["validation_loss"]) == finals[0]["loss"]
    write_csv(DATA / "validation.csv", validation)
    write_csv(DATA / "gamma.csv", gamma)
    extract_records += [
        {"file": "data/validation.csv", "sources": [f"runs/{r}/metrics.jsonl" for r in run_names],
         "selection": "event in {validation,final_validation}; original scalar values parsed from JSON; split retained; no smoothing"},
        {"file": "data/gamma.csv", "sources": [f"runs/{AC}/metrics.jsonl"],
         "selection": "all train events; logged step, epoch, tokens, LR, mixing_step and mixing_gamma; no interpolation of unlogged steps"},
    ]
    write_json(DATA / "results.json", result_map)
    extract_records.append({"file": "data/results.json", "sources": [f"runs/{r}/result.json" for r in run_names],
                            "selection": "all complete result objects keyed by run name; original JSON scalar values"})
    provenance = {
        "description": "Portable numerical extracts for Adaptive Consensus: From Diagnostics to Training Dynamics",
        "source_root": "repository root (paths relative, independent of checkout location)",
        "sources": list(source_records.values()),
        "extracts": extract_records,
        "diagnostics": {
            "main_deck_case": "unseen: held-out training-cache data, not the official validation split",
            "norm_ratio": "average_noise_norm / mean_gradient_norm at first and last recorded checkpoint",
            "alignment": "sum_i(d_i^T H d_i) / sum_i ||d_i||^2 (not a cosine or a [0,1]-bounded statistic)",
            "uncertainty_band": "minimum to maximum of six worker pairs; descriptive range, not a confidence interval",
            "concentration": "100 × worst-worker maximum-coordinate fraction of accumulated consensus error energy; observed checkpoint measurement",
            "validation": "subset trajectories use validation events only; final full-validation values are separate and use final_validation events",
            "smoothing": "none; lines connect stored measurements, and every stored point in selected intervals is retained",
        },
        "figures": {
            "noise": {"data": ["alignment-sync.csv", "alignment-plain.csv"], "selection": "case=unseen", "series": ["average_noise_norm", "mean_gradient_norm", "average_batch_gradient_norm"]},
            "baseline-loss": {"data": ["validation.csv"], "selection": f"run in {{{PLAIN}, {AC}}}; split=subset; all epochs", "transformation": "second panel is paired adaptive loss minus plain loss at identical epochs"},
            "alignment": {"data": ["alignment-adaptive.csv"], "selection": "case=unseen", "series": ["normalized_noise_alignment", "normalized_consensus_alignment", "normalized_random_alignment"], "scale": "log y"},
            "scaled-alignment": {"data": ["scaled-alignment-adaptive.csv"], "selection": "case=unseen; direction in {delta,delta_mul_sqrt_v,delta_mul_v}", "series": ["normalized_alignment"], "scale": "log y"},
            "moment-similarity": {"data": ["similarity-aggregate.csv"], "selection": "category=second_moment", "series": ["cosine_mean", "cosine_min", "cosine_max", "relative_l2_mean", "relative_l2_min", "relative_l2_max"]},
            "scaled-loss": {"data": ["validation.csv"], "selection": f"run in {{{FULL}, {SCALED}, {SCALED}-t2, {SCALED}-t5}}; split=subset; epoch>=10"},
            "diagnosis": {"data": ["checkpoint-summary.csv"], "selection": f"run={SCALED}; observed checkpoints only", "series": ["largest_coordinate_error_energy_fraction × 100", "second_moment_cosine"]},
            "gamma": {"data": ["gamma.csv"], "selection": "all extracted logged train events", "series": ["gamma"], "transformation": "tokens / 1e6"},
        },
    }
    for record in extract_records:
        record["sha256"] = sha256(HERE / record["file"])
    write_json(HERE / "provenance.json", provenance)


def selected(name, **filters):
    return [row for row in read_csv(DATA / name) if all(row[k] == str(v) for k, v in filters.items())]


def values(rows, column):
    return np.asarray([float(row[column]) for row in rows])


def callouts():
    noise = {}
    for key, data_key in [("sync", "sync"), ("packed", "plain")]:
        rows = selected(f"alignment-{data_key}.csv", case="unseen")
        first, last = min(rows, key=lambda r: int(r["epoch"])), max(rows, key=lambda r: int(r["epoch"]))
        noise[key] = {"first_epoch": int(first["epoch"]), "last_epoch": int(last["epoch"]),
                      "ratio_first": float(first["average_noise_norm"]) / float(first["mean_gradient_norm"]),
                      "ratio_last": float(last["average_noise_norm"]) / float(last["mean_gradient_norm"])}
    full = {row["run"]: float(row["loss"]) for row in selected("validation.csv", split="full")}
    a20 = {r["direction"]: float(r["normalized_alignment"]) for r in selected("scaled-alignment-adaptive.csv", epoch=20, case="unseen")}
    sim = {r["category"]: r for r in selected("similarity-aggregate.csv", epoch=40)}
    observed = {r["epoch"]: r for r in selected("checkpoint-summary.csv", run=SCALED)}
    results = json.loads((DATA / "results.json").read_text())
    out = {
        "noise": noise,
        "baseline": {"plain_full": full[PLAIN], "adaptive_full": full[AC], "full_difference": full[AC] - full[PLAIN]},
        "alignment_epoch20": {"delta": a20["delta"], "sqrt_v": a20["delta_mul_sqrt_v"], "v": a20["delta_mul_v"],
                              "sqrt_v_ratio": a20["delta_mul_sqrt_v"] / a20["delta"], "v_ratio": a20["delta_mul_v"] / a20["delta"]},
        "similarity_final": {"v_cosine": float(sim["second_moment"]["cosine_mean"]), "v_relative_l2": float(sim["second_moment"]["relative_l2_mean"]),
                             "m_cosine": float(sim["first_moment"]["cosine_mean"]), "adam_direction_cosine": float(sim["adam_direction"]["cosine_mean"])},
        "diagnosis": {"final_coordinate_percent": 100 * float(observed["40"]["largest_coordinate_error_energy_fraction"]),
                      "epoch24_v_cosine": float(observed["24"]["second_moment_cosine"]), "final_v_cosine": float(observed["40"]["second_moment_cosine"]),
                      "epoch10_v_cosine": float(observed["10"]["second_moment_cosine"])},
        "scaled_full_validation": {"full_mixing": full[FULL], "alpha_half": full[SCALED], "alpha_point_one": full[SCALED + "-t2"], "alpha_minus_half": full[SCALED + "-t5"]},
        "setup": {"parameters": results[AC]["parameters"], "packed_tokens": results[AC]["tokens"], "sync_tokens": results[SYNC]["tokens"],
                  "epochs": results[AC]["epochs"], "packed_steps": results[AC]["step"], "full_validation_tokens": results[AC]["final_validation"]["tokens"]},
    }
    # Numerical assertions target externally specified claims, not plot details.
    assert [round(noise["sync"][k], 2) for k in ("ratio_first", "ratio_last")] == [1.28, 2.61]
    assert [round(noise["packed"][k], 2) for k in ("ratio_first", "ratio_last")] == [2.78, 6.41]
    assert round(out["alignment_epoch20"]["sqrt_v_ratio"]) == 41
    assert round(out["alignment_epoch20"]["v_ratio"]) == 2178
    assert round(out["similarity_final"]["v_cosine"], 4) == 0.9981
    assert round(out["diagnosis"]["final_coordinate_percent"], 1) == 95.7
    assert round(out["baseline"]["plain_full"], 4) == 3.5959
    assert round(out["baseline"]["adaptive_full"], 4) == 3.6062
    write_json(DATA / "callouts.json", out)
    return out


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "DejaVu Sans"],
        "font.size": 16, "axes.labelsize": 16, "axes.titlesize": 17,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
        "axes.titleweight": "normal", "axes.titlecolor": NAVY,
        "text.color": NAVY, "axes.labelcolor": NAVY,
        "xtick.color": GRAY, "ytick.color": GRAY, "axes.edgecolor": "#CBD5E1",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E5E7EB", "grid.alpha": 0.8,
        "grid.linewidth": 0.7, "axes.axisbelow": True,
        "lines.linewidth": 2.5, "lines.markersize": 4,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.2, "legend.labelspacing": 0.3,
        "legend.columnspacing": 1.1, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "path", "svg.hashsalt": "adaptive-consensus-investigation",
        "mathtext.fontset": "dejavusans", "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "figure.dpi": 150,
    })


def figure(cols=1, widths=None):
    fig, axs = plt.subplots(1, cols, figsize=FIGSIZE, layout="constrained", squeeze=False,
                            gridspec_kw={"width_ratios": widths} if widths else None)
    fig.get_layout_engine().set(w_pad=0.055, h_pad=0.04, wspace=0.07, hspace=0.02)
    return fig, axs[0]


def epoch_axis(ax, xmin=1):
    ax.set_xlabel("Epoch")
    ax.set_xlim(xmin, 40)
    ax.set_xticks([10, 20, 30, 40] if xmin == 10 else [1, 10, 20, 30, 40])


def activation(ax, label=False):
    ax.axvline(10, color=GRAY, linestyle=(0, (3, 3)), linewidth=1.3, zorder=1)
    if label:
        ax.annotate("AC activation", xy=(10, 0.96), xycoords=("data", "axes fraction"),
                    xytext=(7, 0), textcoords="offset points", va="top", fontsize=16, color=GRAY)


def log_axis(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[]))


def save(fig, name):
    ASSETS.mkdir(exist_ok=True, parents=True)
    # A fixed canvas keeps the final font sizes predictable at 28 cm width.
    fig.savefig(ASSETS / f"{name}.pdf", metadata={"Creator": "build_figures.py", "CreationDate": None, "ModDate": None})
    fig.savefig(ASSETS / f"{name}.svg", metadata={"Creator": "build_figures.py", "Date": None})
    plt.close(fig)


def plot_noise():
    fig, axs = figure(2)
    for ax, source, title in zip(axs, ["sync", "plain"], ["Synchronous · batch 32,768", "Packed worker · batch 8,192"]):
        rows = selected(f"alignment-{source}.csv", case="unseen")
        x = values(rows, "epoch")
        for key, label, color, line in [
            ("average_noise_norm", "Noise", BLUE, "-"),
            ("mean_gradient_norm", "Mean gradient", GREEN, "-"),
            ("average_batch_gradient_norm", "Batch gradient", GRAY, "--"),
        ]:
            ax.plot(x, values(rows, key), label=label, color=color, linestyle=line,
                    marker="o" if key != "average_batch_gradient_norm" else None,
                    linewidth=2.5 if key != "average_batch_gradient_norm" else 1.5)
        ax.set_title(title, loc="left", pad=10)
        ax.set_ylabel("Gradient norm")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        epoch_axis(ax)
    axs[0].legend(loc="upper right", handlelength=1.2)
    save(fig, "noise")


def plot_baseline():
    fig, (ax, gap_ax) = figure(2, [2.0, 1.05])
    rows_by_run = {}
    for run, label, color, line in [(PLAIN, "Plain decentralized", BLUE, "-"), (AC, "Adaptive consensus", ORANGE, "--")]:
        rows = selected("validation.csv", run=run, split="subset")
        rows_by_run[run] = rows
        ax.plot(values(rows, "epoch"), values(rows, "loss"), label=label, color=color, linestyle=line)
    ax.set_ylabel("Subset validation loss")
    ax.legend(loc="upper right")
    ax.set_ylim(3.48, 5.62)
    epoch_axis(ax)
    activation(ax)
    a = {r["epoch"]: float(r["loss"]) for r in rows_by_run[AC]}
    p = {r["epoch"]: float(r["loss"]) for r in rows_by_run[PLAIN]}
    assert a.keys() == p.keys()
    x = sorted(map(int, a))
    gap_ax.plot(x, [a[str(e)] - p[str(e)] for e in x], color=ORANGE)
    gap_ax.axhline(0, color=GRAY, linewidth=1)
    gap_ax.set_title("Adaptive − plain", loc="left", pad=10)
    gap_ax.set_ylabel("Loss difference")
    gap_ax.yaxis.set_major_locator(MaxNLocator(4))
    epoch_axis(gap_ax)
    activation(gap_ax)
    save(fig, "baseline-loss")


def plot_alignment():
    fig, (ax,) = figure()
    rows = selected("alignment-adaptive.csv", case="unseen")
    for key, label, color, line in [
        ("normalized_noise_alignment", "Gradient noise", BLUE, "-"),
        ("normalized_consensus_alignment", r"Consensus error $\delta$", ORANGE, "-"),
        ("normalized_random_alignment", "Random reference", GRAY, "--"),
    ]:
        assert np.all(values(rows, key) > 0)
        ax.plot(values(rows, "epoch"), values(rows, key), color=color, linestyle=line, marker="o", label=label)
    log_axis(ax)
    ax.set_ylabel("Directional curvature")
    epoch_axis(ax)
    ax.legend(loc="upper right", ncol=3, bbox_to_anchor=(1, 1.21))
    save(fig, "alignment")


def plot_scaled_alignment():
    fig, (ax,) = figure()
    for direction, label, color in [("delta", r"$\delta$", ORANGE), ("delta_mul_sqrt_v", r"$\delta\odot\sqrt{v}$", PURPLE), ("delta_mul_v", r"$\delta\odot v$", GREEN)]:
        rows = selected("scaled-alignment-adaptive.csv", case="unseen", direction=direction)
        assert np.all(values(rows, "normalized_alignment") > 0)
        ax.plot(values(rows, "epoch"), values(rows, "normalized_alignment"), marker="o", color=color, label=label)
    log_axis(ax)
    ax.set_ylabel("Directional curvature")
    epoch_axis(ax)
    ax.legend(loc="upper right", ncol=3, bbox_to_anchor=(1, 1.21))
    save(fig, "scaled-alignment")


def plot_similarity():
    fig, axs = figure(2)
    rows = selected("similarity-aggregate.csv", category="second_moment")
    x = values(rows, "epoch")
    for ax, key, title, ylabel, color in [
        (axs[0], "cosine", r"Local $v$ · cosine", "Pairwise cosine", GREEN),
        (axs[1], "relative_l2", r"Local $v$ · relative distance", "Relative L2", BLUE),
    ]:
        ax.fill_between(x, values(rows, key + "_min"), values(rows, key + "_max"), color=color, alpha=0.14, linewidth=0)
        ax.plot(x, values(rows, key + "_mean"), color=color, label="Six-pair mean")
        ax.set_title(title, loc="left", pad=10)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        epoch_axis(ax)
    axs[0].set_ylim(0.990, 1.0004)
    axs[1].set_ylim(bottom=0)
    save(fig, "moment-similarity")


def plot_scaled_loss():
    fig, (ax,) = figure()
    for run, label, color, line in [
        (FULL, r"Full mixing ($\gamma=1$)", GREEN, "-"),
        (SCALED, r"$\alpha=0.5$", ORANGE, "-"),
        (SCALED + "-t2", r"$\alpha=0.1$", PURPLE, "-"),
        (SCALED + "-t5", r"$\alpha=-0.5$", BLUE, "--"),
    ]:
        rows = [r for r in selected("validation.csv", run=run, split="subset") if int(r["epoch"]) >= 10]
        ax.plot(values(rows, "epoch"), values(rows, "loss"), color=color, label=label, linestyle=line, marker="o", markersize=3)
    ax.set_ylabel("Subset validation loss")
    ax.set_ylim(3.52, 4.30)
    epoch_axis(ax, xmin=10)
    activation(ax)
    ax.legend(loc="upper right", ncol=4, bbox_to_anchor=(1, 1.23))
    ax.annotate("Activation", (10, 4.25), xytext=(6, 0), textcoords="offset points", color=GRAY, va="top", fontsize=16)
    save(fig, "scaled-loss")


def plot_diagnosis():
    fig, axs = figure(2)
    rows = selected("checkpoint-summary.csv", run=SCALED)
    x = values(rows, "epoch")
    axs[0].plot(x, values(rows, "largest_coordinate_error_energy_fraction") * 100, color=ORANGE, marker="o")
    axs[0].set_title("One coordinate dominates", loc="left", pad=10)
    axs[0].set_ylabel("Consensus error energy")
    axs[0].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axs[0].set_ylim(-3, 103)
    axs[0].set_yticks([0, 50, 100])
    axs[1].plot(x, values(rows, "second_moment_cosine"), color=GREEN, marker="o")
    axs[1].set_title(r"Local $v$ diverges", loc="left", pad=10)
    axs[1].set_ylabel("Mean pairwise cosine")
    axs[1].set_ylim(0.69, 1.015)
    axs[1].set_yticks([0.7, 0.8, 0.9, 1.0])
    for ax in axs:
        epoch_axis(ax, xmin=10)
        activation(ax)
    save(fig, "diagnosis")


def plot_gamma():
    fig, (ax,) = figure()
    rows = read_csv(DATA / "gamma.csv")
    ax.plot(values(rows, "tokens") / 1e6, values(rows, "gamma"), color=ORANGE, label=r"Adaptive consensus ($p=2$)")
    ax.axhline(1, color=BLUE, linestyle="--", label=r"Plain decentralized ($\gamma=1$)")
    ax.axvline(408.068096 * 0.25, color=GRAY, linestyle=(0, (3, 3)), linewidth=1.3)
    ax.set_xlabel("Training tokens (millions)")
    ax.set_ylabel(r"Mixing strength $\gamma$")
    ax.set_xlim(0, 410)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1.22), ncol=2)
    save(fig, "gamma")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="refresh included data from original run files")
    parser.add_argument("--repo-root", type=Path, default=REPO, help="original run checkout for --refresh")
    args = parser.parse_args()
    if args.refresh:
        refresh(args.repo_root.resolve())
    if not (DATA / "validation.csv").exists():
        parser.error("bundled data is missing; use --refresh with the source runs available")
    provenance = json.loads((HERE / "provenance.json").read_text())
    for record in provenance["extracts"]:
        assert sha256(HERE / record["file"]) == record["sha256"], f"Bundled data integrity check failed: {record['file']}"
    numbers = callouts()
    callout_record = {
        "file": "data/callouts.json", "sha256": sha256(DATA / "callouts.json"),
        "inputs": ["data/alignment-sync.csv", "data/alignment-plain.csv", "data/scaled-alignment-adaptive.csv",
                   "data/validation.csv", "data/similarity-aggregate.csv", "data/checkpoint-summary.csv", "data/results.json"],
        "selection": "computed numerical slide anchors; formulas specified in diagnostics and build_figures.py; full stored floating-point values",
    }
    provenance["extracts"] = [r for r in provenance["extracts"] if r["file"] != "data/callouts.json"] + [callout_record]
    write_json(HERE / "provenance.json", provenance)
    style()
    for plotter in [plot_noise, plot_baseline, plot_alignment, plot_scaled_alignment,
                    plot_similarity, plot_scaled_loss, plot_diagnosis, plot_gamma]:
        plotter()
    print("Built eight PDF/SVG figures from included data; numerical callouts validated.")
    print(f"Final full-validation loss: plain={numbers['baseline']['plain_full']:.6f}; adaptive={numbers['baseline']['adaptive_full']:.6f}")


if __name__ == "__main__":
    main()
