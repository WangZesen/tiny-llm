"""Plot all AWC beta2=0.99 results and their matched ATC comparisons."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_numeric(path):
    with path.open() as handle:
        return [
            {k: float(v) for k, v in row.items() if k != "campaign"}
            for row in csv.DictReader(handle)
        ]


def main():
    root = Path(__file__).resolve().parent
    all_awc = read_numeric(root.parent / "recipe_sweep_packed4_20m_128k" / "summary.csv")
    rows = read_numeric(root / "summary.csv")
    matched = read_numeric(root / "atc-comparison.csv")
    results = json.loads((root / "results.json").read_text())
    assert len(all_awc) == 43 and len(rows) == 10 and len(matched) == 6
    assert all(r["seeds"] == 3 and r["beta2"] == 0.99 for r in rows)
    fixed = sorted((r for r in rows if r["beta1"] == 0.9), key=lambda r: r["lr"])
    betas = sorted((r for r in rows if r["lr"] == 0.008), key=lambda r: r["beta1"])
    assert len(fixed) == 7 and len(betas) == 4
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, (response, sensitivity, comparison) = plt.subplots(
        1, 3, figsize=(17, 5), layout="constrained"
    )
    lrs = [r["lr"] for r in fixed]
    for beta, color, marker in zip(
        [0.95, 0.98, 0.99, 0.999], ["#0072B2", "#D55E00", "#009E73", "#CC79A7"], "os^D", strict=True
    ):
        group = sorted(
            (
                r
                for r in all_awc
                if r["beta1"] == 0.9 and r["beta2"] == beta and r["lr"] <= max(lrs)
            ),
            key=lambda r: r["lr"],
        )
        response.errorbar(
            [r["lr"] for r in group],
            [r["mean_loss"] for r in group],
            yerr=[r["std_loss"] for r in group],
            color=color,
            marker=marker,
            capsize=3,
            label=rf"$\beta_2={beta}$",
        )
    winner = results["fixed_beta1_09_winner"]
    response.scatter(
        winner["lr"],
        winner["mean_loss"],
        marker="*",
        s=160,
        color="#F0E442",
        edgecolor="#333333",
        zorder=5,
        label="Best at β₁ = 0.9, β₂ = 0.99",
    )
    response.set_title("AWC learning-rate response · β₁ = 0.9", loc="left", fontsize=10)
    sensitivity.errorbar(
        [r["beta1"] for r in betas],
        [r["mean_loss"] for r in betas],
        yerr=[r["std_loss"] for r in betas],
        color="#009E73",
        marker="^",
        capsize=3,
        label="AWC, β₂ = 0.99",
    )
    winner = results["winner"]
    sensitivity.scatter(
        winner["beta1"],
        winner["mean_loss"],
        marker="*",
        s=160,
        color="#F0E442",
        edgecolor="#333333",
        zorder=5,
        label="Best tested AWC",
    )
    sensitivity.set_title("AWC beta1 response · LR = 0.008", loc="left", fontsize=10)
    sensitivity.set_xticks([r["beta1"] for r in betas])
    sensitivity.set_xlabel("AdamW β₁")
    matched.sort(key=lambda r: r["lr"])
    comparison.errorbar(
        [r["lr"] for r in fixed],
        [r["mean_loss"] for r in fixed],
        yerr=[r["std_loss"] for r in fixed],
        color="#0072B2",
        marker="o",
        capsize=3,
        label="AWC, β₂ = 0.99",
    )
    comparison.errorbar(
        [r["lr"] for r in matched],
        [r["atc_mean"] for r in matched],
        yerr=[r["atc_std"] for r in matched],
        color="#D55E00",
        marker="s",
        capsize=3,
        label="ATC, β₂ = 0.99",
    )
    comparison.set_title("AWC versus ATC · β₁ = 0.9, β₂ = 0.99", loc="left", fontsize=10)
    for ax in (response, comparison):
        ax.set_xscale("log")
        ax.set_xticks(lrs, [f"{lr:g}" for lr in lrs], rotation=40, ha="right")
        ax.minorticks_off()
        ax.set_xlabel("Peak learning rate (log scale)")
    for ax in (response, sensitivity, comparison):
        ax.set_ylabel("Final full-validation cross-entropy (nats)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Four 20M workers · 131,072 targets per batch · AWC β₂ = 0.99 analysis\n"
        "30 AWC runs · 10 configurations · mean ± sample SD over three seeds",
        fontsize=12,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(root / f"tuning.{suffix}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
