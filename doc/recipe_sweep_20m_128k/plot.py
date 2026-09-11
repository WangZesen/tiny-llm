"""Regenerate the tuning figure from the adjacent, audited summary CSV."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parent
    with (root / "summary.csv").open() as handle:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
    assert len(rows) == 24 and all(row["seeds"] == 3 for row in rows)
    winner = min(rows, key=lambda r: (r["mean_loss"], r["lr"], r["beta2"]))
    lrs = sorted({row["lr"] for row in rows})
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), layout="constrained")
    colors = ["#0072B2", "#D55E00", "#009E73"]
    for ax, title, limit in zip(
        axes,
        ["Full search range", "Detail: lower learning rates"],
        [max(lrs), 0.0056],
        strict=True,
    ):
        for beta2, color, marker in zip([0.95, 0.98, 0.999], colors, ["o", "s", "^"], strict=True):
            group = sorted(
                (r for r in rows if r["beta2"] == beta2 and r["lr"] <= limit),
                key=lambda r: r["lr"],
            )
            ax.errorbar(
                [r["lr"] for r in group],
                [r["mean_loss"] for r in group],
                yerr=[r["std_loss"] for r in group],
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.5,
                capsize=3,
                label=rf"$\beta_2 = {beta2}$",
            )
        ax.scatter(
            winner["lr"],
            winner["mean_loss"],
            marker="*",
            s=180,
            facecolor="#F0E442",
            edgecolor="#333333",
            linewidth=0.8,
            zorder=5,
            label="Selected recipe",
        )
        ax.set_xscale("log")
        ticks = [lr for lr in lrs if lr <= limit]
        ax.set_xticks(ticks, [f"{lr:g}" for lr in ticks], rotation=35, ha="right")
        ax.minorticks_off()
        ax.set_xlabel("Peak learning rate (log scale)")
        ax.set_ylabel("Final full-validation cross-entropy (nats)")
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(axis="y", alpha=0.2)
        ax.margins(x=0.08, y=0.12)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    axes[1].annotate(
        "Winner: LR 0.0056, β₂ 0.98\n3.55869 ± 0.00489",
        xy=(winner["lr"], winner["mean_loss"]),
        xytext=(0.0037, 3.70),
        fontsize=9,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1},
    )
    fig.suptitle(
        "Synchronous 20M · 131,072 tokens per batch · microbatch 128\n"
        "72 runs · error bars: mean ± sample SD over three seeds · lower is better",
        fontsize=12,
    )
    for suffix in ["pdf", "png"]:
        fig.savefig(root / f"tuning.{suffix}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
