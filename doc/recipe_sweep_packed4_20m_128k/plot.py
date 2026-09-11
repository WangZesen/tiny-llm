"""Regenerate the tuning figure from the adjacent, audited summary CSV."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    root = Path(__file__).resolve().parent
    with (root / "summary.csv").open() as handle:
        rows = [
            {k: float(v) for k, v in row.items() if k != "campaign"}
            for row in csv.DictReader(handle)
        ]
    assert len(rows) == 33 and all(row["seeds"] == 3 for row in rows)
    all_rows = rows
    rows = [row for row in rows if row["beta1"] == 0.9]
    assert len(rows) == 24
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
    plot_betas(all_rows, root)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), layout="constrained")
    colors = ["#0072B2", "#D55E00", "#009E73"]
    for ax, title, limit in zip(
        axes,
        ["Full search range", "Detail: lower learning rates"],
        [max(lrs), 0.008],
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
        "Winner: LR 0.008, β₂ 0.98\n3.60191 ± 0.00684",
        xy=(winner["lr"], winner["mean_loss"]),
        xytext=(0.0045, 3.73),
        fontsize=9,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1},
    )
    fig.suptitle(
        "Packed-4 20M · 131,072 tokens per batch · microbatch 32 per model\n"
        "β₁ = 0.9 · 24 configurations · 72 runs · mean ± sample SD (three seeds)",
        fontsize=12,
    )
    for suffix in ["pdf", "png"]:
        fig.savefig(root / f"tuning.{suffix}", dpi=200)
    plt.close(fig)


def plot_betas(rows, root):
    """Plot the 12 tested beta1/beta2 configurations at fixed LR 0.008."""
    rows = [r for r in rows if r["lr"] == 0.008]
    assert len(rows) == 12
    beta1s = sorted({r["beta1"] for r in rows})
    beta2s = sorted({r["beta2"] for r in rows})
    winner = min(rows, key=lambda r: (r["mean_loss"], r["beta1"], r["beta2"]))
    fig, (ax, heat) = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for b2, color, marker in zip(
        beta2s, ["#0072B2", "#D55E00", "#009E73"], ["o", "s", "^"], strict=True
    ):
        group = sorted((r for r in rows if r["beta2"] == b2), key=lambda r: r["beta1"])
        ax.errorbar(
            [r["beta1"] for r in group],
            [r["mean_loss"] for r in group],
            yerr=[r["std_loss"] for r in group],
            color=color,
            marker=marker,
            markersize=5,
            linewidth=1.5,
            capsize=3,
            label=rf"$\beta_2 = {b2}$",
        )
    ax.scatter(
        winner["beta1"],
        winner["mean_loss"],
        marker="*",
        s=180,
        facecolor="#F0E442",
        edgecolor="#333333",
        zorder=5,
        label="Best tested recipe",
    )
    ax.set_xticks(beta1s, [f"{b:g}" for b in beta1s])
    ax.set_xlabel(r"AdamW $\beta_1$")
    ax.set_ylabel("Final full-validation cross-entropy (nats)")
    ax.set_title("Response to higher beta1", loc="left", fontsize=11)
    ax.grid(axis="y", alpha=0.2)
    ax.margins(x=0.12, y=0.15)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    lookup = {(r["beta1"], r["beta2"]): r for r in rows}
    values = [[lookup[(b1, b2)]["mean_loss"] for b1 in beta1s] for b2 in beta2s]
    color = heat.imshow(values, cmap="cividis_r", aspect="auto")
    for y, b2 in enumerate(beta2s):
        for x, b1 in enumerate(beta1s):
            r = lookup[(b1, b2)]
            rgb = color.cmap(color.norm(r["mean_loss"]))[:3]
            luminance = sum(a * b for a, b in zip(rgb, [0.2126, 0.7152, 0.0722], strict=True))
            heat.text(
                x,
                y,
                f"{r['mean_loss']:.5f}\n± {r['std_loss']:.5f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black" if luminance > 0.5 else "white",
            )
    heat.add_patch(
        Rectangle(
            (beta1s.index(winner["beta1"]) - 0.5, beta2s.index(winner["beta2"]) - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#D55E00",
            linewidth=2.5,
        )
    )
    heat.set_xticks(range(len(beta1s)), [f"{b:g}" for b in beta1s])
    heat.set_yticks(range(len(beta2s)), [f"{b:g}" for b in beta2s])
    heat.set_xlabel(r"AdamW $\beta_1$")
    heat.set_ylabel(r"AdamW $\beta_2$")
    heat.set_title("Mean ± sample SD; outlined cell is best", loc="left", fontsize=11)
    fig.colorbar(color, ax=heat, label="Mean loss (nats)", shrink=0.85)
    fig.suptitle(
        "Packed-4 20M · fixed LR 0.008 · 131,072 tokens per batch\n"
        "12 configurations · 36 runs · mean ± sample SD (three seeds) · lower is better",
        fontsize=12,
    )
    for suffix in ["pdf", "png"]:
        fig.savefig(root / f"betas.{suffix}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
