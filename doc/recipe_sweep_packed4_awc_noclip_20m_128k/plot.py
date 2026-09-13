"""Regenerate four-worker AWC without clipping heatmaps and LR response figures."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent


def save(fig, name):
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"{name}.{suffix}", dpi=200)
    plt.close(fig)


def main():
    data = json.loads((ROOT / "results.json").read_text())
    rows = data["groups"]
    assert len(rows) == 80 and all(r["seeds"] == 3 for r in rows)
    lrs = sorted({r["lr"] for r in rows})
    b1s = sorted({r["beta1"] for r in rows})
    b2s = sorted({r["beta2"] for r in rows})
    lookup = {(r["lr"], r["beta1"], r["beta2"]): r for r in rows}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    norm = Normalize(min(r["mean_loss"] for r in rows), max(r["mean_loss"] for r in rows))
    cmap = plt.get_cmap("viridis_r")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    for ax, lr in zip(axes.flat[:5], lrs, strict=True):
        grid = np.array([[lookup[lr, b1, b2]["mean_loss"] for b1 in b1s] for b2 in b2s])
        im = ax.imshow(grid, cmap=cmap, norm=norm)
        ax.set_xticks(range(4), [f"{v:g}" for v in b1s])
        ax.set_yticks(range(4), [f"{v:g}" for v in b2s])
        ax.set_xlabel("AdamW β₁")
        ax.set_ylabel("AdamW β₂")
        ax.set_title(f"LR {lr:g}")
        best = min((r for r in rows if r["lr"] == lr), key=lambda r: r["mean_loss"])
        ax.add_patch(
            Rectangle(
                (b1s.index(best["beta1"]) - 0.5, b2s.index(best["beta2"]) - 0.5),
                1,
                1,
                fill=False,
                linewidth=3,
                edgecolor="#E69F00",
            )
        )
        for y, b2 in enumerate(b2s):
            for x, b1 in enumerate(b1s):
                r = lookup[lr, b1, b2]
                rgb = cmap(norm(r["mean_loss"]))[:3]
                color = "black" if np.dot(rgb, [0.2126, 0.7152, 0.0722]) > 0.5 else "white"
                ax.text(
                    x,
                    y,
                    f"{r['mean_loss']:.4f}\n±{r['std_loss']:.4f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )
    axes.flat[-1].axis("off")
    w = data["winner"]
    axes.flat[-1].text(
        0.05,
        0.65,
        f"Best tested recipe\nLR {w['lr']:g}, β₁ {w['beta1']:g}, β₂ {w['beta2']:g}\n"
        f"{w['mean_loss']:.6f} ± {w['std_loss']:.6f} nats\n\n"
        "Orange outline: lowest mean at each LR\nShared color scale across all panels\n"
        "Every cell: mean ± sample SD, three seeds\nLower loss is better",
        transform=axes.flat[-1].transAxes,
        va="center",
        fontsize=12,
    )
    fig.colorbar(im, ax=list(axes.flat[:5]), shrink=0.75, label="Full-validation loss (nats)")
    fig.suptitle(
        "AWC · four 20M workers · no clipping · 131,072 targets per batch\n"
        "Complete LR × β₁ × β₂ grid · 80 configurations · 240 runs",
        fontsize=15,
    )
    save(fig, "heatmaps")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained", sharex=True, sharey=True)
    for ax, b1 in zip(axes.flat, b1s, strict=True):
        for b2, color, marker in zip(
            b2s, ["#0072B2", "#D55E00", "#009E73", "#CC79A7"], "os^D", strict=True
        ):
            group = [lookup[lr, b1, b2] for lr in lrs]
            ax.errorbar(
                lrs,
                [r["mean_loss"] for r in group],
                yerr=[r["std_loss"] for r in group],
                color=color,
                marker=marker,
                capsize=3,
                label=f"β₂ = {b2:g}",
            )
        if b1 == w["beta1"]:
            ax.scatter(
                w["lr"],
                w["mean_loss"],
                marker="*",
                s=180,
                color="#F0E442",
                edgecolor="black",
                zorder=5,
                label="Best tested recipe",
            )
        ax.set_title(f"β₁ = {b1:g}")
        ax.set_xscale("log")
        ax.set_xticks(lrs, [f"{lr:g}" for lr in lrs], rotation=30, ha="right")
        ax.minorticks_off()
        ax.set_xlabel("Peak learning rate (log scale)")
        ax.set_ylabel("Full-validation cross-entropy (nats)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=9)
        ax.tick_params(labelbottom=True)
    fig.suptitle(
        "AWC · four 20M workers · no clipping · learning-rate response\n"
        "Mean ± sample SD across seeds 42, 43, 44 · shared loss scale",
        fontsize=15,
    )
    save(fig, "tuning")
    matched = sorted(data["matched_clipped"], key=lambda r: (r["lr"], r["beta1"], r["beta2"]))
    assert len(matched) == 28
    fig, ax = plt.subplots(figsize=(10, 11), layout="constrained")
    for i, row in enumerate(matched):
        delta = row["difference_unclipped_minus_clipped"]
        ax.errorbar(
            delta,
            i,
            xerr=row["paired_std"],
            fmt="o",
            capsize=3,
            color="#D55E00" if delta >= 0 else "#009E73",
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(
        range(len(matched)), [f"{r['lr']:g} / {r['beta1']:g} / {r['beta2']:g}" for r in matched]
    )
    ax.invert_yaxis()
    ax.set_ylabel("Learning rate / β₁ / β₂")
    ax.set_xlabel("Full-validation loss difference: no clipping − clipping at 1.0 (nats)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_title(
        "Four-worker AWC · 28 matched configurations, seeds 42–44\n"
        "Mean ± sample SD of seed-paired differences · positive favors clipping"
    )
    save(fig, "comparison")


if __name__ == "__main__":
    main()
