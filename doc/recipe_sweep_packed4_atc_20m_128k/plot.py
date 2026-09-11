"""Regenerate ATC tuning figures from the adjacent audited CSV files."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def main():
    root = Path(__file__).resolve().parent
    with (root / "summary.csv").open() as handle:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
    provenance = json.loads((root / "results.json").read_text())
    lrs = sorted({r["lr"] for r in rows})
    betas = sorted({r["beta2"] for r in rows})
    assert len(rows) == len(lrs) * len(betas) == 24
    assert all(r["seeds"] == 3 for r in rows)
    winner = min(rows, key=lambda r: (r["mean_loss"], r["lr"], r["beta2"]))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, (response, heatmap) = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    for beta, color, marker in zip(
        betas, ["#0072B2", "#D55E00", "#009E73", "#CC79A7"], "os^D", strict=True
    ):
        group = sorted((r for r in rows if r["beta2"] == beta), key=lambda r: r["lr"])
        response.errorbar(
            [r["lr"] for r in group],
            [r["mean_loss"] for r in group],
            yerr=[r["std_loss"] for r in group],
            marker=marker,
            color=color,
            capsize=3,
            label=rf"$\beta_2={beta}$",
        )
    response.scatter(
        winner["lr"],
        winner["mean_loss"],
        marker="*",
        s=180,
        color="#F0E442",
        edgecolor="#333333",
        zorder=5,
        label="Selected ATC",
    )
    response.axhline(
        provenance["awc_reference"]["winner"]["mean_loss"],
        color="#666666",
        ls="--",
        lw=1,
        label=f"AWC best mean (β₁={provenance['awc_reference']['winner']['beta1']:g})",
    )
    response.set_xscale("log")
    response.set_xticks(lrs, [f"{lr:g}" for lr in lrs], rotation=35, ha="right")
    response.minorticks_off()
    response.set_xlabel("Peak learning rate (log scale)")
    response.set_ylabel("Final full-validation cross-entropy (nats)")
    response.set_title("Learning-rate response", loc="left")
    response.legend(frameon=False, fontsize=8)
    response.grid(axis="y", alpha=0.2)
    lookup = {(r["lr"], r["beta2"]): r for r in rows}
    values = np.array([[lookup[lr, b]["mean_loss"] for lr in lrs] for b in betas])
    rendered = heatmap.imshow(values, aspect="auto", cmap="viridis_r")
    for j, beta in enumerate(betas):
        for i, lr in enumerate(lrs):
            row = lookup[lr, beta]
            shade = rendered.cmap(rendered.norm(row["mean_loss"]))
            luminance = sum(a * b for a, b in zip(shade[:3], [0.299, 0.587, 0.114], strict=True))
            heatmap.text(
                i,
                j,
                f"{row['mean_loss']:.4f}\n±{row['std_loss']:.4f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if luminance > 0.5 else "white",
            )
    heatmap.add_patch(
        Rectangle(
            (lrs.index(winner["lr"]) - 0.5, betas.index(winner["beta2"]) - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#222222",
            linewidth=2,
        )
    )
    heatmap.set_xticks(range(len(lrs)), [f"{lr:g}" for lr in lrs], rotation=35, ha="right")
    heatmap.set_yticks(range(len(betas)), [f"{b:g}" for b in betas])
    heatmap.set_xlabel("Peak learning rate")
    heatmap.set_ylabel(r"AdamW $\beta_2$")
    heatmap.set_title("Mean ± sample SD; selected cell outlined", loc="left")
    fig.colorbar(rendered, ax=heatmap, label="Mean loss (nats)", shrink=0.85)
    fig.suptitle(
        "ATC · four 20M workers · 131,072 targets per batch · β₁ = 0.9\n"
        "72 runs · three seeds per configuration · lower is better",
        fontsize=12,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(root / f"tuning.{suffix}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
