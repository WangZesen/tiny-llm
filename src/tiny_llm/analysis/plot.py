"""Regenerate analysis tables and figures using saved scalar results only."""

import csv
import json
from pathlib import Path


def plot_analysis(output: Path, training_norms: bool = False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    rows, seen = [], set()
    for checkpoint in manifest["checkpoints"]:
        for case in ("seen", "unseen"):
            path = output / f"{checkpoint['weights_hash']}-{case}.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            if result["identity"] != manifest["identity"]:
                raise ValueError(f"incompatible result {path}")
            rows.append(
                dict(
                    checkpoint=checkpoint["name"],
                    tokens=checkpoint["tokens"],
                    epoch=checkpoint["epoch"],
                    step=checkpoint["step"],
                    case=case,
                    weights_hash=checkpoint["weights_hash"],
                    **result["statistics"],
                )
            )
    if not rows:
        return []
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "summary.csv")
    curves = []
    for row in sorted(rows, key=lambda row: (row["tokens"], row["checkpoint"])):
        key = (row["weights_hash"], row["tokens"], row["case"])
        if key not in seen:
            curves.append(row)
            seen.add(key)
    specs = [
        (
            "normalized-alignment",
            "Normalized alignment",
            [
                ("normalized_noise_alignment", "Gradient noise"),
                ("normalized_random_alignment", "Random direction"),
            ],
        ),
        (
            "noise-alignment",
            "Unnormalized noise alignment",
            [
                ("noise_alignment", "Gradient noise"),
            ],
        ),
        (
            "gradient-norms",
            "Norm",
            [
                ("average_noise_norm", "Average noise norm"),
                ("average_batch_gradient_norm", "Average batch-gradient norm"),
                ("mean_gradient_norm", "Mean-gradient norm"),
            ],
        ),
    ]
    for name, ylabel, fields in specs:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for color, (field, label) in enumerate(fields):
            for case, linestyle in (("seen", "-"), ("unseen", "--")):
                values = [row for row in curves if row["case"] == case]
                if values:
                    ax.plot(
                        [row["tokens"] for row in values],
                        [float("nan") if row[field] is None else row[field] for row in values],
                        color=f"C{color}",
                        linestyle=linestyle,
                        marker="o",
                        markersize=3,
                        label=f"{label} ({case})",
                    )
        if training_norms and name == "gradient-norms":
            path = Path(manifest["run"]) / "metrics.jsonl"
            logged = {}
            if path.exists():
                for line in path.read_text().splitlines():
                    if line.strip():
                        row = json.loads(line)
                        if row.get("event") == "train" and "grad_norm" in row:
                            logged[row["tokens"]] = row
            if logged:
                values = [logged[key] for key in sorted(logged)]
                label = (
                    "Logged maximum worker norm (pre-clipping)"
                    if "local_grad_norms" in values[0]
                    else "Logged training batch norm (pre-clipping)"
                )
                ax.plot(
                    [row["tokens"] for row in values],
                    [row["grad_norm"] for row in values],
                    color="gray",
                    alpha=0.5,
                    linewidth=1,
                    label=label,
                )
            else:
                ax.text(
                    0.02,
                    0.98,
                    "Training gradient logs unavailable",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=8,
                )
        ax.set(xlabel="Training tokens", ylabel=ylabel, title=Path(manifest["run"]).name)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        for extension in ("pdf", "png"):
            temporary = output / f"{name}.{extension}.tmp"
            fig.savefig(temporary, format=extension, dpi=180)
            temporary.replace(output / f"{name}.{extension}")
        plt.close(fig)
    return rows
