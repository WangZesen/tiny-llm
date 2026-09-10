"""Regenerate analysis tables and figures using saved scalar results only."""

import csv
import json
from pathlib import Path

from tiny_llm.analysis.core import result_key, validate_consensus


def plot_analysis(output: Path, training_norms: bool = False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    rows, seen = [], set()
    for checkpoint in manifest["checkpoints"]:
        for case in ("seen", "unseen"):
            path = output / f"{result_key(checkpoint)}-{case}.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            if result["identity"] != manifest["identity"]:
                raise ValueError(f"incompatible result {path}")
            validate_consensus(result, checkpoint)
            rows.append(
                dict(
                    checkpoint=checkpoint["name"],
                    tokens=checkpoint["tokens"],
                    epoch=checkpoint["epoch"],
                    step=checkpoint["step"],
                    case=case,
                    weights_hash=checkpoint["weights_hash"],
                    result_key=result_key(checkpoint),
                    **result["statistics"],
                )
            )
    if not rows:
        return []
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "summary.csv")
    curves = []
    for row in sorted(rows, key=lambda row: (row["tokens"], row["checkpoint"])):
        key = (row["result_key"], row["tokens"], row["case"])
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
    if any("consensus_alignment" in row for row in curves):
        specs[0][2].append(("normalized_consensus_alignment", "Consensus error"))
        specs[1] = (specs[1][0], "Unnormalized alignment", specs[1][2])
        specs[1][2].append(("consensus_alignment", "Consensus error"))
        specs[2][2].append(("average_consensus_norm", "Average consensus-error norm"))
    for name, ylabel, fields in specs:
        normalized = name == "normalized-alignment"
        fig, axes = plt.subplots(
            2 if normalized else 1,
            1,
            figsize=(9, 9 if normalized else 5),
            sharex=True,
            squeeze=False,
            constrained_layout=True,
        )
        ax = axes[0, 0]
        log_ax = axes[1, 0] if normalized else None
        nonpositive = False
        positive = False
        for color, (field, label) in enumerate(fields):
            for case, linestyle in (("seen", "-"), ("unseen", "--")):
                values = [row for row in curves if row["case"] == case]
                if values:
                    x = [row["tokens"] for row in values]
                    y = [float("nan") if row.get(field) is None else row[field] for row in values]
                    style = dict(
                        color=f"C{color}",
                        linestyle=linestyle,
                        marker="o",
                        markersize=3,
                        label=f"{label} ({case})",
                    )
                    ax.plot(x, y, **style)
                    if log_ax is not None:
                        nonpositive |= any(value <= 0 for value in y)
                        positive |= any(value > 0 for value in y)
                        log_ax.plot(
                            x, [value if value > 0 else float("nan") for value in y], **style
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
        if log_ax is not None:
            ax.set_xlabel("")
            ax.set_title(f"{Path(manifest['run']).name} — linear scale")
            log_ax.set_yscale("log")
            log_ax.set(xlabel="Training tokens", ylabel=ylabel, title="Logarithmic scale")
            if not positive:
                log_ax.set_ylim(1e-3, 1)
                log_ax.text(
                    0.02, 0.98, "No positive alignments", transform=log_ax.transAxes, va="top"
                )
            elif nonpositive:
                log_ax.text(
                    0.02,
                    0.98,
                    "Nonpositive values omitted",
                    transform=log_ax.transAxes,
                    va="top",
                    fontsize=8,
                )
            log_ax.grid(alpha=0.2, which="both")
        for extension in ("pdf", "png"):
            temporary = output / f"{name}.{extension}.tmp"
            fig.savefig(temporary, format=extension, dpi=180)
            temporary.replace(output / f"{name}.{extension}")
        plt.close(fig)
    return rows
