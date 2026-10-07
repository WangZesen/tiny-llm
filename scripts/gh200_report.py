"""Export the completed, frozen GH200 qualification evidence for review."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from gh200_campaign import read, sha, write
from gh200_compare import collect, verify_candidate


def csv_file(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(root, candidate, output):
    manifest, info = verify_candidate(root, candidate)
    collect(root, candidate)
    summary = read(candidate / "qualification-summary.json")
    expected = len(manifest["cases"])
    # A failed gate is reportable; an unfinished campaign must not look final.
    required = [(kind, row["id"], repeat)
                for kind in ("performance", "numerical", "convergence")
                for row in manifest["convergence" if kind == "convergence" else "cases"]
                for repeat in (range(3) if kind == "performance" else [0])]
    missing = [item for item in required if not (candidate / "results" / item[0]
                                               / f"{item[1]}-r{item[2]}/result.json").exists()]
    if missing:
        raise ValueError(f"campaign unfinished: {len(missing)} results missing")
    output.mkdir(parents=True, exist_ok=True)
    timings, numerics, finals, curves = [], [], [], []
    serialized_controls = []
    sources = {}

    def record(path):
        sources[str(path)] = sha(path)
        return read(path)

    for row in manifest["cases"]:
        for repeat in range(3):
            result = record(candidate / "results/performance" / f"{row['id']}-r{repeat}/result.json")
            if result["status"] != "ok":
                continue
            pair = result["pairs"]
            baseline = record(root / "baseline/throughput" / f"{row['id']}-r{repeat}/result.json")
            timings.append(dict(
                case=row["id"], category=row["category"], repeat=repeat,
                allocation=result["allocation"], order=" then ".join(result["order"]),
                steps_per_window=result["count"],
                native_tokens_per_second=pair["native"]["tokens_per_second"],
                candidate_tokens_per_second=pair["candidate"]["tokens_per_second"],
                speedup=result["speedup"],
                original_baseline_tokens_per_second=baseline["tokens_per_second"],
                **{f"{backend}_{key}": pair[backend][key]
                   for backend in ("native", "candidate")
                   for key in ("startup_seconds", "elapsed_seconds", "training_seconds",
                               "peak_allocated_bytes", "peak_reserved_bytes")},
            ))
        result = record(candidate / "results/numerical" / f"{row['id']}-r0/result.json")
        control_directory = root / "baseline/numerical" / f"{row['id']}-r0"
        control = record(control_directory / "result.json")
        control_updates = record(control_directory / control["updates_file"])
        if control["kind"] == "native_native_serialized_control":
            serialized_controls.append(row["id"])
        if result["status"] == "ok":
            updates = result["updates"]
            numerics.append(dict(
                case=row["id"], status=result["gate"], updates=len(updates),
                workers=len(updates[0]["workers"]),
                maximum_absolute_loss_difference=max(abs(a - b) for update in updates
                    for a, b in zip(update["native_losses"], update["candidate_losses"], strict=True)),
                **{f"maximum_{field}_relative_l2": max(worker[field]["relative_l2"]
                    for update in updates for worker in update["workers"])
                   for field in ("parameters", "gradients", "updates", "moments")},
                **{f"native_control_{field}_relative_l2": max(worker[field]["relative_l2"]
                    for update in control_updates for worker in update["workers"])
                   for field in ("parameters", "gradients", "updates", "moments")},
            ))
    for row in manifest["convergence"]:
        a = record(root / "baseline/convergence" / f"{row['id']}-r0/result.json")
        b = record(candidate / "results/convergence" / f"{row['id']}-r0/result.json")
        if b["status"] != "ok":
            continue
        values = dict(recipe=row["recipe"], seed=row["seed"], tokens=a["training"]["tokens"],
                      native_loss=a["training"]["final_validation"]["loss"],
                      candidate_loss=b["training"]["final_validation"]["loss"])
        values["difference"] = values["candidate_loss"] - values["native_loss"]
        for backend, result, directory in (
            ("native", a, root / "baseline/convergence" / f"{row['id']}-r0/training"),
            ("candidate", b, candidate / "results/convergence" / f"{row['id']}-r0/candidate/training"),
        ):
            metrics = directory / "metrics.jsonl"
            sources[str(metrics)] = sha(metrics)
            validations = [value for line in metrics.read_text().splitlines()
                           if (value := json.loads(line))["event"] == "validation"]
            values[f"{backend}_clipping_count"] = sum(v["grad_clip_count"] for v in validations)
            values[f"{backend}_training_seconds"] = result["training"]["training_seconds"]
            values[f"{backend}_training_elapsed_seconds"] = result["training"]["training_elapsed_seconds"]
            for value in validations:
                curves.append(dict(recipe=row["recipe"], seed=row["seed"], backend=backend,
                                   epoch=value["epoch"], tokens=value["tokens"], loss=value["loss"],
                                   clipping_count=value["grad_clip_count"]))
        finals.append(values)
    for name, rows in (("performance", timings), ("numerical", numerics),
                       ("convergence", finals), ("validation-curves", curves)):
        if rows:
            csv_file(output / f"{name}.csv", rows)
    write(output / "qualification.json", summary)
    write(output / "provenance.json", dict(manifest_identity=manifest["identity"],
                                           candidate_identity=info["source_identity"],
                                           contract_sha256=sha(root / "acceptance/contract.json"),
                                           inputs=sources))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    categories = summary["performance"].get("categories", {})
    for index, mode in enumerate(("sync", "packed4", "packed8")):
        ratios = categories.get(mode, {}).get("allocation_medians", [])
        if ratios:
            axes[0].scatter([index] * len(ratios), ratios, color="#126b82", s=35)
            axes[0].hlines(statistics.median(ratios), index - .18, index + .18, color="#126b82", lw=2)
    axes[0].axhline(1.05, color="#a83a32", ls="--", label="Required: 1.05×")
    axes[0].set(xticks=range(3), xticklabels=["Sync", "Packed-4", "Packed-8"],
                ylabel="Median paired throughput ratio", title="Small-batch categories: three repeats")
    axes[0].legend(frameon=False)
    recipes = list(dict.fromkeys(row["recipe"] for row in manifest["convergence"]))
    labels = {"20m-sync-production": "Sync production", "20m-packed4-production": "Packed-4 production",
              "20m-packed8-production": "Packed-8 production", "20m-sync-small": "Sync B16/C512",
              "20m-packed4-small": "Packed-4 B32/C512"}
    for index, recipe in enumerate(recipes):
        gate = summary["convergence"][recipe]
        if "mean_difference" not in gate:
            continue
        mean, upper = gate["mean_difference"], gate["upper_95"]
        color = "#126b82" if gate["status"] == "pass" else "#a83a32"
        axes[1].plot([mean, upper], [index, index], color=color)
        axes[1].scatter([mean], [index], color=color, s=25)
        axes[1].scatter([upper], [index], color=color, marker="|", s=100)
    axes[1].axvline(.005, color="#a83a32", ls="--")
    axes[1].axvline(0., color=".5", lw=.8)
    axes[1].set(yticks=range(len(recipes)), yticklabels=[labels[key] for key in recipes],
                xlabel="Candidate − native final loss (nats)",
                title="Paired mean and one-sided 95% upper bound")
    axes[1].invert_yaxis()
    axes[1].set_xticks([-.005, 0., .005, .01])
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(output / f"qualification.{suffix}", dpi=180)
    plt.close(fig)

    lines = ["---", "title: GH200 backend qualification", "---", "",
             "The GH200 backend is integrated for synchronous and decentralized training. "
             + ("All measurement gates passed." if summary["measurement_gates_passed"] else
                "Native retirement is blocked by the frozen acceptance gates.")
             + " Following this qualification, GH200 became the default at the user's request; "
             "native remains explicitly selectable. This default change does not alter the "
             "measurements or acceptance thresholds below.",
             "", "![Throughput and convergence qualification](gh200/qualification.svg)", "",
             "[Performance pairs CSV](gh200/performance.csv) · [Numerical cases CSV](gh200/numerical.csv) · "
             "[Convergence CSV](gh200/convergence.csv) · [Validation curves CSV](gh200/validation-curves.csv) · "
             "[Gate results](gh200/qualification.json) · [Provenance](gh200/provenance.json)", "",
             "## Frozen protocol", "",
             f"The fresh native campaign completed before backend implementation: {expected * 3} throughput "
             f"measurements, {expected} native/native numerical controls, and 15 full-budget training runs. "
             "The user's modified packed-8 recipe was included in the executable snapshot. "
             "Both convergence campaigns use seeds 45–47, approximately 408M training targets per run, "
             "and the full 197,411,295-target validation split.", "",
             "Throughput pairs run in the same GH200 allocation with alternating order and equal update counts. "
             "Each side executes at least 20 warmup updates and five windows of at least one second. "
             "Measurements include the production loader, transfers, schedules, clipping, consensus and logging. "
             "Every workload has three independent allocations. Startup, timed training, elapsed time, allocated "
             "memory and reserved memory are separate CSV columns.", "",
             "Startup includes runner setup, compilation/capture and warmup. Timed training is the sum of "
             "the five scored windows; elapsed time additionally includes setup and result collection. "
             "Interpreter imports and scheduler queue time are outside these timers. These boundaries "
             "are identical for both sides of a pair.", "",
             "For every small-batch category, each repeat's median candidate/native ratio across workloads must be at least 1.05. "
             "Every required production case must remain at or above 0.98 in all three repeats. "
             "The matrix covers 20M/50M/90M, synchronous/packed-4/packed-8, local batches 4/8/16, "
             "contexts 512/1024, clipping 1/off, production recipes, accumulation remainders, ATC and adaptive consensus.", "",
             "## Performance", "", "| Category | Repeat 1 | Repeat 2 | Repeat 3 | Gate |",
             "| --- | ---: | ---: | ---: | --- |"]
    for mode in ("sync", "packed4", "packed8"):
        if mode not in categories:
            continue
        values = categories[mode]
        ratios = " | ".join(f"{value:.3f}×" for value in values["allocation_medians"])
        lines.append(f"| {mode} | {ratios} | {'pass' if values['passed'] else 'not qualified'} |")
    lines += ["", f"Overall performance gate: **{summary['performance']['status']}**. "
              f"The production regression gate {'passed' if summary['performance'].get('production_pass') else 'did not pass'}. "
              "Individual cases and allocation variability, including any regressions, are retained in the CSV.",
              "", "## Numerical precision", "",
              f"{summary['numerical_passed']}/{expected} workloads passed five independent matched local updates "
              "per worker. Trajectories are never realigned. The frozen gates are loss atol/rtol 0.002, "
              "gradient relative L2 below 5%, parameter relative L2 below 2%, and update relative L2 below 10%. "
              "Both Adam moments and per-parameter diagnostics are retained. Supplied-gradient optimizer and "
              "isolated-consensus tests use rtol 1e-6 and atol 1e-8.", ""]
    if numerics:
        lines += ["| Quantity | Candidate/native worst relative L2 | Native/native control worst relative L2 |",
                  "| --- | ---: | ---: |"]
        for field in ("parameters", "gradients", "updates", "moments"):
            lines.append(f"| {field} | {max(row[f'maximum_{field}_relative_l2'] for row in numerics):.6g} | "
                         f"{max(row[f'native_control_{field}_relative_l2'] for row in numerics):.6g} |")
    if serialized_controls:
        lines += ["", "The two largest 90M packed-8 native/native controls used independent CPU-offloaded "
                  "states with one shared compiled executor to fit GPU memory. The other controls used "
                  "independently constructed executors. The serialized cases are "
                  + ", ".join(f"`{name}`" for name in serialized_controls) + "."]
    lines += ["", "## Independent convergence", "",
              "For each recipe the paired differences are candidate minus native. The one-sided 95% upper "
              "bound is mean + 2.920 × sample SD / sqrt(3), and must be strictly below 0.005 nats. "
              "An inconclusive bound blocks retirement even when the mean improves. Thresholds were not changed.",
              "", "| Recipe | Native mean | Candidate mean | Mean difference | Upper 95% | Gate |",
              "| --- | ---: | ---: | ---: | ---: | --- |"]
    for recipe in recipes:
        gate = summary["convergence"][recipe]
        selected = [value for value in finals if value["recipe"] == recipe]
        if "mean_difference" in gate:
            lines.append(f"| {labels[recipe]} | {statistics.mean(v['native_loss'] for v in selected):.6f} | "
                         f"{statistics.mean(v['candidate_loss'] for v in selected):.6f} | "
                         f"{gate['mean_difference']:+.6f} | {gate['upper_95']:.6f} | {gate['status']} |")
    lines += ["", "## Reproduction and retained evidence", "",
              f"Manifest: `{manifest['identity']}`. Candidate: `{info['source_identity']}`.", "",
              "The executable native snapshot, lockfile, working-tree patch, resolved configurations, "
              "cache identity, GPU/software metadata, allocation receipts, full logs, checkpoints and "
              "per-parameter comparisons remain outside the maintained training implementation in "
              f"`{root}`. The measured candidate is `{candidate}`. "
              "Initial canonical states and exact numerical input batches are retained; temporary large "
              "comparison tensors can be regenerated from them. Failed development attempts are retained separately.",
              "", "Use `scripts/gh200_campaign.py` for native freezing/baselines, "
              "`scripts/gh200_compare.py` for candidate freezing/submission/collection, and "
              "`scripts/gh200_report.py` to regenerate this report. The frozen acceptance implementation "
              "in `acceptance/` supplies the gate calculations.", ""]
    (output.parent / "gh200-integration.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("doc/performance/gh200"))
    args = parser.parse_args()
    export(args.root.resolve(), args.candidate.resolve(), args.output)


if __name__ == "__main__":
    main()
