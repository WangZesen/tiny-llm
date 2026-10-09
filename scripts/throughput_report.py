#!/usr/bin/env python3
"""Combine GH200 throughput/optimizer evidence and project a 20-token/parameter budget.

No GPU work is launched. Rates are measured; budget times and optimizer acceleration
scenarios are steady-state projections, with no claims about convergence.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import shlex
import statistics
import sys
import textwrap
from pathlib import Path

MODELS = ("20m", "90m", "250m", "500m")
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
CONTEXT_COLORS = {512: "#CC79A7", 1024: "#0072B2", 2048: "#D55E00", 4096: "#009E73"}
MARKERS = ("o", "s", "^", "D")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Inputs:
    def __init__(self, repo: Path):
        self.repo = repo
        self.hashes: dict[str, str] = {}

    def bytes(self, path):
        path = Path(path).resolve()
        data = path.read_bytes()
        self.hashes[path.relative_to(self.repo).as_posix()] = hashlib.sha256(data).hexdigest()
        return data

    def json(self, path):
        return json.loads(self.bytes(path))

    def campaign(self, root):
        manifest = self.json(root / "manifest.json")
        require(digest({k: v for k, v in manifest.items() if k != "identity"})
                == manifest["identity"], "campaign manifest identity changed")
        package = root / "source/tiny_llm"
        source_hash = hashlib.sha256()
        for path in sorted(package.rglob("*.py")):
            source_hash.update(path.relative_to(package).as_posix().encode())
            source_hash.update(self.bytes(path))
        require(source_hash.hexdigest() == manifest["source_hash"], "frozen source changed")
        require(hashlib.sha256(self.bytes(root / "source/throughput_sweep.py")).hexdigest()
                == manifest["driver_sha256"], "frozen campaign driver changed")
        report = self.json(root / "results.json")
        require(report["complete"] and report["campaign_identity"] == manifest["identity"],
                "source campaign is incomplete or mismatched")
        if (root / "audit.json").exists():
            audit = self.json(root / "audit.json")
            require(audit["passed"], "source campaign audit failed")
            if "results_sha256" in audit:
                require(audit["results_sha256"] == hashlib.sha256(self.bytes(root / "results.json")).hexdigest(),
                        "source audit refers to different results")
            if "audit_script_sha256" in audit:
                require(audit["audit_script_sha256"]
                        == hashlib.sha256(self.bytes(root / "source/throughput_audit.py")).hexdigest(),
                        "frozen audit script changed")
        return manifest, report


def measured(raw, receipt, manifest):
    """Independently recompute rates and state/count invariants from raw windows."""
    require(digest(raw["config"]) == receipt["config_identity"], "configuration digest mismatch")
    require(raw["environment"]["source_hash"] == manifest["source_hash"], "source mismatch")
    for key in ("warmup", "windows", "window_seconds"):
        require(raw["protocol"][key] == manifest["protocol"][key], "measurement protocol mismatch")
    require(raw["status"] == "ok", "missing successful measurement")
    require(receipt["accounting"]["state"] == "COMPLETED"
            and receipt["accounting"]["exit_code"] == "0:0"
            and receipt["accounting"]["timelimit"] == "00:14:00", "allocation completion or limit mismatch")
    rows = raw["measurements"]
    require(len(rows) in (5, 10), "incomplete timing windows")
    require(all(r["steps"] >= 20 and math.isfinite(r["seconds"]) and r["seconds"] > 0
                for r in rows), "invalid window")
    require(raw["successful_updates"] == 25 + sum(r["steps"] for r in rows), "update count mismatch")
    require(all(math.isfinite(raw[k]) for k in ("final_loss", "final_grad_norm")), "nonfinite state")
    q = receipt["batch_size"] * receipt["context_length"]
    config = raw["config"]
    require(all(config["runtime"][key] == value for key, value in
                dict(seed=42, cpu_threads=8, training_backend="gh200", amp=True,
                     compile=True, attention_backend="sdpa").items()), "runtime protocol changed")
    require(all(config["optimizer"][key] == value for key, value in
                dict(lr=0.001, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.1,
                     grad_clip=1.).items()), "optimizer protocol changed")
    require(config["training"]["batch_tokens"] == q
            and config["training"]["micro_batch_size"] == receipt["batch_size"]
            and config["model"]["context_length"] == receipt["context_length"]
            and config["decentralized"] is None, "not a single, unaccumulated update")
    rates = [q * r["steps"] / r["seconds"] for r in rows]
    median = statistics.median(rates)
    cv = statistics.stdev(rates) / statistics.mean(rates)
    for expected, actual in ((median, raw["tokens_per_second"]),
                             (1000 * q / median, raw["milliseconds_per_update"]),
                             (cv, raw["coefficient_of_variation"])):
        require(math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-14),
                "raw throughput arithmetic mismatch")
    require(cv <= 0.05, "unstable input: report needs explicit unstable-point handling")
    return dict(rate=median, update_ms=1000 * q / median, cv=cv,
                min_rate=min(rates), max_rate=max(rates), windows=len(rows))


def optimizer_measurement(raw):
    require(raw["protocol"].get("optimizer_timing"), "optimizer event timing missing")
    means = []
    for row in raw["measurements"]:
        samples = row["optimizer_samples"]
        require(sum(s["steps"] for s in samples) == row["steps"], "sample coverage mismatch")
        require(all(s["steps"] > 0 and 0 < s["optimizer_ms"] <= s["graph_ms"]
                    and math.isfinite(s["graph_ms"]) for s in samples), "invalid event time")
        require(math.isclose(sum(s["seconds"] for s in samples), row["seconds"], rel_tol=1e-10),
                "burst duration mismatch")
        mean = sum(s["steps"] * s["optimizer_ms"] for s in samples) / row["steps"]
        require(math.isclose(mean, row["optimizer_ms"], rel_tol=1e-10), "event weighting mismatch")
        means.append(mean)
    median = statistics.median(means)
    cv = statistics.stdev(means) / statistics.mean(means)
    require(math.isclose(median, raw["optimizer_milliseconds_per_update"], rel_tol=1e-10),
            "optimizer median mismatch")
    require(cv <= 0.05, "unstable optimizer timing")
    return median, cv


def parameter_count(model):
    w = model["width"]
    return model["vocab_size"] * w + model["layers"] * (
        4 * w * w + 3 * w * model["ffn_width"] + 2 * w) + w


def project(parameters, tokens_per_parameter, batch, length, update_ms, optimizer_ms):
    """Float64 time arithmetic, exact integer token/update counts; no extrapolated batches."""
    require(all(isinstance(x, int) and x > 0
                for x in (parameters, tokens_per_parameter, batch, length)), "positive integers required")
    require(math.isfinite(update_ms) and 0 < optimizer_ms < update_ms, "invalid step decomposition")
    budget = parameters * tokens_per_parameter
    q = batch * length
    updates = (budget + q - 1) // q
    fractional_updates = budget / q
    total = fractional_updates * update_ms / 1000
    optimizer = fractional_updates * optimizer_ms / 1000
    residual = total - optimizer
    return dict(
        token_budget=budget, tokens_per_update=q, whole_updates=updates,
        rounded_processed_tokens=updates * q, rounded_excess_tokens=updates * q - budget,
        projected_seconds=total, projected_hours=total / 3600,
        rounded_projected_seconds=updates * update_ms / 1000,
        hours_per_billion_tokens=1e9 * update_ms / (q * 3.6e6),
        optimizer_projected_hours=optimizer / 3600,
        residual_projected_hours=residual / 3600,
        optimizer_fraction=optimizer_ms / update_ms,
        ideal_2x_optimizer_hours=(residual + optimizer / 2) / 3600,
        ideal_zero_cost_optimizer_hours=residual / 3600,
        ideal_2x_optimizer_speedup=total / (residual + optimizer / 2),
        ideal_zero_cost_optimizer_speedup=total / residual,
    )


def point_key(row):
    return row["model"], row["context_length"], row["batch_size"]


def unique_ok(points):
    result = {}
    for row in points:
        if row["status"] == "ok":
            key = point_key(row)
            require(key not in result, f"duplicate successful point: {key}")
            result[key] = row
    return result


def normalized_config(config):
    result = copy.deepcopy(config)
    result["runtime"].pop("output_dir")
    return result


def campaign_pair(inputs, throughput, optimizer, tokens_per_parameter):
    old_manifest, old_report = inputs.campaign(throughput)
    new_manifest, new_report = inputs.campaign(optimizer)
    require(new_manifest["original_campaign"]["identity"] == old_manifest["identity"],
            "optimizer campaign refers to a different sweep")
    require(old_manifest["contexts"] == new_manifest["contexts"], "paired contexts differ")
    for path in (throughput / "source/tiny_llm/gh200").glob("*.py"):
        require(inputs.bytes(path) == inputs.bytes(optimizer / "source/tiny_llm/gh200" / path.name),
                "production compute/optimizer implementation changed between campaigns")
    old_points, new_points = unique_ok(old_report["points"]), unique_ok(new_report["points"])
    require(old_points.keys() == new_points.keys(), "successful grids differ")
    rows, failures, window_counts = [], [], set()
    for key in sorted(old_points, key=lambda k: (MODELS.index(k[0]), k[1], k[2])):
        old_point, new_point = old_points[key], new_points[key]
        old_dir, new_dir = throughput / old_point["directory"], optimizer / new_point["directory"]
        old_receipt, receipt = inputs.json(old_dir / "receipt.json"), inputs.json(new_dir / "receipt.json")
        original = inputs.json(old_dir / "result.json")
        baseline, timed = inputs.json(new_dir / "baseline.json"), inputs.json(new_dir / "result.json")
        require(normalized_config(original["config"]) == normalized_config(baseline["config"])
                and baseline["config"] == timed["config"], "cross-campaign configurations differ")
        a, b, c = (measured(original, old_receipt, old_manifest),
                   measured(baseline, receipt, new_manifest), measured(timed, receipt, new_manifest))
        window_counts.update((a["windows"], b["windows"], c["windows"]))
        opt_ms, opt_cv = optimizer_measurement(timed)
        require(baseline["environment"]["hostname"] == timed["environment"]["hostname"]
                and baseline["environment"]["gpu_status"].splitlines()[1].split(",")[0]
                == timed["environment"]["gpu_status"].splitlines()[1].split(",")[0],
                "baseline and optimizer timing used different GPUs")
        params = parameter_count(baseline["config"]["model"])
        require(params == baseline["parameters"] == timed["parameters"] == original["parameters"],
                "parameter count mismatch")
        row = dict(model=key[0], context_length=key[1], batch_size=key[2], parameters=params,
                   **project(params, tokens_per_parameter, key[2], key[1], b["update_ms"], opt_ms))
        row.update(
            primary_tokens_per_second=b["rate"], original_tokens_per_second=a["rate"],
            baseline_update_ms=b["update_ms"], optimizer_update_ms=opt_ms,
            baseline_window_cv=b["cv"], optimizer_window_cv=opt_cv,
            instrumented_window_cv=c["cv"], original_window_cv=a["cv"],
            projected_hours_window_min=row["token_budget"] / b["max_rate"] / 3600,
            projected_hours_window_max=row["token_budget"] / b["min_rate"] / 3600,
            rate_change_from_original=b["rate"] / a["rate"] - 1,
            instrumented_wall_change=c["update_ms"] / b["update_ms"] - 1,
            original_projected_hours=row["token_budget"] / a["rate"] / 3600,
            original_peak_reserved_gib=max(original["setup_peak_reserved_bytes"],
                                           original["measurement_peak_reserved_bytes"]) / 2**30,
            original_peak_allocated_gib=max(original["setup_peak_allocated_bytes"],
                                            original["measurement_peak_allocated_bytes"]) / 2**30,
            baseline_peak_reserved_gib=max(baseline["setup_peak_reserved_bytes"],
                                           baseline["measurement_peak_reserved_bytes"]) / 2**30,
            device_capacity_gib=baseline["total_memory_bytes"] / 2**30,
            observed_baseline_setup_seconds=baseline["setup_seconds"],
            throughput_directory=old_dir.relative_to(inputs.repo).as_posix(),
            optimizer_directory=new_dir.relative_to(inputs.repo).as_posix(),
            throughput_job_id=old_receipt["job_id"], optimizer_job_id=receipt["job_id"],
        )
        # Orthogonal arithmetic identity between rate- and update-based projections.
        require(math.isclose(row["projected_seconds"], row["token_budget"] / b["rate"], rel_tol=1e-12),
                "budget/rate and update-count projections disagree")
        rows.append(row)
    for old in old_report["points"]:
        if old["status"] == "cuda_oom":
            path = throughput / old["directory"]
            raw, receipt = inputs.json(path / "result.json"), inputs.json(path / "receipt.json")
            require(raw["status"] == "cuda_oom" and receipt["status"] == "cuda_oom", "unconfirmed OOM")
            failures.append(dict(model=old["model"], context_length=old["context_length"],
                                 batch_size=old["batch_size"], stage=raw["stage"],
                                 directory=path.relative_to(inputs.repo).as_posix()))
    return rows, failures, window_counts, old_manifest, new_manifest


def combine(inputs, throughputs, optimizers, tokens_per_parameter):
    if isinstance(throughputs, Path):
        throughputs = [throughputs]
    if isinstance(optimizers, Path):
        optimizers = [optimizers]
    require(len(throughputs) == len(optimizers) > 0, "supply matching throughput/optimizer pairs")
    rows, failures, window_counts, contexts, campaigns = [], [], set(), set(), []
    for throughput, optimizer in zip(throughputs, optimizers, strict=True):
        points, oom, counts, old, new = campaign_pair(inputs, throughput, optimizer, tokens_per_parameter)
        require(not contexts.intersection(old["contexts"]), "overlapping campaign contexts")
        contexts.update(old["contexts"])
        rows.extend(points)
        failures.extend(oom)
        window_counts.update(counts)
        campaigns.append(dict(throughput=old["identity"], optimizer=new["identity"],
                              throughput_path=throughput.relative_to(inputs.repo).as_posix(),
                              optimizer_path=optimizer.relative_to(inputs.repo).as_posix(),
                              contexts=old["contexts"]))
        for path in (throughputs[0] / "source/tiny_llm/gh200").glob("*.py"):
            require(inputs.bytes(path) == inputs.bytes(throughput / "source/tiny_llm/gh200" / path.name),
                    "production implementation changed across context campaigns")
    lengths = tuple(sorted(contexts))
    require(1024 in lengths and len(lengths) > 1 and set(lengths) <= set(CONTEXT_COLORS),
            "report requires context 1024 as reference and at least one other supported context")
    require({(r["model"], r["context_length"]) for r in rows}
            == {(m, length) for m in MODELS for length in lengths}, "unexpected campaign coverage")
    require(len({point_key(r) for r in rows}) == len(rows), "duplicate combined measurements")
    rows.sort(key=lambda r: (MODELS.index(r["model"]), r["context_length"], r["batch_size"]))
    curves = []
    for model in MODELS:
        group = [r for r in rows if r["model"] == model]
        require(len({r["parameters"] for r in group}) == 1, "model budget changes across setups")
        best_model = min(group, key=lambda r: r["projected_hours"])
        for row in group:
            row["relative_time_model_best"] = row["projected_hours"] / best_model["projected_hours"]
        for length in lengths:
            points = [r for r in group if r["context_length"] == length]
            require(points, f"missing curve {model}/{length}")
            best = min(points, key=lambda r: r["projected_hours"])
            original_best = max(points, key=lambda r: r["original_tokens_per_second"])
            largest = max(points, key=lambda r: r["batch_size"])
            first = min(points, key=lambda r: r["batch_size"])
            oom = [r for r in failures if r["model"] == model and r["context_length"] == length]
            require(len(oom) == 1 and oom[0]["batch_size"] == 2 * largest["batch_size"],
                    "memory boundary is not established")
            require(sorted(p["batch_size"] for p in points)
                    == [1 << i for i in range(largest["batch_size"].bit_length())], "batch grid has gaps")
            for row in points:
                row["relative_time_curve_best"] = row["projected_hours"] / best["projected_hours"]
                row["speedup_over_batch1"] = first["projected_hours"] / row["projected_hours"]
            near_best = min((p for p in points if p["primary_tokens_per_second"]
                             >= 0.95 * best["primary_tokens_per_second"]), key=lambda p: p["batch_size"])
            curves.append(dict(model=model, context_length=length, token_budget=best["token_budget"],
                               best_batch=best["batch_size"], original_best_batch=original_best["batch_size"],
                               largest_batch=largest["batch_size"], first_oom_batch=oom[0]["batch_size"],
                               smallest_batch_within_5pct=near_best["batch_size"],
                               best_hours=best["projected_hours"],
                               original_best_hours=original_best["original_projected_hours"],
                               best_hours_per_billion_tokens=best["hours_per_billion_tokens"],
                               relative_time_model_best=best["relative_time_model_best"],
                               best_optimizer_fraction=best["optimizer_fraction"],
                               best_optimizer_hours=best["optimizer_projected_hours"],
                               optimizer_2x_hours=best["ideal_2x_optimizer_hours"],
                               optimizer_free_hours=best["ideal_zero_cost_optimizer_hours"],
                               batch1_hours=first["projected_hours"],
                               batching_speedup=first["projected_hours"] / best["projected_hours"],
                               largest_tokens_per_update=largest["tokens_per_update"]))
    matched = matched_contexts(rows)
    diagnostics = dict(
        successful_points=len(rows), oom_points=len(failures), curves=len(curves),
        window_counts=sorted(window_counts),
        maximum_original_window_cv=max(r["original_window_cv"] for r in rows),
        maximum_baseline_window_cv=max(r["baseline_window_cv"] for r in rows),
        maximum_optimizer_window_cv=max(r["optimizer_window_cv"] for r in rows),
        median_cross_campaign_rate_change=statistics.median(r["rate_change_from_original"] for r in rows),
        maximum_absolute_cross_campaign_rate_change=max(abs(r["rate_change_from_original"]) for r in rows),
        maximum_absolute_instrumented_wall_change=max(abs(r["instrumented_wall_change"]) for r in rows),
    )
    require(len(failures) == len(MODELS) * len(lengths), "unexpected OOM coverage")
    return dict(tokens_per_parameter=tokens_per_parameter, contexts=lengths, points=rows, curves=curves,
                oom=failures, matched_contexts=matched, diagnostics=diagnostics,
                source_campaigns=campaigns)


def matched_contexts(rows):
    lookup = {(r["model"], r["context_length"], r["tokens_per_update"]): r for r in rows}
    result = []
    for row in rows:
        if row["context_length"] == 1024:
            continue
        reference = lookup.get((row["model"], 1024, row["tokens_per_update"]))
        if reference:
            result.append(dict(model=row["model"], context_length=row["context_length"],
                               tokens_per_update=row["tokens_per_update"], batch_size=row["batch_size"],
                               reference_batch_size=reference["batch_size"],
                               relative_time_vs_1024=row["projected_hours"] / reference["projected_hours"],
                               projected_hours=row["projected_hours"],
                               reference_hours=reference["projected_hours"]))
    return result


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


CAPTIONS = {
    "performance": "Figure 1. Complete-update throughput (top) and peak reserved GPU memory "
    "across preparation and measurement (bottom). Solid curves use the uninstrumented paired "
    "baselines; faint dotted curves retain the original throughput sweep. Memory and OOM outcomes "
    "come from the original sweep. Crosses identify first OOM batches, drawn at the 95.00 GiB "
    "capacity line as outcome symbols, not measured memory use. Batch size counts sequences; no accumulation.",
    "optimizer": "Figure 2. Optimizer time and its fraction of a complete iteration, measured inside "
    "the captured full-update graph. The boundary includes gradient gathering, norm/clipping, finite "
    "checks, AdamW, and metric commit. Percentages use the same-allocation uninstrumented baseline. "
    "Optimizer cost is nearly independent of batch size; its share declines as forward/backward work grows.",
    "budget": "Figure 3. Ideal steady-state time to process 20 tokens per parameter. Each model's "
    "token budget is fixed across setups, but differs between models. The horizontal axis aligns "
    "tokens per update Q = B L. Solid curves extrapolate measured complete-update rates; dashed curves "
    "assume zero optimizer cost and unchanged remaining work. Shading is the observed baseline-window "
    "range, not a confidence interval. Startup, data delivery, evaluation and checkpoints are excluded.",
    "relative": "Figure 4. Projected time relative to the fastest observed paired baseline for the "
    "same model (1.00x), at the same 20-token/parameter budget. All measured cells are shown. OOM marks "
    "the first confirmed failing batch; dashes are larger unmeasured batches beyond that boundary. "
    "Close differences are descriptive, not statistically established rankings; windows are not independent runs.",
    "context": "Figure 5. Context-length cost with tokens per update held equal: B is adjusted so "
    "Q = B L matches the L = 1024 reference. The model and total training-token budget are also fixed. "
    "A ratio above one means slower training. Only directly measured pairs are compared; no interpolation "
    "or projection beyond the measured batch grid is used. This removes the tokens-per-update confound "
    "present when comparing contexts at the same sequence batch size.",
}


def figures(output, data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, ScalarFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 8.5, "axes.titlesize": 9.5,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "lines.linewidth": 1.25, "lines.markersize": 3.3,
        "mathtext.fontset": "dejavuserif", "svg.fonttype": "none", "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.hashsalt": "gh200-training-cost-v1",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    points = data["points"]
    contexts = tuple(data["contexts"])
    n_contexts = len(contexts)
    panel_width = 2.4 * n_contexts + 0.2
    model_handles = [Line2D([], [], color=color, marker=marker, label=model.upper())
                     for model, color, marker in zip(MODELS, COLORS, MARKERS, strict=True)]

    def style(ax, *, tokens=False):
        ax.set_xscale("log", base=2)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.45, zorder=0)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1024:g}" if tokens else f"{x:g}"))
        ax.tick_params(width=0.6, length=3)

    def curve(model, length):
        return sorted((r for r in points if r["model"] == model and r["context_length"] == length),
                      key=lambda r: r["batch_size"])

    def decorate(fig, title, name, *, handles=None):
        fig.suptitle(title, y=0.985, fontsize=11)
        if handles:
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.951),
                       ncol=len(handles), frameon=False, handlelength=2.2, columnspacing=1.4)
        fig.text(0.075, 0.027, textwrap.fill(CAPTIONS[name], 102), fontsize=7.6, va="bottom",
                 linespacing=1.4)

    def check_page_text(fig):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for item in fig.texts:
            box = item.get_window_extent(renderer)
            require(box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width + 1
                    and box.y1 <= fig.bbox.height + 1,
                    f"page text extends outside export bounds: {item.get_text()[:60]!r}; {box.bounds}")

    with PdfPages(output / "report.pdf", metadata={"Title": "Single-GH200 training cost",
                  "Author": "tiny-llm benchmark", "CreationDate": None, "ModDate": None}) as pdf:
        # A readable summary page makes the PDF usable without the Markdown report.
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.09, 0.94, "Single-GH200 training cost", fontsize=20)
        fig.text(0.09, 0.902, "Measured throughput, memory, optimizer cost and budget projections",
                 fontsize=11)
        fig.text(0.09, 0.86, textwrap.fill(
            f"{len(points)} fitting configurations and {len(data['oom'])} confirmed CUDA OOM probes; "
            f"four model sizes, contexts {'/'.join(map(str, contexts))}, "
            "and power-of-two sequence batches. Single NVIDIA GH200, "
            "95.00 GiB CUDA capacity, BF16 computation, FP32 parameters/moments, compiled SDPA, "
            "complete-update CUDA graphs, GPU-resident synthetic data and no gradient accumulation.", 84),
            fontsize=10, va="top", linespacing=1.5)
        fig.text(0.09, 0.75, r"$D=20P,\quad Q=BL,\quad T=D/R=(D/Q)t_{\mathrm{update}}$", fontsize=15)
        ax = fig.add_axes((0.07, 0.345, 0.87, 0.37))
        ax.axis("off")
        table_rows = [[r["model"].upper(), str(r["context_length"]), str(r["best_batch"]),
                       f"{r['token_budget'] / 1e9:.3f}", f"{r['best_hours']:.3f}",
                       f"{r['relative_time_model_best']:.2f}×", f"{100*r['best_optimizer_fraction']:.2f}"]
                      for r in data["curves"]]
        table = ax.table(cellText=table_rows,
                         colLabels=["Model", "Context", "Best B", "Budget (B)", "Time (h)",
                                    "Relative", "Opt. (%)"], loc="center", cellLoc="right")
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.45)
        for (i, _), cell in table.get_celld().items():
            cell.set_edgecolor("#dddddd")
            cell.set_linewidth(0.4)
            if i == 0:
                cell.set_facecolor("#edf1f4")
                cell.set_text_props(weight="bold")
        paragraphs = [
            "Table 1. Fastest observed paired baseline per model/context, projected to 20 tokens "
            "per parameter. Relative times use the fastest setup for that model. The token budget "
            "therefore remains fixed within each model, not across models. Best B is an observed "
            "ranking; no statistically significant winner is claimed for near ties.",
            "Primary rates use the optimizer campaign's uninstrumented baselines. Original sweep "
            "rates remain separate replication evidence; memory limits use the original OOM sweep. "
            "Each rate is a median of five synchronized windows after warmup. Raw window ranges "
            "describe within-run variation and do not estimate between-run uncertainty.",
            "All times are ideal steady-state estimates. They exclude startup, compilation, CPU "
            "data delivery, evaluation, checkpoints and queue time. They do not imply equal "
            "convergence or quality across model, context or batch choices. Optimizer-free curves "
            "are counterfactual lower bounds with unchanged remaining work, not training recipes.",
        ]
        for y, paragraph in zip((0.315, 0.225, 0.125), paragraphs, strict=True):
            fig.text(0.09, y, textwrap.fill(paragraph, 96), fontsize=9, va="top", linespacing=1.5)
        check_page_text(fig)
        pdf.savefig(fig)
        plt.close(fig)

        def save(fig, name):
            check_page_text(fig)
            for extension in ("pdf", "svg", "png"):
                metadata = ({"CreationDate": None, "ModDate": None} if extension == "pdf"
                            else {"Date": None} if extension == "svg" else None)
                fig.savefig(output / f"{name}.{extension}", dpi=450, metadata=metadata)
            pdf.savefig(fig)
            plt.close(fig)

        fig, axes = plt.subplots(2, n_contexts, figsize=(panel_width, 6.5), sharex="col", sharey="row")
        fig.subplots_adjust(left=0.095, right=0.98, bottom=0.23, top=0.865, hspace=0.25, wspace=0.14)
        for col, length in enumerate(contexts):
            for model, color, marker in zip(MODELS, COLORS, MARKERS, strict=True):
                rows = curve(model, length)
                x = [r["batch_size"] for r in rows]
                axes[0, col].plot(x, [r["primary_tokens_per_second"] / 1e6 for r in rows],
                                  color=color, marker=marker)
                axes[0, col].plot(x, [r["original_tokens_per_second"] / 1e6 for r in rows],
                                  color=color, linestyle=":", alpha=0.45, linewidth=0.8)
                axes[1, col].plot(x, [r["original_peak_reserved_gib"] for r in rows],
                                  color=color, marker=marker)
                oom = next(r for r in data["oom"] if r["model"] == model and r["context_length"] == length)
                axes[1, col].plot(oom["batch_size"], rows[0]["device_capacity_gib"],
                                  color=color, marker="x", markersize=5, linestyle="none")
            axes[0, col].set_yscale("log")
            axes[0, col].yaxis.set_major_formatter(ScalarFormatter())
            axes[1, col].axhline(points[0]["device_capacity_gib"], color="#777777", linestyle="--", lw=0.75)
            axes[1, col].set(ylim=(0, 104), xlabel="Batch size (sequences)")
            for row in (0, 1):
                axes[row, col].set_title(f"({chr(97 + n_contexts*row + col)}) Context {length}", loc="left")
                style(axes[row, col])
        axes[0, 0].set_ylabel("Throughput (million tokens/s)")
        axes[1, 0].set_ylabel("Peak reserved memory (GiB)")
        decorate(fig, "Measured complete-update performance", "performance", handles=model_handles)
        save(fig, "performance")

        fig, axes = plt.subplots(2, n_contexts, figsize=(panel_width, 6.2), sharex="col", sharey="row")
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.86, hspace=0.24, wspace=0.14)
        for col, length in enumerate(contexts):
            for model, color, marker in zip(MODELS, COLORS, MARKERS, strict=True):
                rows = curve(model, length)
                x = [r["batch_size"] for r in rows]
                axes[0, col].plot(x, [r["optimizer_update_ms"] for r in rows], color=color, marker=marker)
                axes[1, col].plot(x, [100*r["optimizer_fraction"] for r in rows], color=color, marker=marker)
            for row in (0, 1):
                axes[row, col].set_title(f"({chr(97 + n_contexts*row + col)}) Context {length}", loc="left")
                axes[row, col].set_ylim(bottom=0)
                style(axes[row, col])
            axes[1, col].set_xlabel("Batch size (sequences)")
        axes[0, 0].set_ylabel("Optimizer time (ms/update)")
        axes[1, 0].set_ylabel("Optimizer share of iteration (%)")
        decorate(fig, "Optimizer cost within the full update", "optimizer", handles=model_handles)
        save(fig, "optimizer")

        fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.7))
        fig.subplots_adjust(left=0.1, right=0.98, bottom=0.22, top=0.815, hspace=0.4, wspace=0.27)
        context_handles = [Line2D([], [], color=CONTEXT_COLORS[length], marker=m, label=f"Context {length}")
                           for length, m in zip(contexts, MARKERS, strict=False)]
        for i, (model, ax) in enumerate(zip(MODELS, axes.flat, strict=True)):
            for length, marker in zip(contexts, MARKERS, strict=False):
                color = CONTEXT_COLORS[length]
                rows = curve(model, length)
                x = [r["tokens_per_update"] for r in rows]
                ax.plot(x, [r["projected_hours"] for r in rows], color=color, marker=marker)
                ax.fill_between(x, [r["projected_hours_window_min"] for r in rows],
                                [r["projected_hours_window_max"] for r in rows], color=color, alpha=0.15)
                ax.plot(x, [r["ideal_zero_cost_optimizer_hours"] for r in rows], color=color,
                        linestyle="--", alpha=0.85, linewidth=1)
            budget = curve(model, 1024)[0]["token_budget"]
            ax.set_title(f"({chr(97+i)}) {model.upper()} · {budget/1e9:.3f}B tokens", loc="left")
            ax.set(xlabel="Tokens per update (×1024)", ylabel="Projected training time (h)", ylim=(0, None))
            style(ax, tokens=True)
        decorate(fig, "Fixed-budget training time: 20 tokens per parameter", "budget", handles=context_handles)
        fig.legend(handles=[Line2D([], [], color="#333333", label="Measured-rate projection"),
                            Line2D([], [], color="#333333", linestyle="--", label="Ideal zero-cost optimizer")],
                   loc="upper center", bbox_to_anchor=(0.5, 0.9), ncol=2, frameon=False)
        save(fig, "budget")

        fig, axes = plt.subplots(4, 1, figsize=(7.4, 5.2 + n_contexts))
        fig.subplots_adjust(left=0.1, right=0.865, bottom=0.19, top=0.92, hspace=0.5)
        batches = [1 << i for i in range(max(r["batch_size"] for r in data["oom"]).bit_length())]
        limit = max(r["relative_time_model_best"] for r in points)
        norm = LogNorm(vmin=1, vmax=limit)
        cmap = plt.get_cmap("cividis_r").copy()
        cmap.set_bad("#eeeeee")
        for i, (model, ax) in enumerate(zip(MODELS, axes, strict=True)):
            values = np.full((n_contexts, len(batches)), np.nan)
            for r in (r for r in points if r["model"] == model):
                values[contexts.index(r["context_length"]), batches.index(r["batch_size"])] = r["relative_time_model_best"]
            ax.imshow(np.ma.masked_invalid(values), norm=norm, cmap=cmap, aspect="auto")
            for y, length in enumerate(contexts):
                oom = next(r["batch_size"] for r in data["oom"] if r["model"] == model and r["context_length"] == length)
                for x, batch in enumerate(batches):
                    value = values[y, x]
                    label = f"{value:.2f}×" if math.isfinite(value) else "OOM" if batch == oom else "—"
                    color = "white" if math.isfinite(value) and norm(value) > 0.55 else "#202020"
                    ax.text(x, y, label, ha="center", va="center", fontsize=7.4, color=color)
            ax.set(xticks=range(len(batches)), xticklabels=[str(b) for b in batches], yticks=range(n_contexts),
                   yticklabels=[str(length) for length in contexts], ylabel="Context")
            ax.set_title(f"({chr(97+i)}) {model.upper()} — reference: fastest setup for this model", loc="left")
            ax.tick_params(length=0)
        axes[-1].set_xlabel("Batch size (sequences)")
        cax = fig.add_axes((0.89, 0.3, 0.017, 0.5))
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        ticks = [x for x in (1, 1.25, 1.5, 2, 3, 4, 6, 8, 12, 16) if x <= limit]
        bar.set_ticks(ticks)
        bar.set_ticklabels([f"{x:.2f}" for x in ticks])
        bar.set_label("Relative training time")
        decorate(fig, "Relative time at each model's fixed token budget", "relative")
        save(fig, "relative")

        comparisons = [length for length in contexts if length != 1024]
        fig, axes = plt.subplots(1, len(comparisons), figsize=(7.4, 4.3), sharey=True, squeeze=False)
        fig.subplots_adjust(left=0.1, right=0.98, bottom=0.32, top=0.80, wspace=0.16)
        for i, (length, ax) in enumerate(zip(comparisons, axes[0], strict=True)):
            for model, color, marker in zip(MODELS, COLORS, MARKERS, strict=True):
                rows = sorted((r for r in data["matched_contexts"] if r["model"] == model
                               and r["context_length"] == length), key=lambda r: r["tokens_per_update"])
                ax.plot([r["tokens_per_update"] for r in rows], [r["relative_time_vs_1024"] for r in rows],
                        color=color, marker=marker)
            ax.axhline(1, color="#777777", linestyle="--", lw=0.75)
            ax.set_title(f"({chr(97+i)}) L={length} / L=1024", loc="left")
            ax.set_xlabel("Tokens per update (×1024)")
            style(ax, tokens=True)
        axes[0, 0].set_ylabel("Relative training time\n(equal tokens per update)")
        decorate(fig, "Controlled comparison of context lengths", "context", handles=model_handles)
        save(fig, "context")


def report_text(data, output, throughputs, optimizers, command):
    rows, curves, stats = data["points"], data["curves"], data["diagnostics"]
    originals = os.path.relpath(throughputs[0], output)
    paired = os.path.relpath(optimizers[0], output)
    contexts = data["contexts"]
    shortest = min(contexts)
    winners = [min((c for c in curves if c["model"] == model), key=lambda c: c["best_hours"])
               for model in MODELS]
    durations = [f"{60*r['best_hours']:.2f} minutes ({r['model'].upper()})"
                 if r["best_hours"] < 0.1 else f"{r['best_hours']:.2f} hours ({r['model'].upper()})"
                 for r in winners]
    batch1_fractions = [next(r["optimizer_fraction"] for r in rows
                            if point_key(r) == (model, shortest, 1)) for model in MODELS]
    largest_points = [next(r for r in rows if point_key(r) ==
                          (c["model"], c["context_length"], c["largest_batch"])) for c in curves]
    fraction_min = min(r["optimizer_fraction"] for r in largest_points)
    fraction_max = max(r["optimizer_fraction"] for r in largest_points)
    changed = [c for c in curves if c["best_batch"] != c["original_best_batch"]]
    sensitivity = []
    for c in changed:
        other = next(r for r in rows if point_key(r) ==
                     (c["model"], c["context_length"], c["original_best_batch"]))
        delta = 100 * (other["projected_hours"] / c["best_hours"] - 1)
        sensitivity.append(f"For {c['model'].upper()}/context {c['context_length']}, the paired "
                           f"repeat selects batch {c['best_batch']} and the original sweep selects "
                           f"batch {c['original_best_batch']}; their projected times in the paired "
                           f"repeat differ by {delta:.2f}%.")
    text = [
        "# Single-GH200 training: throughput, memory, optimizer cost and fixed-budget time",
        "",
        f"This combined report covers **{len(rows)} fitting configurations and {len(data['oom'])} "
        f"confirmed CUDA OOM probes**, at contexts **{', '.join(map(str, contexts))}**. "
        "The budget is **20 tokens per parameter**, fixed within each model across all batch/context "
        "setups. All source timing windows passed their stability threshold. Projections use the "
        "measured configurations without extrapolating to unmeasured batch sizes.",
        "",
        "[Publication PDF](report.pdf) · [All merged measurements and projections (CSV)](measurements.csv) "
        "· [JSON](results.json) · [Curve summaries](curves.csv) · [Matched-context comparisons](matched-contexts.csv)",
        "",
        "## Main results",
        "",
        "The fastest observed paired baselines across all contexts project to **" + ", ".join(durations)
        + "**. These are steady-state "
        "compute estimates, excluding data delivery and other training overheads. Cross-model totals "
        "include different token budgets; hours per billion tokens separates that effect.",
        "",
        "| Model | Exact parameters | Training tokens (20P) | Best context | Best batch | Time | h / billion tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, r in zip(MODELS, winners, strict=True):
        p = next(p for p in rows if p["model"] == model)
        time_label = f"{60*r['best_hours']:.2f} min" if r["best_hours"] < 0.1 else f"{r['best_hours']:.2f} h"
        text.append(f"| {model.upper()} | {p['parameters']:,} | {r['token_budget']:,} | "
                    f"{r['context_length']} | {r['best_batch']} | {time_label} | {r['best_hours_per_billion_tokens']:.3f} |")
    text += [
        "",
        "The optimizer accounts for **" + ", ".join(f"{100*f:.1f}%" for f in batch1_fractions)
        + "** of batch-1 iterations "
        f"at context {shortest} for 20M/90M/250M/500M. Its per-update cost changes little with batch size, "
        "so larger batches amortize it. At the largest fitting batch on each curve, it is only "
        f"**{100*fraction_min:.2f}%–{100*fraction_max:.2f}%** of iteration time. A zero-cost optimizer "
        f"would therefore improve those largest-batch configurations by only about "
        f"**{1/(1-fraction_min):.3f}×–{1/(1-fraction_max):.3f}×**, if all other costs stayed fixed.",
        "",
        "**Ranking sensitivity:** " + " ".join(sensitivity)
        + (" All other curve winners agree across campaigns." if changed else
           "All curve winners agree across campaigns.")
        + " Differences below 5% are treated as a throughput plateau, not a statistically established "
        "winner. The table below also shows the smallest measured batch within 5% of each curve's peak.",
        "",
        "## Measurement sources and protocol",
        "",
        f"The [original throughput sweep]({originals}/README.md) supplies the independent throughput "
        f"repeat, GPU memory and first-OOM boundaries. The [optimizer campaign]({paired}/README.md) "
        "supplies the primary uninstrumented throughput and its paired optimizer timing. "
        "The reports are joined exactly on model, context and sequence batch. Their rates are kept "
        "separate rather than averaged. Production compute/optimizer source and resolved settings "
        "were checked for agreement between campaigns.",
        "",
        "Campaign pairs: " + "; ".join(
            f"[throughput]({os.path.relpath(t, output)}/README.md) / "
            f"[optimizer]({os.path.relpath(o, output)}/README.md) "
            f"(contexts {', '.join(map(str, source['contexts']))})"
            for t, o, source in zip(throughputs, optimizers, data["source_campaigns"], strict=True))
        + (". The context-512 additions are new GPU measurements. Previous raw campaigns are retained."
           if 512 in contexts else "."),
        "",
        "Single NVIDIA GH200 (`nvidia_gh200_120gb:1`), **95.00 GiB CUDA-reported capacity**; "
        "8 CPU threads; BF16 autocast, FP32 parameters/moments; compiled SDPA and complete-update "
        "CUDA graphs; synthetic inputs/targets resident on GPU. Batch B means sequences in one "
        "optimizer update, with **no accumulation**. Seed 42, LR 0.001, AdamW betas (0.9, 0.99), "
        "epsilon 1e-8, weight decay 0.1 and clipping 1.0. All allocations requested 14 minutes.",
        "",
        "Each worker excludes setup, compilation, capture and 20 warmup updates, calibrates with "
        "five updates, then measures five synchronized windows targeting 10 seconds (minimum 20 "
        f"updates). Unstable windows trigger five more; observed window counts per worker are "
        f"{stats['window_counts']}. "
        "Optimizer timing uses external CUDA events inside the full graph, sampling the last replay "
        "of each approximately 0.1-second burst and weighting by burst update count. Its boundary "
        "includes gradient gathering, norm/clipping, finite checks, coefficients, AdamW moments/weights "
        "and metric commit. The denominator is the same-allocation uninstrumented baseline.",
        "",
        f"Maximum baseline window CV: **{100*stats['maximum_baseline_window_cv']:.3f}%**; "
        f"maximum optimizer CV: **{100*stats['maximum_optimizer_window_cv']:.3f}%**. "
        f"Median paired-baseline throughput change from the original campaign: "
        f"**{100*stats['median_cross_campaign_rate_change']:+.2f}%**; maximum absolute change: "
        f"**{100*stats['maximum_absolute_cross_campaign_rate_change']:.2f}%**. Maximum absolute "
        f"instrumented wall-time change: **{100*stats['maximum_absolute_instrumented_wall_change']:.3f}%**.",
        "",
        "Windows are repeated measurements within a single allocation, not independent training "
        "runs. Shaded ranges show observed window extrema, not confidence intervals. Cross-campaign "
        "differences include GPU/node and time drift. Close rankings are descriptive.",
        "",
        "![Measured throughput and memory](performance.png)", "", CAPTIONS["performance"],
        "", "[Vector PDF](performance.pdf) · [Editable SVG](performance.svg)",
        "", "![Optimizer cost](optimizer.png)", "", CAPTIONS["optimizer"],
        "", "[Vector PDF](optimizer.pdf) · [Editable SVG](optimizer.svg)",
        "",
        "## Fixed-budget projection",
        "",
        "Let P be the exact parameter count, D = 20P the training-token budget, L the context "
        "length, B the sequence batch, Q = BL tokens per update, t the baseline update time, "
        "and o the measured optimizer time. With consistent time units:",
        "",
        "```text\nR = Q / t\nT = D / R = (D / Q) × t\n"
        "T_optimizer = (D / Q) × o\nT_residual = T − T_optimizer\n"
        "T(s) = (D / Q) × [(t − o) + o / s]\n"
        "speedup(s) = 1 / [(1 − f) + f / s],  f = o / t\n```",
        "",
        "s = 2 models a twice-as-fast optimizer; s → ∞ gives the ideal zero-cost optimizer "
        "limit. The residual includes all other timed work, including device copies, forward, "
        "loss, backward and dispatch; it is not an isolated forward/backward kernel measurement. "
        "These acceleration scenarios assume unchanged residual costs and update semantics. "
        "They do not describe training with the optimizer removed.",
        "",
        "The main tables use the continuous token-normalized estimate D/R. CSV/JSON also give "
        "the exact integer count `ceil(D/Q)`, rounded processed tokens, and the time for that many "
        "complete updates. This accounts for a final full batch overshooting the target; a partial "
        "final batch was not benchmarked. Rates are extrapolated only in duration, never to unmeasured "
        "batch sizes, contexts or architectures.",
        "",
        "Startup, compilation, validation, checkpoints, data loading, CPU tokenization, queue time "
        "and interruptions are absent. Observed baseline setup time is retained separately in the "
        "CSV, but is not added to the estimate: it does not cover all end-to-end overheads. This "
        "distinction matters particularly for the short 20M projection. Synthetic timing does not "
        "establish real-data throughput, convergence, equal quality, or optimal training hyperparameters.",
        "",
        "![Training budget](budget.png)", "", CAPTIONS["budget"],
        "", "[Vector PDF](budget.pdf) · [Editable SVG](budget.svg)",
        "",
        "| Model | Context | Best B | Within 5% B | Largest B / first OOM | Time (h) | Relative to model best | Speedup over B=1 | Optimizer (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in curves:
        text.append(f"| {r['model'].upper()} | {r['context_length']} | {r['best_batch']} | "
                    f"{r['smallest_batch_within_5pct']} | {r['largest_batch']} / {r['first_oom_batch']} | "
                    f"{r['best_hours']:.4f} | {r['relative_time_model_best']:.3f}× | "
                    f"{r['batching_speedup']:.2f}× | {100*r['best_optimizer_fraction']:.2f} |")
    text += [
        "", "![Relative training cost](relative.png)", "", CAPTIONS["relative"],
        "", "[Vector PDF](relative.pdf) · [Editable SVG](relative.svg)",
        "",
        "## Context length with equal tokens per update",
        "",
        "At a fixed sequence batch, changing context changes tokens per update and the number "
        "of optimizer steps needed to finish the budget. Longer contexts can therefore appear "
        "faster at small B by amortizing the optimizer and fixed costs. The controlled comparison "
        "below holds Q fixed and adjusts B. This measures the context effect for the same model, "
        "token budget and number of updates.",
        "",
        "| Model | Common Q at memory boundary | Context | Batch | Time / L=1024 at equal Q |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        group = [r for r in curves if r["model"] == model]
        q = min(r["largest_tokens_per_update"] for r in group)
        for length in contexts:
            ratio = 1. if length == 1024 else next(r["relative_time_vs_1024"] for r in data["matched_contexts"]
                       if r["model"] == model and r["context_length"] == length
                       and r["tokens_per_update"] == q)
            text.append(f"| {model.upper()} | {q:,} | {length} | {q//length} | {ratio:.3f}× |")
    text += [
        "", "![Matched context cost](context.png)", "", CAPTIONS["context"],
        "", "[Vector PDF](context.pdf) · [Editable SVG](context.svg)",
        "",
        "## Ideal optimizer acceleration",
        "",
        "The next table uses each model's fastest observed context, showing batch 1 and its best "
        "batch. Each counterfactual holds model, context, batch and total token budget fixed. Only "
        "optimizer cost changes. Other contexts and every "
        "batch are available in `measurements.csv`.",
        "",
        "| Model | Context | Batch | Baseline time (h) | With 2× optimizer (h) | Zero-cost optimizer (h) | Ideal maximum speedup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, winner in zip(MODELS, winners, strict=True):
        for batch in sorted({1, winner["best_batch"]}):
            length = winner["context_length"]
            r = next(r for r in rows if point_key(r) == (model, length, batch))
            text.append(f"| {model.upper()} | {length} | {batch} | {r['projected_hours']:.4f} | "
                        f"{r['ideal_2x_optimizer_hours']:.4f} | {r['ideal_zero_cost_optimizer_hours']:.4f} | "
                        f"{r['ideal_zero_cost_optimizer_speedup']:.3f}× |")
    text += [
        "",
        "## Reproduction and evidence",
        "",
        "```bash", command, "```",
        "",
        "The report generator independently recomputes rates and optimizer event weighting, "
        "checks exact parameter counts, configuration/source identities, finite state, completed "
        "update counts, grid coverage and first-OOM boundaries, and verifies D/R against the "
        "update-count calculation. Its input hashes cover raw results, receipts, frozen package "
        "source and campaign manifests. Figures use a white background, consistent colorblind-safe "
        "encodings, labeled units, panel identifiers, log2 batch axes and common comparison scales. "
        "PDF/SVG preserve vectors and text; PNG previews are rendered at 450 dpi.",
        "",
        "[Analysis checks and input hashes](analysis.json) · [Execution manifest](manifest.json)", "",
        "Campaign audits: " + "; ".join(
            f"[throughput]({os.path.relpath(t, output)}/audit.json) / "
            f"[optimizer]({os.path.relpath(o, output)}/audit.json) "
            f"(contexts {', '.join(map(str, source['contexts']))})"
            for t, o, source in zip(throughputs, optimizers, data["source_campaigns"], strict=True)) + ".", "",
    ]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--throughput", type=Path, nargs="+", default=[Path("runs/gh200-throughput-20261008")],
                        help="completed throughput campaigns, in the same order as --optimizer")
    parser.add_argument("--optimizer", type=Path, nargs="+", default=[Path("runs/gh200-optimizer-timing-20261009")])
    parser.add_argument("--output", type=Path, default=Path("runs/gh200-training-cost-20261009"))
    parser.add_argument("--list-inputs", action="store_true", help="validate sources and print dependency paths")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    throughputs, optimizers = [p.resolve() for p in args.throughput], [p.resolve() for p in args.optimizer]
    output = args.output.resolve()
    inputs = Inputs(repo)
    inputs.bytes(Path(__file__))
    data = combine(inputs, throughputs, optimizers, tokens_per_parameter=20)
    if args.list_inputs:
        print(json.dumps(sorted(inputs.hashes)))
        return
    require(output not in throughputs + optimizers, "combined output must not overwrite source campaigns")
    require(not (output / "manifest.json").exists(), "use a new output directory for an audited report")
    output.mkdir(parents=True, exist_ok=True)
    command = shlex.join([".venv-x86_64/bin/python", "scripts/throughput_report.py", "--throughput",
                          *[str(p.relative_to(repo)) for p in throughputs], "--optimizer",
                          *[str(p.relative_to(repo)) for p in optimizers],
                          "--output", str(output.relative_to(repo))])
    write_json(output / "results.json", data)
    for name, rows in (("measurements", data["points"]), ("curves", data["curves"]),
                       ("matched-contexts", data["matched_contexts"]), ("oom", data["oom"])):
        write_csv(output / f"{name}.csv", rows)
    figures(output, data)
    (output / "README.md").write_text(report_text(data, output, throughputs, optimizers, command))
    write_json(output / "analysis.json", dict(
        passed=True, arithmetic="Python float64 for times; exact integers for counts and budgets",
        diagnostics=data["diagnostics"], input_sha256=inputs.hashes, command=command,
        python=sys.version, software={p: importlib.metadata.version(p) for p in ("numpy", "matplotlib")},
        projection_contract="D=20P, Q=BL, T=D/R; ideal optimizer changes hold residual time fixed",
        checks=["raw throughput and optimizer arithmetic", "finite state and complete update counts",
                "exact parameter counts", "matching configurations and production source",
                "same GPU for paired timings", "contiguous grids and first OOM boundaries",
                "Slurm completion and fourteen-minute allocation limits",
                "rate versus update-count projection identity"],
    ))
    print(json.dumps(data["diagnostics"], indent=2))


if __name__ == "__main__":
    main()
