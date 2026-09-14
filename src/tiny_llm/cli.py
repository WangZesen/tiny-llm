import argparse
import json
import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path

from tiny_llm.config import PRESETS, Config, ModelConfig, load_config
from tiny_llm.runtime import setup_logging


def _add_benchmark_arguments(parser: argparse.ArgumentParser, *, packed: bool) -> None:
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=3 if packed else 20)
    parser.add_argument("--steps", type=int, default=8 if packed else 100)
    if not packed:
        parser.add_argument("--data-mode", choices=("synthetic", "real"), default="synthetic")
        parser.add_argument("--windows", type=int, default=3)
        parser.add_argument("--profile", action="store_true")


def _add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--checkpoints", nargs="+", default=["all"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-case", choices=("seen", "unseen", "both"), default="both")
    parser.add_argument("--noise-samples", default="32")
    parser.add_argument("--random-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="BF16 forward autocast with FP32 parameters and gradient accumulation",
    )
    parser.add_argument("--compile-hvp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hvp-batch-size", type=int)
    parser.add_argument(
        "--consensus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also analyze every consensus-error direction for packed runs",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small Llama pretraining and research experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    analysis = commands.add_parser("analyze")
    _add_analysis_arguments(analysis)
    analysis.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    analysis.add_argument("--checkpoint-manifest", type=Path, help=argparse.SUPPRESS)
    plotting = commands.add_parser("plot-analysis")
    plotting.add_argument("--output", required=True, type=Path)
    plotting.add_argument("--training-norms", action="store_true")
    submission = commands.add_parser("submit-analysis")
    _add_analysis_arguments(submission)
    submission.set_defaults(amp=True, hvp_batch_size=64, device="cuda")
    submission.add_argument("--jobs", required=True, type=int)
    submission.add_argument("--walltime-hours", type=int)
    submission.add_argument("--dry-run", action="store_true")
    submission.add_argument("--resume", action="store_true")
    for name in (
        "prepare",
        "train",
        "evaluate",
        "benchmark",
        "benchmark-worker",
        "benchmark-packed",
        "benchmark-packed-worker",
        "sweep",
        "report",
    ):
        child = commands.add_parser(name)
        child.add_argument(
            "--config",
            type=Path,
            action="append",
            default=[],
            help="YAML file; repeat to merge in order",
        )
        child.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        if name == "train":
            child.add_argument("--resume", type=Path)
        if name == "evaluate":
            child.add_argument("--checkpoint", required=True, type=Path)
            child.add_argument("--full", action="store_true")
        if name in ("benchmark", "benchmark-worker", "benchmark-packed", "benchmark-packed-worker"):
            _add_benchmark_arguments(child, packed=name.startswith("benchmark-packed"))
        if name == "benchmark-packed":
            child.add_argument("--num-models", type=int, nargs="+", default=[4, 8])
        if name == "benchmark-packed-worker":
            child.add_argument("--execution", choices=("packed", "sequential"), required=True)
        if name == "benchmark":
            child.add_argument("--all-presets", action="store_true")
            child.add_argument("--gh200", action="store_true")
            child.add_argument("--budget-minutes", type=float, default=75)
        if name == "sweep":
            child.add_argument("--output", type=Path, default=Path("runs/campaign"))
            child.add_argument("--benchmarks", type=Path, default=Path("runs/benchmarks"))
            child.add_argument("--gpus", default="0,1")
        if name == "report":
            child.add_argument("--runs", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging()
    if args.command in ("analyze", "submit-analysis"):
        from tiny_llm.analysis import AnalysisOptions, analyze

        try:
            samples = "all" if args.noise_samples == "all" else int(args.noise_samples)
            options = AnalysisOptions(
                data_case=args.data_case,
                noise_samples=samples,
                random_samples=args.random_samples,
                seed=args.seed,
                device=args.device,
                dtype=args.dtype,
                tf32=args.tf32,
                amp=args.amp,
                compile_hvp=args.compile_hvp,
                hvp_batch_size=args.hvp_batch_size,
                consensus=args.consensus,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.command == "analyze":
            expected = (
                json.loads(args.checkpoint_manifest.read_text())
                if args.checkpoint_manifest
                else None
            )
            analyze(
                args.run,
                args.checkpoints,
                args.output,
                options,
                plots=args.plots,
                expected_checkpoints=expected,
            )
        else:
            from tiny_llm.analysis.slurm import orchestrate

            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
            try:
                result = orchestrate(
                    args.run,
                    args.checkpoints,
                    args.output,
                    args.jobs,
                    options,
                    dry_run=args.dry_run,
                    resume=args.resume,
                    walltime_hours=args.walltime_hours,
                )
            except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
                parser.exit(1, f"{exc}\n")
            print(json.dumps(result, indent=2))
        return
    if args.command == "plot-analysis":
        from tiny_llm.analysis.plot import plot_analysis

        plot_analysis(args.output, args.training_norms)
        return
    config = load_config(args.config, args.set)
    dispatch(args, config, parser)


def dispatch(args: argparse.Namespace, config: Config, parser: argparse.ArgumentParser) -> None:
    match args.command:
        case "prepare":
            from tiny_llm.data import prepare

            prepare(config)
        case "train":
            from tiny_llm.train import train

            result = train(config, args.resume)
            if result["status"] != "complete":
                raise SystemExit(130)
        case "evaluate":
            from tiny_llm.train import evaluate_checkpoint

            print(json.dumps(evaluate_checkpoint(config, args.checkpoint, args.full), indent=2))
        case "benchmark":
            from tiny_llm.benchmark import benchmark

            if args.gh200:
                if args.all_presets:
                    parser.error("--gh200 tunes one preset at a time")
                from tiny_llm.benchmark import tune_gh200

                tune_gh200(config, args.output, budget_minutes=args.budget_minutes)
            else:
                candidates = [(config, args.output)]
                if args.all_presets:
                    candidates = []
                    for preset in PRESETS:
                        candidate = config.model_copy(deep=True)
                        candidate.model = ModelConfig(
                            **(config.model.model_dump() | PRESETS[preset])
                        )
                        candidates.append((candidate, args.output / preset))
                for candidate, output in candidates:
                    benchmark(
                        candidate,
                        output,
                        data_mode=args.data_mode,
                        warmup=args.warmup,
                        steps=args.steps,
                        windows=args.windows,
                        profile=args.profile,
                    )
        case "benchmark-worker":
            from tiny_llm.benchmark import benchmark_worker

            benchmark_worker(
                config,
                args.output,
                warmup=args.warmup,
                steps=args.steps,
                windows=args.windows,
                data_mode=args.data_mode,
                profile=args.profile,
            )
        case "benchmark-packed":
            from tiny_llm.packed_benchmark import benchmark_packed

            benchmark_packed(
                config,
                args.output,
                num_models=args.num_models,
                warmup=args.warmup,
                steps=args.steps,
            )
        case "benchmark-packed-worker":
            from tiny_llm.packed_benchmark import benchmark_packed_worker

            benchmark_packed_worker(
                config, args.output, execution=args.execution, warmup=args.warmup, steps=args.steps
            )
        case "sweep":
            from tiny_llm.experiments import sweep

            sweep(config, args.output, args.benchmarks, args.gpus.split(","))
        case "report":
            from tiny_llm.experiments import report

            report(args.runs)
