import argparse
import json
from pathlib import Path

from tiny_llm.config import PRESETS, ModelConfig, load_config
from tiny_llm.runtime import setup_logging


def _add_benchmark_arguments(parser, *, packed):
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=3 if packed else 20)
    parser.add_argument("--steps", type=int, default=8 if packed else 100)
    if not packed:
        parser.add_argument("--data-mode", choices=("synthetic", "real"), default="synthetic")
        parser.add_argument("--windows", type=int, default=3)
        parser.add_argument("--profile", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small Llama pretraining and research experiments")
    commands = parser.add_subparsers(dest="command", required=True)
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
        child.add_argument("--config", type=Path)
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


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging()
    config = load_config(args.config, args.set)
    dispatch(args, config, parser)


def dispatch(args, config, parser):
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
