import argparse
import json
from pathlib import Path

from tiny_llm.config import PRESETS, ModelConfig, load_config
from tiny_llm.runtime import setup_logging


def main():
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
            child.add_argument("--output", required=True, type=Path)
        if name in ("benchmark-packed", "benchmark-packed-worker"):
            child.add_argument("--warmup", type=int, default=3)
            child.add_argument("--steps", type=int, default=8)
        if name == "benchmark-packed":
            child.add_argument("--num-models", type=int, nargs="+", default=[4, 8])
        if name == "benchmark-packed-worker":
            child.add_argument("--execution", choices=("packed", "sequential"), required=True)
        if name == "benchmark":
            child.add_argument("--all-presets", action="store_true")
        if name == "sweep":
            child.add_argument("--output", type=Path, default=Path("runs/campaign"))
            child.add_argument("--benchmarks", type=Path, default=Path("runs/benchmarks"))
            child.add_argument("--gpus", default="0,1")
        if name == "report":
            child.add_argument("--runs", required=True, type=Path)
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config, args.set)
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
            from tiny_llm.experiments import benchmark

            if args.all_presets:
                for preset in PRESETS:
                    candidate = config.model_copy(deep=True)
                    candidate.model = ModelConfig(**(config.model.model_dump() | PRESETS[preset]))
                    benchmark(candidate, args.output / preset)
            else:
                benchmark(config, args.output)
        case "benchmark-worker":
            from tiny_llm.experiments import benchmark_worker

            benchmark_worker(config, args.output)
        case "benchmark-packed":
            from tiny_llm.packed_benchmark import benchmark_packed

            benchmark_packed(config, args.output, args.num_models, args.warmup, args.steps)
        case "benchmark-packed-worker":
            from tiny_llm.packed_benchmark import benchmark_packed_worker

            benchmark_packed_worker(config, args.output, args.execution, args.warmup, args.steps)
        case "sweep":
            from tiny_llm.experiments import sweep

            sweep(config, args.output, args.benchmarks, args.gpus.split(","))
        case "report":
            from tiny_llm.experiments import report

            report(args.runs)
