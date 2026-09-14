from pathlib import Path

import pytest

from tiny_llm.cli import build_parser, main
from tiny_llm.config import PRESETS, Config, save_config


def test_repeated_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import tiny_llm.cli as module

    calls = []
    monkeypatch.setattr(module, "dispatch", lambda args, config, parser: calls.append(config))
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("optimizer: {lr: 0.003}\n")
    main(
        [
            "train",
            "--config",
            str(recipe),
            "--config",
            "configs/cosine.yaml",
            "--config",
            "configs/wsd.yaml",
            "--set",
            "lr_schedule.decay_fraction=0.2",
            "--set",
            "lr_schedule.warmup_steps=500",
        ]
    )
    assert len(calls) == 1
    assert calls[0].optimizer.lr == 0.003
    assert calls[0].lr_schedule.name == "wsd"
    assert calls[0].lr_schedule.decay_fraction == 0.2
    assert calls[0].lr_schedule.warmup_steps == 500


@pytest.mark.parametrize(
    "command,extra,warmup,steps",
    [
        ("benchmark", [], 20, 100),
        ("benchmark-worker", [], 20, 100),
        ("benchmark-packed", [], 3, 8),
        ("benchmark-packed-worker", ["--execution", "sequential"], 3, 8),
    ],
)
def test_benchmark_defaults(command, extra, warmup, steps):
    args = build_parser().parse_args([command, "--output", "out", *extra])
    assert (args.warmup, args.steps, args.output) == (warmup, steps, Path("out"))
    if command in ("benchmark", "benchmark-worker"):
        assert (args.data_mode, args.windows, args.profile) == ("synthetic", 3, False)
    if command == "benchmark":
        assert (args.all_presets, args.gh200, args.budget_minutes) == (False, False, 75)
    if command == "benchmark-packed":
        assert args.num_models == [4, 8]


@pytest.mark.parametrize(
    "command,function,extra,expected",
    [
        (
            "benchmark",
            "benchmark",
            ["--data-mode", "real", "--windows", "2", "--profile"],
            dict(data_mode="real", windows=2, profile=True),
        ),
        (
            "benchmark-worker",
            "benchmark_worker",
            ["--data-mode", "real", "--windows", "2", "--profile"],
            dict(data_mode="real", windows=2, profile=True),
        ),
        (
            "benchmark-packed",
            "benchmark_packed",
            ["--num-models", "1", "4"],
            dict(num_models=[1, 4]),
        ),
        (
            "benchmark-packed-worker",
            "benchmark_packed_worker",
            ["--execution", "sequential"],
            dict(execution="sequential"),
        ),
    ],
)
def test_benchmark_dispatch(
    tiny_config: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command,
    function,
    extra,
    expected,
):
    import tiny_llm.benchmark as ordinary
    import tiny_llm.packed_benchmark as packed

    calls = []
    module = packed if "packed" in command else ordinary
    monkeypatch.setattr(module, function, lambda *args, **kwargs: calls.append((args, kwargs)))
    path = tmp_path / "recipe.yaml"
    save_config(tiny_config, path)
    main(
        [command, "--config", str(path), "--output", "out", "--warmup", "5", "--steps", "7", *extra]
    )
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (tiny_config, Path("out"))
    assert kwargs == dict(warmup=5, steps=7, **expected)


def test_all_presets_dispatch(tiny_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import tiny_llm.benchmark as module

    calls = []
    monkeypatch.setattr(module, "benchmark", lambda *args, **kwargs: calls.append((args, kwargs)))
    path = tmp_path / "recipe.yaml"
    save_config(tiny_config, path)
    main(["benchmark", "--config", str(path), "--output", "out", "--all-presets"])
    assert len(calls) == len(PRESETS)
    for ((config, output), options), (name, preset) in zip(calls, PRESETS.items(), strict=True):
        assert output == Path("out") / name
        assert all(getattr(config.model, key) == value for key, value in preset.items())
        assert config.optimizer == tiny_config.optimizer
        assert options == dict(
            data_mode="synthetic", warmup=20, steps=100, windows=3, profile=False
        )
    assert len({id(args[0]) for args, _ in calls}) == len(PRESETS)


def test_tuner_dispatch_and_conflict(
    tiny_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import tiny_llm.benchmark as module

    calls = []
    monkeypatch.setattr(module, "tune_gh200", lambda *args, **kwargs: calls.append((args, kwargs)))
    path = tmp_path / "recipe.yaml"
    save_config(tiny_config, path)
    argv = [
        "benchmark",
        "--config",
        str(path),
        "--output",
        "out",
        "--gh200",
        "--budget-minutes",
        "2",
    ]
    main(argv)
    assert calls == [((tiny_config, Path("out")), dict(budget_minutes=2))]
    with pytest.raises(SystemExit) as error:
        main([*argv, "--all-presets"])
    assert error.value.code == 2
    assert len(calls) == 1
