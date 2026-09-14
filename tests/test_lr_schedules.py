from pathlib import Path

import pytest
import yaml

from tiny_llm.config import (
    Config,
    CosineScheduleConfig,
    WSDScheduleConfig,
    load_config,
    save_config,
)
from tiny_llm.data import TokenCache
from tiny_llm.train import learning_rate, recipe_identity


@pytest.mark.parametrize("name,cls", [("cosine", CosineScheduleConfig), ("wsd", WSDScheduleConfig)])
def test_schedule_config(name, cls, tmp_path: Path):
    config = Config.model_validate({"lr_schedule": {"name": name}})
    assert isinstance(config.lr_schedule, cls)
    assert config.lr_schedule == cls()
    assert config.lr_schedule.warmup_steps == 1000
    path = tmp_path / "resolved.yaml"
    save_config(config, path)
    assert load_config(path) == config
    assert yaml.safe_load(path.read_text())["lr_schedule"]["name"] == name


def test_default_schedule():
    assert Config().lr_schedule == CosineScheduleConfig()
    assert WSDScheduleConfig().decay_fraction == 0.1
    assert load_config() == load_config([]) == Config()


@pytest.mark.parametrize(
    "schedule",
    [
        {},
        {"warmup_steps": 100},
        {"name": "linear"},
        {"name": "cosine", "decay_fraction": 0.1},
        {"name": "wsd", "min_lr_ratio": 0},
        {"name": "wsd", "decay_fraction": -0.1},
        {"name": "wsd", "decay_fraction": 1.1},
        {"name": "wsd", "decay_fraction": float("nan")},
        {"name": "wsd", "decay_fraction": float("inf")},
        {"name": "cosine", "min_lr_ratio": -0.1},
        {"name": "cosine", "min_lr_ratio": 1.1},
        {"name": "cosine", "min_lr_ratio": float("nan")},
    ],
)
def test_invalid_schedule_config(schedule):
    with pytest.raises(ValueError):
        Config.model_validate({"lr_schedule": schedule})


@pytest.mark.parametrize("name", ["cosine", "wsd"])
@pytest.mark.parametrize("warmup", [-1, 0.5, 1.0, True, "1000", float("nan"), float("inf")])
def test_invalid_warmup(name, warmup):
    with pytest.raises(ValueError):
        Config.model_validate({"lr_schedule": {"name": name, "warmup_steps": warmup}})


@pytest.mark.parametrize("name", ["cosine", "wsd"])
def test_warmup_fraction_rejected(name):
    with pytest.raises(ValueError, match="Extra inputs"):
        Config.model_validate({"lr_schedule": {"name": name, "warmup_fraction": 0.1}})


@pytest.mark.parametrize("field", ["warmup_fraction", "min_lr_ratio"])
def test_legacy_optimizer_schedule_rejected(field):
    with pytest.raises(ValueError, match="Extra inputs"):
        Config.model_validate({"optimizer": {field: 0.1}})


def test_composed_config(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "optimizer: {lr: 0.002, beta2: 0.98}\n"
        "training: {batch_tokens: 1024, checkpoint_policy: explicit, checkpoint_epochs: [1, 2]}\n"
        "lr_schedule: {name: wsd, warmup_steps: 200, decay_fraction: 0.4}\n"
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "optimizer: {lr: 0.003}\n"
        "training: {micro_batch_size: 1, checkpoint_epochs: [3]}\n"
        "lr_schedule: {decay_fraction: 0.5}\n"
    )
    # The base alone has an invalid batch shape; validation occurs after composition.
    config = load_config([str(base), overlay], ["optimizer.lr=0.004"])
    assert config.optimizer.lr == 0.004
    assert config.optimizer.beta2 == 0.98
    assert config.training.batch_tokens == 1024
    assert config.training.micro_batch_size == 1
    assert config.training.checkpoint_epochs == [3]
    assert config.lr_schedule == WSDScheduleConfig(warmup_steps=200, decay_fraction=0.5)
    resolved = tmp_path / "resolved.yaml"
    save_config(config, resolved)
    assert load_config(resolved) == load_config(str(resolved)) == load_config([resolved]) == config


@pytest.mark.parametrize("first,second", [("cosine", "wsd"), ("wsd", "cosine")])
def test_scheduler_switching(first, second, tmp_path: Path):
    initial = tmp_path / "initial.yaml"
    initial.write_text(f"lr_schedule: {{name: {first}, warmup_steps: 300}}\n")
    paths = [initial, Path(f"configs/{first}.yaml"), Path(f"configs/{second}.yaml")]
    assert load_config(paths).lr_schedule == load_config(paths[-1]).lr_schedule
    assert (
        load_config(paths[:2], [f"lr_schedule.name={second}"]).lr_schedule
        == load_config(paths[-1]).lr_schedule
    )
    config = load_config(
        paths,
        [f"lr_schedule.name={first}", "lr_schedule.warmup_steps=200"],
    )
    assert config.lr_schedule.name == first
    assert config.lr_schedule.warmup_steps == 200


def test_same_scheduler_retains_fields(tmp_path: Path):
    patch = tmp_path / "patch.yaml"
    patch.write_text("lr_schedule: {name: wsd, decay_fraction: 0.3}\n")
    base = tmp_path / "base.yaml"
    base.write_text("lr_schedule: {name: wsd, warmup_steps: 200}\n")
    config = load_config([base, patch], ["lr_schedule.name=wsd"])
    assert config.lr_schedule == WSDScheduleConfig(warmup_steps=200, decay_fraction=0.3)


@pytest.mark.parametrize("contents", ["[]", "3", "a string"])
def test_invalid_config_file(contents, tmp_path: Path):
    path = tmp_path / "invalid.yaml"
    path.write_text(contents)
    with pytest.raises(ValueError, match="YAML mapping"):
        load_config([Path("configs/wsd.yaml"), path])


def test_empty_config_and_scalar_replacement(tmp_path: Path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    assert load_config([empty]) == Config()
    base = tmp_path / "base.yaml"
    base.write_text("lr_schedule: null\n")
    assert load_config([base, Path("configs/wsd.yaml")]).lr_schedule == WSDScheduleConfig()
    with pytest.raises(ValueError, match="cannot descend into scalar"):
        load_config(base, ["lr_schedule.name=wsd"])


@pytest.mark.parametrize("name", ["cosine", "wsd"])
@pytest.mark.parametrize(
    "recipe",
    sorted(path for path in Path("configs").glob("*.yaml") if path.stem not in {"cosine", "wsd"}),
    ids=lambda path: path.stem,
)
def test_recipes_are_scheduler_independent(recipe: Path, name):
    raw = yaml.safe_load(recipe.read_text())
    assert "lr_schedule" not in raw
    assert (
        not {"warmup_fraction", "warmup_steps", "min_lr_ratio", "decay_fraction"}
        & raw.get("optimizer", {}).keys()
    )
    plain = load_config(recipe)
    composed = load_config([recipe, Path(f"configs/{name}.yaml")])
    assert composed.lr_schedule.name == name
    assert composed.model_dump(exclude={"lr_schedule"}) == plain.model_dump(exclude={"lr_schedule"})


@pytest.mark.parametrize(
    "step,ratio", [(0, 0), (25, 0.5), (50, 1), (525, 0.55), (1000, 0.1), (1100, 0.1)]
)
def test_cosine_values(tiny_config: Config, step, ratio):
    tiny_config.lr_schedule = CosineScheduleConfig(warmup_steps=50)
    batch = tiny_config.training.batch_tokens
    assert learning_rate(tiny_config, step * batch, 1000 * batch) == pytest.approx(
        tiny_config.optimizer.lr * ratio
    )


@pytest.mark.parametrize(
    "step,ratio",
    [(0, 0), (25, 0.5), (50, 1), (500, 1), (900, 1), (925, 0.5), (964, 0.2), (1000, 0), (1100, 0)],
)
def test_wsd_values(tiny_config: Config, step, ratio):
    tiny_config.lr_schedule = WSDScheduleConfig(warmup_steps=50)
    batch = tiny_config.training.batch_tokens
    assert learning_rate(tiny_config, step * batch, 1000 * batch) == pytest.approx(
        tiny_config.optimizer.lr * ratio
    )


@pytest.mark.parametrize(
    "warmup,decay,steps,ratios",
    [
        (0, 0.1, [0, 900, 925, 1000, 1100], [1, 1, 0.5, 0, 0]),
        (100, 0, [0, 50, 100, 1000, 1100], [0, 0.5, 1, 1, 1]),
        (0, 0, [0, 1, 1000, 1100], [1, 1, 1, 1]),
        (500, 0.5, [0, 250, 500, 625, 1000], [0, 0.5, 1, 0.5, 0]),
        (0, 1, [0, 250, 640, 1000], [1, 0.5, 0.2, 0]),
        (50, 1e-20, [0, 50, 999, 1000], [0, 1, 1, 0]),
        (900, 0.2, [0, 800, 900, 925, 1000], [0, 8 / 9, 1, 0.5, 0]),
    ],
)
def test_wsd_degenerate_phases(tiny_config: Config, warmup, decay, steps, ratios):
    tiny_config.lr_schedule = WSDScheduleConfig(warmup_steps=warmup, decay_fraction=decay)
    batch = tiny_config.training.batch_tokens
    assert [
        learning_rate(tiny_config, step * batch, 1000 * batch) / tiny_config.optimizer.lr
        for step in steps
    ] == pytest.approx(ratios)


@pytest.mark.parametrize("cls", [CosineScheduleConfig, WSDScheduleConfig])
@pytest.mark.parametrize("total_steps", [2000, 4000])
@pytest.mark.parametrize("batch_tokens", [16, 32])
def test_default_warmup_is_independent_of_budget_and_batch(
    tiny_config: Config, cls, total_steps, batch_tokens
):
    tiny_config.lr_schedule = cls()
    tiny_config.training.batch_tokens = batch_tokens
    assert [
        learning_rate(tiny_config, step * batch_tokens, total_steps * batch_tokens)
        / tiny_config.optimizer.lr
        for step in (0, 1, 500, 1000)
    ] == pytest.approx([0, 0.001, 0.5, 1])


@pytest.mark.parametrize("cls", [CosineScheduleConfig, WSDScheduleConfig])
@pytest.mark.parametrize("total_steps", [500, 1000])
def test_run_finishes_during_warmup(tiny_config: Config, cls, total_steps):
    tiny_config.lr_schedule = cls()
    batch = tiny_config.training.batch_tokens
    assert learning_rate(tiny_config, total_steps * batch, total_steps * batch) == pytest.approx(
        tiny_config.optimizer.lr * total_steps / 1000
    )
    terminal = 0.1 if cls is CosineScheduleConfig else 0
    assert learning_rate(tiny_config, 1001 * batch, total_steps * batch) == pytest.approx(
        tiny_config.optimizer.lr * terminal
    )


def test_schedule_identity(tiny_config: Config, cache_dir: Path):
    cache = TokenCache(cache_dir)
    cosine = recipe_identity(tiny_config, cache)
    tiny_config.lr_schedule = WSDScheduleConfig()
    wsd = recipe_identity(tiny_config, cache)
    assert wsd != cosine
    tiny_config.lr_schedule.decay_fraction = 0.2
    assert recipe_identity(tiny_config, cache) != wsd
    tiny_config.lr_schedule = WSDScheduleConfig(warmup_steps=100)
    assert recipe_identity(tiny_config, cache) != wsd
