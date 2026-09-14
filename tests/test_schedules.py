from fractions import Fraction
from pathlib import Path

import pytest

from tiny_llm.config import Config, TrainingConfig, load_config, save_config
from tiny_llm.data import training_boundaries


def test_decimal_divisibility_and_halfway_rounding(tiny_config: Config) -> None:
    tiny_config.training = TrainingConfig(
        tokens_per_parameters=0.3,
        epoch_tokens_per_parameters=0.1,
        batch_tokens=4,
        micro_batch_size=1,
    )
    assert training_boundaries(tiny_config, 100) == [3, 5, 8]
    with pytest.raises(ValueError, match="divisible"):
        TrainingConfig(tokens_per_parameters=0.31, epoch_tokens_per_parameters=0.1)


@pytest.mark.parametrize("epoch_ratio", [0.001, 0.01])
def test_empty_epochs_rejected(tiny_config: Config, epoch_ratio: float) -> None:
    tiny_config.training = TrainingConfig(
        tokens_per_parameters=epoch_ratio * 2,
        epoch_tokens_per_parameters=epoch_ratio,
        batch_tokens=16,
        micro_batch_size=2,
    )
    with pytest.raises(ValueError, match="at least one global batch"):
        training_boundaries(tiny_config, tiny_config.model.parameter_count)


@pytest.mark.parametrize("workers", [1, 2, 4, 8, 16, 32, 64, 128])
def test_extended_budget_preserves_batches(workers: int) -> None:
    config = load_config(
        "configs/20m.yaml",
        [
            f"decentralized.num_models={workers}",
            f"training.micro_batch_size={128 // workers}",
        ],
    )
    parameters, batch = config.model.parameter_count, config.training.batch_tokens
    short = training_boundaries(config, parameters)
    config.training.tokens_per_parameters = 40
    long = training_boundaries(config, parameters)
    assert short == long[: len(short)]
    assert [b - a for a, b in zip([0] + short[:-1], short, strict=True)][:6] == [
        batches * 128 for batches in (78, 78, 77, 78, 78, 78)
    ]
    for epoch, boundary in enumerate(long, 1):
        tokens = boundary * config.model.context_length
        assert boundary % workers == 0
        assert tokens % batch == 0
        assert abs(tokens - epoch * Fraction("0.5") * parameters) <= batch // 2


def test_smoke_budget() -> None:
    config = load_config("configs/smoke.yaml")
    assert training_boundaries(config, config.model.parameter_count) == [128, 256]


@pytest.mark.parametrize(
    "policy,epochs,expected",
    [
        ("interval", 1, list(range(1, 41))),
        ("interval", 7, [7, 14, 21, 28, 35]),
        ("interval", 50, []),
        ("explicit", [10, 2, 10, 40], [2, 10, 40]),
        ("final", 1, []),
        ("none", 1, []),
    ],
)
def test_checkpoint_schedule(
    policy: str, epochs: object, expected: list[int], tmp_path: Path
) -> None:
    config = Config.model_validate(
        {"training": {"checkpoint_policy": policy, "checkpoint_epochs": epochs}}
    )
    assert sorted(config.training.saved_epochs()) == expected
    assert isinstance(config.training.checkpoint_epochs, list)
    path = tmp_path / "resolved.yaml"
    save_config(config, path)
    assert load_config(path) == config


@pytest.mark.parametrize(
    "policy,epochs",
    [
        ("interval", [1, 2]),
        ("interval", [2, 2]),
        ("explicit", [41]),
        ("explicit", []),
        ("interval", 0),
        ("explicit", [-1]),
        ("interval", True),
        ("interval", 1.5),
        ("all", 1),
    ],
)
def test_invalid_checkpoint_schedule(policy: str, epochs: object) -> None:
    with pytest.raises(ValueError):
        TrainingConfig.model_validate({"checkpoint_policy": policy, "checkpoint_epochs": epochs})


@pytest.mark.parametrize(
    "field",
    [
        "tokens_per_parameter",
        "epoch_tokens_per_parameter",
        "max_tokens",
        "epoch_tokens",
        "checkpoint_every",
    ],
)
def test_removed_fields_rejected(field: str) -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        TrainingConfig.model_validate({field: 1})
