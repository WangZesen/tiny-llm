"""Shared fixtures for small token budgets and exact saved-state comparisons."""

import numpy as np
import torch

from tiny_llm.config import Config, TrainingConfig


def set_budget(config: Config, total: int, epoch: int) -> None:
    config.training = TrainingConfig.model_validate(
        config.training.model_dump()
        | {
            "tokens_per_parameters": total / config.model.parameter_count,
            "epoch_tokens_per_parameters": epoch / config.model.parameter_count,
        }
    )


def assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


def gradients(model: torch.nn.Module) -> list[torch.Tensor]:
    values = []
    for parameter in model.parameters():
        assert parameter.grad is not None
        values.append(parameter.grad)
    return values
