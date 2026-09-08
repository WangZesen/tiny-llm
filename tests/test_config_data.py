import json
import math

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from tiny_llm.config import PRESETS, Config, ModelConfig, load_config
from tiny_llm.data import TokenCache, training_boundaries, validation_indices
from tiny_llm.model import Llama
from tiny_llm.runtime import setup_runtime


def test_strict_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("optimizer:\n  lr: 0.001\n")
    config = load_config(path, ["optimizer.lr=0.003", "runtime.deterministic=false"])
    assert config.optimizer.lr == 0.003
    assert config.runtime.deterministic is False
    with pytest.raises(ValidationError):
        load_config(path, ["optimizer.learning_rate=0.1"])
    with pytest.raises(ValidationError):
        load_config(path, ["optimizer.beta2=1"])
    with pytest.raises(ValueError):
        load_config(path, ["missing_equals"])
    with pytest.raises(ValidationError):
        load_config(path, ["model.heads=7"])


@pytest.mark.parametrize(
    "preset,expected", [("20m", 20403520), ("50m", 48507392), ("90m", 91605120)]
)
def test_parameters_and_epochs(preset, expected):
    config = Config(model=ModelConfig(**PRESETS[preset]))
    # Meta tensors avoid allocating 90M weights just to count them.
    with torch.device("meta"):
        model = Llama(config.model)
    assert model.parameter_count == expected == config.model.parameter_count
    boundaries = training_boundaries(config, expected)
    assert len(boundaries) == 40
    assert 0 <= boundaries[-1] * 1024 - 20 * expected < 1024
    assert all(
        abs((end - start) * 1024 - 0.5 * expected) < 1024
        for start, end in zip([0] + boundaries[:-1], boundaries, strict=True)
    )
    assert "embedding.weight" in model.state_dict()
    assert not any("lm_head" in key for key in model.state_dict())


def test_boundary_rounding(tiny_config):
    tiny_config.training.max_tokens = 67
    tiny_config.training.epoch_tokens = 23
    assert training_boundaries(tiny_config, 100) == [6, 12, 17]


def test_cross_shard_shift_and_partial(cache_dir, tiny_config):
    cache = TokenCache(cache_dir)
    cache.verify()
    np.testing.assert_array_equal(cache.read("train", 8, 26), np.arange(8, 26) % 17)
    x, y = cache.batch("train", [2, 3], 4, torch.device("cpu"))
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert y[0, -1] == x[1, 0]
    full = validation_indices(cache, tiny_config, full=True)
    assert len(full) == math.ceil(29 / 4)
    _, targets = cache.batch("validation", full, 4, torch.device("cpu"))
    assert (targets != -100).sum() == 29
    assert (targets[-1, 1:] == -100).all()
    first = validation_indices(cache, tiny_config, full=False)
    assert len(first) == 3
    np.testing.assert_array_equal(first, validation_indices(cache, tiny_config, full=False))


def test_cache_corruption(cache_dir):
    cache = TokenCache(cache_dir)
    shard = cache_dir / cache.manifest["splits"]["train"]["shards"][0]["file"]
    with shard.open("r+b") as handle:
        handle.write(b"\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="checksum"):
        cache.verify()
    shard.write_bytes(b"bad")
    with pytest.raises(ValueError, match="truncated"):
        TokenCache(cache_dir)


def test_cache_manifest_and_config(cache_dir, tiny_config):
    cache = TokenCache(cache_dir)
    tiny_config.data.shuffle_seed = 12
    with pytest.raises(ValueError, match="preprocessing"):
        cache.validate_config(tiny_config)
    path = cache_dir / "manifest.json"
    value = json.loads(path.read_text())
    value["vocab_size"] = 10
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identity"):
        TokenCache(cache_dir)


def test_deterministic_default_and_override(tiny_config):
    setup_runtime(tiny_config)
    assert not torch.are_deterministic_algorithms_enabled()
    tiny_config.runtime.deterministic = True
    setup_runtime(tiny_config)
    assert torch.are_deterministic_algorithms_enabled()
