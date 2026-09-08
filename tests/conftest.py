import json

import numpy as np
import pytest
import torch

from tiny_llm.config import Config, ModelConfig
from tiny_llm.data import TokenWriter, fingerprint


@pytest.fixture(autouse=True)
def runtime_policy():
    deterministic = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.use_deterministic_algorithms(deterministic)
    torch.set_num_threads(threads)


@pytest.fixture
def tiny_config(tmp_path):
    config = Config(
        model=ModelConfig(vocab_size=17, layers=1, width=8, heads=2, ffn_width=16, context_length=4)
    )
    config.runtime.device = "cpu"
    config.runtime.amp = False
    config.runtime.cpu_threads = 1
    config.runtime.output_dir = tmp_path / "run"
    config.data.cache_dir = tmp_path / "cache"
    config.training.batch_tokens = 16
    config.training.micro_batch_size = 2
    config.training.max_tokens = 64
    config.training.epoch_tokens = 32
    config.training.checkpoint_every = 2
    config.training.log_every = 1
    config.evaluation.subset_blocks = 3
    config.evaluation.batch_size = 2
    return config


@pytest.fixture
def cache_dir(tiny_config):
    directory = tiny_config.data.cache_dir
    directory.mkdir()
    manifest = dict(
        version=1,
        dataset=tiny_config.data.dataset,
        dataset_revision="fixture-dataset",
        tokenizer=tiny_config.data.tokenizer,
        tokenizer_revision="fixture-tokenizer",
        tokenizer_fingerprint="fixture",
        vocab_size=17,
        eos_token_id=2,
        dtype="<u2",
        preprocessing=dict(
            shuffle_seed=42, shuffle_buffer=10000, append_eos=True, add_special_tokens=False
        ),
        validation_complete=True,
        splits={},
    )
    for split, count in (("train", 129), ("validation", 30)):
        writer = TokenWriter(directory, split, shard_tokens=11)
        writer.write(np.arange(count) % 17)
        manifest["splits"][split] = writer.finish()
    manifest["identity"] = fingerprint(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory
