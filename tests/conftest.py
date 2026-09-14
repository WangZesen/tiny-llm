import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tiny_llm.config import Config, ModelConfig, TrainingConfig
from tiny_llm.data import TokenWriter, fingerprint
from tiny_llm.state import CacheContent, CacheManifest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that start tokenizer worker processes",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run CPU compiler and figure-rendering checks",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for marker in ("integration", "slow"):
        option = f"--run-{marker}"
        if not config.getoption(option):
            skip = pytest.mark.skip(reason=f"enable with {option}")
            for item in items:
                if item.get_closest_marker(marker) is not None:
                    item.add_marker(skip)


@pytest.fixture(scope="session")
def single_threaded_session():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(autouse=True)
def runtime_policy(single_threaded_session):
    deterministic = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    if torch.are_deterministic_algorithms_enabled() != deterministic:
        torch.use_deterministic_algorithms(deterministic)
    if torch.get_num_threads() != threads:
        torch.set_num_threads(threads)


@pytest.fixture(autouse=True)
def cpu_run_metadata(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Use fixed host metadata for CPU tests while preserving benchmark profiling."""
    if request.node.get_closest_marker("cuda") is not None:
        return

    def environment():
        return dict(source_hash="fixture", versions={"torch": torch.__version__}, gpu=None)

    for module in ("train", "benchmark", "packed_benchmark"):
        monkeypatch.setattr(f"tiny_llm.{module}.environment", environment)


@pytest.fixture
def tiny_config(tmp_path: Path) -> Config:
    config = Config(
        model=ModelConfig(vocab_size=17, layers=1, width=8, heads=2, ffn_width=16, context_length=4)
    )
    config.runtime.device = "cpu"
    config.runtime.amp = False
    config.runtime.compile = False
    config.runtime.cpu_threads = 1
    config.runtime.output_dir = tmp_path / "run"
    config.data.cache_dir = tmp_path / "cache"
    config.data.shard_tokens = 11
    # Tiny runs should exercise decay rather than the production 1,000-step warmup.
    config.lr_schedule.warmup_steps = 0
    config.training.batch_tokens = 16
    config.training.micro_batch_size = 2
    config.training = TrainingConfig(
        tokens_per_parameters=0.08,
        epoch_tokens_per_parameters=0.04,
        batch_tokens=16,
        micro_batch_size=2,
        log_every=1,
    )
    config.training.log_every = 1
    config.evaluation.subset_blocks = 3
    config.evaluation.batch_size = 2
    return config


@pytest.fixture
def cache_dir(tiny_config: Config) -> Path:
    directory = tiny_config.data.cache_dir
    directory.mkdir()
    manifest: CacheContent = CacheContent(
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
            shuffle_seed=tiny_config.data.shuffle_seed,
            shuffle_buffer=tiny_config.data.shuffle_buffer,
            append_eos=True,
            add_special_tokens=False,
        ),
        validation_complete=True,
        splits={},
    )
    for split, count in (("train", 133), ("validation", 30)):
        writer = TokenWriter(directory, split, shard_tokens=tiny_config.data.shard_tokens)
        writer.write(np.arange(count) % 17)
        manifest["splits"][split] = writer.finish()
    completed_manifest = CacheManifest(**manifest, identity=fingerprint(manifest))
    (directory / "manifest.json").write_text(json.dumps(completed_manifest))
    return directory
