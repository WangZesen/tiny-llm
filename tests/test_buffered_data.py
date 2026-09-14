import threading
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch

from tiny_llm.config import Config
from tiny_llm.data import (
    BufferedTokenLoader,
    TokenCache,
    fingerprint,
    subset_batch,
    validation_indices,
)
from tiny_llm.model import Llama, token_losses
from tiny_llm.train import evaluate, recipe_identity, train

CPU = torch.device("cpu")
# One physical shard per group; sequence boundaries may cross shard boundaries.
GROUP_SIZE = 1


def collect(
    cache,
    *,
    seed: int | None = 42,
    prefetch=True,
    cursor=0,
    count=13,
    batch=5,
    group_size=GROUP_SIZE,
):
    with BufferedTokenLoader(
        cache, "train", 4, count, group_size, seed=seed, prefetch=prefetch, cursor=cursor
    ) as loader:
        pairs = []
        while loader.cursor < count:
            x, y = loader.next_batch(min(batch, count - loader.cursor), CPU)
            pairs.extend(zip(x.tolist(), y.tolist(), strict=True))
        return pairs


def test_coverage_boundaries_and_seeds(cache_dir: Path):
    prefetch = True
    cache = TokenCache(cache_dir)
    expected = cache.batch("train", list(range(14)), 4, CPU)
    expected = list(zip(expected[0].tolist(), expected[1].tolist(), strict=True))
    actual = collect(cache, prefetch=prefetch, count=14)
    assert sorted(actual) == sorted(expected)
    assert actual == collect(cache, prefetch=not prefetch, batch=1, count=14)
    assert actual != collect(cache, seed=43, prefetch=prefetch, count=14)
    assert collect(cache, seed=None, prefetch=prefetch, count=14) == expected


def test_resume_every_cursor_without_reading_previous_ranges(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    cache = TokenCache(cache_dir)
    expected = collect(cache)
    reads = []
    original = cache.read_into

    def record(split, start, stop, destination):
        reads.append((start, stop))
        return original(split, start, stop, destination)

    monkeypatch.setattr(cache, "read_into", record)
    for cursor in range(14):
        reads.clear()
        assert collect(cache, cursor=cursor, prefetch=False) == expected[cursor:]
        with BufferedTokenLoader(
            cache, "train", 4, 13, GROUP_SIZE, seed=42, cursor=cursor
        ) as loader:
            remaining = loader.groups[loader.group_position :] if cursor < 13 else []
            assert reads == [(first * 4, end * 4 + 1) for _, first, end in remaining]
            # State is small metadata only, independent of prefetch timing.
            assert loader.state_dict()["cursor"] == cursor
            loader.validate_state(loader.state_dict())
            with pytest.raises(ValueError, match="incompatible"):
                loader.validate_state({**loader.state_dict(), "seed": 99})


def test_contiguous_file_reads_and_two_buffer_residency(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    cache = TokenCache(cache_dir)
    original_open = Path.open
    original_read = cache.read_into
    references, ranges, file_reads = [], [], []
    high_water = 0

    class Reader:
        def __init__(self, path, handle):
            self.path, self.handle = path, handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def seek(self, offset):
            return self.handle.seek(offset)

        def readinto(self, output):
            start = self.handle.tell()
            # Deliberate short reads verify the loop doesn't seek or skip bytes.
            size = self.handle.readinto(output[:3])
            file_reads.append((self.path, start, size))
            return size

    def open_file(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        return Reader(path, handle) if path.suffix == ".bin" else handle

    def record(split, start, stop, destination):
        nonlocal high_water
        references.append(weakref.ref(destination))
        high_water = max(high_water, sum(ref() is not None for ref in references))
        assert destination.dtype == np.dtype("<u2")
        assert destination.nbytes <= 26
        ranges.append((start, stop))
        return original_read(split, start, stop, destination)

    monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(cache, "read_into", record)
    collect(cache, prefetch=True, batch=2)
    assert high_water <= 2
    assert all(ref() is None for ref in references)
    # Every range reads its entire contiguous span, including exactly one halo.
    expected = []
    for start, stop in ranges:
        while start < stop:
            shard = int(np.searchsorted(cache.offsets["train"], start, side="right") - 1)
            end = min(stop, cache.offsets["train"][shard + 1])
            offset = (start - cache.offsets["train"][shard]) * 2
            remaining = (end - start) * 2
            while remaining:
                count = min(remaining, 3)
                expected.append((cache.paths["train"][shard], offset, count))
                offset += count
                remaining -= count
            start = end
    assert file_reads == expected
    assert not any(t.name.startswith("token-reader") for t in threading.enumerate())


def test_reader_failure_and_interruption_cleanup(cache_dir: Path, monkeypatch: pytest.MonkeyPatch):
    cache = TokenCache(cache_dir)
    original = cache.read_into

    def fail_second(split, start, stop, destination):
        if start == 12:
            raise OSError("simulated reader failure")
        return original(split, start, stop, destination)

    monkeypatch.setattr(cache, "read_into", fail_second)
    loader = BufferedTokenLoader(cache, "train", 4, 13, GROUP_SIZE)
    with pytest.raises(OSError, match="simulated reader failure"):
        with loader:
            loader.next_batch(5, CPU)
    assert loader._active is None and loader._future is None
    assert not any(t.name.startswith("token-reader") for t in threading.enumerate())
    monkeypatch.setattr(cache, "read_into", original)
    for cursor in (2, 3, 4):
        loader = BufferedTokenLoader(cache, "train", 4, 13, GROUP_SIZE)
        with pytest.raises(InterruptedError):
            with loader:
                loader.next_batch(cursor, CPU)
                raise InterruptedError
        loader.close()  # Idempotent even at a prefetch transition.
        assert loader._active is None and loader._future is None
        assert not any(t.name.startswith("token-reader") for t in threading.enumerate())
    cache.paths["train"][0].write_bytes(b"\0\0")
    with pytest.raises(OSError, match="unexpected EOF"):
        cache.read("train", 0, 5)


def test_validation_scan_subset_reuse_and_loss(
    cache_dir: Path, tiny_config: Config, monkeypatch: pytest.MonkeyPatch
):
    cache = TokenCache(cache_dir)
    tiny_config.data.shuffle_group_size = GROUP_SIZE
    indices = validation_indices(cache, tiny_config, full=False)
    expected_x, expected_y = cache.batch("validation", indices, 4, CPU)
    original = cache.read_into
    reads = []

    def record(split, start, stop, destination):
        reads.append((start, stop))
        return original(split, start, stop, destination)

    monkeypatch.setattr(cache, "read_into", record)
    tokens, valid = cache.validation_subset(tiny_config)
    assert reads == [(0, 13), (12, 25), (24, 30)]
    assert tokens.dtype == np.dtype("<u2")
    assert tokens.shape == (3, 5)
    actual_x, actual_y = subset_batch(tokens, valid, CPU)
    assert torch.equal(actual_x, expected_x) and torch.equal(actual_y, expected_y)
    model = Llama(tiny_config.model, "reference")
    with torch.no_grad():
        expected_loss = token_losses(model(expected_x), expected_y).mean().item()
    result = evaluate(model, cache, tiny_config, CPU, full=False)
    assert result["loss"] == pytest.approx(expected_loss, abs=1e-6)
    assert reads == [(0, 13), (12, 25), (24, 30)]
    expected = cache.batch("validation", list(range(8)), 4, CPU)
    with BufferedTokenLoader(cache, "validation", 4, 8, GROUP_SIZE) as loader:
        x, y = loader.next_batch(8, CPU)
    assert torch.equal(x, expected[0]) and torch.equal(y, expected[1])
    assert (y != -100).sum() == 29


def test_ordering_compatibility_and_legacy_rejection(cache_dir: Path, tiny_config: Config):
    cache = TokenCache(cache_dir)
    identity = recipe_identity(tiny_config, cache)
    tiny_config.data.prefetch = False
    assert recipe_identity(tiny_config, cache) == identity
    tiny_config.data.shuffle_group_size = GROUP_SIZE
    assert recipe_identity(tiny_config, cache) != identity
    checkpoint = cache_dir.parent / "legacy.pt"
    torch.save({"version": 1}, checkpoint)
    with pytest.raises(ValueError, match="incompatible training checkpoint version"):
        train(tiny_config, checkpoint)


@pytest.mark.parametrize("group_size,seed", [(1, 0), (2, 42), (4, 43)])
def test_every_short_budget_is_prefix_of_long_run(cache_dir: Path, group_size, seed):
    cache = TokenCache(cache_dir)
    expected = collect(cache, seed=seed, count=33, group_size=group_size, prefetch=False)
    for count in range(1, 34):
        actual = collect(cache, seed=seed, count=count, group_size=group_size, batch=2)
        assert actual == expected[:count]
        cursor = count // 2
        assert (
            collect(cache, seed=seed, count=count, cursor=cursor, group_size=group_size, batch=3)
            == expected[cursor:count]
        )


@pytest.mark.parametrize("group_size", [1, 2, 4])
def test_shard_groups_advance_in_file_order(cache_dir: Path, group_size):
    cache = TokenCache(cache_dir)
    with BufferedTokenLoader(cache, "train", 4, 33, group_size, seed=42) as loader:
        for _, first, end in loader.groups:
            actual = loader.next_batch(end - first, CPU)
            expected = cache.batch("train", list(range(first, end)), 4, CPU)
            assert sorted(zip(*[v.tolist() for v in actual], strict=True)) == sorted(
                zip(*[v.tolist() for v in expected], strict=True)
            )


def test_incomplete_final_shuffle_group_is_rejected(cache_dir: Path):
    import json

    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Remove only lookahead: the nominal budget still fits, but shuffling the
    # final group would otherwise depend on where preparation stopped.
    split = manifest["splits"]["train"]
    split["shards"] = split["shards"][:-1]
    split["tokens"] = 132
    manifest["identity"] = fingerprint(
        {key: value for key, value in manifest.items() if key != "identity"}
    )
    manifest_path.write_text(json.dumps(manifest))
    cache = TokenCache(cache_dir)
    assert cache.blocks("train", 4) == 32
    with pytest.raises(ValueError, match="complete shuffle groups"):
        collect(cache, count=32, group_size=2)
    # Earlier complete groups are still usable.
    assert len(collect(cache, count=27, group_size=2)) == 27


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True, "2"])
def test_shuffle_group_size_must_be_positive_integer(invalid):
    with pytest.raises(ValueError):
        Config.model_validate({"data": {"shuffle_group_size": invalid}})


def test_shuffle_group_config_default_serialization_and_legacy_rejection():
    assert Config().data.shuffle_group_size == 2
    cfg = Config.model_validate({"data": {"shuffle_group_size": 3}})
    assert Config.model_validate_json(cfg.model_dump_json()) == cfg
    with pytest.raises(ValueError, match="extra_forbidden"):
        Config.model_validate({"data": {"buffer_size_mib": 64}})
