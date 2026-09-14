import threading
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch

from tiny_llm.config import Config
from tiny_llm.data import BufferedTokenLoader, TokenCache, subset_batch, validation_indices
from tiny_llm.model import Llama, token_losses
from tiny_llm.train import evaluate, recipe_identity, train

CPU = torch.device("cpu")
# Three four-token sequences, plus a separate lookahead token.
SIZE = 24 / 2**20


def collect(cache, *, seed: int | None = 42, prefetch=True, cursor=0, count=13, batch=5):
    with BufferedTokenLoader(
        cache, "train", 4, count, SIZE, seed=seed, prefetch=prefetch, cursor=cursor
    ) as loader:
        pairs = []
        while loader.cursor < count:
            x, y = loader.next_batch(min(batch, count - loader.cursor), CPU)
            pairs.extend(zip(x.tolist(), y.tolist(), strict=True))
        return pairs


@pytest.mark.parametrize("prefetch", [False, True])
def test_coverage_boundaries_and_seeds(cache_dir: Path, prefetch):
    cache = TokenCache(cache_dir)
    expected = cache.batch("train", list(range(13)), 4, CPU)
    expected = list(zip(expected[0].tolist(), expected[1].tolist(), strict=True))
    actual = collect(cache, prefetch=prefetch)
    assert sorted(actual) == sorted(expected)
    assert actual == collect(cache, prefetch=not prefetch, batch=1)
    assert actual != collect(cache, seed=43, prefetch=prefetch)
    assert collect(cache, seed=None, prefetch=prefetch) == expected


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
        with BufferedTokenLoader(cache, "train", 4, 13, SIZE, seed=42, cursor=cursor) as loader:
            remaining = loader.order[loader.range_position :]
            assert reads == [(int(i) * 12, min(13, (int(i) + 1) * 3) * 4 + 1) for i in remaining]
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
    loader = BufferedTokenLoader(cache, "train", 4, 13, SIZE)
    with pytest.raises(OSError, match="simulated reader failure"):
        with loader:
            loader.next_batch(5, CPU)
    assert loader._active is None and loader._future is None
    assert not any(t.name.startswith("token-reader") for t in threading.enumerate())
    monkeypatch.setattr(cache, "read_into", original)
    for cursor in (2, 3, 4):
        loader = BufferedTokenLoader(cache, "train", 4, 13, SIZE)
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
    tiny_config.data.buffer_size_mib = SIZE
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
    with BufferedTokenLoader(cache, "validation", 4, 8, SIZE) as loader:
        x, y = loader.next_batch(8, CPU)
    assert torch.equal(x, expected[0]) and torch.equal(y, expected[1])
    assert (y != -100).sum() == 29


def test_ordering_compatibility_and_legacy_rejection(cache_dir: Path, tiny_config: Config):
    cache = TokenCache(cache_dir)
    identity = recipe_identity(tiny_config, cache)
    tiny_config.data.prefetch = False
    assert recipe_identity(tiny_config, cache) == identity
    tiny_config.data.buffer_size_mib = SIZE
    assert recipe_identity(tiny_config, cache) != identity
    checkpoint = cache_dir.parent / "legacy.pt"
    torch.save({"version": 1}, checkpoint)
    with pytest.raises(ValueError, match="incompatible training checkpoint version"):
        train(tiny_config, checkpoint)
