import multiprocessing
import sys
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

import tiny_llm.data as data
from tiny_llm.config import Config
from tiny_llm.state import TokenArray


class IntegerTokenizer:
    eos_token_id = 2

    def __call__(self, texts, *, add_special_tokens, return_attention_mask):
        assert add_special_tokens is False
        assert return_attention_mask is False
        return {"input_ids": [[int(token) for token in text.split()] for text in texts]}


@pytest.fixture
def offline_source(monkeypatch: pytest.MonkeyPatch):
    import huggingface_hub
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({str(i): i for i in range(17)}, unk_token="0"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="0", eos_token="2")
    documents = {split: ["3 4 5", "", "6 7"] * 10 + ["8"] for split in ("train", "validation")}
    closed = []
    shuffles = []

    class API:
        def dataset_info(self, *args, **kwargs):
            return SimpleNamespace(sha="fixture-dataset")

        def model_info(self, *args, **kwargs):
            return SimpleNamespace(sha="fixture-tokenizer")

        def list_repo_files(self, *args, **kwargs):
            return [f"en/c4-{split}.00000.json.gz" for split in documents]

    class Stream:
        def __init__(self, split):
            self.split = split
            self.texts = list(documents[split])

        def shuffle(self, seed, buffer_size):
            shuffles.append((self.split, seed, buffer_size))
            order = np.random.default_rng(seed).permutation(len(self.texts))
            self.texts = [self.texts[index] for index in order]
            return self

        def iter(self, batch_size):
            try:
                for offset in range(0, len(self.texts), batch_size):
                    yield {"text": self.texts[offset : offset + batch_size]}
            finally:
                closed.append(self.split)

    def load(*args, **kwargs):
        assert kwargs["streaming"] is True
        assert len(kwargs["data_files"][kwargs["split"]]) == 1
        return Stream(kwargs["split"])

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load))
    return SimpleNamespace(
        tokenizer=tokenizer, documents=documents, closed=closed, shuffles=shuffles
    )


def forbid_pool(*args, **kwargs):
    raise AssertionError("serial preparation and cache reuse must not start workers")


@pytest.mark.integration
def test_spawned_preparation_matches_serial(
    tiny_config: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offline_source,
):
    tiny_config.data.tokenize_batch_size = 4  # 31 documents leave a partial final batch.
    tiny_config.data.prepare_validation_tokens = 13
    tiny_config.data.prepare_workers = 1
    serial_path = tiny_config.data.cache_dir
    with monkeypatch.context() as patch:
        patch.setattr(data, "ProcessPoolExecutor", forbid_pool)
        serial = data.prepare(tiny_config)

    tiny_config.data.cache_dir = tmp_path / "parallel"
    tiny_config.data.prepare_workers = 2
    children_before = {child.pid for child in multiprocessing.active_children()}
    parallel = data.prepare(tiny_config)
    assert {child.pid for child in multiprocessing.active_children()} <= children_before
    assert parallel == serial
    assert "prepare_workers" not in parallel["preprocessing"]
    cache = data.TokenCache(tiny_config.data.cache_dir)
    cache.verify()
    for info in parallel["splits"].values():
        for shard in info["shards"]:
            assert (serial_path / shard["file"]).read_bytes() == (
                cache.directory / shard["file"]
            ).read_bytes()
    assert parallel["splits"]["train"]["tokens"] == 69
    assert parallel["splits"]["validation"]["tokens"] == 13
    assert not parallel["validation_complete"]
    np.testing.assert_array_equal(cache.read("validation", 0, 8), [3, 4, 5, 2, 2, 6, 7, 2])
    assert offline_source.closed == ["train", "validation"] * 2
    assert offline_source.shuffles == [("train", 42, 100000)] * 2

    tiny_config.data.prepare_workers = 8
    with monkeypatch.context() as patch:
        patch.setattr(data, "ProcessPoolExecutor", forbid_pool)
        assert data.prepare(tiny_config) == parallel


def test_parallel_batches_are_ordered_and_bounded(monkeypatch: pytest.MonkeyPatch):
    later_finished = Event()
    completion_order = []
    pulled = consumed = max_ahead = 0

    def encode(texts):
        index = int(texts[0])
        if index == 0:
            assert later_finished.wait(timeout=10)
        completion_order.append(index)
        if index == 1:
            later_finished.set()
        return np.array([index], dtype="<u2")

    def source():
        nonlocal pulled, max_ahead
        for index in range(20):
            pulled += 1
            max_ahead = max(max_ahead, pulled - consumed)
            assert pulled - consumed <= 4
            yield [str(index)]

    monkeypatch.setattr(data, "_tokenize_worker", encode)
    output = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        with closing(data._tokenized_batches(source(), IntegerTokenizer(), pool, 2)) as batches:
            for values in batches:
                output.extend(values.tolist())
                consumed += 1
    assert output == list(range(20))
    assert completion_order.index(1) < completion_order.index(0)
    assert max_ahead == 4


def test_early_close_cancels_pending_and_drains_running_work():
    cancelled = Event()

    class RecordingFuture(Future[TokenArray]):
        def cancel(self):
            result = super().cancel()
            if result:
                cancelled.set()
            return result

    class ControlledPool(Executor):
        def __init__(self):
            self.futures: list[RecordingFuture] = []

        def submit(self, fn, /, *args, **kwargs):
            future = RecordingFuture()
            self.futures.append(future)
            if len(self.futures) == 1:
                future.set_result(np.array([0], dtype="<u2"))
            elif len(self.futures) == 2:
                future.set_running_or_notify_cancel()
            return future

    pool = ControlledPool()
    batches = data._tokenized_batches(([str(i)] for i in range(100)), IntegerTokenizer(), pool, 2)
    np.testing.assert_array_equal(next(batches), [0])
    assert len(pool.futures) == 4
    with ThreadPoolExecutor(max_workers=1) as closer:
        closed = closer.submit(batches.close)
        try:
            assert cancelled.wait(timeout=10)
            assert not closed.done()
        finally:
            pool.futures[1].set_result(np.array([1], dtype="<u2"))
        closed.result(timeout=10)
    assert all(future.cancelled() for future in pool.futures[2:])
    assert len(pool.futures) == 4


def test_tokenization_rejects_uint16_overflow():
    with pytest.raises(ValueError, match="uint16"):
        data._tokenize_documents(IntegerTokenizer(), ["65536"])


def test_token_writer_closes_shard_after_fsync_failure(tmp_path: Path, monkeypatch):
    writer = data.TokenWriter(tmp_path, "train", shard_tokens=2)
    handles = []

    def fail_fsync(fd):
        handles.append(writer.handle)
        raise OSError("fixture fsync failure")

    monkeypatch.setattr(data.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="fsync failure"):
        writer.write([3, 4])
    assert writer.handle is None
    assert handles and all(handle.closed for handle in handles)
