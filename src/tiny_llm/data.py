"""C4 token caches and bounded, sequential readers with in-memory shuffling."""

import hashlib
import json
import math
import os
import shutil
import time
from bisect import bisect_right
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO, Protocol

import numpy as np
import torch
from loguru import logger
from numpy.typing import ArrayLike

from tiny_llm.config import Config
from tiny_llm.state import (
    Batch,
    CacheContent,
    CacheManifest,
    IndexArray,
    LoaderIdentity,
    LoaderState,
    Shard,
    SplitManifest,
    TokenArray,
)


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class TokenWriter:
    def __init__(self, directory: Path, split: str, shard_tokens: int):
        self.directory, self.split, self.shard_tokens = directory, split, shard_tokens
        self.shards: list[Shard] = []
        self.count = 0
        self.handle: BinaryIO | None = None
        self.in_shard = 0

    def write(self, tokens: ArrayLike) -> None:
        values = np.asarray(tokens)
        if values.size and (values.min() < 0 or values.max() > 65535):
            raise ValueError("token ID cannot be represented as uint16")
        values = values.astype("<u2")
        while len(values):
            if self.handle is None:
                self.path = self.directory / f"{self.split}-{len(self.shards):05d}.bin"
                self.handle = self.path.open("wb")
                self.in_shard = 0
            take = min(len(values), self.shard_tokens - self.in_shard)
            self.handle.write(values[:take].tobytes())
            self.in_shard += take
            self.count += take
            values = values[take:]
            if self.in_shard == self.shard_tokens:
                self._close_shard()

    def _close_shard(self) -> None:
        if self.handle is None:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.shards.append(
            Shard(file=self.path.name, tokens=self.in_shard, sha256=file_digest(self.path))
        )
        self.handle = None

    def finish(self) -> SplitManifest:
        self._close_shard()
        return SplitManifest(tokens=self.count, shards=self.shards)


def training_boundaries(config: Config, parameters: int) -> list[int]:
    """Nearest cumulative global batches, independent of the final epoch count."""
    length = config.model.context_length
    train = config.training
    epoch_batches = (
        Fraction(str(train.epoch_tokens_per_parameters)) * parameters / train.batch_tokens
    )
    batch_blocks = train.batch_tokens // length
    boundaries = [
        math.floor(k * epoch_batches + Fraction(1, 2)) * batch_blocks
        for k in range(1, train.epoch_count + 1)
    ]
    if boundaries[0] == 0 or len(set(boundaries)) != len(boundaries):
        raise ValueError("virtual epochs must each contain at least one global batch")
    return boundaries


def prepare(config: Config) -> CacheManifest:
    """Prepare privately, then atomically publish both splits and their manifest."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    cfg = config.data
    required = (
        cfg.prepare_train_tokens
        or training_boundaries(config, config.model.parameter_count)[-1]
        * config.model.context_length
    )
    destination = cfg.cache_dir
    if destination.exists():
        cache = TokenCache(destination)
        cache.validate_config(config)
        if cache.manifest["splits"]["train"]["tokens"] < required + 1:
            raise ValueError("existing cache is too small; prepare a larger cache at a new path")
        if cfg.prepare_validation_tokens is None and not cache.manifest["validation_complete"]:
            raise ValueError("existing cache has truncated validation; use a new cache path")
        cache.verify()
        logger.info("Verified and reused cache {}", destination)
        return cache.manifest

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    # An exclusive lock protects preparation shared by multiple GPU processes.
    lock_path = destination.with_name(destination.name + ".lock")
    import fcntl

    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if temporary.exists():
            logger.warning(
                "Discarding incomplete preparation at {}; completed caches are immutable", temporary
            )
            shutil.rmtree(temporary)
        temporary.mkdir()
        api = HfApi()
        dataset_sha = api.dataset_info(cfg.dataset, revision=cfg.revision).sha
        tokenizer_sha = api.model_info(cfg.tokenizer, revision=cfg.tokenizer_revision).sha
        if dataset_sha is None or tokenizer_sha is None:
            raise ValueError("could not resolve dataset/tokenizer commit SHA")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer, revision=tokenizer_sha, use_fast=True
        )
        if len(tokenizer) != config.model.vocab_size or tokenizer.eos_token_id is None:
            raise ValueError("model vocabulary must match the tokenizer, which must define EOS")
        tokenizer.save_pretrained(temporary / "tokenizer")
        tokenizer_hash = fingerprint(
            {
                "vocab": tokenizer.get_vocab(),
                "eos": tokenizer.eos_token_id,
                "backend": tokenizer.backend_tokenizer.to_str(),
            }
        )
        paths = sorted(api.list_repo_files(cfg.dataset, repo_type="dataset", revision=dataset_sha))
        manifest: CacheContent = {
            "version": 1,
            "dataset": cfg.dataset,
            "dataset_revision": dataset_sha,
            "tokenizer": cfg.tokenizer,
            "tokenizer_revision": tokenizer_sha,
            "tokenizer_fingerprint": tokenizer_hash,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "dtype": "<u2",
            "preprocessing": {
                "shuffle_seed": cfg.shuffle_seed,
                "shuffle_buffer": cfg.shuffle_buffer,
                "append_eos": True,
                "add_special_tokens": False,
            },
            "validation_complete": cfg.prepare_validation_tokens is None,
            "splits": {},
        }
        for split, limit in (
            ("train", required + 1),
            ("validation", cfg.prepare_validation_tokens),
        ):
            selected = [
                p for p in paths if p.startswith(f"en/c4-{split}.") and p.endswith(".json.gz")
            ]
            if not selected:
                raise ValueError(f"no English C4 {split} shards at {dataset_sha}")
            urls = [f"hf://datasets/{cfg.dataset}@{dataset_sha}/{p}" for p in selected]
            stream = load_dataset("json", data_files={split: urls}, split=split, streaming=True)
            if split == "train":
                stream = stream.shuffle(seed=cfg.shuffle_seed, buffer_size=cfg.shuffle_buffer)
            writer = TokenWriter(temporary, split, cfg.shard_tokens)
            start, last_log = time.monotonic(), 0.0
            for batch in stream.iter(batch_size=cfg.tokenize_batch_size):
                encoded = tokenizer(
                    batch["text"], add_special_tokens=False, return_attention_mask=False
                )["input_ids"]
                flattened = [
                    token for document in encoded for token in (*document, tokenizer.eos_token_id)
                ]
                if limit is not None:
                    flattened = flattened[: max(0, limit - writer.count)]
                writer.write(flattened)
                now = time.monotonic()
                if now - last_log > 15:
                    logger.info(
                        "Preparing {}: {:,} tokens ({:,.0f} tokens/s)",
                        split,
                        writer.count,
                        writer.count / (now - start),
                    )
                    last_log = now
                if limit is not None and writer.count >= limit:
                    break
            manifest["splits"][split] = writer.finish()
            if writer.count < 2 or (limit is not None and writer.count < limit):
                raise ValueError(f"insufficient {split} tokens: {writer.count}")
        completed_manifest = CacheManifest(**manifest, identity=fingerprint(manifest))
        (temporary / "manifest.json").write_text(json.dumps(completed_manifest, indent=2) + "\n")
        temporary.rename(destination)
        logger.success("Published cache {} ({})", destination, completed_manifest["identity"][:12])
        return completed_manifest


class TokenCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.manifest: CacheManifest = json.loads((self.directory / "manifest.json").read_text())
        content = {k: v for k, v in self.manifest.items() if k != "identity"}
        if self.manifest.get("version") != 1 or fingerprint(content) != self.manifest.get(
            "identity"
        ):
            raise ValueError("invalid cache manifest version or identity")
        self.paths: dict[str, list[Path]] = {}
        self.offsets: dict[str, list[int]] = {}
        self._subset_key: tuple[int, int, int] | None = None
        self._subset: tuple[TokenArray, IndexArray] | None = None
        for split, info in self.manifest["splits"].items():
            offsets, paths = [0], []
            for shard in info["shards"]:
                path = self.directory / shard["file"]
                if path.stat().st_size != shard["tokens"] * 2:
                    raise ValueError(f"truncated token shard: {path}")
                paths.append(path)
                offsets.append(offsets[-1] + shard["tokens"])
            if offsets[-1] != info["tokens"]:
                raise ValueError("manifest token count does not match shards")
            self.paths[split], self.offsets[split] = paths, offsets

    def validate_config(self, config: Config) -> None:
        data, manifest = config.data, self.manifest
        if manifest["vocab_size"] != config.model.vocab_size:
            raise ValueError("cache/model vocabulary mismatch")
        for key, value in (("dataset", data.dataset), ("tokenizer", data.tokenizer)):
            if manifest[key] != value:
                raise ValueError(f"cache {key} mismatch")
        for expected, key in (
            (data.revision, "dataset_revision"),
            (data.tokenizer_revision, "tokenizer_revision"),
        ):
            if expected != "main" and expected != manifest[key]:
                raise ValueError(f"cache {key} mismatch; specify the resolved commit SHA")
        if (
            manifest["preprocessing"]["shuffle_seed"] != data.shuffle_seed
            or manifest["preprocessing"]["shuffle_buffer"] != data.shuffle_buffer
        ):
            raise ValueError("cache preprocessing mismatch")

    def verify(self) -> None:
        for info in self.manifest["splits"].values():
            for shard in info["shards"]:
                if file_digest(self.directory / shard["file"]) != shard["sha256"]:
                    raise ValueError(f"checksum mismatch: {shard['file']}")

    def blocks(self, split: str, length: int, partial: bool = False) -> int:
        targets = self.manifest["splits"][split]["tokens"] - 1
        return math.ceil(targets / length) if partial else targets // length

    def read(self, split: str, start: int, stop: int) -> TokenArray:
        """Read a contiguous interval into a compact owning array, without mmap."""
        if not 0 <= start < stop <= self.offsets[split][-1]:
            raise IndexError("token range outside cache")
        values = np.empty(stop - start, dtype="<u2")
        self.read_into(split, start, stop, values)
        return values

    def read_into(self, split: str, start: int, stop: int, destination: TokenArray) -> None:
        """Fill a preallocated uint16 array using sequential reads within each file."""
        if not 0 <= start < stop <= self.offsets[split][-1]:
            raise IndexError("token range outside cache")
        if destination.dtype != np.dtype("<u2") or not destination.flags.c_contiguous:
            raise ValueError("destination must be a contiguous uint16 array")
        if destination.size < stop - start:
            raise ValueError("destination is too small")
        output = memoryview(destination).cast("B")
        written = 0
        while start < stop:
            index = bisect_right(self.offsets[split], start) - 1
            end = min(stop, self.offsets[split][index + 1])
            remaining = 2 * (end - start)
            with self.paths[split][index].open("rb", buffering=0) as handle:
                handle.seek(2 * (start - self.offsets[split][index]))
                while remaining:
                    size = handle.readinto(output[written : written + remaining])
                    if not size:
                        raise OSError(f"unexpected EOF in token shard: {self.paths[split][index]}")
                    written += size
                    remaining -= size
            start = end

    def batch(
        self, split: str, indices: Sequence[int] | IndexArray, length: int, device: torch.device
    ) -> Batch:
        inputs = np.zeros((len(indices), length), dtype=np.int64)
        targets = np.full_like(inputs, -100)
        count = self.offsets[split][-1]
        for row, index in enumerate(indices):
            start = int(index) * length
            values = self.read(split, start, min(start + length + 1, count))
            inputs[row, : len(values) - 1] = values[:-1]
            targets[row, : len(values) - 1] = values[1:]
        return transfer_batch(inputs, targets, device)

    def validation_subset(
        self, config: Config, should_stop: Callable[[], bool] | None = None
    ) -> tuple[TokenArray, IndexArray]:
        """Collect the fixed subset in one sequential scan, retaining only its tokens."""
        length = config.model.context_length
        key = (length, config.evaluation.seed, config.evaluation.subset_blocks)
        if key == self._subset_key and self._subset is not None:
            return self._subset
        indices = validation_indices(self, config, full=False)
        tokens = np.zeros((len(indices), length + 1), dtype="<u2")
        valid = np.minimum(length, self.offsets["validation"][-1] - 1 - indices * length)
        capacity = buffer_blocks(config.data.buffer_size_mib, length)
        total = self.blocks("validation", length, partial=True)
        for first in range(0, total, capacity):
            if should_stop is not None and should_stop():
                raise InterruptedError("validation subset preparation interrupted")
            end = min(total, first + capacity)
            values = self.read(
                "validation", first * length, min(end * length + 1, self.offsets["validation"][-1])
            )
            left, right = np.searchsorted(indices, [first, end])
            for row in range(left, right):
                offset = (int(indices[row]) - first) * length
                count = int(valid[row]) + 1
                tokens[row, :count] = values[offset : offset + count]
            del values
        self._subset_key, self._subset = key, (tokens, valid)
        return self._subset


def transfer_batch(inputs: IndexArray, targets: IndexArray, device: torch.device) -> Batch:
    result = [torch.from_numpy(values) for values in (inputs, targets)]
    if device.type == "cuda":
        result = [values.pin_memory() for values in result]
    return result[0].to(device, non_blocking=True), result[1].to(device, non_blocking=True)


def subset_batch(tokens: TokenArray, valid: IndexArray, device: torch.device) -> Batch:
    inputs, targets = tokens[:, :-1].astype(np.int64), tokens[:, 1:].astype(np.int64)
    mask = np.arange(targets.shape[1])[None, :] >= valid[:, None]
    inputs[mask], targets[mask] = 0, -100
    return transfer_batch(inputs, targets, device)


def buffer_blocks(size_mib: float, length: int) -> int:
    capacity = int(size_mib * 2**20) // (2 * length)
    if capacity < 1:
        raise ValueError("buffer must hold at least one uint16 sequence")
    return capacity


class TokenSource(Protocol):
    manifest: CacheManifest
    offsets: dict[str, list[int]]

    def blocks(self, split: str, length: int, partial: bool = False) -> int: ...

    def read_into(self, split: str, start: int, stop: int, destination: TokenArray) -> None: ...


class BufferedTokenLoader:
    """Fill, shuffle by row index, and drain; at most two compact token buffers.

    A range contains complete original blocks plus one lookahead token (two extra
    bytes). Range order and row permutations have independent seed namespaces.
    Only the committed sample cursor is needed to resume, even with prefetching.
    """

    VERSION = 1

    def __init__(
        self,
        cache: TokenSource,
        split: str,
        length: int,
        blocks: int,
        size_mib: float = 64,
        seed: int | None = None,
        prefetch: bool = True,
        cursor: int = 0,
    ):
        if not 0 <= cursor <= blocks or not 0 < blocks <= cache.blocks(split, length, partial=True):
            raise ValueError("invalid buffered loader block budget or cursor")
        self.cache, self.split, self.length = cache, split, length
        self.blocks, self.cursor, self.seed = blocks, cursor, seed
        self.capacity = buffer_blocks(size_mib, length)
        ranges = math.ceil(blocks / self.capacity)
        self.order = (
            np.arange(ranges)
            if seed is None
            else np.random.default_rng(np.random.SeedSequence([seed, 0])).permutation(ranges)
        )
        sizes = np.minimum(self.capacity, blocks - self.order * self.capacity)
        self.ends = np.cumsum(sizes)
        self.range_position = int(np.searchsorted(self.ends, cursor, side="right"))
        self.identity: LoaderIdentity = LoaderIdentity(
            version=self.VERSION,
            cache_identity=cache.manifest["identity"],
            split=split,
            length=length,
            blocks=blocks,
            buffer_blocks=self.capacity,
            seed=seed,
        )
        self.prefetch = prefetch
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[TokenArray] | None = None
        self._active: TokenArray | None = None
        self._windows: TokenArray | None = None
        self._row_order: IndexArray | None = None
        self._closed = False

    def _load(self, position: int) -> TokenArray:
        first = int(self.order[position]) * self.capacity
        end = min(self.blocks, first + self.capacity)
        values = np.empty((end - first) * self.length + 1, dtype="<u2")
        stop = min(end * self.length + 1, self.cache.offsets[self.split][-1])
        self.cache.read_into(self.split, first * self.length, stop, values)
        values[stop - first * self.length :] = 0
        return values

    def _activate(self) -> tuple[TokenArray, IndexArray]:
        # Release all views before promoting the future and scheduling another read.
        self._windows = self._active = self._row_order = None
        if self._future is None:
            self._active = self._load(self.range_position)
        else:
            self._active = self._future.result()
            self._future = None
        count = (len(self._active) - 1) // self.length
        self._windows = np.lib.stride_tricks.sliding_window_view(self._active, self.length + 1)[
            :: self.length
        ]
        range_id = int(self.order[self.range_position])
        self._row_order = (
            np.arange(count)
            if self.seed is None
            else np.random.default_rng(
                np.random.SeedSequence([self.seed, 1, range_id])
            ).permutation(count)
        )
        if self.prefetch and self.range_position + 1 < len(self.order):
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="token-reader"
                )
            self._future = self._executor.submit(self._load, self.range_position + 1)

        return self._windows, self._row_order

    def next_batch(self, count: int, device: torch.device) -> Batch:
        if self._closed:
            raise RuntimeError("loader is closed")
        if count <= 0 or self.cursor + count > self.blocks:
            raise ValueError("batch exceeds remaining token blocks")
        inputs = np.empty((count, self.length), dtype=np.int64)
        targets = np.empty_like(inputs)
        filled = 0
        while filled < count:
            if self._windows is None or self._row_order is None:
                windows, row_order = self._activate()
            else:
                windows, row_order = self._windows, self._row_order
            start = int(self.ends[self.range_position - 1]) if self.range_position else 0
            offset = self.cursor - start
            take = min(count - filled, int(self.ends[self.range_position]) - self.cursor)
            rows = row_order[offset : offset + take]
            values = windows[rows]  # Only this microbatch is gathered/copied.
            inputs[filled : filled + take] = values[:, :-1]
            targets[filled : filled + take] = values[:, 1:]
            first = int(self.order[self.range_position]) * self.capacity
            if (first + len(row_order)) * self.length >= self.cache.offsets[self.split][-1]:
                valid = self.cache.offsets[self.split][-1] - 1 - (first + rows) * self.length
                mask = np.arange(self.length)[None, :] >= valid[:, None]
                inputs[filled : filled + take][mask] = 0
                targets[filled : filled + take][mask] = -100
            self.cursor += take
            filled += take
            if self.cursor == self.ends[self.range_position]:
                self.range_position += 1
                self._windows = self._active = self._row_order = None
            del windows, row_order
        return transfer_batch(inputs, targets, device)

    def state_dict(self, committed_cursor: int | None = None) -> LoaderState:
        cursor = self.cursor if committed_cursor is None else committed_cursor
        if not 0 <= cursor <= self.cursor:
            raise ValueError("committed cursor is ahead of the loader")
        return LoaderState(**self.identity, cursor=cursor)

    def validate_state(self, state: LoaderState) -> None:
        if state != self.state_dict():
            raise ValueError("incompatible buffered loader state")

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._future = self._windows = self._active = self._row_order = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def validation_indices(cache: TokenCache, config: Config, full: bool) -> IndexArray:
    length = config.model.context_length
    if full:
        return np.arange(cache.blocks("validation", length, partial=True))
    count = cache.blocks("validation", length)
    if not count:
        return np.arange(cache.blocks("validation", length, partial=True))
    return np.sort(
        np.random.default_rng(config.evaluation.seed).choice(
            count,
            size=min(config.evaluation.subset_blocks, count),
            replace=False,
        )
    )
