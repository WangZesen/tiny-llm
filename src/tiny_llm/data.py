"""C4 token caches and bounded, sequential readers with in-memory shuffling."""

import hashlib
import json
import math
import multiprocessing
import os
import shutil
import time
from bisect import bisect_right
from collections import deque
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import closing, nullcontext
from fractions import Fraction
from itertools import islice
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
        try:
            self.handle.flush()
            os.fsync(self.handle.fileno())
        finally:
            self.close()
        self.shards.append(
            Shard(file=self.path.name, tokens=self.in_shard, sha256=file_digest(self.path))
        )

    def close(self) -> None:
        """Release an open shard, including when preparation fails."""
        if self.handle is not None:
            handle, self.handle = self.handle, None
            handle.close()

    def finish(self) -> SplitManifest:
        self._close_shard()
        return SplitManifest(tokens=self.count, shards=self.shards)


class _DocumentTokenizer(Protocol):
    @property
    def eos_token_id(self) -> int | None: ...

    def __call__(
        self, texts: list[str], *, add_special_tokens: bool, return_attention_mask: bool
    ) -> Mapping[str, Sequence[Sequence[int]]]: ...


_worker_tokenizer: _DocumentTokenizer | None = None


def _initialize_tokenizer(tokenizer_path: str) -> None:
    # Each spawned worker owns one tokenizer and uses a single tokenizer thread.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    global _worker_tokenizer
    _worker_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=True, local_files_only=True
    )


def _tokenize_documents(tokenizer: _DocumentTokenizer, texts: list[str]) -> TokenArray:
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer must define EOS")
    encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    values = np.asarray(
        [token for document in encoded for token in (*document, eos)], dtype=np.int64
    )
    if values.size and (values.min() < 0 or values.max() > 65535):
        raise ValueError("token ID cannot be represented as uint16")
    return values.astype("<u2")


def _tokenize_worker(texts: list[str]) -> TokenArray:
    if _worker_tokenizer is None:
        raise RuntimeError("tokenizer worker is not initialized")
    return _tokenize_documents(_worker_tokenizer, texts)


def _tokenized_batches(
    batches: Iterable[list[str]],
    tokenizer: _DocumentTokenizer,
    pool: Executor | None,
    workers: int,
) -> Generator[TokenArray, None, None]:
    """Consume bounded parallel work in source order, independent of worker timing."""
    if pool is None:
        for batch in batches:
            yield _tokenize_documents(tokenizer, batch)
        return

    source = iter(batches)
    pending: deque[Future[TokenArray]] = deque()
    try:
        for batch in islice(source, 2 * workers):
            pending.append(pool.submit(_tokenize_worker, batch))
        while pending:
            # Keep the current future in the queue until it completes so an
            # interruption also drains this task before the next split starts.
            values = pending[0].result()
            pending.popleft()
            yield values
            for batch in islice(source, 1):
                pending.append(pool.submit(_tokenize_worker, batch))
    finally:
        for future in pending:
            future.cancel()
        wait([future for future in pending if not future.cancelled()])


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
    length = config.model.context_length
    required_blocks = (required + length - 1) // length
    # Groups own sequences by their start token. Include the whole last group,
    # align its end to a sequence boundary, and retain the lookahead token.
    group_tokens = cfg.shuffle_group_size * cfg.shard_tokens
    group_end = ((required_blocks - 1) * length // group_tokens + 1) * group_tokens
    train_limit = ((group_end + length - 1) // length) * length + 1
    destination = cfg.cache_dir
    if destination.exists():
        cache = TokenCache(destination)
        cache.validate_config(config)
        groups = shuffle_groups(cache, "train", length, cfg.shuffle_group_size, complete=True)
        if not groups or groups[-1][2] < required_blocks:
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
        pool_context = (
            ProcessPoolExecutor(
                max_workers=cfg.prepare_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_tokenizer,
                initargs=(str((temporary / "tokenizer").resolve()),),
            )
            if cfg.prepare_workers > 1
            else nullcontext(None)
        )
        with pool_context as pool:
            for split, limit in (
                ("train", train_limit),
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
                with (
                    closing(stream.iter(batch_size=cfg.tokenize_batch_size)) as source,
                    closing(
                        _tokenized_batches(
                            (batch["text"] for batch in source),
                            tokenizer,
                            pool,
                            cfg.prepare_workers,
                        )
                    ) as batches,
                    closing(writer),
                ):
                    for flattened in batches:
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
        for _, first, end in shuffle_groups(
            self, "validation", length, config.data.shuffle_group_size
        ):
            if should_stop is not None and should_stop():
                raise InterruptedError("validation subset preparation interrupted")
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


class TokenSource(Protocol):
    manifest: CacheManifest
    offsets: dict[str, list[int]]

    def blocks(self, split: str, length: int, partial: bool = False) -> int: ...

    def read_into(self, split: str, start: int, stop: int, destination: TokenArray) -> None: ...


def shuffle_groups(
    cache: TokenSource, split: str, length: int, group_size: int, *, complete: bool = False
) -> list[tuple[int, int, int]]:
    """Return (physical group ID, first block, end block), independent of run budget.

    A sequence belongs to the group containing its first token. Seeded training
    needs every shard of a group and its aligned lookahead, so extending a cache
    cannot change a previously usable group's permutation domain.
    """
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("shuffle group size must be a positive integer")
    offsets = cache.offsets[split]
    total = cache.blocks(split, length, partial=True)
    groups = []
    for shard in range(0, len(offsets) - 1, group_size):
        last = min(shard + group_size, len(offsets) - 1)
        first = (offsets[shard] + length - 1) // length
        end = (offsets[last] + length - 1) // length
        if complete and (last - shard < group_size or end * length + 1 > offsets[-1]):
            break
        end = min(end, total)
        if first < end:
            groups.append((shard // group_size, first, end))
    return groups


class BufferedTokenLoader:
    """Fill, shuffle by row index, and drain; at most two compact token buffers.

    Consecutive shard groups are visited in file order. Each group contains
    original sequences plus lookahead, with a budget-independent row permutation.
    Only the committed sample cursor is needed to resume, even with prefetching.
    """

    VERSION = 2

    def __init__(
        self,
        cache: TokenSource,
        split: str,
        length: int,
        blocks: int,
        group_size: int = 2,
        seed: int | None = None,
        prefetch: bool = True,
        cursor: int = 0,
    ):
        if not 0 <= cursor <= blocks or not 0 < blocks <= cache.blocks(split, length, partial=True):
            raise ValueError("invalid buffered loader block budget or cursor")
        self.cache, self.split, self.length = cache, split, length
        self.blocks, self.cursor, self.seed = blocks, cursor, seed
        available = shuffle_groups(cache, split, length, group_size, complete=seed is not None)
        if not available or available[-1][2] < blocks:
            raise ValueError(
                "cache too small for complete shuffle groups; prepare a larger cache at a new path"
            )
        self.groups = [group for group in available if group[1] < blocks]
        self.ends = np.array([end for _, _, end in self.groups], dtype=np.int64)
        self.group_position = int(np.searchsorted(self.ends, cursor, side="right"))
        self.identity: LoaderIdentity = LoaderIdentity(
            version=self.VERSION,
            cache_identity=cache.manifest["identity"],
            split=split,
            length=length,
            blocks=blocks,
            shuffle_group_size=group_size,
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
        _, first, end = self.groups[position]
        values = np.empty((end - first) * self.length + 1, dtype="<u2")
        stop = min(end * self.length + 1, self.cache.offsets[self.split][-1])
        self.cache.read_into(self.split, first * self.length, stop, values)
        values[stop - first * self.length :] = 0
        return values

    def _activate(self) -> tuple[TokenArray, IndexArray]:
        # Release all views before promoting the future and scheduling another read.
        self._windows = self._active = self._row_order = None
        if self._future is None:
            self._active = self._load(self.group_position)
        else:
            self._active = self._future.result()
            self._future = None
        count = (len(self._active) - 1) // self.length
        self._windows = np.lib.stride_tricks.sliding_window_view(self._active, self.length + 1)[
            :: self.length
        ]
        group_id = self.groups[self.group_position][0]
        self._row_order = (
            np.arange(count)
            if self.seed is None
            else np.random.default_rng(
                np.random.SeedSequence([self.seed, 1, group_id])
            ).permutation(count)
        )
        if self.prefetch and self.group_position + 1 < len(self.groups):
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="token-reader"
                )
            self._future = self._executor.submit(self._load, self.group_position + 1)

        return self._windows, self._row_order

    def next_batch(self, count: int, device: torch.device) -> Batch:
        if self._closed:
            raise RuntimeError("loader is closed")
        if count <= 0 or self.cursor + count > self.blocks:
            raise ValueError("batch exceeds remaining token blocks")
        host_batch: Batch | None = None
        if device.type == "cuda":
            # Fill pinned storage directly instead of copying ordinary arrays into it.
            shape = (count, self.length)
            host_batch = (
                torch.empty(shape, dtype=torch.int64, pin_memory=True),
                torch.empty(shape, dtype=torch.int64, pin_memory=True),
            )
            inputs, targets = (tensor.numpy() for tensor in host_batch)
        else:
            inputs = np.empty((count, self.length), dtype=np.int64)
            targets = np.empty_like(inputs)
        filled = 0
        while filled < count:
            if self._windows is None or self._row_order is None:
                windows, row_order = self._activate()
            else:
                windows, row_order = self._windows, self._row_order
            start = int(self.ends[self.group_position - 1]) if self.group_position else 0
            offset = self.cursor - start
            take = min(count - filled, int(self.ends[self.group_position]) - self.cursor)
            rows = row_order[offset : offset + take]
            values = windows[rows]  # Only this microbatch is gathered/copied.
            inputs[filled : filled + take] = values[:, :-1]
            targets[filled : filled + take] = values[:, 1:]
            first = self.groups[self.group_position][1]
            if (first + len(row_order)) * self.length >= self.cache.offsets[self.split][-1]:
                valid = self.cache.offsets[self.split][-1] - 1 - (first + rows) * self.length
                mask = np.arange(self.length)[None, :] >= valid[:, None]
                inputs[filled : filled + take][mask] = 0
                targets[filled : filled + take][mask] = -100
            self.cursor += take
            filled += take
            if self.cursor == self.ends[self.group_position]:
                self.group_position += 1
                self._windows = self._active = self._row_order = None
            del windows, row_order
        if host_batch is not None:
            return (
                host_batch[0].to(device, non_blocking=True),
                host_batch[1].to(device, non_blocking=True),
            )
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
