"""One-time C4 tokenization and random-access packed token caches."""

import hashlib
import json
import math
import os
import shutil
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from tiny_llm.config import Config


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TokenWriter:
    def __init__(self, directory: Path, split: str, shard_tokens: int):
        self.directory, self.split, self.shard_tokens = directory, split, shard_tokens
        self.shards: list[dict] = []
        self.count = 0
        self.handle = None
        self.in_shard = 0

    def write(self, tokens):
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

    def _close_shard(self):
        if self.handle is None:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.shards.append(
            dict(file=self.path.name, tokens=self.in_shard, sha256=file_digest(self.path))
        )
        self.handle = None

    def finish(self) -> dict:
        self._close_shard()
        return dict(tokens=self.count, shards=self.shards)


def training_boundaries(config: Config, parameters: int) -> list[int]:
    """Cumulative block boundaries, avoiding per-epoch rounding drift."""
    length = config.model.context_length
    train = config.training
    total = train.max_tokens or math.ceil(parameters * train.tokens_per_parameter)
    epoch = train.epoch_tokens or parameters * train.epoch_tokens_per_parameter
    count = math.ceil(total / epoch)
    boundaries = [math.ceil(min(total, i * epoch) / length) for i in range(1, count + 1)]
    if len(set(boundaries)) != len(boundaries):
        raise ValueError("virtual epochs must each contain at least one block")
    return boundaries


def prepare(config: Config) -> dict:
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
        manifest = {
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
        manifest["identity"] = fingerprint(manifest)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.rename(destination)
        logger.success("Published cache {} ({})", destination, manifest["identity"][:12])
        return manifest


class TokenCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        content = {k: v for k, v in self.manifest.items() if k != "identity"}
        if self.manifest.get("version") != 1 or fingerprint(content) != self.manifest.get(
            "identity"
        ):
            raise ValueError("invalid cache manifest version or identity")
        self.maps: dict[str, list] = {}
        self.offsets: dict[str, list[int]] = {}
        for split, info in self.manifest["splits"].items():
            offsets, arrays = [0], []
            for shard in info["shards"]:
                path = self.directory / shard["file"]
                if path.stat().st_size != shard["tokens"] * 2:
                    raise ValueError(f"truncated token shard: {path}")
                arrays.append(np.memmap(path, dtype="<u2", mode="r"))
                offsets.append(offsets[-1] + shard["tokens"])
            if offsets[-1] != info["tokens"]:
                raise ValueError("manifest token count does not match shards")
            self.maps[split], self.offsets[split] = arrays, offsets

    def validate_config(self, config: Config):
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

    def verify(self):
        for info in self.manifest["splits"].values():
            for shard in info["shards"]:
                if file_digest(self.directory / shard["file"]) != shard["sha256"]:
                    raise ValueError(f"checksum mismatch: {shard['file']}")

    def blocks(self, split: str, length: int, partial: bool = False) -> int:
        targets = self.manifest["splits"][split]["tokens"] - 1
        return math.ceil(targets / length) if partial else targets // length

    def read(self, split: str, start: int, stop: int) -> np.ndarray:
        if not 0 <= start < stop <= self.offsets[split][-1]:
            raise IndexError("token range outside cache")
        pieces = []
        while start < stop:
            index = bisect_right(self.offsets[split], start) - 1
            end = min(stop, self.offsets[split][index + 1])
            pieces.append(
                self.maps[split][index][
                    start - self.offsets[split][index] : end - self.offsets[split][index]
                ]
            )
            start = end
        return np.concatenate(pieces).astype(np.int64)

    def batch(self, split: str, indices, length: int, device: torch.device):
        inputs = np.zeros((len(indices), length), dtype=np.int64)
        targets = np.full_like(inputs, -100)
        count = self.offsets[split][-1]
        for row, index in enumerate(indices):
            start = int(index) * length
            values = self.read(split, start, min(start + length + 1, count))
            inputs[row, : len(values) - 1] = values[:-1]
            targets[row, : len(values) - 1] = values[1:]
        result = [torch.from_numpy(values) for values in (inputs, targets)]
        if device.type == "cuda":
            result = [values.pin_memory() for values in result]
        return tuple(values.to(device, non_blocking=True) for values in result)


def validation_indices(cache: TokenCache, config: Config, full: bool):
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
