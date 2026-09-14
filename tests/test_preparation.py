import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tiny_llm.config import Config
from tiny_llm.data import BufferedTokenLoader, TokenCache, prepare


@pytest.mark.parametrize(
    "shard_tokens,group_size,expected_tokens", [(7, 2, 73), (1, 1, 65), (11, 3, 69)]
)
def test_preparation_eos_determinism_and_reuse(
    tiny_config: Config,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    shard_tokens,
    group_size,
    expected_tokens,
):

    class API:
        def dataset_info(self, *args, **kwargs):
            return SimpleNamespace(sha="fixture-dataset")

        def model_info(self, *args, **kwargs):
            return SimpleNamespace(sha="fixture-tokenizer")

        def list_repo_files(self, *args, **kwargs):
            return [
                "en/c4-train.00000.json.gz",
                "en/c4-validation.00000.json.gz",
                "en.noclean/c4-train.00000.json.gz",
            ]

    class Tokenizer:
        eos_token_id = 2
        backend_tokenizer = SimpleNamespace(to_str=lambda: "fixture tokenizer")

        def __len__(self):
            return 17

        def get_vocab(self):
            return {str(i): i for i in range(17)}

        def save_pretrained(self, path):
            path.mkdir()
            (path / "tokenizer.json").write_text("fixture tokenizer")

        def __call__(self, texts, **kwargs):
            assert kwargs["add_special_tokens"] is False
            return {"input_ids": [[int(token) for token in text.split()] for text in texts]}

    class Stream:
        def shuffle(self, seed, buffer_size):
            assert seed == 42 and buffer_size == 100000
            return self

        def iter(self, batch_size):
            for _ in range(10):
                yield {"text": ["3 4 5", "", "6 7"]}

    def load(*args, **kwargs):
        assert kwargs["streaming"] is True
        assert len(kwargs["data_files"][kwargs["split"]]) == 1
        return Stream()

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=API))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())
        ),
    )
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load))
    tiny_config.data.shard_tokens = shard_tokens
    tiny_config.data.shuffle_group_size = group_size
    tiny_config.data.prepare_workers = 1
    first = prepare(tiny_config)
    cache = TokenCache(tiny_config.data.cache_dir)
    np.testing.assert_array_equal(cache.read("train", 0, 8), [3, 4, 5, 2, 2, 6, 7, 2])
    assert first["splits"]["train"]["tokens"] == expected_tokens
    assert first["splits"]["validation"]["tokens"] == 80
    assert first["validation_complete"]
    assert prepare(tiny_config) == first
    tiny_config.data.cache_dir = tmp_path / "second"
    assert prepare(tiny_config) == first
    manifest = json.loads((tiny_config.data.cache_dir / "manifest.json").read_text())
    assert manifest["dataset_revision"] == "fixture-dataset"
    tiny_config.data.prepare_train_tokens = 100
    with pytest.raises(ValueError, match="too small"):
        prepare(tiny_config)

    # Separately prepared short and long caches preserve the same sample prefix,
    # including when a sequence spans multiple tiny shards.
    tiny_config.data.cache_dir = tmp_path / "short"
    tiny_config.data.prepare_train_tokens = 32
    short_manifest = prepare(tiny_config)
    assert short_manifest["splits"]["train"]["tokens"] < expected_tokens
    short_cache = TokenCache(tiny_config.data.cache_dir)
    with (
        BufferedTokenLoader(short_cache, "train", 4, 8, group_size, seed=42) as short,
        BufferedTokenLoader(cache, "train", 4, 16, group_size, seed=42) as long,
    ):
        for actual, expected in zip(
            short.next_batch(8, torch.device("cpu")),
            long.next_batch(8, torch.device("cpu")),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
