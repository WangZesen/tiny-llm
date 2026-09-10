import copy
import json

import numpy as np
import pytest
import torch

from tiny_llm.analysis import load_checkpoint
from tiny_llm.analysis.core import result_key
from tiny_llm.checkpoints import file_hash, load_training_checkpoint, save_epoch_checkpoint
from tiny_llm.config import Config
from tiny_llm.data import TokenCache
from tiny_llm.packed import PackedLlama
from tiny_llm.packed_optimizer import PackedAdamW
from tiny_llm.runtime import rng_state
from tiny_llm.train import evaluate_checkpoint, recipe_identity, train


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_equal(x, y)
    else:
        assert a == b


@pytest.fixture
def packed_epoch(tiny_config, cache_dir):
    raw = tiny_config.model_dump()
    raw["decentralized"] = dict(num_models=4, topology="one_peer_exponential")
    raw["training"].update(
        checkpoint_policy="all", micro_batch_size=1, max_tokens=96, epoch_tokens=48
    )
    raw["runtime"]["deterministic"] = True
    config = Config.model_validate(raw)
    train(config)
    return config, config.runtime.output_dir / "epoch-001.pt"


@pytest.mark.parametrize("workers", [1, 4])
def test_epoch_continuation_and_retention(tiny_config, cache_dir, workers):
    raw = tiny_config.model_dump()
    raw["training"].update(checkpoint_policy="all", max_tokens=96, epoch_tokens=48)
    raw["runtime"]["deterministic"] = True
    if workers > 1:
        raw["decentralized"] = dict(num_models=workers, topology="one_peer_exponential")
        raw["training"]["micro_batch_size"] = 1
    config = Config.model_validate(raw)
    assert config.training.save_epoch_training_state
    train(config)
    output = config.runtime.output_dir
    epoch = output / "epoch-001.pt"
    original = load_training_checkpoint(epoch)
    assert original["completed_epochs"] == original["best_epoch"] == 1
    assert original["cursor"] == 12 and original["step"] == 3
    assert original["best_loss"] < float("inf")
    expected = load_training_checkpoint(output / "final.pt")
    # Omitting the new field is compatible with historical checkpoints/configuration.
    legacy = copy.deepcopy(original)
    legacy["config"]["training"].pop("save_epoch_training_state")
    legacy.pop("worker_files", None)
    legacy_path = output / "legacy.pt"
    torch.save(legacy, legacy_path)
    branch = Config.model_validate(legacy["config"])
    branch.runtime.output_dir = output.parent / "branch"
    branch.data.prefetch = False
    branch.training.save_epoch_training_state = False
    assert recipe_identity(branch, TokenCache(cache_dir)) == original["recipe_identity"]
    train(branch, epoch)
    actual = load_training_checkpoint(branch.runtime.output_dir / "final.pt")
    for key in (
        "model",
        "optimizer",
        "rng",
        "loader",
        "step",
        "cursor",
        "completed_epochs",
        "best_loss",
        "best_epoch",
    ):
        assert_equal(actual[key], expected[key])
    assert not list(branch.runtime.output_dir.glob("epoch-*.pt"))
    assert (branch.runtime.output_dir / "epoch-002.safetensors").is_file()
    events = [
        json.loads(line)
        for line in (branch.runtime.output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert {row["epoch"] for row in events if row["event"] == "train"} == {2}
    # Analysis accepts retention changes and recognizes manifests and ordinary weights alike.
    branch.training.checkpoint_policy = "final"
    a, info = load_checkpoint(branch, epoch)
    b, alias = load_checkpoint(branch, legacy_path)
    assert result_key(info) == result_key(alias)
    for x, y in zip(a.parameters(), b.parameters(), strict=True):
        assert_equal(x, y)
    assert (
        evaluate_checkpoint(config, epoch, True)["loss"]
        == evaluate_checkpoint(config, legacy_path, True)["loss"]
    )
    legacy_branch = branch.model_copy(deep=True)
    legacy_branch.runtime.output_dir = output.parent / "legacy-branch"
    train(legacy_branch, legacy_path)
    assert_equal(
        load_training_checkpoint(legacy_branch.runtime.output_dir / "final.pt")["optimizer"],
        expected["optimizer"],
    )
    before = epoch.read_bytes()
    with pytest.raises(ValueError, match="committed epoch"):
        train(config, epoch)
    assert epoch.read_bytes() == before


def test_worker_files_compact_and_validated(packed_epoch):
    config, path = packed_epoch
    root = torch.load(path, weights_only=False)
    assert root["version"] == 4 and root["kind"] == "packed_epoch"
    assert "model" not in root and "optimizer" not in root
    assert len(root["workers"]) == 4
    before = rng_state()
    combined = load_training_checkpoint(path)
    assert_equal(before, rng_state())
    model = PackedLlama(config.model, 4)
    model.load_packed_state_dict(combined["model"])
    for member in root["workers"]:
        local_path = path.parent / member["path"]
        local = torch.load(local_path, weights_only=False)
        assert local["worker"] == member["worker"]
        assert_equal(local["model"], model.local_state_dict(member["worker"]))
        for entry in model.layout:
            for tensor in (
                local["model"][entry.name],
                *local["optimizer"]["state"][entry.name].values(),
            ):
                assert tensor.device.type == "cpu"
                assert tensor.storage_offset() == 0
                assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
        with pytest.raises(ValueError, match="not resumable alone"):
            load_training_checkpoint(local_path)
    worker_path = path.parent / root["workers"][0]["path"]
    original = worker_path.read_bytes()
    local = torch.load(worker_path, weights_only=False)
    for damage in (
        "missing",
        "truncated",
        "swapped",
        "shape",
        "names",
        "groups",
        "position",
        "counter",
        "moment",
    ):
        broken = copy.deepcopy(local)
        manifest = copy.deepcopy(root)
        if damage == "missing":
            worker_path.unlink()
        elif damage == "truncated":
            worker_path.write_bytes(original[:32])
        elif damage == "swapped":
            worker_path.write_bytes((path.parent / root["workers"][1]["path"]).read_bytes())
        else:
            name = next(iter(broken["model"]))
            if damage == "shape":
                broken["model"][name] = broken["model"][name].flatten()[:1].clone()
            elif damage == "names":
                broken["optimizer"]["state"].pop(name)
            elif damage == "groups":
                broken["optimizer"]["param_groups"][0]["params"].reverse()
            elif damage == "position":
                broken["cursor"] += 4
            elif damage == "counter":
                broken["optimizer"]["state"][name]["step"].add_(1)
            else:
                broken["optimizer"]["state"][name]["exp_avg_sq"].fill_(float("nan"))
            torch.save(broken, worker_path)
        if damage not in ("missing", "truncated"):
            manifest["workers"][0]["sha256"] = file_hash(worker_path)
        torch.save(manifest, path)
        with pytest.raises((ValueError, FileNotFoundError)):
            load_training_checkpoint(path)
        worker_path.write_bytes(original)
        torch.save(root, path)
    assert_equal(load_training_checkpoint(path)["optimizer"], combined["optimizer"])


def test_interrupted_worker_export_and_immutability(packed_epoch, monkeypatch):
    import tiny_llm.checkpoints as checkpoints

    config, original = packed_epoch
    state = load_training_checkpoint(original)
    model = PackedLlama(config.model, 4)
    model.load_packed_state_dict(state["model"])
    optimizer = PackedAdamW(model, config)
    optimizer.load_state_dict(state["optimizer"])
    directory = original.parent.parent / "partial"
    directory.mkdir()
    path = directory / original.name
    real_save = checkpoints.atomic_checkpoint
    calls = 0

    def interrupted(destination, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated interruption")
        real_save(destination, value)

    monkeypatch.setattr(checkpoints, "atomic_checkpoint", interrupted)
    with pytest.raises(OSError, match="simulated"):
        save_epoch_checkpoint(path, state, model, optimizer)
    assert not path.exists()
    assert (directory / "node-000" / path.name).exists()
    monkeypatch.setattr(checkpoints, "atomic_checkpoint", real_save)
    save_epoch_checkpoint(path, state, model, optimizer)
    assert_equal(load_training_checkpoint(path)["optimizer"], state["optimizer"])
    with pytest.raises(ValueError, match="committed epoch"):
        save_epoch_checkpoint(path, state, model, optimizer)
