import copy
import json
import os
import signal
from pathlib import Path

import pytest
import torch
from helpers import assert_nested_equal, set_budget

from tiny_llm.analysis import load_checkpoint
from tiny_llm.analysis.core import result_key
from tiny_llm.checkpoints import load_training_checkpoint, save_epoch_checkpoint
from tiny_llm.config import Config
from tiny_llm.data import TokenCache, file_digest
from tiny_llm.packed import PackedLlama
from tiny_llm.runtime import rng_state
from tiny_llm.train import evaluate_checkpoint, make_optimizer, recipe_identity, train


@pytest.mark.parametrize(
    "workers,policy,epochs,selected,training_state",
    [
        (1, "interval", [2], {2, 4}, True),
        (1, "explicit", [1, 3], {1, 3}, False),
        (4, "interval", [2], {2, 4}, False),
        (4, "explicit", [1, 3], {1, 3}, True),
    ],
)
def test_scheduled_artifacts(
    tiny_config: Config,
    cache_dir: Path,
    workers: int,
    policy: str,
    epochs: list[int],
    selected: set[int],
    training_state: bool,
) -> None:
    set_budget(tiny_config, 128, 32)
    raw = tiny_config.model_dump()
    raw["training"].update(
        checkpoint_policy=policy,
        checkpoint_epochs=epochs,
        save_epoch_training_state=training_state,
        micro_batch_size=1,
    )
    if workers > 1:
        raw["decentralized"] = {"num_models": workers}
    config = Config.model_validate(raw)
    train(config)
    output = config.runtime.output_dir
    expected_weights = {f"epoch-{epoch:03d}.safetensors" for epoch in selected}
    assert {path.name for path in output.glob("*.safetensors")} == expected_weights
    expected_states = {f"epoch-{epoch:03d}.pt" for epoch in selected} if training_state else set()
    assert {path.name for path in output.glob("*.pt")} == expected_states | {"final.pt"}
    for worker in range(workers if workers > 1 else 0):
        directory = output / f"node-{worker:03d}"
        assert {path.name for path in directory.glob("*.safetensors")} == expected_weights
        assert {path.name for path in directory.glob("*.pt")} == expected_states
    best = json.loads((output / "best.json").read_text())
    assert best["weights"] == (
        f"epoch-{best['epoch']:03d}.safetensors" if best["epoch"] in selected else None
    )
    events = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in events if row["event"] == "validation"] == [1, 2, 3, 4]


@pytest.mark.parametrize("during", ["training", "evaluation"])
def test_interruption_keeps_only_committed_epochs(
    tiny_config: Config, cache_dir: Path, monkeypatch: pytest.MonkeyPatch, during: str
) -> None:
    import tiny_llm.train as module

    tiny_config.training.checkpoint_policy = "interval"
    if during == "training":
        original = module.optimizer_update
        calls = 0

        def update(*args, **kwargs):
            nonlocal calls
            result = original(*args, **kwargs)
            calls += 1
            if calls == 3:
                os.kill(os.getpid(), signal.SIGTERM)
            return result

        monkeypatch.setattr(module, "optimizer_update", update)
    else:
        original_evaluate = module.evaluate
        evaluations = 0

        def evaluate(*args, **kwargs):
            nonlocal evaluations
            evaluations += 1
            if evaluations == 2:
                os.kill(os.getpid(), signal.SIGTERM)
            return original_evaluate(*args, **kwargs)

        monkeypatch.setattr(module, "evaluate", evaluate)
    assert train(tiny_config)["status"] == "interrupted"
    output = tiny_config.runtime.output_dir
    assert {path.name for path in output.glob("*.pt")} == {"epoch-001.pt"}
    assert {path.name for path in output.glob("*.safetensors")} == {"epoch-001.safetensors"}


@pytest.fixture(params=["adamw", "accumadamw"])
def packed_epoch(tiny_config: Config, cache_dir: Path, request: pytest.FixtureRequest):
    tiny_config.optimizer.name = request.param
    raw = tiny_config.model_dump()
    raw["decentralized"] = dict(num_models=4, topology="one_peer_exponential")
    raw["training"].update(
        checkpoint_policy="interval",
        micro_batch_size=1,
        tokens_per_parameters=96 / 800,
        epoch_tokens_per_parameters=48 / 800,
    )
    raw["runtime"]["deterministic"] = True
    config = Config.model_validate(raw)
    train(config)
    return config, config.runtime.output_dir / "epoch-001.pt"


@pytest.mark.parametrize("workers,checkpoint_format", [(1, "epoch"), (4, "epoch"), (4, "combined")])
@pytest.mark.parametrize(
    "optimizer_name,accum_iter", [("adamw", 4), ("accumadamw", 3), ("accumadamw", 4)]
)
def test_epoch_continuation_and_retention(
    tiny_config: Config, cache_dir: Path, workers, checkpoint_format, optimizer_name, accum_iter
):
    tiny_config.optimizer.name = optimizer_name
    tiny_config.optimizer.accum_iter = accum_iter
    raw = tiny_config.model_dump()
    raw["training"].update(
        checkpoint_policy="interval",
        tokens_per_parameters=96 / 800,
        epoch_tokens_per_parameters=48 / 800,
    )
    raw["runtime"]["deterministic"] = True
    if workers > 1:
        raw["decentralized"] = dict(num_models=workers, topology="one_peer_exponential")
        raw["training"]["micro_batch_size"] = 1
    config = Config.model_validate(raw)
    assert config.training.save_epoch_training_state
    train(config)
    output = config.runtime.output_dir
    epoch = output / "epoch-001.pt"
    if optimizer_name == "adamw":
        legacy = torch.load(epoch, weights_only=False)
        legacy["config"]["optimizer"].pop("name")
        legacy["config"]["optimizer"].pop("accum_iter")
        torch.save(legacy, epoch)
    original = load_training_checkpoint(epoch)
    assert original["completed_epochs"] == original["best_epoch"] == 1
    assert original["cursor"] == 12 and original["step"] == 3
    assert original["best_loss"] < float("inf")
    expected = load_training_checkpoint(output / "final.pt")
    branch = Config.model_validate(original["config"])
    branch.runtime.output_dir = output.parent / "branch"
    branch.data.prefetch = False
    branch.training.save_epoch_training_state = False
    assert recipe_identity(branch, TokenCache(cache_dir)) == original["recipe_identity"]
    continuation = epoch
    if checkpoint_format == "combined":
        continuation = output / "combined.pt"
        torch.save(original, continuation)
    train(branch, continuation)
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
        assert_nested_equal(actual[key], expected[key])
    assert not list(branch.runtime.output_dir.glob("epoch-*.pt"))
    assert (branch.runtime.output_dir / "epoch-002.safetensors").is_file()
    events = [
        json.loads(line)
        for line in (branch.runtime.output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert {row["epoch"] for row in events if row["event"] == "train"} == {2}
    # Analysis accepts retention changes and reads the same epoch weights.
    branch.training.checkpoint_policy = "final"
    a, info = load_checkpoint(branch, epoch)
    b, alias = load_checkpoint(branch, epoch.with_suffix(".safetensors"))
    assert result_key(info) == result_key(alias)
    for x, y in zip(a.parameters(), b.parameters(), strict=True):
        assert_nested_equal(x, y)
    assert (
        evaluate_checkpoint(config, epoch, True)["loss"]
        == evaluate_checkpoint(config, epoch.with_suffix(".safetensors"), True)["loss"]
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
    assert_nested_equal(before, rng_state())
    model = PackedLlama(config.model, 4)
    model.load_packed_state_dict(combined["model"])
    for member in root["workers"]:
        local_path = path.parent / member["path"]
        local = torch.load(local_path, weights_only=False)
        assert local["worker"] == member["worker"]
        assert_nested_equal(local["model"], model.local_state_dict(member["worker"]))
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
    damages = [
        "missing",
        "truncated",
        "swapped",
        "shape",
        "names",
        "groups",
        "position",
        "counter",
        "moment",
    ]
    if config.optimizer.name == "accumadamw":
        damages.extend(["buffer", "buffer_shape", "buffer_missing", "window", "optimizer"])
    for damage in damages:
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
            elif damage == "buffer":
                broken["optimizer"]["state"][name]["accum_grad"].fill_(float("nan"))
            elif damage == "buffer_shape":
                broken["optimizer"]["state"][name]["accum_grad"] = torch.zeros(1)
            elif damage == "buffer_missing":
                broken["optimizer"]["state"][name].pop("accum_grad")
            elif damage == "window":
                broken["optimizer"]["param_groups"][0]["accum_iter"] = 2
            elif damage == "optimizer":
                broken["optimizer"]["optimizer"] = "adamw"
            else:
                broken["optimizer"]["state"][name]["exp_avg_sq"].fill_(float("nan"))
            torch.save(broken, worker_path)
        if damage not in ("missing", "truncated"):
            manifest["workers"][0]["sha256"] = file_digest(worker_path)
        torch.save(manifest, path)
        with pytest.raises((ValueError, FileNotFoundError)):
            load_training_checkpoint(path)
        worker_path.write_bytes(original)
        torch.save(root, path)
    assert_nested_equal(load_training_checkpoint(path)["optimizer"], combined["optimizer"])


def test_interrupted_worker_export_and_immutability(packed_epoch, monkeypatch: pytest.MonkeyPatch):
    import tiny_llm.checkpoints as checkpoints

    config, original = packed_epoch
    state = load_training_checkpoint(original)
    model = PackedLlama(config.model, 4)
    model.load_packed_state_dict(state["model"])
    optimizer = make_optimizer(model, config, torch.device("cpu"))
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
    assert_nested_equal(load_training_checkpoint(path)["optimizer"], state["optimizer"])
    with pytest.raises(ValueError, match="committed epoch"):
        save_epoch_checkpoint(path, state, model, optimizer)
