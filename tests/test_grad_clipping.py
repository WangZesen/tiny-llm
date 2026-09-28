import json
from pathlib import Path

import pytest
import torch
from helpers import set_budget

from tiny_llm.config import Config, load_config
from tiny_llm.data import TokenCache
from tiny_llm.runtime import clip_grad_norm_
from tiny_llm.train import recipe_identity, train


def test_null_clipping_preserves_gradients():
    parameters = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(1))]
    parameters[0].grad = torch.tensor([3.0, 4.0])
    before = parameters[0].grad.clone()
    version = parameters[0].grad._version
    assert clip_grad_norm_(iter(parameters), None).item() == 5.0
    assert torch.equal(parameters[0].grad, before)
    assert parameters[0].grad._version == version
    assert parameters[1].grad is None
    assert clip_grad_norm_(iter(parameters), 1.0).item() == 5.0
    assert parameters[0].grad.norm() <= 1.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("maximum", [1.0, None])
def test_nonfinite_gradients_raise(bad, maximum):
    parameter = torch.nn.Parameter(torch.zeros(1))
    parameter.grad = torch.tensor([bad])
    with pytest.raises(RuntimeError, match="non-finite"):
        clip_grad_norm_([parameter], maximum)


def test_clipping_config(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n")
    default = load_config(path)
    explicit = load_config(path, ["optimizer.grad_clip=1.0"])
    assert explicit == default
    assert load_config(path, ["optimizer.grad_clip=null"]).optimizer.grad_clip is None
    for invalid in ["0", "-1"]:
        with pytest.raises(ValueError):
            load_config(path, [f"optimizer.grad_clip={invalid}"])


@pytest.mark.parametrize("workers", [1, 4])
def test_clipping_identity_resume_and_logging(tiny_config: Config, cache_dir: Path, workers):
    from tiny_llm.config import Config

    raw = tiny_config.model_dump()
    if workers == 4:
        raw["decentralized"] = dict(num_models=4, topology="one_peer_exponential")
        raw["training"].update(batch_tokens=32, micro_batch_size=2)
    config = Config.model_validate(raw)
    config.training.checkpoint_policy = "final"
    cache = TokenCache(cache_dir)
    clipped_identity = recipe_identity(config, cache)
    config.optimizer.grad_clip = None
    assert recipe_identity(config, cache) != clipped_identity
    result = train(config)
    checkpoint = config.runtime.output_dir / "final.pt"
    state = torch.load(checkpoint, weights_only=False)
    assert state["config"]["optimizer"]["grad_clip"] is None
    assert load_config(config.runtime.output_dir / "resolved.yaml").optimizer.grad_clip is None
    resumed = train(config, checkpoint)
    assert resumed["final_validation"]["loss"] == result["final_validation"]["loss"]
    metrics = [
        json.loads(line)
        for line in (config.runtime.output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    updates = [row for row in metrics if "grad_norm" in row]
    assert updates
    assert all(row["grad_norm"] >= 0 for row in updates)
    if workers == 4:
        assert all(len(row["local_grad_norms"]) == 4 for row in updates)
    for row in metrics:
        if row["event"] == "validation":
            assert row["grad_clip_count"] == 0
            if workers == 4:
                assert row["local_grad_clip_counts"] == [0] * workers
    config.optimizer.grad_clip = 1.0
    with pytest.raises(ValueError, match="recipe"):
        train(config, checkpoint)


@pytest.mark.parametrize("scheme", [None, "awc", "atc"])
@pytest.mark.parametrize("maximum", [1.0, None])
def test_epoch_clipping_counts_include_unlogged_updates(
    tiny_config: Config, cache_dir: Path, monkeypatch: pytest.MonkeyPatch, scheme, maximum
):
    # Script the reported pre-clipping norms, while preserving the real optimizer update.
    # Exact-threshold norms do not count, and workers have deliberately different counts.
    norms = [
        [2.0, 0.5, 1.0, 2.0],
        [0.5, 2.0, 0.5, 2.0],
        [1.0, 0.5, 3.0, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 2.0],
        [2.0, 1.0, 0.5, 2.0],
        [2.0, 1.0, 0.5, 2.0],
        [1.0, 1.0, 0.5, 1.0],
    ]
    workers = 4 if scheme else 1
    values = iter(value for row in norms for value in row[:workers])
    set_budget(tiny_config, 128, 64)
    raw = tiny_config.model_dump()
    raw["training"].update(micro_batch_size=1, log_every=100)
    raw["optimizer"]["grad_clip"] = maximum
    if scheme:
        raw["decentralized"] = dict(num_models=workers, scheme=scheme)
    config = Config.model_validate(raw)

    def scripted(parameters, maximum):
        norm = clip_grad_norm_(parameters, maximum)
        return norm.new_tensor(next(values))

    module = "packed_optimizer" if scheme else "train"
    monkeypatch.setattr(f"tiny_llm.{module}.clip_grad_norm_", scripted)
    train(config)
    assert next(values, None) is None
    events = [
        json.loads(line)
        for line in (config.runtime.output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert len([row for row in events if row["event"] == "train"]) == 2
    epochs = [row for row in events if row["event"] == "validation"]
    expected = [[1, 1, 1, 2], [2, 0, 0, 3]] if maximum else [[0] * 4] * 2
    for row, counts in zip(epochs, expected, strict=True):
        assert row["grad_clip_count"] == sum(counts[:workers])
        if scheme:
            assert row["local_grad_clip_counts"] == counts
        else:
            assert "local_grad_clip_counts" not in row
    log = (config.runtime.output_dir / "run.log").read_text()
    assert "gradient clips" in log
    if scheme:
        assert "per worker" in log


@pytest.mark.parametrize("scheme", [None, "awc", "atc"])
@pytest.mark.parametrize("maximum,clipped", [(1e-8, True), (1e8, False)])
def test_epoch_counts_match_actual_norms(
    tiny_config: Config, cache_dir: Path, scheme, maximum, clipped
):
    raw = tiny_config.model_dump()
    raw["optimizer"]["grad_clip"] = maximum
    if scheme:
        raw["decentralized"] = dict(num_models=4, scheme=scheme)
        raw["training"]["micro_batch_size"] = 1
    config = Config.model_validate(raw)
    train(config)
    rows = [
        json.loads(line)
        for line in (config.runtime.output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    for epoch in (1, 2):
        updates = [row for row in rows if row["event"] == "train" and row["epoch"] == epoch]
        summary = next(
            row for row in rows if row["event"] == "validation" and row["epoch"] == epoch
        )
        counts = [0] * (4 if scheme else 1)
        for row in updates:
            norms = row.get("local_grad_norms", [row["grad_norm"]])
            for worker, norm in enumerate(norms):
                assert (norm > maximum) == clipped
                counts[worker] += int(norm > maximum)
        assert summary["grad_clip_count"] == sum(counts)
        if scheme:
            assert summary["local_grad_clip_counts"] == counts
