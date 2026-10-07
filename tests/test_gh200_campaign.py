"""Numerical gate and frozen-production harness contracts, without a GPU."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from tiny_llm.config import Config
from tiny_llm.train import train


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


campaign = load_script("gh200_campaign")
worker = load_script("gh200_worker")


def test_convergence_gate_requires_all_paired_seeds_and_upper_bound():
    native = {45: 3.5, 46: 3.6, 47: 3.7}
    assert campaign.convergence_gate(native, {45: 3.5})["status"] == "incomplete"
    assert campaign.convergence_gate(native, native)["status"] == "pass"
    assert campaign.convergence_gate(native, {k: v + 0.006 for k, v in native.items()})[
        "status"
    ] == "not_qualified"
    noisy = {45: 3.49, 46: 3.6, 47: 3.71}
    result = campaign.convergence_gate(native, noisy)
    assert abs(result["mean_difference"]) < 1e-10
    assert result["status"] == "not_qualified"


def test_performance_gate_rejects_missing_independence_and_regressions():
    ids = {"sync", "packed4", "packed8", "production"}
    pairs = [dict(id=name, mode=name, category="production" if name == "production" else "small_batch",
                  repeat=repeat, allocation=f"job{repeat}", native=100, candidate=110)
             for name in ids for repeat in range(3)]
    assert campaign.performance_gate(pairs, ids)["status"] == "pass"
    assert campaign.performance_gate(pairs[:-1], ids)["status"] == "incomplete"
    production = next(p for p in pairs if p["id"] == "production")
    production["candidate"] = 97
    assert campaign.performance_gate(pairs, ids)["status"] == "not_qualified"
    production["candidate"] = 110
    bad = [p | {"allocation": "one-job"} for p in pairs]
    with pytest.raises(ValueError, match="independent"):
        campaign.performance_gate(bad, ids)


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("scheme", [None, "awc", "atc", "adaptive"])
def test_baseline_adapter_matches_actual_production_training(
    tiny_config: Config, cache_dir: Path, scheme
):
    from tiny_llm.config import AdaptiveConsensusConfig, DecentralizedConfig

    cfg = tiny_config.model_copy(deep=True)
    cfg.training.micro_batch_size = 3  # Global batch four: a genuine remainder.
    cfg.training.checkpoint_policy = "final"
    if scheme is not None:
        cfg.training.micro_batch_size = 2
        cfg.decentralized = DecentralizedConfig(
            num_models=2, topology="one_peer_exponential",
            scheme="awc" if scheme == "adaptive" else scheme,
            adaptive_consensus=AdaptiveConsensusConfig(start_frac=0.5, p=1)
            if scheme == "adaptive" else None,
        )
    cfg = Config.model_validate(cfg.model_dump())
    runner = worker.NativeRunner(cfg)
    try:
        while runner.cursor < runner.boundaries[-1]:
            runner.update(inspect=True)
        expected = runner.state()
    finally:
        runner.close()
    train(cfg)
    actual = torch.load(cfg.runtime.output_dir / "final.pt", weights_only=False)
    assert_nested_equal(expected["model"], actual["model"])
    assert_nested_equal(expected["optimizer"], actual["optimizer"])
    assert runner.step == actual["step"]
    assert runner.cursor == actual["cursor"]


def test_tensor_comparison_records_near_zero_absolute_errors():
    result = worker.tensor_metrics({"x": torch.zeros(2)}, {"x": torch.tensor([0.0, 1e-7])})
    assert result["maximum_absolute"] > 0
    assert result["parameters"]["x"]["relative_l2"] > 1
    with pytest.raises(FloatingPointError):
        worker.tensor_metrics({"x": torch.ones(1)}, {"x": torch.tensor([float("nan")])})


@pytest.mark.parametrize("packed", [False, True])
def test_offloaded_controls_keep_independent_states_and_identical_inputs(
    tiny_config, cache_dir, tmp_path, monkeypatch, packed
):
    from tiny_llm.config import DecentralizedConfig

    if packed:
        tiny_config.decentralized = DecentralizedConfig(num_models=2)
    monkeypatch.setattr("tiny_llm.runtime.environment", lambda: {"source_hash": "fixture"})
    result = worker.numerical_serialized(tiny_config, {"local_updates": 2}, tmp_path)
    assert result["status"] == "ok"
    assert result["kind"] == "native_native_serialized_control"
    updates = campaign.read(tmp_path / "updates.json")
    for row in updates:
        for values in row["workers"]:
            assert values["left_counters"] == values["right_counters"]
            for field in ("parameters", "gradients", "moments", "updates"):
                assert values[field]["relative_l2"] == 0
