import importlib.util
import math
from pathlib import Path

import pytest


@pytest.fixture
def report():
    spec = importlib.util.spec_from_file_location(
        "throughput_report", Path(__file__).parents[1] / "scripts/throughput_report.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_projection_budget_rounding_and_optimizer_limits(report):
    result = report.project(101, 20, 2, 16, 10., 4.)
    assert result["token_budget"] == 2020
    assert result["tokens_per_update"] == 32
    assert result["whole_updates"] == 64
    assert result["rounded_processed_tokens"] == 2048
    assert result["rounded_excess_tokens"] == 28
    assert result["projected_seconds"] == pytest.approx(0.63125)
    assert result["rounded_projected_seconds"] == pytest.approx(0.64)
    assert result["ideal_2x_optimizer_hours"] * 3600 == pytest.approx(0.505)
    assert result["ideal_zero_cost_optimizer_hours"] * 3600 == pytest.approx(0.37875)
    assert result["ideal_2x_optimizer_speedup"] == pytest.approx(1.25)
    assert result["ideal_zero_cost_optimizer_speedup"] == pytest.approx(5 / 3)
    assert result["projected_hours"] == pytest.approx(
        result["residual_projected_hours"] + result["optimizer_projected_hours"])


def test_budget_scaling_leaves_relative_cost_and_per_token_time_unchanged(report):
    a = report.project(100, 20, 4, 1024, 5., 1.)
    b = report.project(100, 40, 4, 1024, 5., 1.)
    assert b["projected_hours"] == 2 * a["projected_hours"]
    assert b["optimizer_projected_hours"] == 2 * a["optimizer_projected_hours"]
    assert b["hours_per_billion_tokens"] == a["hours_per_billion_tokens"]
    assert b["ideal_zero_cost_optimizer_speedup"] == a["ideal_zero_cost_optimizer_speedup"]


@pytest.mark.parametrize("batch,update,opt", [(0, 10., 1.), (1, math.inf, 1.),
                                            (1, 10., 0.), (1, 10., 10.), (1, 10., math.nan)])
def test_reject_impossible_projection(report, batch, update, opt):
    with pytest.raises(ValueError):
        report.project(100, 20, batch, 1024, update, opt)


def test_context_comparison_matches_tokens_not_sequence_batch(report):
    rows = [dict(model="20m", context_length=1024, batch_size=4,
                 tokens_per_update=4096, projected_hours=2.),
            dict(model="20m", context_length=1024, batch_size=1,
                 tokens_per_update=1024, projected_hours=5.),
            dict(model="20m", context_length=4096, batch_size=1,
                 tokens_per_update=4096, projected_hours=3.),
            dict(model="20m", context_length=4096, batch_size=2,
                 tokens_per_update=8192, projected_hours=1.5)]
    result = report.matched_contexts(rows)
    assert len(result) == 1
    assert result[0]["reference_batch_size"] == 4
    assert result[0]["relative_time_vs_1024"] == 1.5


def test_context_512_comparison_uses_twice_the_reference_batch(report):
    rows = [dict(model="20m", context_length=1024, batch_size=4,
                 tokens_per_update=4096, projected_hours=2.),
            dict(model="20m", context_length=512, batch_size=8,
                 tokens_per_update=4096, projected_hours=1.8),
            dict(model="20m", context_length=512, batch_size=1,
                 tokens_per_update=512, projected_hours=10.)]
    matched = report.matched_contexts(rows)
    assert len(matched) == 1
    assert matched[0]["batch_size"] == 8 and matched[0]["reference_batch_size"] == 4
    assert matched[0]["relative_time_vs_1024"] == pytest.approx(0.9)


def test_multiple_campaigns_recompute_model_reference_and_check_coverage(report, tmp_path, monkeypatch):
    def pair(inputs, throughput, optimizer, tokens_per_parameter):
        length = int(throughput.name)
        rows, oom = [], []
        for model in report.MODELS:
            for batch in (1, 2):
                projected = report.project(100, tokens_per_parameter, batch, length,
                                            5. if length == 512 else 20., 1.)
                row: dict = dict(model=model, context_length=length, batch_size=batch, parameters=100,
                           **projected)
                row.update(primary_tokens_per_second=projected["token_budget"] / projected["projected_seconds"],
                           original_tokens_per_second=projected["token_budget"] / projected["projected_seconds"],
                           original_projected_hours=projected["projected_hours"],
                           original_window_cv=0., baseline_window_cv=0., optimizer_window_cv=0.,
                           rate_change_from_original=0., instrumented_wall_change=0.)
                rows.append(row)
            oom.append(dict(model=model, context_length=length, batch_size=4))
        manifest = dict(contexts=[length], identity=str(length))
        return rows, oom, {5}, manifest, manifest

    monkeypatch.setattr(report, "campaign_pair", pair)
    sources = [tmp_path / "1024", tmp_path / "512"]
    data = report.combine(report.Inputs(tmp_path), sources, sources, 20)
    assert data["contexts"] == (512, 1024)
    assert len(data["curves"]) == 8 and len(data["points"]) == 16
    for curve in data["curves"]:
        assert curve["relative_time_model_best"] == (1. if curve["context_length"] == 512 else 2.)
        assert curve["first_oom_batch"] == 4
    with pytest.raises(ValueError, match="overlapping campaign contexts"):
        report.combine(report.Inputs(tmp_path), sources * 2, sources * 2, 20)
    with pytest.raises(ValueError, match="matching throughput/optimizer"):
        report.combine(report.Inputs(tmp_path), sources, sources[:1], 20)


@pytest.mark.parametrize("width,layers,ffn,count", [(320, 8, 896, 20403520),
                                                   (640, 14, 1792, 91605120),
                                                   (960, 20, 2560, 251943360),
                                                   (1280, 24, 3456, 516814080)])
def test_independent_parameter_count(report, width, layers, ffn, count):
    assert report.parameter_count(dict(width=width, layers=layers, ffn_width=ffn,
                                       vocab_size=32000)) == count


def test_optimizer_sample_weighting_validation(report):
    samples = [dict(steps=3, seconds=0.03, optimizer_ms=1., graph_ms=9.),
               dict(steps=1, seconds=0.01, optimizer_ms=3., graph_ms=9.)]
    row = dict(steps=4, seconds=0.04, optimizer_ms=1.5, optimizer_samples=samples)
    raw: dict = dict(protocol=dict(optimizer_timing=True), measurements=[row] * 5,
               optimizer_milliseconds_per_update=1.5)
    assert report.optimizer_measurement(raw) == (1.5, 0.)
    raw["measurements"][0]["optimizer_ms"] = 2.
    with pytest.raises(ValueError, match="weighting mismatch"):
        report.optimizer_measurement(raw)


def test_duplicate_successful_measurements_are_not_silently_averaged(report):
    row = dict(model="20m", context_length=1024, batch_size=1, status="ok")
    with pytest.raises(ValueError, match="duplicate successful point"):
        report.unique_ok([row, dict(row)])
