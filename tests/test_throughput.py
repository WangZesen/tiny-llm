import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from tiny_llm.cli import main
from tiny_llm.config import PRESETS, Config, ModelConfig, load_config
from tiny_llm.throughput import (
    benchmark_throughput,
    classify_failure,
    measurement_summary,
    optimizer_summary,
    optimizer_window,
)


@pytest.fixture
def driver():
    spec = importlib.util.spec_from_file_location(
        "throughput_sweep", Path(__file__).parents[1] / "scripts/throughput_sweep.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("preset,count", [("250m", 251943360), ("500m", 516814080)])
def test_larger_preset_files(preset, count):
    config = load_config(Path(__file__).parents[1] / f"configs/{preset}.yaml")
    assert config.model == ModelConfig(**PRESETS[preset])
    assert config.model.parameter_count == count
    assert config.model.width // config.model.heads == 64
    assert config.runtime.training_backend == "gh200"


def test_summary_weights_tokens_and_reports_variation():
    rows = [dict(steps=20, seconds=2), dict(steps=20, seconds=4), dict(steps=20, seconds=2)]
    result = measurement_summary(rows, 4, 1024)
    assert result["tokens_per_second"] == 40960
    assert result["sequences_per_second"] == 40
    assert result["milliseconds_per_update"] == 100
    assert not result["stable"]
    assert result["coefficient_of_variation"] > 0.05


@pytest.mark.parametrize("error,status", [
    (torch.cuda.OutOfMemoryError("allocation failed"), "cuda_oom"),
    (RuntimeError("CUDA error: out of memory"), "cuda_oom"),
    (MemoryError("out of memory"), "error"),
    (RuntimeError("CPU allocator out of memory"), "error"),
    (FloatingPointError("nonfinite update"), "nonfinite"),
    (TimeoutError("deadline"), "timeout"),
    (RuntimeError("invalid kernel launch"), "error"),
])
def test_failure_categories(error, status):
    wrapper = RuntimeError("compiler wrapper")
    wrapper.__cause__ = error
    assert classify_failure(wrapper) == status


def test_no_accumulation_and_positive_protocol(tmp_path):
    config = Config()
    config.training.micro_batch_size = 16
    with pytest.raises(ValueError, match="no gradient accumulation"):
        benchmark_throughput(config, tmp_path / "a.json")
    config.training.micro_batch_size = config.training.batch_tokens // config.model.context_length
    for seconds in (0, -1, math.inf, math.nan):
        with pytest.raises(ValueError, match="positive"):
            benchmark_throughput(config, tmp_path / "a.json", window_seconds=seconds)


@pytest.mark.parametrize("instrumented", [False, True])
def test_throughput_cli_dispatch(tmp_path, monkeypatch, instrumented):
    calls = []

    def worker(config, destination, **kwargs):
        calls.append((destination, kwargs))
        return {"status": "cuda_oom"}

    monkeypatch.setattr("tiny_llm.throughput.benchmark_throughput", worker)
    main(["benchmark-throughput-worker", "--output", str(tmp_path / "r.json")]
         + (["--optimizer-timing"] if instrumented else []))
    kwargs = dict(warmup=20, windows=5, window_seconds=10.0)
    if instrumented:
        kwargs["optimizer_timing"] = True
    assert calls == [(tmp_path / "r.json", kwargs)]


@pytest.mark.parametrize("length", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("batch", [1, 2, 128])
def test_sweep_configuration(driver, tmp_path, length, batch):
    raw = driver.point_config(Config().model_dump(mode="json"), length, batch, tmp_path)
    config = Config.model_validate(raw)
    assert config.training.micro_batch_size == batch
    assert config.training.batch_tokens == batch * length
    assert config.training.checkpoint_policy == "none"


def test_context_selection_and_optimizer_inheritance(driver):
    assert driver.selected_contexts([512]) == (512,)
    assert driver.selected_contexts(None, {"contexts": [1024, 2048, 4096]}) == (1024, 2048, 4096)
    assert driver.selected_contexts(None, {"contexts": [512]}) == (512,)
    for contexts in ([], [512, 512], [256]):
        with pytest.raises(ValueError, match="subset"):
            driver.selected_contexts(contexts)
    with pytest.raises(ValueError, match="original campaign"):
        driver.selected_contexts([512], {"contexts": [1024, 2048, 4096]})


def test_sixteen_curves_cap_concurrency_and_resume(driver, tmp_path, monkeypatch):
    manifest = dict(campaign_id="test", identity="identity", resources=driver.RESOURCES,
                    models={model: Config().model_dump(mode="json") for model in driver.MODELS},
                    repo=str(tmp_path), contexts=list(driver.CONTEXTS), concurrency=12)
    state = dict(curves=[dict(id=f"{model}-c{length}", model=model, context_length=length,
                             next_batch=1, complete=False, blocked=None, current=None, attempts=[])
                        for model in driver.MODELS for length in driver.CONTEXTS])
    monkeypatch.setattr(driver, "prepare", lambda root: (manifest, state))
    monkeypatch.setattr(driver, "validate_manifest", lambda root: manifest)
    submissions = []

    def command(argv):
        assert argv[0] == "sbatch"
        submissions.append(argv)
        return str(100 + len(submissions))

    def interrupted(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(driver, "command", command)
    monkeypatch.setattr(driver.time, "sleep", interrupted)
    monkeypatch.setattr(driver, "scheduler_rows", lambda ids: {
        job: dict(state="RUNNING") for job in ids})
    with pytest.raises(KeyboardInterrupt):
        driver.run(tmp_path)
    assert len(submissions) == 12
    with pytest.raises(KeyboardInterrupt):
        driver.run(tmp_path, resume=True)
    assert len(submissions) == 12
    with pytest.raises(ValueError, match="frozen contexts"):
        driver.run(tmp_path, resume=True, contexts=[512])


def test_independent_optimizer_points_resume_and_retry(driver, tmp_path, monkeypatch):
    manifest = dict(campaign_id="paired", identity="identity", resources=driver.RESOURCES,
                    models={model: Config().model_dump(mode="json") for model in driver.MODELS},
                    repo=str(tmp_path), contexts=[512], concurrency=12, mode="optimizer_timing",
                    batch_limits={f"{model}-c512": 8 for model in driver.MODELS})
    curves = [dict(id=f"{model}-c512", model=model, context_length=512, next_batch=1,
                   complete=False, blocked=None, current=None, attempts=[]) for model in driver.MODELS]
    state = dict(curves=curves, work_items=driver.optimizer_work_items(curves, manifest["batch_limits"]))
    assert [item["fixed_batch"] for item in state["work_items"]] == [1]*4 + [2]*4 + [4]*4 + [8]*4
    monkeypatch.setattr(driver, "prepare", lambda root: (manifest, state))
    monkeypatch.setattr(driver, "validate_manifest", lambda root: manifest)
    monkeypatch.setattr(driver, "collect", lambda root: None)
    submissions = []

    def command(argv):
        submissions.append(argv)
        return str(100 + len(submissions))

    def interrupted(seconds):
        raise KeyboardInterrupt

    def result(directory, manifest, receipt):
        failed = receipt["model"] == "20m" and receipt["batch_size"] == 2 and receipt["attempt"] == 1
        return {"status": "compiler_error" if failed else "ok"}

    monkeypatch.setattr(driver, "command", command)
    monkeypatch.setattr(driver.time, "sleep", interrupted)
    monkeypatch.setattr(driver, "scheduler_rows", lambda ids: {
        job: dict(state="COMPLETED", exit_code="0:0") for job in ids})
    monkeypatch.setattr(driver, "validated_result", result)
    with pytest.raises(KeyboardInterrupt):
        driver.run(tmp_path)
    assert len(submissions) == 12
    saved = driver.read(tmp_path / "state.json")
    assert all(len(curve["active_points"]) == 3 for curve in saved["curves"])
    monkeypatch.setattr(driver.time, "sleep", lambda seconds: None)
    assert not driver.run(tmp_path, resume=True)
    assert len(submissions) == 15
    saved = driver.read(tmp_path / "state.json")
    assert saved["curves"][0]["blocked"] == "compiler_error"
    assert all(curve["complete"] for curve in saved["curves"][1:])
    # Retrying the failed point must succeed before its curve's unscheduled B=8 starts.
    monkeypatch.setattr(driver.time, "sleep", interrupted)
    with pytest.raises(KeyboardInterrupt):
        driver.run(tmp_path, resume=True, retry_failed=True)
    assert len(submissions) == 16
    saved = driver.read(tmp_path / "state.json")
    assert saved["curves"][0]["active_points"] == ["points/20m-c512/b2/attempt-2"]
    monkeypatch.setattr(driver.time, "sleep", lambda seconds: None)
    assert driver.run(tmp_path, resume=True)
    assert len(submissions) == 17
    saved = driver.read(tmp_path / "state.json")
    assert all(curve["complete"] for curve in saved["curves"])
    assert [len(curve["attempts"]) for curve in saved["curves"]] == [5, 4, 4, 4]


@pytest.mark.parametrize("status,next_batch,complete,blocked", [
    ("ok", 8, False, None), ("cuda_oom", 4, True, None),
    ("nonfinite", 4, False, "nonfinite"), ("timeout", 4, False, "timeout"),
])
def test_curve_advances_only_after_complete_update(driver, tmp_path, monkeypatch,
                                                  status, next_batch, complete, blocked):
    directory = tmp_path / "point"
    directory.mkdir()
    curve = dict(id="20m-c1024", current="point", next_batch=4, complete=False, blocked=None)
    receipt = dict(batch_size=4)
    monkeypatch.setattr(driver, "validated_result", lambda *args: {"status": status})
    driver.finish_point(tmp_path, {}, curve, receipt, {"state": "COMPLETED", "exit_code": "0:0"})
    assert curve == dict(id="20m-c1024", current=None, next_batch=next_batch,
                         complete=complete, blocked=blocked)


@pytest.mark.parametrize("state,status", [("TIMEOUT", "timeout"), ("OUT_OF_MEMORY", "host_oom"),
                                         ("NODE_FAIL", "infrastructure_failure")])
def test_scheduler_failures_are_not_gpu_oom(driver, tmp_path, state, status):
    (tmp_path / "point").mkdir()
    curve = dict(id="500m-c4096", current="point", next_batch=1, complete=False, blocked=None)
    receipt = dict(batch_size=1)
    driver.finish_point(tmp_path, {}, curve, receipt, {"state": state, "exit_code": "1:0"})
    assert curve["blocked"] == status
    assert not curve["complete"] and curve["next_batch"] == 1


def test_resume_reconciles_intent_without_resubmitting(driver, monkeypatch):
    calls = []

    def command(argv):
        calls.append(argv)
        return "123|tp-case"

    monkeypatch.setattr(driver, "command", command)
    receipt = dict(job_name="tp-case", submitted="2026-10-08T10:00:00+00:00", job_id=None)
    driver.reconcile_intent(receipt)
    assert receipt["job_id"] == "123" and receipt["status"] == "submitted"
    assert all(argv[0] != "sbatch" for argv in calls)
    monkeypatch.setattr(driver, "command", lambda argv: "")
    with pytest.raises(RuntimeError, match="unresolved submission intent"):
        driver.reconcile_intent(receipt)


def test_result_rejects_incomplete_timing(driver, tmp_path):
    config = Config().model_dump(mode="json")
    receipt = dict(config_identity=driver.digest(config), batch_size=1, context_length=1024)
    manifest = dict(source_hash="source", protocol=driver.PROTOCOL)
    result = dict(config=config, environment={"source_hash": "source"}, protocol=driver.PROTOCOL,
                  status="ok", successful_updates=25, measurements=[])
    driver.write(tmp_path / "result.json", result)
    with pytest.raises(ValueError, match="incomplete measurement"):
        driver.validated_result(tmp_path, manifest, receipt)


@pytest.mark.parametrize("instrumented", [False, True])
def test_watcher_resume_does_not_duplicate_jobs(driver, tmp_path, monkeypatch, instrumented):
    manifest: dict = dict(campaign_id="test", identity="identity", resources=driver.RESOURCES,
                    models={"20m": Config().model_dump(mode="json")}, repo=str(tmp_path))
    if instrumented:
        manifest.update(mode="optimizer_timing", batch_limits={"20m-c1024": 2})
    state = dict(curves=[dict(id="20m-c1024", model="20m", context_length=1024,
                             next_batch=1, complete=False, blocked=None, current=None, attempts=[])])
    monkeypatch.setattr(driver, "prepare", lambda root: (manifest, state))
    monkeypatch.setattr(driver, "validate_manifest", lambda root: manifest)
    submitted = []

    def command(argv):
        assert argv[0] == "sbatch"
        submitted.append(argv)
        return str(100 + len(submitted))

    def interrupted(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(driver, "command", command)
    monkeypatch.setattr(driver, "scheduler_rows", lambda ids: {
        job: dict(state="COMPLETED", exit_code="0:0") for job in ids})
    monkeypatch.setattr(driver, "validated_result", lambda directory, manifest, receipt: {
        "status": "ok" if instrumented or receipt["batch_size"] == 1 else "cuda_oom"})
    monkeypatch.setattr(driver, "collect", lambda root: None)
    monkeypatch.setattr(driver.time, "sleep", interrupted)
    with pytest.raises(KeyboardInterrupt):
        driver.run(tmp_path)
    assert len(submitted) == 1
    assert driver.read(tmp_path / "state.json")["curves"][0]["current"] is not None
    monkeypatch.setattr(driver.time, "sleep", lambda seconds: None)
    assert driver.run(tmp_path, resume=True)
    assert len(submitted) == 2
    final = driver.read(tmp_path / "state.json")["curves"][0]
    assert final["complete"] and final["next_batch"] == (4 if instrumented else 2)
    assert len(final["attempts"]) == 2


def test_optimizer_burst_weighting_and_share(driver):
    samples = [dict(steps=3, seconds=0.03, optimizer_ms=1., graph_ms=9.),
               dict(steps=1, seconds=0.01, optimizer_ms=3., graph_ms=9.)]
    window = dict(steps=4, seconds=0.04, **optimizer_window(samples))
    assert window["optimizer_ms"] == 1.5
    summary = optimizer_summary([window] * 5)
    assert summary["optimizer_milliseconds_per_update"] == 1.5
    assert summary["optimizer_fraction_instrumented"] == pytest.approx(0.15)
    assert summary["optimizer_fraction_graph"] == pytest.approx(1.5 / 9)
    assert summary["optimizer_stable"]
    pair = driver.paired_optimizer_summary(dict(summary, milliseconds_per_update=10., stable=True),
                                           dict(milliseconds_per_update=8., stable=True,
                                                coefficient_of_variation=0.))
    assert pair["optimizer_fraction"] == pytest.approx(1.5 / 8)
    assert pair["instrumentation_warning"]


def test_optimizer_result_validation(driver, tmp_path):
    config = Config().model_dump(mode="json")
    receipt = dict(config_identity=driver.digest(config), batch_size=1, context_length=1024)
    manifest = dict(source_hash="source", protocol=driver.PROTOCOL, mode="optimizer_timing")
    rows = [dict(steps=20, seconds=2.) for _ in range(5)]
    baseline = dict(config=config, environment={"source_hash": "source"}, protocol=driver.PROTOCOL,
                    status="ok", successful_updates=125, measurements=rows,
                    **measurement_summary(rows, 1, 1024))
    driver.write(tmp_path / "baseline.json", baseline)
    timed_rows = [dict(row, **optimizer_window([
        dict(steps=20, seconds=2., optimizer_ms=10., graph_ms=99.)])) for row in rows]
    result: dict = dict(baseline, measurements=timed_rows, protocol=dict(driver.PROTOCOL, optimizer_timing=True),
                  **optimizer_summary(timed_rows))
    result.update(driver.paired_optimizer_summary(result, baseline))
    driver.write(tmp_path / "result.json", result)
    assert driver.validated_result(tmp_path, manifest, receipt)["optimizer_fraction"] == 0.1
    result["optimizer_fraction"] = 0.2
    driver.write(tmp_path / "result.json", result)
    with pytest.raises(ValueError, match="paired baseline arithmetic"):
        driver.validated_result(tmp_path, manifest, receipt)
    result["measurements"][0]["optimizer_samples"][0]["steps"] = 19
    driver.write(tmp_path / "result.json", result)
    with pytest.raises(ValueError, match="incomplete optimizer samples"):
        driver.validated_result(tmp_path, manifest, receipt)


def test_optimizer_followup_oom_is_a_failure(driver, tmp_path, monkeypatch):
    (tmp_path / "point").mkdir()
    curve = dict(id="500m-c4096", current="point", next_batch=8, complete=False, blocked=None)
    monkeypatch.setattr(driver, "validated_result", lambda *args: {"status": "cuda_oom"})
    driver.finish_point(tmp_path, {"mode": "optimizer_timing"}, curve, {"batch_size": 8},
                        {"state": "COMPLETED", "exit_code": "0:0"})
    assert not curve["complete"] and curve["blocked"] == "cuda_oom"


def test_paired_workers_share_deadline_and_enable_events_only_second(driver, tmp_path, monkeypatch):
    driver.write(tmp_path / "manifest.json", dict(mode="optimizer_timing", protocol=driver.PROTOCOL))
    waits, commands = [], []
    moments = iter([0., 0., 51., 120.])
    monkeypatch.setattr(driver.time, "monotonic", lambda: next(moments))

    class Process:
        def __init__(self, argv, start_new_session):
            assert start_new_session
            commands.append(argv)
            self.path = Path(argv[argv.index("--output") + 1])

        def wait(self, timeout):
            waits.append(timeout)
            result = dict(status="ok", milliseconds_per_update=10., coefficient_of_variation=0.,
                          stable=True, optimizer_milliseconds_per_update=1.)
            driver.write(self.path, result)
            return 0

    monkeypatch.setattr(driver.subprocess, "Popen", Process)
    assert driver.execute(tmp_path, "point") == 0
    assert waits == [780., 729.]
    assert "--optimizer-timing" not in commands[0] and "--optimizer-timing" in commands[1]
    result = driver.read(tmp_path / "point/result.json")
    assert result["optimizer_fraction"] == 0.1 and result["paired_elapsed_seconds"] == 120.


@pytest.mark.cuda
@pytest.mark.parametrize("preset", ["250m", "500m"])
def test_full_size_optimizer_reduction(preset):
    from tiny_llm.gh200.model import GH200Model
    from tiny_llm.gh200.optimizer import ArenaAdamW
    from tiny_llm.runtime import setup_runtime

    if not torch.cuda.is_available() or "GH200" not in torch.cuda.get_device_name():
        pytest.skip("GH200 required")
    config = Config(model=ModelConfig(**PRESETS[preset]))
    device = setup_runtime(config)
    model = GH200Model(config.model, 1, device)
    optimizer = ArenaAdamW(model, config)
    # Uniform, known gradients test all descriptor blocks and the full-size norm reduction.
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.001)
    optimizer.step(torch.ones(1, device=device))
    optimizer.inspect(1)
    expected_norm = math.sqrt(config.model.parameter_count) * 0.001
    assert optimizer.norms.item() == pytest.approx(expected_norm, rel=2e-6)
    assert optimizer.scales.item() == pytest.approx(1 / (expected_norm + 1e-6), rel=2e-6)
    for entry in model.layout:
        moment = model.parameter_view(optimizer.first_moment_storage, entry)
        expected = 0.001 / (expected_norm + 1e-6) * (1 - config.optimizer.beta1)
        torch.testing.assert_close(moment, torch.full_like(moment, expected), rtol=2e-6, atol=1e-12)


@pytest.mark.cuda
@pytest.mark.parametrize("preset", ["250m", "500m"])
def test_large_captured_throughput(preset, tmp_path):
    if not torch.cuda.is_available() or "GH200" not in torch.cuda.get_device_name():
        pytest.skip("GH200 required")
    config = Config.model_validate(dict(
        model=ModelConfig(**PRESETS[preset]).model_dump(),
        training=dict(micro_batch_size=1, batch_tokens=1024),
    ))
    result = benchmark_throughput(config, tmp_path / "result.json", warmup=2, windows=2,
                                  window_seconds=0.1)
    assert result["status"] == "ok", result
    assert result["successful_updates"] >= 47
    assert result["tokens_per_second"] > 0
    assert result["measurement_peak_allocated_bytes"] > config.model.parameter_count * 16
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "ok"


@pytest.mark.cuda
def test_optimizer_event_capture_preserves_updates():
    from tiny_llm.gh200.engine import GH200Update
    from tiny_llm.runtime import setup_runtime
    from tiny_llm.throughput import CapturedUpdateTimer

    if not torch.cuda.is_available() or "GH200" not in torch.cuda.get_device_name():
        pytest.skip("GH200 required")
    config = Config.model_validate(dict(
        model=dict(vocab_size=32, layers=1, width=320, heads=5, ffn_width=896,
                   context_length=16),
        training=dict(micro_batch_size=1, batch_tokens=16),
    ))
    engines = []
    timer = None
    for instrumented in (False, True):
        device = setup_runtime(config)
        engine = GH200Update(config, device)
        if instrumented:
            timer = CapturedUpdateTimer(engine)
        engine.prepare()
        engines.append(engine)
    assert timer is not None
    x = torch.arange(16, device="cuda").view(1, 16)

    def batch(count, device):
        return x, (x + 1) % 32

    samples = []
    for step in range(1, 5):
        for engine in engines:
            engine.execute(batch, 0.001)
            engine.inspect(step)
        torch.cuda.synchronize()
        samples.append(timer.sample())
        # Event nodes change neither canonical weights nor optimizer state/counters.
        for key, value in engines[0]._persistent().items():
            torch.testing.assert_close(value, engines[1]._persistent()[key], rtol=0, atol=0)
    assert all(0 < row["optimizer_ms"] < row["graph_ms"] for row in samples)


@pytest.mark.cuda
def test_optimizer_timed_worker(tmp_path):
    if not torch.cuda.is_available() or "GH200" not in torch.cuda.get_device_name():
        pytest.skip("GH200 required")
    config = Config.model_validate(dict(
        model=ModelConfig(**PRESETS["20m"]).model_dump(),
        training=dict(micro_batch_size=1, batch_tokens=1024),
    ))
    result = benchmark_throughput(config, tmp_path / "timed.json", warmup=2, windows=2,
                                  window_seconds=0.2, optimizer_timing=True)
    assert result["status"] == "ok", result
    assert 0 < result["optimizer_fraction_instrumented"] < 1
    assert result["optimizer_milliseconds_per_update"] > 0
    for row in result["measurements"]:
        assert sum(sample["steps"] for sample in row["optimizer_samples"]) == row["steps"]
    assert result["successful_updates"] == 7 + sum(row["steps"] for row in result["measurements"])
