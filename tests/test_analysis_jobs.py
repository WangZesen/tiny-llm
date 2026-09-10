import copy
import json
import subprocess
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file

from tiny_llm.analysis import AnalysisOptions, analyze, load_checkpoint
from tiny_llm.analysis import slurm as jobs
from tiny_llm.config import save_config
from tiny_llm.model import Llama


@pytest.fixture
def timing():
    return dict(profile="fixture", checkpoint_seconds=4293.6, startup_seconds=60)


def checkpoints(count=40):
    rows = [
        dict(name=f"epoch-{index + 1:03d}.safetensors", weights_hash=str(index), tokens=index)
        for index in range(count)
    ]
    return rows + [dict(rows[-1], name="final.pt"), dict(rows[-1], name="latest.pt")]


def test_static_shards_aliases_and_limits(timing):
    plan = jobs.plan_shards(checkpoints(), 4, timing)
    assert [row["unique_checkpoints"] for row in plan["jobs"]] == [10] * 4
    assert plan["checkpoint_files"] == 42 and plan["unique_checkpoints"] == 40
    assert all(row["hours"] == 25 for row in plan["jobs"])
    assert [row["name"] for row in plan["jobs"][0]["checkpoints"]][:2] == [
        "epoch-001.safetensors",
        "epoch-005.safetensors",
    ]
    assert {row["name"] for row in plan["jobs"][3]["checkpoints"]} >= {
        "epoch-040.safetensors",
        "final.pt",
        "latest.pt",
    }
    assert plan["estimated_gpu_hours"] * 3600 == pytest.approx(
        sum(row["estimated_seconds"] for row in plan["jobs"])
    )
    uneven = jobs.plan_shards(checkpoints(5), 3, timing)
    assert [row["unique_checkpoints"] for row in uneven["jobs"]] == [2, 2, 1]
    for count in (0, -1, 41):
        with pytest.raises(ValueError, match="jobs must"):
            jobs.plan_shards(checkpoints(), count, timing)
    with pytest.raises(ValueError, match="at least --jobs 2"):
        jobs.plan_shards(checkpoints(), 1, timing)
    with pytest.raises(ValueError, match="one checkpoint exceeds"):
        jobs.plan_shards(checkpoints(), 4, timing, max_hours=1)
    same_average = [
        dict(name=name, tokens=i, weights_hash="average", result_key=key)
        for i, (name, key) in enumerate((("one", "a"), ("two", "b"), ("alias", "a")))
    ]
    split = jobs.plan_shards(same_average, 2, None, walltime_hours=2)
    assert split["unique_checkpoints"] == 2
    assert [row["name"] for row in split["jobs"][0]["checkpoints"]] == ["one", "alias"]


def test_measured_profile_and_explicit_walltime(tiny_config):
    from dataclasses import replace

    from tiny_llm.config import PRESETS, Config, ModelConfig

    cfg = Config(model=ModelConfig(**PRESETS["20m"]))
    cfg.training.micro_batch_size = 32
    cfg.training.batch_tokens = 32768
    data = [dict(blocks=9963, available_noise_batches=312)] * 2
    options = AnalysisOptions(device="cuda", amp=True, hvp_batch_size=64)
    timing = jobs.measured_timing(cfg, data, options)
    assert timing["checkpoint_seconds"] == pytest.approx(4293.6)
    assert jobs.measured_timing(cfg, data, replace(options, amp=False)) is None
    assert jobs.measured_timing(tiny_config, data, options) is None
    all_samples = jobs.measured_timing(cfg, data, replace(options, noise_samples="all"))
    assert all_samples["checkpoint_seconds"] > timing["checkpoint_seconds"]
    with pytest.raises(ValueError, match="provide --walltime-hours"):
        jobs.plan_shards(checkpoints(), 4, None)
    explicit = jobs.plan_shards(checkpoints(), 4, None, walltime_hours=12)
    assert all(row["hours"] == 12 for row in explicit["jobs"])
    assert explicit["estimated_gpu_hours"] is None
    for hours in (0, 73):
        with pytest.raises(ValueError, match="walltime must"):
            jobs.plan_shards(checkpoints(), 4, None, walltime_hours=hours)


@pytest.fixture
def saved_shards(tmp_path, tiny_config, cache_dir, monkeypatch, request):
    import tiny_llm.analysis.core as analysis
    from tiny_llm.config import DecentralizedConfig
    from tiny_llm.packed import PackedLlama

    cfg = tiny_config
    packed = getattr(request, "param", False)
    if packed:
        cfg.decentralized = DecentralizedConfig(num_models=2)
    run = cfg.runtime.output_dir
    run.mkdir()
    save_config(cfg, run / "resolved.yaml")
    infos = []
    for epoch in (1, 2):
        path = run / f"epoch-{epoch:03d}.safetensors"
        model = Llama(cfg.model)
        if packed:
            local = PackedLlama(cfg.model, 2)
            for index, delta in enumerate((-0.125, 0.125)):
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                state["norm.weight"].add_(delta)
                local.load_local_state_dict(index, state)
                directory = run / f"node-{index:03d}"
                directory.mkdir(exist_ok=True)
                save_file(state, directory / path.name)
            local.copy_average_to(model)
        save_file(model.state_dict(), str(path))
        _, info = load_checkpoint(cfg, path)
        infos.append(info)
    source_hash = jobs.source_digest(jobs.Path(jobs.__file__).parents[1])
    monkeypatch.setattr(
        analysis,
        "environment",
        lambda: dict(
            source_hash=source_hash, versions={"torch": "fixture"}, gpu=None, hostname="worker"
        ),
    )
    options = AnalysisOptions(device="cpu", noise_samples=1, random_samples=1, compile_hvp=False)
    root = tmp_path / "campaign"
    root.mkdir()
    shards = []
    for index, info in enumerate(infos):
        output = root / "shards" / str(index)
        from tiny_llm.analysis import EpochData, summarize
        from tiny_llm.data import TokenCache

        output.mkdir(parents=True)
        data = [
            EpochData(TokenCache(cache_dir), cfg, case).metadata() for case in ("seen", "unseen")
        ]
        manifest = dict(
            identity="fixture",
            run=str(run),
            options=asdict(options),
            data=data,
            micro_batch_size=cfg.training.micro_batch_size,
            environment=dict(source_hash=source_hash),
            checkpoints=[info],
        )
        jobs.atomic_json(output / "manifest.json", manifest)
        for row in data:
            noise = [dict(batch_index=0, quadratic=1.0, noise_norm_squared=1.0, gradient_norm=2.0)]
            random = [dict(index=0, quadratic=2.0)]
            result = dict(
                identity="fixture",
                weights_hash=info["weights_hash"],
                case=row["case"],
                data=row,
                noise=noise,
                random=random,
                statistics=summarize(noise, random, cfg.model.parameter_count, 0.5),
            )
            if packed:
                consensus = [
                    dict(
                        worker=i,
                        norm_squared=1.0,
                        norm=1.0,
                        quadratic=-0.5,
                        normalized_alignment=-0.5,
                        hvp_seconds=1.0,
                        total_seconds=1.1,
                    )
                    for i in range(2)
                ]
                result["consensus"] = consensus
                result["statistics"].update(analysis.consensus_statistics(consensus))
                result["result_key"] = jobs.result_key(info)
            jobs.atomic_json(output / f"{jobs.result_key(info)}-{row['case']}.json", result)
        shards.append(
            dict(
                index=index,
                checkpoints=[info],
                output=str(output),
                hours=2,
                unique_checkpoints=1,
                attempts=[dict(job_id=str(100 + index))],
            )
        )
    return dict(
        run=str(run),
        output=str(root),
        options=asdict(options),
        jobs=shards,
        data=manifest["data"],
        micro_batch_size=cfg.training.micro_batch_size,
        source_hash=source_hash,
        source_snapshot=str(root / "source"),
        version=2,
    )


@pytest.mark.parametrize("saved_shards", [False, True], indirect=True, ids=["ordinary", "packed"])
def test_collect_matches_serial_and_ignores_worker_provenance(saved_shards, monkeypatch):
    plan = saved_shards
    # Only this integration test performs actual analysis and plotting.
    for shard in plan["jobs"]:
        directory = jobs.Path(shard["output"])
        for path in directory.glob("*.json"):
            path.unlink()
        analyze(
            jobs.Path(plan["run"]),
            [r["name"] for r in shard["checkpoints"]],
            directory,
            AnalysisOptions(**plan["options"]),
            plots=False,
        )
    second = jobs.Path(plan["jobs"][1]["output"]) / "manifest.json"
    metadata = json.loads(second.read_text())
    metadata["environment"].update(hostname="different-node", slurm={"SLURM_JOB_ID": "101"})
    second.write_text(json.dumps(metadata))
    combined = jobs.collect_results(plan)
    assert len(combined["workers"]) == 2 and len(combined["checkpoints"]) == 2
    root = jobs.Path(plan["output"])
    assert len(list(root.glob("*.png"))) == len(list(root.glob("*.pdf"))) == 3
    assert (root / "summary.csv").is_file()
    serial = root / "serial"
    analyze(
        jobs.Path(plan["run"]), ["all"], serial, AnalysisOptions(**plan["options"]), plots=False
    )
    for path in root.glob("*-seen.json"):
        parallel = json.loads(path.read_text())
        single = json.loads((serial / path.name).read_text())
        assert parallel["statistics"] == single["statistics"]
        assert [row["batch_index"] for row in parallel["noise"]] == [
            row["batch_index"] for row in single["noise"]
        ]
        for a, b in zip(parallel.get("consensus", []), single.get("consensus", []), strict=True):
            assert {k: v for k, v in a.items() if not k.endswith("seconds")} == {
                k: v for k, v in b.items() if not k.endswith("seconds")
            }
    # Running with plotting disabled does not change numerical identity or resume behavior.
    assert combined["identity"] == json.loads((serial / "manifest.json").read_text())["identity"]
    import tiny_llm.analysis.core as analysis

    monkeypatch.setattr(analysis, "measure", lambda *args: pytest.fail("recomputed saved cases"))
    analyze(
        jobs.Path(plan["run"]), ["all"], serial, AnalysisOptions(**plan["options"]), plots=False
    )


@pytest.mark.parametrize("saved_shards", [True], indirect=True)
def test_consensus_collection_validation_and_preflight(saved_shards, monkeypatch):
    from tiny_llm.analysis import slurm

    plan = saved_shards
    directory = jobs.Path(plan["jobs"][0]["output"])
    path = next(directory.glob("*-seen.json"))
    original = json.loads(path.read_text())
    for damage in ("missing", "worker", "scalar", "statistic", "key"):
        row = copy.deepcopy(original)
        if damage == "missing":
            row["consensus"].pop()
        elif damage == "worker":
            row["consensus"][1]["worker"] = 0
        elif damage == "scalar":
            row["consensus"][0]["norm"] = float("inf")
        elif damage == "key":
            row["result_key"] = "wrong"
        else:
            row["statistics"]["consensus_alignment"] = 9
        path.write_text(json.dumps(row))
        with pytest.raises(ValueError):
            jobs.collect_results(plan)
    path.write_text(json.dumps(original))
    monkeypatch.setattr(slurm, "partition_hours", lambda: 72)
    options = AnalysisOptions(**(plan["options"] | dict(device="cuda")))
    prepared = slurm.prepare_plan(
        jobs.Path(plan["run"]), ["all"], directory / "new", 2, options, walltime_hours=2
    )
    assert prepared["timing"] is None
    assert all(shard["additional_hvp_passes"] == 4 for shard in prepared["jobs"])
    assert all(list(shard["consensus_workers"].values()) == [2] for shard in prepared["jobs"])
    next((jobs.Path(plan["run"]) / "node-001").glob("*.safetensors")).unlink()
    with pytest.raises(ValueError, match="missing worker"):
        slurm.prepare_plan(
            jobs.Path(plan["run"]), ["all"], directory / "new", 2, options, walltime_hours=2
        )


@pytest.mark.parametrize("damage", ["missing", "identity", "samples", "checkpoint", "options"])
def test_collection_rejects_incomplete_or_incompatible(saved_shards, damage):
    plan = saved_shards
    directory = jobs.Path(plan["jobs"][1]["output"])
    path = next(directory.glob("*-seen.json"))
    if damage == "missing":
        path.unlink()
    elif damage in ("identity", "samples"):
        result = json.loads(path.read_text())
        if damage == "identity":
            result["identity"] = "bad"
        else:
            result["noise"] = []
        path.write_text(json.dumps(result))
    else:
        path = directory / "manifest.json"
        result = json.loads(path.read_text())
        if damage == "checkpoint":
            result["checkpoints"][0]["tokens"] += 1
        else:
            result["options"]["seed"] += 1
        path.write_text(json.dumps(result))
    with pytest.raises((ValueError, FileNotFoundError)):
        jobs.collect_results(plan)
    assert not (jobs.Path(plan["output"]) / "summary.csv").exists()


def test_scheduler_accounting_lag_and_step_rows(monkeypatch):
    def execute(command, **kwargs):
        if command[0] == "squeue":
            return SimpleNamespace(stdout="100|COMPLETING\n")
        return SimpleNamespace(
            stdout=("100|COMPLETED|0:0\n100.batch|FAILED|1:0\n101|CANCELLED by 123|0:15\n")
        )

    monkeypatch.setattr(jobs.subprocess, "run", execute)
    states = jobs.scheduler_states(["100", "101", "102"])
    assert states["100"] == dict(state="COMPLETING", exit_code=None)
    assert states["101"]["state"] == "CANCELLED"
    assert states["102"]["state"] == "UNKNOWN"


def test_waits_for_every_job_before_failure(saved_shards, monkeypatch):
    plan = saved_shards
    calls = iter(
        [
            {
                "100": dict(state="FAILED", exit_code="1:0"),
                "101": dict(state="RUNNING", exit_code=None),
            },
            {
                "100": dict(state="FAILED", exit_code="1:0"),
                "101": dict(state="COMPLETED", exit_code="0:0"),
            },
        ]
    )
    monkeypatch.setattr(jobs, "scheduler_states", lambda ids: next(calls))
    sleeps = []
    monkeypatch.setattr(jobs.time, "sleep", sleeps.append)
    receipt = jobs.Path(plan["output"]) / "submission.json"
    with pytest.raises(RuntimeError, match="analysis jobs failed"):
        jobs.wait_and_collect(plan, receipt)
    assert sleeps == [60]
    assert json.loads(receipt.read_text())["status"] == "failed"
    assert not (jobs.Path(plan["output"]) / "summary.csv").exists()


def test_wait_unknown_then_success(saved_shards, monkeypatch):
    plan = saved_shards
    calls = iter(
        [
            {
                "100": dict(state="UNKNOWN", exit_code=None),
                "101": dict(state="COMPLETED", exit_code="0:0"),
            },
            {key: dict(state="COMPLETED", exit_code="0:0") for key in ("100", "101")},
        ]
    )
    monkeypatch.setattr(jobs, "scheduler_states", lambda ids: next(calls))
    monkeypatch.setattr(jobs.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(jobs, "plot_analysis", lambda output: None)
    result = jobs.wait_and_collect(plan, jobs.Path(plan["output"]) / "submission.json")
    assert result["status"] == "complete"


def test_submission_resume_and_commands(saved_shards, monkeypatch):
    plan = saved_shards
    states = {
        "100": dict(state="COMPLETED", exit_code="0:0"),
        "101": dict(state="TIMEOUT", exit_code="0:0"),
    }
    monkeypatch.setattr(jobs, "scheduler_states", lambda ids: states)
    commands = []
    monkeypatch.setattr(
        jobs.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command) or SimpleNamespace(stdout="102\n"),
    )
    receipt = jobs.Path(plan["output"]) / "submission.json"
    jobs.submit_pending(plan, receipt, resume=True)
    assert len(commands) == 1
    command = commands[0]
    assert "--gpus=1" in command and "--no-plots" in command
    assert "--consensus" in command
    assignment = jobs.Path(command[command.index("--checkpoint-manifest") + 1])
    assert json.loads(assignment.read_text()) == plan["jobs"][1]["checkpoints"]
    assert not any("dependency" in value for value in command)
    assert "epoch-002.safetensors" in command and "epoch-001.safetensors" not in command
    assert len(plan["jobs"][0]["attempts"]) == 1
    assert len(plan["jobs"][1]["attempts"]) == 2
    states["102"] = dict(state="RUNNING", exit_code=None)
    jobs.submit_pending(plan, receipt, resume=True)
    assert len(commands) == 1  # Reattachment does not submit again.


def test_partial_submission_and_interruption(saved_shards, monkeypatch):
    plan = saved_shards
    for shard in plan["jobs"]:
        shard["attempts"] = []
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, command, stderr="scheduler unavailable")
        return SimpleNamespace(stdout="110\n")

    monkeypatch.setattr(jobs.subprocess, "run", execute)
    receipt = jobs.Path(plan["output"]) / "submission.json"
    with pytest.raises(subprocess.CalledProcessError):
        jobs.submit_pending(plan, receipt, resume=False)
    saved = json.loads(receipt.read_text())
    assert saved["jobs"][0]["attempts"][0]["job_id"] == "110"
    assert not saved["jobs"][1]["attempts"]
    assert not saved["jobs"][1]["submission_uncertain"]
    monkeypatch.setattr(
        jobs, "scheduler_states", lambda ids: {"110": dict(state="RUNNING", exit_code=None)}
    )
    monkeypatch.setattr(
        jobs.time, "sleep", lambda seconds: (_ for _ in ()).throw(KeyboardInterrupt)
    )
    with pytest.raises(KeyboardInterrupt):
        jobs.wait_and_collect(dict(plan, jobs=plan["jobs"][:1]), receipt)
    assert len(calls) == 2  # No implicit cancellation.


def test_uncertain_submission_is_not_duplicated(saved_shards, monkeypatch):
    plan = copy.deepcopy(saved_shards)
    for shard in plan["jobs"]:
        shard["attempts"] = []
    plan["jobs"][0]["submission_uncertain"] = True
    with pytest.raises(RuntimeError, match="uncertain submission"):
        jobs.submit_pending(plan, jobs.Path(plan["output"]) / "submission.json", resume=True)


def test_dry_run_does_not_create_outputs(tmp_path, monkeypatch):
    expected = dict(jobs=[dict(index=0)])
    monkeypatch.setattr(jobs, "prepare_plan", lambda *args: expected)
    output = tmp_path / "absent"
    assert (
        jobs.orchestrate(tmp_path, ["all"], output, 1, AnalysisOptions(device="cuda"), dry_run=True)
        == expected
    )
    assert not output.exists()


def test_resume_checks_saved_settings_and_source(saved_shards, monkeypatch):
    plan = saved_shards
    root = jobs.Path(plan["output"])
    plan.update(mode="independent-single-gpu", selected=["all"], config_hash="not-current")
    plan["options"]["device"] = "cuda"
    (root / "submission.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="resume arguments"):
        jobs.orchestrate(
            jobs.Path(plan["run"]),
            ["all"],
            root,
            3,
            AnalysisOptions(**plan["options"]),
            resume=True,
        )
    with pytest.raises(ValueError, match="saved source or resolved.yaml changed"):
        jobs.orchestrate(
            jobs.Path(plan["run"]),
            ["all"],
            root,
            2,
            AnalysisOptions(**plan["options"]),
            resume=True,
        )


def test_orchestrator_lock_is_exclusive(saved_shards):
    plan = saved_shards
    root = jobs.Path(plan["output"])
    with (root / ".orchestrator.lock").open("a") as lock:
        jobs.fcntl.flock(lock, jobs.fcntl.LOCK_EX | jobs.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            jobs.orchestrate(
                jobs.Path(plan["run"]), ["all"], root, 2, AnalysisOptions(device="cuda")
            )


def test_repository_resolution_from_saved_source(tmp_path, monkeypatch):
    import tiny_llm.analysis.slurm as module

    repository = tmp_path / "repo"
    (repository / "scripts").mkdir(parents=True)
    (repository / "scripts/slurm.sh").write_text("#!/bin/sh\n")
    (repository / "pyproject.toml").write_text("")
    snapshot = repository / "runs/analysis/source/tiny_llm/analysis/slurm.py"
    monkeypatch.setattr(module, "__file__", str(snapshot))
    assert module.repository_root() == repository
    # Source snapshots may also live outside the checkout.
    monkeypatch.setattr(module, "__file__", str(tmp_path / "external/source/tiny_llm/module.py"))
    monkeypatch.chdir(repository)
    assert module.repository_root() == repository


def test_recursive_source_snapshot(tmp_path):
    package = tmp_path / "package"
    nested = package / "analysis"
    nested.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    child = nested / "core.py"
    child.write_text("value = 1\n")
    before = jobs.source_digest(package)
    child.write_text("value = 2\n")
    assert jobs.source_digest(package) != before
    output = tmp_path / "output"
    plan = dict(
        source_snapshot=str(output / "source"),
        source_hash=jobs.source_digest(jobs.Path(jobs.__file__).parents[1]),
    )
    jobs.snapshot_source(plan)
    assert (output / "source/tiny_llm/analysis/core.py").is_file()
    assert (output / "source/tiny_llm/cli.py").is_file()


@pytest.mark.parametrize("exit_code", [0, 7])
def test_launcher_temporary_caches_and_cleanup(tmp_path, exit_code):
    import os

    setup = tmp_path / "setup.sh"
    setup.write_text("""source() { :; }
uv() { :; }
srun() {
    printf '%s\\n%s\\n' "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" > "$ANALYSIS_TEST_REPORT"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
    return "$ANALYSIS_TEST_EXIT"
}
""")
    scratch = tmp_path / "node-scratch"
    scratch.mkdir()
    report = tmp_path / "paths"
    env = dict(
        os.environ,
        BASH_ENV=str(setup),
        SLURM_SUBMIT_DIR=str(tmp_path),
        SLURM_TMPDIR=str(scratch),
        TMPDIR=str(tmp_path),
        ANALYSIS_TEST_REPORT=str(report),
        ANALYSIS_TEST_EXIT=str(exit_code),
    )
    for preferred in (scratch, tmp_path):
        result = subprocess.run(
            ["bash", str(jobs.repository_root() / "scripts/slurm.sh"), "analyze"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == exit_code, result.stderr
        inductor, triton = [jobs.Path(p) for p in report.read_text().splitlines()]
        assert inductor.parent == triton.parent
        assert inductor.parent.parent == preferred
        assert not inductor.parent.exists()
        env["SLURM_TMPDIR"] = ""
