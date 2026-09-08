import json
import math

import pytest

from tiny_llm.config import save_config
from tiny_llm.experiments import sweep
from tiny_llm.runtime import atomic_json


def test_twelve_run_promotions_and_report(tiny_config, cache_dir, tmp_path, monkeypatch):
    import tiny_llm.experiments as module

    benchmarks = tmp_path / "benchmarks"
    root = tmp_path / "campaign"
    for preset in ("20m", "50m", "90m"):
        directory = benchmarks / preset
        directory.mkdir(parents=True)
        atomic_json(
            directory / "summary.json",
            {
                "selected": {
                    "micro_batch_size": 2,
                    "compile": False,
                    "tokens_per_second": 1000,
                }
            },
        )
    stages = []

    def stage(configs, gpus, stage_dir):
        assert gpus == ["0", "1"]
        stages.append(configs)
        for config in configs:
            path = config.runtime.output_dir
            path.mkdir(exist_ok=True)
            save_config(config, path / "resolved.yaml")
            opt = config.optimizer
            loss = 3 + abs(math.log(opt.lr / 0.001)) + opt.weight_decay / 10 + abs(opt.beta2 - 0.99)
            atomic_json(
                path / "result.json",
                dict(
                    status="complete",
                    final_validation={"loss": loss},
                    parameters=config.model.parameter_count,
                    tokens=64,
                    peak_memory_bytes=0,
                ),
            )
            (path / "metrics.jsonl").write_text(
                json.dumps({"event": "validation", "tokens": 64, "loss": loss}) + "\n"
            )

    monkeypatch.setattr(module, "run_stage", stage)
    sweep(tiny_config, root, benchmarks, ["0", "1"])
    assert list(map(len, stages)) == [6, 4, 2]
    assert {c.optimizer.lr for c in stages[1]} == {0.001}
    assert {c.optimizer.weight_decay for c in stages[1]} == {0, 0.1}
    assert {c.optimizer.beta2 for c in stages[1]} == {0.95, 0.99}
    assert {c.optimizer.beta2 for c in stages[2]} == {0.99}
    assert json.loads((root / "complete.json").read_text())["runs"] == 12
    assert len(json.loads((root / "comparison.json").read_text())) == 12
    assert (root / "learning-curves.png").exists()
    assert (root / "selected-90m.yaml").exists()
    tiny_config.optimizer.beta1 = 0.8
    with pytest.raises(ValueError, match="settings changed"):
        sweep(tiny_config, root, benchmarks, ["0", "1"])
