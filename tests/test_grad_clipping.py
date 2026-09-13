import json

import pytest
import torch

from tiny_llm.config import load_config
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


def test_clipping_config(tmp_path):
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
def test_clipping_identity_resume_and_logging(tiny_config, cache_dir, workers):
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
    config.optimizer.grad_clip = 1.0
    with pytest.raises(ValueError, match="recipe"):
        train(config, checkpoint)
