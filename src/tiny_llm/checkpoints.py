"""Full training checkpoints and compact, independently inspectable worker snapshots."""

import hashlib
import math
from pathlib import Path

import torch

from tiny_llm.config import Config
from tiny_llm.data import training_boundaries
from tiny_llm.packed import PackedLlama
from tiny_llm.runtime import atomic_checkpoint, preserve_rng

RETENTION_FIELDS = {"checkpoint_policy", "save_epoch_training_state"}
POSITION_FIELDS = ("recipe_identity", "cursor", "step", "completed_epochs")


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compact_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone(memory_format=torch.contiguous_format)
    if isinstance(value, dict):
        return {key: compact_cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(compact_cpu(item) for item in value)
    return value


def require_new_epoch(path):
    if path.exists():
        raise ValueError(
            f"committed epoch snapshot already exists: {path}; use a new output directory"
        )


def save_epoch_checkpoint(path, state, model, optimizer):
    """Publish the root last; an interrupted uncommitted epoch may be rewritten."""
    path = Path(path)
    require_new_epoch(path)
    if not isinstance(model, PackedLlama):
        atomic_checkpoint(path, state)
        return
    shared = {
        key: value
        for key, value in state.items()
        if key not in ("model", "optimizer", "worker_files")
    }
    shared.update(version=4, kind="packed_epoch", num_workers=model.num_models, workers=[])
    position = {key: state[key] for key in POSITION_FIELDS}
    names = [entry.name for entry in model.layout]
    for worker, (local_optimizer, parameters) in enumerate(
        zip(optimizer.optimizers, optimizer.local_parameters, strict=True)
    ):
        name_by_id = {id(p): name for name, p in zip(names, parameters, strict=True)}
        local = dict(
            version=1,
            kind="packed_worker",
            worker=worker,
            num_workers=model.num_models,
            model_config=state["config"]["model"],
            **position,
            model=model.local_state_dict(worker),
            optimizer=dict(
                state={
                    name: local_optimizer.state[p]
                    for name, p in zip(names, parameters, strict=True)
                },
                param_groups=[
                    {**group, "params": [name_by_id[id(p)] for p in group["params"]]}
                    for group in local_optimizer.param_groups
                ],
            ),
        )
        relative = Path(f"node-{worker:03d}") / path.name
        destination = path.parent / relative
        destination.parent.mkdir(exist_ok=True)
        atomic_checkpoint(destination, compact_cpu(local))
        shared["workers"].append(
            dict(worker=worker, path=relative.as_posix(), sha256=file_hash(destination))
        )
        del local
    atomic_checkpoint(path, shared)


def _validate_worker(local, root, config, model, index):
    if (
        local.get("kind") != "packed_worker"
        or local.get("version") != 1
        or local.get("worker") != index
        or local.get("num_workers") != model.num_models
        or local.get("model_config") != root["config"]["model"]
        or any(local.get(key) != root[key] for key in POSITION_FIELDS)
    ):
        raise ValueError("incompatible worker identity or training position")
    names = {entry.name for entry in model.layout}
    if set(local["model"]) != names or set(local["optimizer"]["state"]) != names:
        raise ValueError("incompatible worker parameter names")
    for entry in model.layout:
        parameter = local["model"][entry.name]
        moments = local["optimizer"]["state"][entry.name]
        if set(moments) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("incompatible worker optimizer state")
        for tensor in (parameter, moments["exp_avg"], moments["exp_avg_sq"]):
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != entry.shape
                or tensor.dtype != model.parameter_storage.dtype
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError("invalid worker parameter/moment tensor")
        step = moments["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.ndim != 0
            or not torch.isfinite(step)
            or step.item() != root["step"]
        ):
            raise ValueError("invalid worker optimizer counter")
        if (moments["exp_avg_sq"] < 0).any():
            raise ValueError("negative worker second moment")
    groups = local["optimizer"]["param_groups"]
    expected_names = [
        [entry.name for entry in model.layout if len(entry.shape) >= 2],
        [entry.name for entry in model.layout if len(entry.shape) < 2],
    ]
    if len(groups) != 2:
        raise ValueError("incompatible worker parameter groups")
    for group, expected, decay in zip(
        groups, expected_names, (config.optimizer.weight_decay, 0.0), strict=True
    ):
        if (
            group["params"] != expected
            or tuple(group["betas"]) != (config.optimizer.beta1, config.optimizer.beta2)
            or group["eps"] != config.optimizer.eps
            or group["weight_decay"] != decay
            or group["amsgrad"]
            or group["maximize"]
            or not math.isfinite(group["lr"])
            or group["lr"] < 0
        ):
            raise ValueError("incompatible worker parameter groups")


def load_training_checkpoint(path):
    """Return the existing combined representation, without changing live training state."""
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("kind") == "packed_worker":
        raise ValueError("a worker file is not resumable alone; load the root epoch checkpoint")
    if state.get("version") != 4 and state.get("kind") != "packed_epoch":
        return state
    try:
        return _load_packed_epoch(path, state)
    except (KeyError, TypeError, IndexError, RuntimeError) as exc:
        raise ValueError(f"invalid packed epoch checkpoint: {path}: {exc}") from exc


def _load_packed_epoch(path, state):
    if state.get("version") != 4 or state.get("kind") != "packed_epoch":
        raise ValueError("unsupported packed epoch format")
    config = Config.model_validate(state["config"])
    count = state["num_workers"]
    if config.decentralized is None or count != config.decentralized.num_models:
        raise ValueError("incompatible packed epoch worker count")
    if "model" in state or "optimizer" in state:
        raise ValueError("packed epoch root must contain only shared state")
    boundaries = training_boundaries(config, config.model.parameter_count)
    epoch = state["completed_epochs"]
    if (
        not 1 <= epoch <= len(boundaries)
        or state["cursor"] != boundaries[epoch - 1]
        or state["loader"]["cursor"] != state["cursor"]
    ):
        raise ValueError("incompatible packed epoch position")
    members = state["workers"]
    if len(members) != count or [member["worker"] for member in members] != list(range(count)):
        raise ValueError("incomplete packed epoch workers")
    # Assemble only into temporary CPU arenas. A caller receives nothing on failure.
    with preserve_rng():
        model = PackedLlama(config.model, count, "reference")
    first = torch.zeros_like(model.parameter_storage)
    second = torch.zeros_like(first)
    optimizers = []
    common_groups = None
    for index, member in enumerate(members):
        expected = f"node-{index:03d}/{path.name}"
        if member["path"] != expected:
            raise ValueError("incompatible worker file reference")
        source = path.parent / expected
        # Hash and deserialize the same open file; an atomic replacement cannot race this read.
        with source.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != member["sha256"]:
                raise ValueError(f"worker checksum mismatch: {source}")
            handle.seek(0)
            local = torch.load(handle, map_location="cpu", weights_only=False)
        _validate_worker(local, state, config, model, index)
        groups = local["optimizer"]["param_groups"]
        if common_groups is not None and groups != common_groups:
            raise ValueError("inconsistent worker parameter groups")
        common_groups = groups
        model.load_local_state_dict(index, local["model"])
        steps = []
        for entry in model.layout:
            moments = local["optimizer"]["state"][entry.name]
            model.parameter_view(first, entry)[index].copy_(moments["exp_avg"])
            model.parameter_view(second, entry)[index].copy_(moments["exp_avg_sq"])
            steps.append(moments["step"].clone())
        optimizers.append(
            dict(
                steps=steps, groups=[{k: v for k, v in g.items() if k != "params"} for g in groups]
            )
        )
        del local
    return dict(
        {
            key: value
            for key, value in state.items()
            if key not in ("kind", "num_workers", "workers")
        },
        version=3,
        model=model.packed_state_dict(),
        optimizer=dict(
            version=1,
            layout=model.layout_metadata(),
            first_moment_storage=first,
            second_moment_storage=second,
            workers=optimizers,
        ),
        worker_files=members,
    )
