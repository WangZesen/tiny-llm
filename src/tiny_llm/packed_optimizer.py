"""Independent local optimizers with shared, inspectable state arenas."""

import torch
from torch import nn

from tiny_llm.config import Config
from tiny_llm.optimizers import (
    AccumAdamW,
    make_local_optimizer,
    validate_accum_group,
    validate_accum_state,
)
from tiny_llm.packed import PackedLlama
from tiny_llm.runtime import clip_grad_norm_
from tiny_llm.state import TorchState


class PackedOptimizer:
    optimizer_name = "adamw"

    def __init__(self, model: PackedLlama, config: Config):
        if config.optimizer.name != self.optimizer_name:
            raise ValueError(f"{type(self).__name__} requires optimizer.name={self.optimizer_name}")
        self.model = model
        self.layout = model.layout
        self._parameters = tuple(model.parameters())
        self._arena = model.parameter_storage
        self.first_moment_storage = torch.zeros_like(self._arena)
        self.second_moment_storage = torch.zeros_like(self._arena)
        self.accum_grad_storage = (
            torch.zeros_like(self._arena) if self.optimizer_name == "accumadamw" else None
        )
        self.optimizers: list[torch.optim.AdamW | AccumAdamW] = []
        self.local_parameters: list[list[nn.Parameter]] = []
        for worker in range(model.num_models):
            parameters = [nn.Parameter(p[worker].detach()) for p in self._parameters]
            optimizer = make_local_optimizer(parameters, config, self._arena.device)
            fused = optimizer.param_groups[0].get("fused", False)
            for entry, parameter in zip(self.layout, parameters, strict=True):
                optimizer.state[parameter] = {
                    "step": torch.zeros(
                        (),
                        dtype=torch.int64
                        if self.accum_grad_storage is not None
                        else torch.float64
                        if not fused and torch.get_default_dtype() == torch.float64
                        else torch.float32,
                        device=self._arena.device if fused else "cpu",
                    ),
                    "exp_avg": model.parameter_view(self.first_moment_storage, entry)[worker],
                    "exp_avg_sq": model.parameter_view(self.second_moment_storage, entry)[worker],
                }
                if self.accum_grad_storage is not None:
                    optimizer.state[parameter]["accum_grad"] = model.parameter_view(
                        self.accum_grad_storage, entry
                    )[worker]
            self.optimizers.append(optimizer)
            self.local_parameters.append(parameters)

    @property
    def param_groups(self):
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def _check_storage(self):
        if self.model.parameter_storage is not self._arena:
            raise RuntimeError(
                "model storage changed; construct the optimizer after model placement"
            )

    def zero_grad(self, set_to_none=True):
        self._check_storage()
        self.model.zero_grad(set_to_none=set_to_none)
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def bind_gradients(self):
        self._check_storage()
        for worker, parameters in enumerate(self.local_parameters):
            for local, packed in zip(parameters, self._parameters, strict=True):
                local.grad = None if packed.grad is None else packed.grad[worker]

    def clip_grad_norm_(self, maximum: float | None):
        self.bind_gradients()
        return torch.stack(
            [clip_grad_norm_(parameters, maximum) for parameters in self.local_parameters]
        )

    def step(self):
        self.bind_gradients()
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> TorchState:
        state: TorchState = {
            "version": 1,
            "layout": self.model.layout_metadata(),
            "first_moment_storage": self.first_moment_storage,
            "second_moment_storage": self.second_moment_storage,
            "workers": [
                {
                    "steps": [optimizer.state[p]["step"].clone() for p in parameters],
                    "groups": [
                        {key: value for key, value in group.items() if key != "params"}
                        for group in optimizer.param_groups
                    ],
                }
                for optimizer, parameters in zip(
                    self.optimizers, self.local_parameters, strict=True
                )
            ],
        }
        if self.accum_grad_storage is not None:
            state.update(
                version=2, optimizer="accumadamw", accum_grad_storage=self.accum_grad_storage
            )
        return state

    @torch.no_grad()
    def load_state_dict(self, state: TorchState) -> None:
        self._check_storage()
        accum = self.accum_grad_storage is not None
        if (
            state.get("version") != (2 if accum else 1)
            or state.get("optimizer", "adamw") != self.optimizer_name
            or state["layout"] != self.model.layout_metadata()
        ):
            raise ValueError("incompatible packed optimizer layout")
        arena_names = ["first_moment_storage", "second_moment_storage"]
        if accum:
            arena_names.append("accum_grad_storage")
        for name in arena_names:
            if state[name].shape != self._arena.shape:
                raise ValueError("incompatible optimizer arena shape")
        if len(state["workers"]) != len(self.optimizers):
            raise ValueError("incompatible optimizer worker count")
        # Validate all workers before changing any live arena.
        for optimizer, parameters, worker in zip(
            self.optimizers, self.local_parameters, state["workers"], strict=True
        ):
            if len(worker["steps"]) != len(parameters) or len(worker["groups"]) != 2:
                raise ValueError("incompatible optimizer counters or groups")
            if accum:
                for group, saved in zip(optimizer.param_groups, worker["groups"], strict=True):
                    validate_accum_group(saved)
                    if saved["accum_iter"] != group["accum_iter"]:
                        raise ValueError("incompatible AccumAdamW window")
        if accum:
            for worker_index, worker in enumerate(state["workers"]):
                for entry, step in zip(self.layout, worker["steps"], strict=True):
                    group = worker["groups"][0 if len(entry.shape) >= 2 else 1]
                    local = {"step": step}
                    for key, arena_name in zip(
                        ("exp_avg", "exp_avg_sq", "accum_grad"), arena_names, strict=True
                    ):
                        local[key] = self.model.parameter_view(state[arena_name], entry)[
                            worker_index
                        ]
                    validate_accum_state(local, entry.shape, self._arena.dtype, group["accum_iter"])
        for name in arena_names:
            getattr(self, name).copy_(state[name])
        for optimizer, parameters, worker in zip(
            self.optimizers, self.local_parameters, state["workers"], strict=True
        ):
            for parameter, step in zip(parameters, worker["steps"], strict=True):
                optimizer.state[parameter]["step"].copy_(step)
            for group, saved in zip(optimizer.param_groups, worker["groups"], strict=True):
                # Execution flags follow the current device; mathematical settings resume.
                keys = ["lr", "betas", "eps", "weight_decay"]
                keys.extend(["accum_iter"] if accum else ["amsgrad", "maximize"])
                for key in keys:
                    group[key] = saved[key]


class PackedAdamW(PackedOptimizer):
    """Packed PyTorch AdamW, preserving the original arena and checkpoint interface."""


class PackedAccumAdamW(PackedOptimizer):
    """Packed AccumAdamW with an additional local gradient-accumulation arena."""

    optimizer_name = "accumadamw"
