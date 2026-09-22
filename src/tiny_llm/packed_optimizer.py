"""Independent local optimizers with shared, inspectable state arenas."""

import math

import torch
from torch import nn

from tiny_llm.config import Config
from tiny_llm.optimizer import AccumAdamW, make_local_optimizer, validate_accum_group
from tiny_llm.packed import PackedLlama
from tiny_llm.runtime import clip_grad_norm_
from tiny_llm.state import TorchState


class PackedOptimizer:
    def __init__(self, model: PackedLlama, config: Config):
        self.model = model
        self.layout = model.layout
        self._parameters = tuple(model.parameters())
        self._arena = model.parameter_storage
        self.first_moment_storage = torch.zeros_like(self._arena)
        self.second_moment_storage = torch.zeros_like(self._arena)
        self.name = config.optimizer.name
        self.accumulation_steps = config.optimizer.accumulation_steps
        self.accumulated_gradient_storage = (
            torch.zeros_like(self._arena) if self.name == "accum_adamw" else None
        )
        self.optimizers: list[torch.optim.Optimizer] = []
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
                        if isinstance(optimizer, AccumAdamW)
                        else torch.float64
                        if not fused and torch.get_default_dtype() == torch.float64
                        else torch.float32,
                        device=self._arena.device if fused else "cpu",
                    ),
                    "exp_avg": model.parameter_view(self.first_moment_storage, entry)[worker],
                    "exp_avg_sq": model.parameter_view(self.second_moment_storage, entry)[worker],
                }
                if self.accumulated_gradient_storage is not None:
                    optimizer.state[parameter]["accum_grad"] = model.parameter_view(
                        self.accumulated_gradient_storage, entry
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
        state = {
            "version": 2 if self.name == "accum_adamw" else 1,
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
        if self.accumulated_gradient_storage is not None:
            state.update(
                optimizer_name=self.name,
                accumulated_gradient_storage=self.accumulated_gradient_storage,
            )
        return state

    @torch.no_grad()
    def load_state_dict(self, state: TorchState) -> None:
        self._check_storage()
        version = 2 if self.name == "accum_adamw" else 1
        if (
            state.get("version") != version
            or state.get("optimizer_name", "adamw") != self.name
            or state["layout"] != self.model.layout_metadata()
        ):
            raise ValueError("incompatible packed optimizer layout")
        arenas = ["first_moment_storage", "second_moment_storage"]
        if self.accumulated_gradient_storage is not None:
            arenas.append("accumulated_gradient_storage")
        for name in arenas:
            value = state.get(name)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != self._arena.shape
                or value.dtype != self._arena.dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError("incompatible optimizer arena shape")
        if (state["second_moment_storage"] < 0).any():
            raise ValueError("negative optimizer second moment")
        if len(state["workers"]) != len(self.optimizers):
            raise ValueError("incompatible optimizer worker count")
        for index, (optimizer, parameters, worker) in enumerate(
            zip(self.optimizers, self.local_parameters, state["workers"], strict=True)
        ):
            if len(worker["steps"]) != len(parameters) or len(worker["groups"]) != len(
                optimizer.param_groups
            ):
                raise ValueError("incompatible optimizer parameters or groups")
            for entry, step in zip(self.layout, worker["steps"], strict=True):
                if (
                    not isinstance(step, torch.Tensor)
                    or step.ndim != 0
                    or not torch.isfinite(step)
                    or step.item() < 0
                    or step.item() != int(step.item())
                ):
                    raise ValueError("invalid optimizer step counter")
                if self.accumulated_gradient_storage is not None:
                    buffer = self.model.parameter_view(
                        state["accumulated_gradient_storage"], entry
                    )[index]
                    if int(step.item()) % self.accumulation_steps == 0 and buffer.count_nonzero():
                        raise ValueError("nonzero AccumAdamW buffer at a window boundary")
            for saved in worker["groups"]:
                if self.name == "accum_adamw":
                    validate_accum_group(saved)
                    if saved["accumulation_steps"] != self.accumulation_steps:
                        raise ValueError("incompatible optimizer accumulation_steps")
                elif (
                    any(key not in saved for key in self._mathematical_settings())
                    or not math.isfinite(saved["lr"])
                    or saved["lr"] < 0
                ):
                    raise ValueError("incompatible optimizer parameter groups")
        # All validation precedes writes, and copying preserves the arena aliases.
        for name in arenas:
            getattr(self, name).copy_(state[name])
        for optimizer, parameters, worker in zip(
            self.optimizers, self.local_parameters, state["workers"], strict=True
        ):
            for parameter, step in zip(parameters, worker["steps"], strict=True):
                optimizer.state[parameter]["step"].copy_(step)
            for group, saved in zip(optimizer.param_groups, worker["groups"], strict=True):
                # Execution flags follow the current device; mathematical settings resume.
                for key in self._mathematical_settings():
                    group[key] = saved[key]

    def _mathematical_settings(self) -> tuple[str, ...]:
        common = ("lr", "betas", "eps", "weight_decay")
        return common + (
            ("accumulation_steps",) if self.name == "accum_adamw" else ("amsgrad", "maximize")
        )


class PackedAdamW(PackedOptimizer):
    """Backward-compatible entry point for independent packed AdamW optimizers."""

    def __init__(self, model: PackedLlama, config: Config):
        if config.optimizer.name != "adamw":
            raise ValueError("PackedAdamW requires optimizer.name=adamw; use PackedOptimizer")
        super().__init__(model, config)
