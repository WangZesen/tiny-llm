"""Independent AdamW optimizers with shared, inspectable moment arenas."""

import torch
from torch import nn

from tiny_llm.config import Config
from tiny_llm.packed import PackedLlama
from tiny_llm.runtime import clip_grad_norm_


class PackedAdamW:
    def __init__(self, model: PackedLlama, config: Config):
        self.model = model
        self.layout = model.layout
        self._parameters = tuple(model.parameters())
        self._arena = model.parameter_storage
        self.first_moment_storage = torch.zeros_like(self._arena)
        self.second_moment_storage = torch.zeros_like(self._arena)
        self.optimizers, self.local_parameters = [], []
        fused = (
            config.runtime.fused_optimizer
            and self._arena.device.type == "cuda"
            and not config.runtime.deterministic
        )
        cfg = config.optimizer
        for worker in range(model.num_models):
            parameters = [nn.Parameter(p[worker].detach()) for p in self._parameters]
            decay = [p for p in parameters if p.ndim >= 2]
            no_decay = [p for p in parameters if p.ndim < 2]
            optimizer = torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": cfg.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
                eps=cfg.eps,
                fused=fused,
                foreach=False if config.runtime.deterministic else None,
            )
            for entry, parameter in zip(self.layout, parameters, strict=True):
                optimizer.state[parameter] = {
                    "step": torch.zeros(
                        (),
                        dtype=torch.float64
                        if not fused and torch.get_default_dtype() == torch.float64
                        else torch.float32,
                        device=self._arena.device if fused else "cpu",
                    ),
                    "exp_avg": model.parameter_view(self.first_moment_storage, entry)[worker],
                    "exp_avg_sq": model.parameter_view(self.second_moment_storage, entry)[worker],
                }
            self.optimizers.append(optimizer)
            self.local_parameters.append(parameters)

    @property
    def param_groups(self):
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def _check_storage(self):
        if self.model.parameter_storage is not self._arena:
            raise RuntimeError("model storage changed; construct PackedAdamW after model placement")

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

    def state_dict(self):
        return {
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

    @torch.no_grad()
    def load_state_dict(self, state):
        self._check_storage()
        if state.get("version") != 1 or state["layout"] != self.model.layout_metadata():
            raise ValueError("incompatible packed optimizer layout")
        for name in ("first_moment_storage", "second_moment_storage"):
            if state[name].shape != self._arena.shape:
                raise ValueError("incompatible optimizer arena shape")
        if len(state["workers"]) != len(self.optimizers):
            raise ValueError("incompatible optimizer worker count")
        for name in ("first_moment_storage", "second_moment_storage"):
            getattr(self, name).copy_(state[name])
        for optimizer, parameters, worker in zip(
            self.optimizers, self.local_parameters, state["workers"], strict=True
        ):
            for parameter, step in zip(parameters, worker["steps"], strict=True):
                optimizer.state[parameter]["step"].copy_(step)
            for group, saved in zip(optimizer.param_groups, worker["groups"], strict=True):
                # Execution flags follow the current device; mathematical settings resume.
                for key in ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize"):
                    group[key] = saved[key]
