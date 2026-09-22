"""Local optimizers shared by ordinary and packed training."""

# PyTorch's optimizers use these same foreach kernels; no public equivalents exist.
# pyright: reportPrivateImportUsage=false

import math
from collections import defaultdict
from collections.abc import Callable, Iterable

import torch
from torch import nn

from tiny_llm.config import Config
from tiny_llm.state import TorchState


def validate_accum_group(group: dict) -> None:
    """Validate mathematical settings, including restored parameter groups."""
    for key in ("lr", "eps", "weight_decay"):
        value = group[key]
        if not math.isfinite(value) or value < 0 or (key == "eps" and value == 0):
            raise ValueError(f"invalid AccumAdamW {key}: {value}")
    if len(group["betas"]) != 2 or any(
        not math.isfinite(beta) or not 0 <= beta < 1 for beta in group["betas"]
    ):
        raise ValueError("invalid AccumAdamW betas")
    steps = group["accumulation_steps"]
    if type(steps) is not int or steps < 1:
        raise ValueError("accumulation_steps must be a positive integer")


def validate_accum_state(state: TorchState, parameter: torch.Tensor, steps: int) -> None:
    """Validate a complete per-parameter state without modifying any live tensors."""
    if set(state) != {"step", "exp_avg", "exp_avg_sq", "accum_grad"}:
        raise ValueError("incompatible AccumAdamW parameter state")
    step = state["step"]
    if (
        not isinstance(step, torch.Tensor)
        or step.ndim != 0
        or not torch.isfinite(step)
        or step.item() < 0
        or step.item() != int(step.item())
    ):
        raise ValueError("invalid AccumAdamW step counter")
    for key in ("exp_avg", "exp_avg_sq", "accum_grad"):
        value = state[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != parameter.shape
            or value.dtype != parameter.dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"invalid AccumAdamW {key} tensor")
    if (state["exp_avg_sq"] < 0).any():
        raise ValueError("negative AccumAdamW second moment")
    if int(step.item()) % steps == 0 and state["accum_grad"].count_nonzero():
        raise ValueError("nonzero AccumAdamW buffer at a window boundary")


class AccumAdamW(torch.optim.Optimizer):
    """AdamW updates with moments committed from every window's mean gradient.

    Appendix C: https://arxiv.org/html/2410.11998v1#A3
    The persistent second moment uses beta2 (the algorithm prints beta1).
    Bias correction uses ceil(step / accumulation_steps), and epsilon is outside
    the square root. Unlike Decent-DP's pre-scaled representation, exp_avg and
    exp_avg_sq store the ordinary, unscaled moments of completed windows.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        accumulation_steps: int = 2,
        *,
        foreach: bool = True,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                accumulation_steps=accumulation_steps,
                foreach=foreach,
            ),
        )

    def add_param_group(self, param_group: dict) -> None:
        validate_accum_group(self.defaults | param_group)
        super().add_param_group(param_group)

    @torch.no_grad()
    def step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, closure: Callable[[], torch.Tensor | float] | None = None
    ) -> torch.Tensor | float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Reject unsupported gradients before updating any parameter or state.
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if p.grad.layout != torch.strided:
                        raise RuntimeError("AccumAdamW does not support sparse gradients")
                    if p.is_complex():
                        raise RuntimeError("AccumAdamW requires real parameters")
        for group in self.param_groups:
            buckets = defaultdict(list)
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state.update(
                        step=torch.zeros((), dtype=torch.int64, device="cpu"),
                        exp_avg=torch.zeros_like(p),
                        exp_avg_sq=torch.zeros_like(p),
                        accum_grad=torch.zeros_like(p),
                    )
                state["step"].add_(1)
                # Missing gradients can put individual parameters in different windows.
                buckets[p.device, p.dtype, int(state["step"].item())].append(p)
            for (_, _, step), params in buckets.items():
                self._update(params, group, step)
        return loss

    def _update(self, params, group, step: int) -> None:
        beta1, beta2 = group["betas"]
        s = group["accumulation_steps"]
        k = (step + s - 1) // s
        step_size = group["lr"] / (1 - beta1**k)
        correction2 = math.sqrt(1 - beta2**k)
        decay = 1 - group["lr"] * group["weight_decay"]
        grads = [p.grad for p in params]
        first = [self.state[p]["exp_avg"] for p in params]
        second = [self.state[p]["exp_avg_sq"] for p in params]
        buffers = [self.state[p]["accum_grad"] for p in params]
        if group["foreach"]:
            m = torch._foreach_mul(first, beta1)
            torch._foreach_add_(m, grads, alpha=1 - beta1)
            v = torch._foreach_mul(second, beta2)
            torch._foreach_addcmul_(v, grads, grads, value=1 - beta2)
            torch._foreach_sqrt_(v)
            torch._foreach_div_(v, correction2)
            torch._foreach_add_(v, group["eps"])
            torch._foreach_mul_(params, decay)
            torch._foreach_addcdiv_(params, m, v, value=-step_size)
            torch._foreach_add_(buffers, grads, alpha=1 / s)
            if step % s == 0:
                torch._foreach_mul_(first, beta1)
                torch._foreach_add_(first, buffers, alpha=1 - beta1)
                torch._foreach_mul_(second, beta2)
                torch._foreach_addcmul_(second, buffers, buffers, value=1 - beta2)
                torch._foreach_zero_(buffers)
        else:
            for p, g, m, v, b in zip(params, grads, first, second, buffers, strict=True):
                update = m.mul(beta1).add_(g, alpha=1 - beta1)
                denom = v.mul(beta2).addcmul_(g, g, value=1 - beta2)
                denom.sqrt_().div_(correction2).add_(group["eps"])
                p.mul_(decay).addcdiv_(update, denom, value=-step_size)
                b.add_(g, alpha=1 / s)
                if step % s == 0:
                    m.mul_(beta1).add_(b, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(b, b, value=1 - beta2)
                    b.zero_()

    def state_dict(self) -> TorchState:
        return super().state_dict() | {"optimizer_name": "accum_adamw", "version": 1}

    def load_state_dict(self, state_dict: TorchState) -> None:
        if state_dict.get("optimizer_name") != "accum_adamw" or state_dict.get("version") != 1:
            raise ValueError("incompatible AccumAdamW optimizer state")
        groups = state_dict["param_groups"]
        if len(groups) != len(self.param_groups):
            raise ValueError("incompatible AccumAdamW parameter groups")
        expected_ids = set()
        for group, saved in zip(self.param_groups, groups, strict=True):
            validate_accum_group(saved)
            if (
                len(group["params"]) != len(saved["params"])
                or group["accumulation_steps"] != saved["accumulation_steps"]
            ):
                raise ValueError("incompatible AccumAdamW parameters or accumulation_steps")
            for p, index in zip(group["params"], saved["params"], strict=True):
                if index in expected_ids:
                    raise ValueError("duplicate AccumAdamW parameter state")
                expected_ids.add(index)
                if index in state_dict["state"]:
                    validate_accum_state(state_dict["state"][index], p, saved["accumulation_steps"])
        if not set(state_dict["state"]).issubset(expected_ids):
            raise ValueError("unknown AccumAdamW parameter state")
        foreach = [group["foreach"] for group in self.param_groups]
        super().load_state_dict(state_dict)
        for group, execution in zip(self.param_groups, foreach, strict=True):
            group["foreach"] = execution
        for state in self.state.values():
            state["step"] = state["step"].to(device="cpu", dtype=torch.int64)


def make_local_optimizer(
    parameters: Iterable[nn.Parameter], config: Config, device: torch.device
) -> torch.optim.Optimizer:
    parameters = list(parameters)
    cfg = config.optimizer
    groups = [
        {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
    ]
    if cfg.name == "accum_adamw":
        return AccumAdamW(
            groups,
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            accumulation_steps=cfg.accumulation_steps,
            foreach=not config.runtime.deterministic,
        )
    return torch.optim.AdamW(
        groups,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        fused=config.runtime.fused_optimizer
        and device.type == "cuda"
        and not config.runtime.deterministic,
        foreach=False if config.runtime.deterministic else None,
    )
