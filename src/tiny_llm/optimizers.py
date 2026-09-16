"""Local optimizers, including the corrected Appendix C AccumAdamW update."""

# PyTorch exposes the tensor-list kernels under underscored names.
# pyright: reportPrivateImportUsage=false

import math
from collections.abc import Callable, Iterable
from typing import Any, TypeVar, overload

import torch
from torch import Tensor

from tiny_llm.config import Config

_Loss = TypeVar("_Loss")


def validate_accum_group(group: dict[str, Any]) -> None:
    for key in ("lr", "eps", "weight_decay"):
        value = group[key]
        if not math.isfinite(value) or value < 0 or (key == "eps" and value == 0):
            raise ValueError(f"invalid AccumAdamW {key}: {value}")
    if len(group["betas"]) != 2 or any(
        not math.isfinite(beta) or not 0 <= beta < 1 for beta in group["betas"]
    ):
        raise ValueError("invalid AccumAdamW betas")
    if type(group["accum_iter"]) is not int or group["accum_iter"] < 1:
        raise ValueError("AccumAdamW accum_iter must be a positive integer")
    if group.get("foreach") is not None and type(group["foreach"]) is not bool:
        raise ValueError("AccumAdamW foreach must be a boolean or None")


def validate_accum_state(
    state: dict[str, Any], shape: tuple[int, ...], dtype: torch.dtype, accum_iter: int
) -> None:
    """Validate saved state before copying it into live tensors or packed arenas."""
    if set(state) != {"step", "exp_avg", "exp_avg_sq", "accum_grad"}:
        raise ValueError("incompatible AccumAdamW state")
    step = state["step"]
    if (
        not isinstance(step, Tensor)
        or step.ndim != 0
        or step.dtype != torch.int64
        or step.item() < 0
    ):
        raise ValueError("invalid AccumAdamW counter")
    for key in ("exp_avg", "exp_avg_sq", "accum_grad"):
        tensor = state[key]
        if (
            not isinstance(tensor, Tensor)
            or tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"invalid AccumAdamW {key} tensor")
    if (state["exp_avg_sq"] < 0).any():
        raise ValueError("negative AccumAdamW second moment")
    if step.item() % accum_iter == 0 and state["accum_grad"].count_nonzero():
        raise ValueError("nonempty AccumAdamW buffer at a window boundary")


class AccumAdamW(torch.optim.Optimizer):
    """Update every step; commit moments of the mean gradient every ``accum_iter`` steps.

    Follows Appendix C of https://openreview.net/pdf?id=lo3nlFHOft, correcting
    beta1 to beta2 in the persistent second moment and placing epsilon outside
    the square root, as in the authors' Decent-DP implementation. Unlike that
    implementation, exp_avg and exp_avg_sq store the unscaled persistent moments.
    Counters and windows advance only for parameters with a gradient.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        accum_iter: int = 4,
        foreach: bool | None = None,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            accum_iter=accum_iter,
            foreach=foreach,
        )
        validate_accum_group(defaults)
        super().__init__(params, defaults)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        validate_accum_group(self.defaults | param_group)
        super().add_param_group(param_group)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], _Loss]) -> _Loss: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], _Loss] | None = None) -> _Loss | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            # A missing gradient can leave a parameter on a different window.
            batches: dict[tuple[torch.device, torch.dtype, int], list[Tensor]] = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse or not p.is_floating_point():
                    raise RuntimeError("AccumAdamW requires dense real floating-point gradients")
                state = self.state[p]
                if not state:
                    state.update(
                        step=torch.zeros((), dtype=torch.int64, device="cpu"),
                        exp_avg=torch.zeros_like(p),
                        exp_avg_sq=torch.zeros_like(p),
                        accum_grad=torch.zeros_like(p),
                    )
                state["step"].add_(1)
                t = int(state["step"].item())
                batches.setdefault((p.device, p.dtype, t), []).append(p)
            for (device, _, t), params in batches.items():
                foreach = group["foreach"]
                if foreach is None:
                    foreach = device.type == "cuda"
                if foreach and not torch.are_deterministic_algorithms_enabled():
                    self._foreach_update(params, group, t)
                else:
                    for p in params:
                        self._single_update(p, group, t)
        return loss

    def _single_update(self, p: Tensor, group: dict[str, Any], t: int) -> None:
        beta1, beta2 = group["betas"]
        s, lr = group["accum_iter"], group["lr"]
        k = (t + s - 1) // s
        state, g = self.state[p], p.grad
        assert g is not None
        m, v, b = (state[key] for key in ("exp_avg", "exp_avg_sq", "accum_grad"))
        current_m = m.mul(beta1).add_(g, alpha=1 - beta1)
        denom = v.mul(beta2).addcmul_(g, g, value=1 - beta2)
        denom.sqrt_().div_(math.sqrt(1 - beta2**k)).add_(group["eps"])
        p.mul_(1 - lr * group["weight_decay"])
        p.addcdiv_(current_m, denom, value=-lr / (1 - beta1**k))
        b.add_(g, alpha=1 / s)
        if t % s == 0:
            m.mul_(beta1).add_(b, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(b, b, value=1 - beta2)
            b.zero_()

    def _foreach_update(self, params: list[Tensor], group: dict[str, Any], t: int) -> None:
        beta1, beta2 = group["betas"]
        s, lr = group["accum_iter"], group["lr"]
        k = (t + s - 1) // s
        grads = [p.grad for p in params if p.grad is not None]
        m = [self.state[p]["exp_avg"] for p in params]
        v = [self.state[p]["exp_avg_sq"] for p in params]
        b = [self.state[p]["accum_grad"] for p in params]
        current_m = torch._foreach_mul(m, beta1)
        torch._foreach_add_(current_m, grads, alpha=1 - beta1)
        denom = torch._foreach_mul(v, beta2)
        torch._foreach_addcmul_(denom, grads, grads, value=1 - beta2)
        torch._foreach_sqrt_(denom)
        torch._foreach_div_(denom, math.sqrt(1 - beta2**k))
        torch._foreach_add_(denom, group["eps"])
        torch._foreach_mul_(params, 1 - lr * group["weight_decay"])
        torch._foreach_addcdiv_(params, current_m, denom, value=-lr / (1 - beta1**k))
        torch._foreach_add_(b, grads, alpha=1 / s)
        if t % s == 0:
            torch._foreach_mul_(m, beta1)
            torch._foreach_add_(m, b, alpha=1 - beta1)
            torch._foreach_mul_(v, beta2)
            torch._foreach_addcmul_(v, b, b, value=1 - beta2)
            torch._foreach_zero_(b)

    def state_dict(self) -> dict[str, Any]:
        return super().state_dict() | {"version": 1, "optimizer": "accumadamw"}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("version") != 1 or state_dict.get("optimizer") != "accumadamw":
            raise ValueError("incompatible AccumAdamW optimizer identity")
        groups = state_dict["param_groups"]
        if len(groups) != len(self.param_groups):
            raise ValueError("incompatible AccumAdamW parameter groups")
        for saved, current in zip(groups, self.param_groups, strict=True):
            validate_accum_group(saved)
            if saved["accum_iter"] != current["accum_iter"] or len(saved["params"]) != len(
                current["params"]
            ):
                raise ValueError("incompatible AccumAdamW window or parameter count")
            for index, p in zip(saved["params"], current["params"], strict=True):
                if index in state_dict["state"]:
                    validate_accum_state(
                        state_dict["state"][index], tuple(p.shape), p.dtype, saved["accum_iter"]
                    )
        execution = [group["foreach"] for group in self.param_groups]
        super().load_state_dict(state_dict)
        for group, foreach in zip(self.param_groups, execution, strict=True):
            group["foreach"] = foreach
            for p in group["params"]:
                if p in self.state and self.state[p]:
                    self.state[p]["step"] = self.state[p]["step"].to("cpu", dtype=torch.int64)


def make_local_optimizer(
    parameters: Iterable[Tensor], config: Config, device: torch.device
) -> torch.optim.AdamW | AccumAdamW:
    cfg = config.optimizer
    parameters = list(parameters)
    groups = [
        {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
    ]
    common: dict[str, Any] = dict(lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
    foreach = False if config.runtime.deterministic else None
    if cfg.name == "accumadamw":
        return AccumAdamW(groups, **common, accum_iter=cfg.accum_iter, foreach=foreach)
    fused = (
        config.runtime.fused_optimizer
        and device.type == "cuda"
        and not config.runtime.deterministic
    )
    return torch.optim.AdamW(groups, **common, fused=fused, foreach=foreach)
