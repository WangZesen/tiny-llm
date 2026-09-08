"""Packed Llama execution and aligned, shared parameter storage."""

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tiny_llm.config import ModelConfig
from tiny_llm.model import Llama, rotary, stable_dtype, token_losses


@dataclass(frozen=True)
class ParameterLayout:
    name: str
    shape: tuple[int, ...]
    start: int
    numel: int
    padded_numel: int


def linear(x: Tensor, weight: Tensor) -> Tensor:
    n, batch, length, width = x.shape
    return torch.bmm(x.reshape(n, batch * length, width), weight.transpose(1, 2)).view(
        n, batch, length, weight.shape[1]
    )


def norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    full = x.to(stable_dtype(x))
    normalized = full * torch.rsqrt(full.square().mean(dim=-1, keepdim=True) + eps)
    return normalized.to(x.dtype) * weight[:, None, None, :]


def local_mean_losses(logits: Tensor, targets: Tensor) -> Tensor:
    """One mean per worker; summing these does not scale gradients by worker count."""
    counts = (targets != -100).sum(dim=(1, 2))
    return token_losses(logits, targets).sum(dim=(1, 2)) / counts.clamp_min(1)


class PackedLlama(nn.Module):
    def __init__(self, config: ModelConfig, num_models: int, attention_backend: str = "sdpa"):
        super().__init__()
        if num_models < 1:
            raise ValueError("num_models must be positive")
        # Exactly the ordinary initialization, including its RNG consumption.
        ordinary = Llama(config, attention_backend)
        self.config, self.num_models, self.backend = config, num_models, attention_backend
        self.embedding, self.blocks, self.norm = ordinary.embedding, ordinary.blocks, ordinary.norm
        entries, offset = [], 0
        for name, parameter in self.named_parameters():
            size = parameter.numel()
            padded = (size + 63) // 64 * 64
            entries.append(ParameterLayout(name, tuple(parameter.shape), offset, size, padded))
            offset += padded
        self.layout = tuple(entries)
        self.storage_numel = offset
        first = next(self.parameters())
        self._parameter_storage = first.new_zeros((num_models, offset))
        for entry in self.layout:
            original = self.get_parameter(entry.name)
            view = self.parameter_view(self._parameter_storage, entry)
            with torch.no_grad():
                view.copy_(original.expand_as(view))
            module_name, _, name = entry.name.rpartition(".")
            self.get_submodule(module_name).register_parameter(name, nn.Parameter(view))
        self._mix_scratch = None

    @property
    def parameter_storage(self) -> Tensor:
        return self._parameter_storage

    @property
    def parameter_count(self) -> int:
        """Total unique trainable parameters across workers, excluding alignment padding."""
        return self.num_models * self.local_parameter_count

    @property
    def local_parameter_count(self) -> int:
        return sum(entry.numel for entry in self.layout)

    def parameter_view(self, storage: Tensor, entry: ParameterLayout) -> Tensor:
        return storage[:, entry.start : entry.start + entry.numel].view(
            self.num_models, *entry.shape
        )

    def layout_metadata(self) -> list[dict]:
        return [asdict(entry) for entry in self.layout]

    def packed_state_dict(self) -> dict:
        return {
            "version": 1,
            "layout": self.layout_metadata(),
            "parameter_storage": self.parameter_storage.detach(),
        }

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if assign:
            raise ValueError("assign=True would break packed storage aliases; use assign=False")
        return super().load_state_dict(state_dict, strict=strict, assign=False)

    @torch.no_grad()
    def load_packed_state_dict(self, state):
        if (
            state.get("version") != 1
            or state["layout"] != self.layout_metadata()
            or state["parameter_storage"].shape != self.parameter_storage.shape
        ):
            raise ValueError("incompatible packed parameter layout")
        self.parameter_storage.copy_(state["parameter_storage"])

    def _apply(self, fn, recurse=True):
        # nn.Module conversion transforms parameters separately. Rebind them to a
        # fresh arena afterward; callers construct optimizers after placement.
        super()._apply(fn, recurse)
        first = next(self.parameters())
        arena = first.new_zeros((self.num_models, self.storage_numel))
        with torch.no_grad():
            for entry in self.layout:
                parameter = self.get_parameter(entry.name)
                view = self.parameter_view(arena, entry)
                view.copy_(parameter)
                parameter.data = view
        self._parameter_storage = arena
        self._mix_scratch = None
        return self

    def forward(self, input_ids: Tensor) -> Tensor:
        cfg = self.config
        if (
            input_ids.ndim != 3
            or input_ids.shape[0] != self.num_models
            or not 0 < input_ids.shape[2] <= cfg.context_length
        ):
            raise ValueError(
                "input_ids must have shape [num_models, batch, length <= context_length]"
            )
        n, batch, length = input_ids.shape
        workers = torch.arange(n, device=input_ids.device)[:, None, None]
        x = self.embedding.weight[workers, input_ids]
        for block in self.blocks:
            z = norm(x, block.attention_norm.weight, cfg.norm_eps)
            q, k, v = [
                linear(z, projection.weight)
                .reshape(n * batch, length, cfg.heads, cfg.width // cfg.heads)
                .transpose(1, 2)
                for projection in (
                    block.attention.q_proj,
                    block.attention.k_proj,
                    block.attention.v_proj,
                )
            ]
            q, k = rotary(q, cfg.rope_theta), rotary(k, cfg.rope_theta)
            if self.backend == "sdpa":
                result = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                with torch.autocast(x.device.type, enabled=False):
                    dtype = stable_dtype(q)
                    scores = q.to(dtype) @ k.to(dtype).transpose(-2, -1) / math.sqrt(q.shape[-1])
                    mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
                    probabilities = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
                    result = (probabilities @ v.to(dtype)).to(v.dtype)
            result = result.transpose(1, 2).contiguous().view(n, batch, length, cfg.width)
            x = x + linear(result, block.attention.out_proj.weight)
            z = norm(x, block.ffn_norm.weight, cfg.norm_eps)
            x = x + linear(
                F.silu(linear(z, block.ffn.gate_proj.weight)) * linear(z, block.ffn.up_proj.weight),
                block.ffn.down_proj.weight,
            )
        return linear(norm(x, self.norm.weight, cfg.norm_eps), self.embedding.weight)

    def local_state_dict(self, worker: int) -> dict[str, Tensor]:
        if not 0 <= worker < self.num_models:
            raise ValueError("worker index out of range")
        return {
            entry.name: self.get_parameter(entry.name)[worker].detach().clone()
            for entry in self.layout
        }

    @torch.no_grad()
    def load_local_state_dict(self, worker: int, state: dict[str, Tensor]):
        if not 0 <= worker < self.num_models:
            raise ValueError("worker index out of range")
        if set(state) != {entry.name for entry in self.layout}:
            raise ValueError("local state dict does not match the model")
        for entry in self.layout:
            if tuple(state[entry.name].shape) != entry.shape:
                raise ValueError(f"invalid shape for {entry.name}")
        for entry in self.layout:
            self.get_parameter(entry.name)[worker].copy_(state[entry.name])

    @torch.no_grad()
    def copy_average_to(self, model: Llama):
        if model.config != self.config:
            raise ValueError("averaged model configuration mismatch")
        # One reduction over the arena. Padding is zero and never copied out.
        average = self.parameter_storage.float().mean(dim=0)
        for entry in self.layout:
            model.get_parameter(entry.name).copy_(
                average[entry.start : entry.start + entry.numel].view(entry.shape)
            )

    @torch.no_grad()
    def mix_(self, topology: str, step: int):
        if topology not in ("complete", "one_peer_ring", "one_peer_exponential") or step < 0:
            raise ValueError("invalid topology or step")
        if self.num_models == 1:
            return
        arena = self.parameter_storage
        if self._mix_scratch is None:
            self._mix_scratch = torch.empty_like(arena)
        if topology == "complete":
            self._mix_scratch.copy_(arena.mean(dim=0, keepdim=True))
        else:
            offset = (
                (1 if step % 2 == 0 else -1)
                if topology == "one_peer_ring"
                else (1 << (step % (self.num_models - 1).bit_length()))
            )
            peers = (torch.arange(self.num_models, device=arena.device) - offset) % self.num_models
            torch.index_select(arena, 0, peers, out=self._mix_scratch)
            self._mix_scratch.add_(arena).mul_(0.5)
        arena.copy_(self._mix_scratch)
