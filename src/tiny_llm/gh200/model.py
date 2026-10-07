"""Fused projections over the original canonical per-worker parameter arena."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tiny_llm.config import ModelConfig
from tiny_llm.packed import PackedLlama


@dataclass(frozen=True)
class FusedLayout:
    name: str
    start: int
    numel: int
    decay: bool


class GH200Model(PackedLlama):
    """Canonical checkpoints and fused trainable leaves share the same storage.

    Placement happens before projection fusion. Once fused, moving/replacing
    storage is prohibited; checkpoint loading always copies into existing views.
    The replica dimension is present internally even for synchronous training.
    """

    def __init__(self, config: ModelConfig, num_models: int, device: torch.device):
        self._fused = False
        super().__init__(config, num_models, "sdpa")
        self.to(device)
        self._entries = {entry.name: entry for entry in self.layout}
        self._canonical_parameters = {entry.name: self.get_parameter(entry.name) for entry in self.layout}
        fused_ranges: dict[str, tuple[int, int]] = {}
        for index, block in enumerate(cast(list[Any], list(self.blocks))):
            for owner, names, fused in (
                (block.attention, ("q_proj", "k_proj", "v_proj"), "qkv_proj"),
                (block.ffn, ("gate_proj", "up_proj"), "gate_up_proj"),
            ):
                family = "attention" if fused == "qkv_proj" else "ffn"
                prefix = f"blocks.{index}.{family}."
                entries = [self._entries[prefix + name + ".weight"] for name in names]
                if any(a.start + a.numel != b.start for a, b in zip(entries, entries[1:], strict=False)):
                    raise ValueError("projection fusion requires adjacent, unpadded canonical matrices")
                start, size = entries[0].start, sum(entry.numel for entry in entries)
                view = self.parameter_storage[:, start:start + size].view(
                    num_models, len(names) * entries[0].shape[0], config.width
                )
                holder = nn.Module()
                holder.register_parameter("weight", nn.Parameter(view))
                owner.add_module(fused, holder)
                for name in names:
                    getattr(owner, name).register_parameter("weight", None)
                fused_ranges[prefix + fused + ".weight"] = (start, size)
        self.fused_layout = tuple(
            FusedLayout(name, *fused_ranges[name], True) if name in fused_ranges else
            FusedLayout(name, self._entries[name].start, self._entries[name].numel,
                        len(self._entries[name].shape) >= 2)
            for name, _ in self.named_parameters()
        )
        self._fused = True
        self.fused_rope = False

    def _apply(self, fn, recurse=True):
        if self._fused:
            raise RuntimeError("place GH200Model at construction; changing its arena is unsupported")
        return super()._apply(fn, recurse)

    def get_parameter(self, target):
        if self._fused and target in self._entries:
            return self._canonical_parameters[target]
        return super().get_parameter(target)

    def projection(self, x: Tensor, weight: Tensor) -> Tensor:
        n, batch, length, _ = x.shape
        if n == 1:
            return F.linear(x[0], weight[0]).unsqueeze(0)
        return torch.bmm(x.flatten(1, 2), weight.transpose(1, 2)).view(n, batch, length, -1)

    def forward(self, input_ids: Tensor) -> Tensor:
        from .gh200_pipeline import selected_norm
        from .gh200_rope import packed_rope
        from .gh200_swiglu import paired_swiglu

        if input_ids.ndim != 3 or input_ids.shape[0] != self.num_models:
            raise ValueError("GH200 inputs must have shape [workers, batch, context]")
        n, batch, length = input_ids.shape
        cfg = self.config
        if length > cfg.context_length:
            raise ValueError("input exceeds configured context")
        if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.bfloat16:
            raise ValueError("GH200 forward requires CUDA BF16 autocast")
        if n == 1:
            hidden = F.embedding(input_ids[0], self.embedding.weight[0]).unsqueeze(0)
        else:
            workers = torch.arange(n, device=input_ids.device)[:, None, None]
            hidden = self.embedding.weight[workers, input_ids]
        rows = 8 if cfg.width == 320 and batch * length >= 4096 else 4
        blocks = cast(list[Any], list(self.blocks))
        first = blocks[0].attention_norm
        hidden, normalized = selected_norm(hidden, None, first.weight, first.eps, rows=rows)
        for index, block in enumerate(blocks):
            attention = block.attention
            qkv = self.projection(normalized, attention.qkv_proj.weight)
            if self.fused_rope:
                q, k, v = packed_rope(qkv.flatten(0, 1), attention._rope_cos,
                                      attention._rope_sin, cfg.heads,
                                      copy_value=batch * length <= 4096)
            else:
                q, k, v = qkv.reshape(n * batch, length, 3, cfg.heads, cfg.width // cfg.heads).unbind(2)
                q, k = attention._rotary(q.transpose(1, 2)), attention._rotary(k.transpose(1, 2))
                v = v.transpose(1, 2)
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attended = attended.transpose(1, 2).contiguous().view(n, batch, length, cfg.width)
            branch = self.projection(attended, attention.out_proj.weight)
            hidden, normalized = selected_norm(branch, hidden, block.ffn_norm.weight,
                                               cfg.norm_eps, rows=rows)
            gate_up = self.projection(normalized, block.ffn.gate_up_proj.weight)
            branch = self.projection(paired_swiglu(gate_up), block.ffn.down_proj.weight)
            norm = blocks[index + 1].attention_norm if index + 1 < len(blocks) else self.norm
            hidden, normalized = selected_norm(branch, hidden, norm.weight, cfg.norm_eps, rows=rows)
        return self.projection(normalized, self.embedding.weight)
