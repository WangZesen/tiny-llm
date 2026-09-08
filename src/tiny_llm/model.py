"""Llama blocks with a shared state dict and an explicit second-order path."""

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tiny_llm.config import ModelConfig


def stable_dtype(x: Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        full = x.to(stable_dtype(x))
        normalized = full * torch.rsqrt(full.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def rotary(x: Tensor, theta: float) -> Tensor:
    """Half-split RoPE. Compute angles in FP32/64, preserving FP64 grad checks."""
    length, dim = x.shape[-2:]
    dtype = stable_dtype(x)
    frequency = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=dtype) / dim)
    angle = torch.arange(length, device=x.device, dtype=dtype)[:, None] * frequency
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    left, right = x.chunk(2, dim=-1)
    return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, backend: str):
        super().__init__()
        self.heads = config.heads
        self.theta = config.rope_theta
        self.backend = backend
        self.q_proj = nn.Linear(config.width, config.width, bias=False)
        self.k_proj = nn.Linear(config.width, config.width, bias=False)
        self.v_proj = nn.Linear(config.width, config.width, bias=False)
        self.out_proj = nn.Linear(config.width, config.width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = [
            projection(x).view(batch, length, self.heads, width // self.heads).transpose(1, 2)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        ]
        q, k = rotary(q, self.theta), rotary(k, self.theta)
        if self.backend == "sdpa":
            result = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Disable autocast here so the explicitly promoted score computation stays FP32.
            with torch.autocast(x.device.type, enabled=False):
                dtype = stable_dtype(q)
                scores = q.to(dtype) @ k.to(dtype).transpose(-2, -1) / math.sqrt(q.shape[-1])
                mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
                probabilities = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
                result = (probabilities @ v.to(dtype)).to(v.dtype)
        return self.out_proj(result.transpose(1, 2).contiguous().view(batch, length, width))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.width, config.ffn_width, bias=False)
        self.up_proj = nn.Linear(config.width, config.ffn_width, bias=False)
        self.down_proj = nn.Linear(config.ffn_width, config.width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig, backend: str):
        super().__init__()
        self.attention_norm = RMSNorm(config.width, config.norm_eps)
        self.attention = Attention(config, backend)
        self.ffn_norm = RMSNorm(config.width, config.norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class Llama(nn.Module):
    def __init__(
        self, config: ModelConfig, attention_backend: Literal["sdpa", "reference"] = "sdpa"
    ):
        super().__init__()
        if attention_backend not in ("sdpa", "reference"):
            raise ValueError(f"unknown attention backend: {attention_backend}")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(
            [Block(config, attention_backend) for _ in range(config.layers)]
        )
        self.norm = RMSNorm(config.width, config.norm_eps)
        # F.linear with embedding.weight is an actual tie, not two serialized aliases.
        self.apply(self._initialize)
        for block in self.blocks:
            for projection in (block.attention.out_proj, block.ffn.down_proj):
                nn.init.normal_(
                    projection.weight, std=config.init_std / math.sqrt(2 * config.layers)
                )

    def _initialize(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.init_std)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.config.context_length:
            raise ValueError("input_ids must have shape [batch, length <= context_length]")
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return F.linear(self.norm(x), self.embedding.weight)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def token_losses(logits: Tensor, targets: Tensor) -> Tensor:
    """Unreduced next-token CE; -100 marks padding. Keeps double precision intact."""
    return F.cross_entropy(
        logits.to(stable_dtype(logits)).flatten(0, 1),
        targets.flatten(),
        reduction="none",
        ignore_index=-100,
    ).view_as(targets)
