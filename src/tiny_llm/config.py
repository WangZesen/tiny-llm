"""Strict YAML configuration, shared by the CLI and Python API."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class ModelConfig(StrictModel):
    vocab_size: int = Field(32000, ge=2, le=65536)
    layers: int = Field(8, gt=0)
    width: int = Field(320, gt=0)
    heads: int = Field(5, gt=0)
    ffn_width: int = Field(896, gt=0)
    context_length: int = Field(1024, gt=0)
    rope_theta: float = Field(10000.0, gt=0)
    norm_eps: float = Field(1e-5, gt=0)
    init_std: float = Field(0.02, gt=0)

    @model_validator(mode="after")
    def dimensions(self):
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width must divide into heads with an even head dimension")
        return self

    @property
    def parameter_count(self) -> int:
        d = self.width
        return self.vocab_size * d + self.layers * (4 * d * d + 3 * d * self.ffn_width + 2 * d) + d


PRESETS = {
    "20m": dict(layers=8, width=320, heads=5, ffn_width=896),
    "50m": dict(layers=10, width=512, heads=8, ffn_width=1408),
    "90m": dict(layers=14, width=640, heads=10, ffn_width=1792),
}


class DataConfig(StrictModel):
    cache_dir: Path = Path("data/c4")
    dataset: str = "allenai/c4"
    revision: str = "main"
    tokenizer: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    tokenizer_revision: str = "main"
    shuffle_seed: int = Field(42, ge=0, lt=2**32)
    shuffle_buffer: int = Field(10000, gt=0)
    shard_tokens: int = Field(16_777_216, gt=0)
    tokenize_batch_size: int = Field(256, gt=0)
    # Runtime buffering; separate from the document shuffle used during preparation.
    buffer_size_mib: float = Field(64.0, gt=0)
    prefetch: bool = True
    # Overrides for preparation/smoke runs; None means the complete validation split.
    prepare_train_tokens: int | None = Field(None, gt=0)
    prepare_validation_tokens: int | None = Field(None, gt=0)


class OptimizerConfig(StrictModel):
    lr: float = Field(1e-3, gt=0)
    beta1: float = Field(0.9, ge=0, lt=1)
    beta2: float = Field(0.95, ge=0, lt=1)
    eps: float = Field(1e-8, gt=0)
    weight_decay: float = Field(0.1, ge=0)
    grad_clip: float = Field(1.0, gt=0)
    warmup_fraction: float = Field(0.05, ge=0, lt=1)
    min_lr_ratio: float = Field(0.1, ge=0, le=1)


class TrainingConfig(StrictModel):
    tokens_per_parameter: float = Field(20.0, gt=0)
    epoch_tokens_per_parameter: float = Field(0.5, gt=0)
    batch_tokens: int = Field(32768, gt=0)
    micro_batch_size: int = Field(8, gt=0)
    checkpoint_every: int = Field(500, gt=0)
    log_every: int = Field(20, gt=0)
    # Explicit short-run overrides. Stored in configs; never silently applied by sweep.
    max_tokens: int | None = Field(None, gt=0)
    epoch_tokens: int | None = Field(None, gt=0)


class EvaluationConfig(StrictModel):
    subset_blocks: int = Field(1024, gt=0)
    batch_size: int = Field(8, gt=0)
    seed: int = Field(12345, ge=0, lt=2**32)


class RuntimeConfig(StrictModel):
    seed: int = Field(42, ge=0, lt=2**32)
    deterministic: bool = False
    device: str = "cuda:0"
    amp: bool = True
    compile: bool = False
    fused_optimizer: bool = True
    attention_backend: Literal["sdpa", "reference"] = "sdpa"
    output_dir: Path = Path("runs/default")
    cpu_threads: int = Field(8, gt=0)


class Config(StrictModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def batch_shape(self):
        length = self.model.context_length
        if self.training.batch_tokens % length:
            raise ValueError("training.batch_tokens must be divisible by context_length")
        if self.training.micro_batch_size > self.training.batch_tokens // length:
            raise ValueError("micro_batch_size exceeds effective batch size")
        if int(self.data.buffer_size_mib * 2**20) < 2 * length:
            raise ValueError("data.buffer_size_mib must hold at least one uint16 sequence")
        return self


def load_config(path: str | Path | None = None, overrides: list[str] = ()) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) if path else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep or not key or any(not part for part in key.split(".")):
            raise ValueError(f"expected dotted.key=value, got {override!r}")
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot descend into scalar configuration: {key}")
        node[parts[-1]] = yaml.safe_load(value)
    return Config.model_validate(raw)


def save_config(config: Config, path: Path) -> None:
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
