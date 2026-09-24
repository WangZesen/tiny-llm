"""Strict YAML configuration, shared by the CLI and Python API."""

from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class ModelConfig(StrictModel):
    vocab_size: int = Field(default=32000, ge=2, le=65536)
    layers: int = Field(default=8, gt=0)
    width: int = Field(default=320, gt=0)
    heads: int = Field(default=5, gt=0)
    ffn_width: int = Field(default=896, gt=0)
    context_length: int = Field(default=1024, gt=0)
    rope_theta: float = Field(default=10000.0, gt=0)
    norm_eps: float = Field(default=1e-5, gt=0)
    init_std: float = Field(default=0.02, gt=0)

    @model_validator(mode="after")
    def dimensions(self) -> Self:
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
    shuffle_seed: int = Field(default=42, ge=0, lt=2**32)
    shuffle_buffer: int = Field(default=100_000, gt=0)
    shard_tokens: int = Field(default=33_554_432, gt=0)
    tokenize_batch_size: int = Field(default=256, gt=0)
    prepare_workers: int = Field(default=8, gt=0, strict=True)
    # Runtime buffering; separate from the document shuffle used during preparation.
    shuffle_group_size: int = Field(default=2, gt=0, strict=True)
    prefetch: bool = True
    # Overrides for preparation/smoke runs; None means the complete validation split.
    prepare_train_tokens: int | None = Field(default=None, gt=0)
    prepare_validation_tokens: int | None = Field(default=None, gt=0)


class OptimizerConfig(StrictModel):
    lr: float = Field(default=1e-3, gt=0)
    beta1: float = Field(default=0.9, ge=0, lt=1)
    beta2: float = Field(default=0.95, ge=0, lt=1)
    eps: float = Field(default=1e-8, gt=0)
    weight_decay: float = Field(default=0.1, ge=0)
    grad_clip: float | None = Field(default=1.0, gt=0)


class CosineScheduleConfig(StrictModel):
    name: Literal["cosine"] = "cosine"
    warmup_steps: int = Field(default=312, ge=0, strict=True)
    min_lr_ratio: float = Field(default=0.0, ge=0, le=1)


class WSDScheduleConfig(StrictModel):
    name: Literal["wsd"] = "wsd"
    warmup_steps: int = Field(default=312, ge=0, strict=True)
    decay_fraction: float = Field(default=0.1, ge=0, le=1)


class TrainingConfig(StrictModel):
    tokens_per_parameters: float = Field(default=20.0, gt=0)
    epoch_tokens_per_parameters: float = Field(default=0.5, gt=0)
    batch_tokens: int = Field(default=32768, gt=0)
    micro_batch_size: int = Field(default=32, gt=0)
    checkpoint_policy: Literal["interval", "explicit", "final", "none"] = "final"
    checkpoint_epochs: list[Annotated[int, Field(strict=True, gt=0)]] = Field(
        default_factory=lambda: [1], min_length=1
    )
    save_epoch_training_state: bool = True
    log_every: int = Field(default=20, gt=0)

    @field_validator("checkpoint_epochs", mode="before")
    @classmethod
    def normalize_epochs(cls, value: object) -> object:
        return [value] if isinstance(value, int) else value

    @field_validator("checkpoint_epochs")
    @classmethod
    def unique_epochs(cls, value: list[int], info: ValidationInfo) -> list[int]:
        return sorted(set(value)) if info.data.get("checkpoint_policy") == "explicit" else value

    @property
    def epoch_count(self) -> int:
        count = Fraction(str(self.tokens_per_parameters)) / Fraction(
            str(self.epoch_tokens_per_parameters)
        )
        if count.denominator != 1:
            raise ValueError(
                "tokens_per_parameters must be divisible by epoch_tokens_per_parameters"
            )
        return count.numerator

    @model_validator(mode="after")
    def schedule(self) -> Self:
        epochs = self.epoch_count
        if self.checkpoint_policy == "interval" and len(self.checkpoint_epochs) != 1:
            raise ValueError("interval checkpoint policy requires exactly one epoch interval")
        if self.checkpoint_policy == "explicit" and self.checkpoint_epochs[-1] > epochs:
            raise ValueError("checkpoint epoch exceeds the training epoch count")
        return self

    def saved_epochs(self) -> frozenset[int]:
        if self.checkpoint_policy == "interval":
            interval = self.checkpoint_epochs[0]
            return frozenset(range(interval, self.epoch_count + 1, interval))
        if self.checkpoint_policy == "explicit":
            return frozenset(self.checkpoint_epochs)
        return frozenset()


class EvaluationConfig(StrictModel):
    subset_blocks: int = Field(default=1024, gt=0)
    batch_size: int = Field(default=128, gt=0)
    seed: int = Field(default=12345, ge=0, lt=2**32)


class RuntimeConfig(StrictModel):
    seed: int = Field(default=42, ge=0, lt=2**32)
    deterministic: bool = False
    device: str = "cuda:0"
    amp: bool = True
    compile: bool = True
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    sdpa_backend: Literal["auto", "flash", "cudnn"] = "auto"
    fused_optimizer: bool = True
    attention_backend: Literal["sdpa", "reference"] = "sdpa"
    output_dir: Path = Path("runs/default")
    cpu_threads: int = Field(default=8, gt=0)


class AdaptiveConsensusConfig(StrictModel):
    start_frac: float = Field(ge=0, le=1)
    p: float = Field(ge=0)


def validate_topology_size(topology: str, num_models: int) -> None:
    if num_models == 1:
        return
    if topology == "one_peer_ring" and num_models % 2:
        raise ValueError("one_peer_ring requires an even num_models (or 1)")
    if topology == "one_peer_exponential" and num_models & (num_models - 1):
        raise ValueError("one_peer_exponential requires a power-of-two num_models (or 1)")


class DecentralizedConfig(StrictModel):
    num_models: int = Field(gt=0)
    topology: Literal["complete", "one_peer_ring", "one_peer_exponential"] = "complete"
    scheme: Literal["awc", "atc"] = "awc"
    adaptive_consensus: AdaptiveConsensusConfig | None = None

    @model_validator(mode="after")
    def topology_size(self) -> Self:
        validate_topology_size(self.topology, self.num_models)
        return self


class Config(StrictModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    lr_schedule: Annotated[
        CosineScheduleConfig | WSDScheduleConfig, Field(discriminator="name")
    ] = Field(default_factory=CosineScheduleConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    decentralized: DecentralizedConfig | None = None

    @model_validator(mode="after")
    def batch_shape(self) -> Self:
        length = self.model.context_length
        if self.training.batch_tokens % length:
            raise ValueError("training.batch_tokens must be divisible by context_length")
        if self.training.micro_batch_size > self.training.batch_tokens // length:
            raise ValueError("micro_batch_size exceeds effective batch size")
        if self.decentralized is not None:
            expected = self.decentralized.num_models * self.training.micro_batch_size * length
            if self.training.batch_tokens != expected:
                raise ValueError(
                    "decentralized training requires batch_tokens = "
                    "num_models * micro_batch_size * context_length (no accumulation)"
                )
        return self


def _merge_config(target: dict, source: dict, *, root: bool = True) -> None:
    for key, value in source.items():
        previous = target.get(key)
        if isinstance(previous, dict) and isinstance(value, dict):
            if (
                root
                and key == "lr_schedule"
                and "name" in previous
                and "name" in value
                and previous["name"] != value["name"]
            ):
                previous.clear()
            _merge_config(previous, value, root=False)
        else:
            target[key] = value


def load_config(
    path: str | Path | Sequence[str | Path] | None = None, overrides: Sequence[str] = ()
) -> Config:
    """Merge YAML files in order, then apply dotted overrides and validate once."""
    paths = () if path is None else (path,) if isinstance(path, (str, Path)) else path
    raw: dict = {}
    for filename in paths:
        value = yaml.safe_load(Path(filename).read_text())
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"configuration must be a YAML mapping: {filename}")
        _merge_config(raw, value)
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
        parsed = yaml.safe_load(value)
        if parts == ["lr_schedule", "name"] and "name" in node and node["name"] != parsed:
            node.clear()
        node[parts[-1]] = parsed
    return Config.model_validate(raw)


def save_config(config: Config, path: Path) -> None:
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
