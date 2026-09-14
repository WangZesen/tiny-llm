"""Token-budgeted training with resumable data position and independent evaluation."""

import math
import signal
import time
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import FrameType
from typing import overload

import torch
from loguru import logger
from safetensors.torch import load_file, save_file

from tiny_llm.checkpoints import (
    RETENTION_FIELDS,
    load_training_checkpoint,
    require_new_epoch,
    save_epoch_checkpoint,
)
from tiny_llm.config import Config, save_config
from tiny_llm.data import (
    BufferedTokenLoader,
    TokenCache,
    fingerprint,
    subset_batch,
    training_boundaries,
)
from tiny_llm.model import Llama, token_losses
from tiny_llm.packed import PackedLlama, local_mean_losses
from tiny_llm.packed_optimizer import PackedAdamW
from tiny_llm.runtime import (
    actual_backend,
    append_metric,
    atomic_checkpoint,
    atomic_json,
    attention_kernels,
    autocast,
    clip_grad_norm_,
    environment,
    preserve_rng,
    restore_rng,
    rng_state,
    setup_logging,
    setup_runtime,
)
from tiny_llm.state import Batch, EvaluationResult, LossFunction, TrainingState


@overload
def make_optimizer(model: Llama, config: Config, device: torch.device) -> torch.optim.AdamW: ...


@overload
def make_optimizer(model: PackedLlama, config: Config, device: torch.device) -> PackedAdamW: ...


def make_optimizer(
    model: Llama | PackedLlama, config: Config, device: torch.device
) -> torch.optim.AdamW | PackedAdamW:
    if isinstance(model, PackedLlama):
        return PackedAdamW(model, config)
    cfg = config.optimizer
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    fused = (
        config.runtime.fused_optimizer
        and device.type == "cuda"
        and not config.runtime.deterministic
    )
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
    logger.info(
        "AdamW fused={} | attention={} | deterministic={}",
        fused,
        actual_backend(config),
        config.runtime.deterministic,
    )
    return optimizer


def learning_rate(config: Config, consumed: int, total: int) -> float:
    cfg = config.optimizer
    warmup = total * cfg.warmup_fraction
    if consumed < warmup:
        return cfg.lr * consumed / warmup
    progress = min(1.0, max(0.0, (consumed - warmup) / (total - warmup)))
    return cfg.lr * (
        cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
    )


@dataclass(frozen=True)
class AdaptiveConsensusSchedule:
    total_steps: int
    start_step: int
    lr_max: float | None
    p: float

    def gamma(self, step: int, lr: float) -> float:
        if step < self.start_step or self.p == 0 or self.lr_max is None:
            return 1.0
        return (lr / self.lr_max) ** self.p


def adaptive_consensus_schedule(
    config: Config, boundaries: list[int]
) -> AdaptiveConsensusSchedule | None:
    """Resolve the full run's active LR maximum, over the realized updates."""
    adaptive = config.decentralized.adaptive_consensus if config.decentralized else None
    if adaptive is None:
        return None
    length = config.model.context_length
    batch = config.training.batch_tokens // length
    previous, total_steps = 0, 0
    for boundary in boundaries:
        total_steps += (boundary - previous + batch - 1) // batch
        previous = boundary
    start_step = math.ceil(adaptive.start_frac * total_steps)
    previous, step, lr_max = 0, 0, None
    for boundary in boundaries:
        for cursor in range(previous, boundary, batch):
            if step >= start_step:
                consumed = min(cursor + batch, boundary) * length
                lr = learning_rate(config, consumed, boundaries[-1] * length)
                lr_max = lr if lr_max is None else max(lr_max, lr)
            step += 1
        previous = boundary
    if lr_max == 0 and adaptive.p > 0:
        raise ValueError("adaptive consensus requires a positive maximum LR in its active steps")
    return AdaptiveConsensusSchedule(total_steps, start_step, lr_max, adaptive.p)


def loss_function(model: Llama | PackedLlama, config: Config, device: torch.device) -> LossFunction:
    def loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with autocast(config, device):
            if isinstance(model, PackedLlama):
                return local_mean_losses(model(inputs), targets)
            return token_losses(model(inputs), targets).sum()

    if config.runtime.compile and not config.runtime.deterministic:
        compiled = torch.compile(loss, mode=config.runtime.compile_mode)

        batch_axis = 1 if isinstance(model, PackedLlama) else 0

        def dispatched(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            # Microbatch remainders do not amortize another compilation.
            if inputs.shape[batch_axis] != config.training.micro_batch_size:
                return loss(inputs, targets)
            return compiled(inputs, targets)

        return dispatched
    return loss


def optimizer_update(
    model: Llama,
    optimizer: torch.optim.AdamW,
    compute_loss: LossFunction,
    next_batch: Callable[[int, torch.device], Batch],
    config: Config,
    device: torch.device,
    step_blocks: int,
    profile: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The production update, also used by synthetic and real-data benchmarks."""

    def region(name):
        return torch.profiler.record_function(name) if profile else nullcontext()

    optimizer.zero_grad(set_to_none=True)
    step_loss = torch.zeros((), device=device)
    for offset in range(0, step_blocks, config.training.micro_batch_size):
        count = min(config.training.micro_batch_size, step_blocks - offset)
        with region("data_and_transfer"):
            x, y = next_batch(count, device)
        with region("forward_and_loss"):
            summed_loss = compute_loss(x, y)
        with region("backward"):
            (summed_loss / (step_blocks * config.model.context_length)).backward()
        step_loss += summed_loss.detach()
    with region("clipping_and_finite_check"):
        grad_norm = clip_grad_norm_(model.parameters(), config.optimizer.grad_clip)
        if not torch.isfinite(step_loss).item():
            raise FloatingPointError("nonfinite training loss")
    with region("optimizer"):
        optimizer.step()
    return step_loss, grad_norm


@torch.no_grad()
def evaluate(
    model: Llama,
    cache: TokenCache,
    config: Config,
    device: torch.device,
    full: bool,
    should_stop: Callable[[], bool] | None = None,
) -> EvaluationResult:
    was_training = model.training
    start = time.monotonic()
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    loader = None
    try:
        with preserve_rng():
            model.eval()
            if full:
                count_blocks = cache.blocks("validation", config.model.context_length, partial=True)
                full_loader = BufferedTokenLoader(
                    cache,
                    "validation",
                    config.model.context_length,
                    count_blocks,
                    config.data.buffer_size_mib,
                    prefetch=config.data.prefetch,
                )
                loader = full_loader
                batches: Iterator[Batch] = (
                    full_loader.next_batch(
                        min(config.evaluation.batch_size, count_blocks - offset), device
                    )
                    for offset in range(0, count_blocks, config.evaluation.batch_size)
                )
            else:
                tokens, valid = cache.validation_subset(config, should_stop)
                count_blocks = len(tokens)
                batches = (
                    subset_batch(
                        tokens[offset : offset + config.evaluation.batch_size],
                        valid[offset : offset + config.evaluation.batch_size],
                        device,
                    )
                    for offset in range(0, count_blocks, config.evaluation.batch_size)
                )
            for offset in range(0, count_blocks, config.evaluation.batch_size):
                if should_stop is not None and should_stop():
                    raise InterruptedError("evaluation interrupted")
                x, y = next(batches)
                with autocast(config, device):
                    losses = token_losses(model(x), y)
                total_loss += losses.double().sum()
                count += int((y != -100).sum().item())
                if full and offset % (config.evaluation.batch_size * 100) == 0:
                    logger.info("Full validation: {}/{} blocks", offset, count_blocks)
    finally:
        if loader is not None:
            loader.close()
        model.train(was_training)
    if count == 0:
        raise ValueError("empty validation set")
    mean = total_loss.item() / count
    if not math.isfinite(mean):
        raise FloatingPointError("nonfinite validation loss")
    return EvaluationResult(
        loss=mean,
        perplexity=math.exp(mean) if mean < 700 else None,
        tokens=count,
        seconds=time.monotonic() - start,
        split="full" if full else "subset",
        validation_complete=cache.manifest["validation_complete"] if full else False,
    )


def recipe_identity(config: Config, cache: TokenCache) -> str:
    value = config.model_dump(mode="json")
    for key in RETENTION_FIELDS:
        value["training"].pop(key)
    for name in ("output_dir", "device", "cpu_threads"):
        value["runtime"].pop(name)
    value["data"].pop("cache_dir")
    value["data"].pop("prefetch")  # Scheduling changes do not alter sample order.
    value["cache_identity"] = cache.manifest["identity"]
    value["loader_version"] = BufferedTokenLoader.VERSION
    return fingerprint(value)


def save_weights(model: Llama, path: Path):
    temporary = path.with_name(path.name + ".tmp")
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
        str(temporary),
    )
    temporary.replace(path)


def train(config: Config, resume: Path | None = None) -> dict:
    import fcntl

    config = Config.model_validate(config.model_dump())
    output = config.runtime.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(config, resume)


def _train(config: Config, resume: Path | None) -> dict:
    output = config.runtime.output_dir
    if (output / "resolved.yaml").exists() and resume is None:
        raise ValueError(f"run already exists at {output}; use --resume or a new output directory")
    setup_logging(output / "run.log")
    device = setup_runtime(config)
    cache = TokenCache(config.data.cache_dir)
    cache.validate_config(config)
    cache.verify()
    config.data.revision = cache.manifest["dataset_revision"]
    config.data.tokenizer_revision = cache.manifest["tokenizer_revision"]
    decentralized = config.decentralized
    num_models = decentralized.num_models if decentralized else 1
    boundaries = training_boundaries(config, config.model.parameter_count)
    selected_epochs = config.training.saved_epochs()
    consensus_schedule = adaptive_consensus_schedule(config, boundaries)
    model = (
        PackedLlama(config.model, num_models, actual_backend(config))
        if decentralized
        else Llama(config.model, actual_backend(config))
    ).to(device)
    kernels = attention_kernels(model, config, device)
    assert model.parameter_count == num_models * config.model.parameter_count
    averaged = None
    if decentralized:
        with preserve_rng():
            averaged = Llama(config.model, actual_backend(config)).to(device)

    def evaluation_model() -> Llama:
        if isinstance(model, PackedLlama):
            assert averaged is not None
            model.copy_average_to(averaged)
            return averaged
        return model

    blocks, length = boundaries[-1], config.model.context_length
    if blocks > cache.blocks("train", length):
        raise ValueError(f"cache too small: need {blocks * length + 1:,} training tokens")
    optimizer = make_optimizer(model, config, device)
    compute_loss = loss_function(model, config, device)
    identity = recipe_identity(config, cache)
    cursor, step, completed_epochs, best_loss = 0, 0, 0, float("inf")
    best_epoch = None
    state = None
    if resume:
        # Resume checkpoints are trusted local pickle artifacts, not untrusted model downloads.
        state = load_training_checkpoint(resume)
        if state.get("version") != (3 if decentralized else 2):
            raise ValueError(f"incompatible training checkpoint version: {state.get('version')}")
        if state["recipe_identity"] != identity:
            raise ValueError("resume checkpoint is incompatible with this recipe or token cache")
        if isinstance(model, PackedLlama):
            model.load_packed_state_dict(state["model"])
        else:
            model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        cursor, step, completed_epochs = state["cursor"], state["step"], state["completed_epochs"]
        best_loss, best_epoch = state["best_loss"], state["best_epoch"]
        restore_rng(state["rng"])
        logger.info(
            "Resuming step {} at {:,} targets, {} completed epochs",
            step,
            cursor * length,
            completed_epochs,
        )
    for epoch in sorted(selected_epochs):
        if epoch <= completed_epochs:
            continue
        suffix = ".pt" if config.training.save_epoch_training_state else ".safetensors"
        require_new_epoch(output / f"epoch-{epoch:03d}{suffix}")
    loader = BufferedTokenLoader(
        cache,
        "train",
        length,
        blocks,
        config.data.buffer_size_mib,
        seed=config.runtime.seed,
        prefetch=config.data.prefetch,
        cursor=cursor,
    )
    if state is not None:
        loader.validate_state(state["loader"])
    save_config(config, output / "resolved.yaml")
    metadata = environment()
    metadata.update(
        parameters=config.model.parameter_count,
        non_embedding_parameters=config.model.parameter_count
        - config.model.vocab_size * config.model.width,
        cache_identity=cache.manifest["identity"],
        recipe_identity=identity,
        attention=actual_backend(config),
        attention_kernels=kernels,
        compiled=config.runtime.compile and not config.runtime.deterministic,
        realized_tokens=blocks * length,
        epochs=len(boundaries),
        deterministic=config.runtime.deterministic,
        loader=loader.identity,
    )
    if decentralized:
        metadata.update(
            num_models=num_models,
            total_parameters=model.parameter_count,
            topology=decentralized.topology,
            scheme=decentralized.scheme,
            storage_numel=model.storage_numel,
        )
    if consensus_schedule is not None:
        metadata["adaptive_consensus"] = asdict(consensus_schedule)
    atomic_json(output / ("environment-resume.json" if resume else "environment.json"), metadata)
    logger.info(
        "Training {:,} parameters ({:,} excluding embeddings and tied LM head) "
        "for {:,} targets in {} virtual epochs",
        model.parameter_count,
        model.parameter_count - model.embedding.weight.numel(),
        blocks * length,
        len(boundaries),
    )
    metrics_path = output / "metrics.jsonl"
    stopped = False

    def stop(signum: int, frame: FrameType | None) -> None:
        nonlocal stopped
        stopped = True
        logger.warning(
            "Signal {} received; stopping with checkpoint policy {}",
            signum,
            config.training.checkpoint_policy,
        )

    previous_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}

    def checkpoint(name: str, *, epoch: bool = False) -> None:
        if config.training.checkpoint_policy == "none":
            return
        state = TrainingState(
            version=3 if decentralized else 2,
            recipe_identity=identity,
            config=config.model_dump(mode="json"),
            model=model.packed_state_dict()
            if isinstance(model, PackedLlama)
            else model.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=rng_state(),
            cursor=cursor,
            step=step,
            completed_epochs=completed_epochs,
            best_loss=best_loss,
            best_epoch=best_epoch,
            loader=loader.state_dict(committed_cursor=cursor),
        )
        if epoch:
            save_epoch_checkpoint(output / name, state, model, optimizer)
        else:
            atomic_checkpoint(output / name, state)

    start = time.monotonic()
    session_initial_tokens = cursor * length
    training_seconds = 0.0
    window_start, window_tokens = start, 0
    window_loss = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        logger.info("Collecting fixed validation subset with a sequential scan")
        cache.validation_subset(config, should_stop=lambda: stopped)
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_start = window_start = time.monotonic()
        packed_metrics: tuple[torch.Tensor, torch.Tensor, float] | None = None
        for epoch_index in range(completed_epochs, len(boundaries)):
            boundary = boundaries[epoch_index]
            while cursor < boundary:
                step_blocks = config.training.batch_tokens // length
                lr = learning_rate(config, (cursor + step_blocks) * length, blocks * length)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                if isinstance(model, PackedLlama):
                    assert isinstance(optimizer, PackedAdamW) and decentralized is not None
                    optimizer.zero_grad(set_to_none=True)
                    local_batch = step_blocks // num_models
                    x, y = loader.next_batch(step_blocks, device)
                    # Assign every Nth sample in the buffered stream to one worker.
                    # Both epoch boundaries and committed cursors are multiples of N.
                    x, y = (
                        tensor.view(local_batch, num_models, length).transpose(0, 1).contiguous()
                        for tensor in (x, y)
                    )
                    means = compute_loss(x, y)
                    means.sum().backward()
                    step_loss = means.detach().sum() * (local_batch * length)
                    local_grad_norms = optimizer.clip_grad_norm_(config.optimizer.grad_clip)
                    grad_norm = local_grad_norms.max()
                    if not torch.isfinite(step_loss).item():
                        raise FloatingPointError(f"nonfinite training loss at step {step}")
                    mixing_gamma = consensus_schedule.gamma(step, lr) if consensus_schedule else 1.0
                    if decentralized.scheme == "atc":
                        optimizer.step()
                    model.mix_(decentralized.topology, step, gamma=mixing_gamma)
                    if decentralized.scheme == "awc":
                        optimizer.step()
                    packed_metrics = (means.detach(), local_grad_norms, mixing_gamma)
                else:
                    assert isinstance(optimizer, torch.optim.AdamW)
                    step_loss, grad_norm = optimizer_update(
                        model,
                        optimizer,
                        compute_loss,
                        loader.next_batch,
                        config,
                        device,
                        step_blocks,
                    )
                cursor += step_blocks
                step += 1
                window_loss += step_loss.item()
                window_tokens += step_blocks * length
                if step % config.training.log_every == 0 or cursor == boundary:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    now = time.monotonic()
                    window_seconds = now - window_start
                    training_seconds += window_seconds
                    row: dict[str, object] = dict(
                        event="train",
                        step=step,
                        epoch=epoch_index + 1,
                        tokens=cursor * length,
                        loss=window_loss / window_tokens,
                        lr=lr,
                        grad_norm=grad_norm.item(),
                        tokens_per_second=window_tokens / window_seconds,
                        training_seconds=window_seconds,
                        elapsed_tokens_per_second=(cursor * length - session_initial_tokens)
                        / (now - training_start),
                        peak_memory_bytes=torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else 0,
                    )
                    if packed_metrics is not None:
                        means, local_grad_norms, mixing_gamma = packed_metrics
                        row.update(
                            local_losses=means.tolist(),
                            local_grad_norms=local_grad_norms.tolist(),
                            tokens_per_model=cursor * length // num_models,
                            mixing_step=step - 1,
                            mixing_gamma=mixing_gamma,
                        )
                    append_metric(metrics_path, row)
                    logger.info(
                        "Epoch {}/{} | step {} | tokens {:,} | loss {:.4f} | lr {:.3g} | {:,.0f} tok/s",
                        epoch_index + 1,
                        len(boundaries),
                        step,
                        cursor * length,
                        row["loss"],
                        lr,
                        row["tokens_per_second"],
                    )
                    window_loss = 0.0
                    window_start, window_tokens = time.monotonic(), 0
                if stopped:
                    result = dict(status="interrupted", step=step, tokens=cursor * length)
                    atomic_json(output / "status.json", result)
                    return result
            epoch_model = evaluation_model()
            evaluation = evaluate(
                epoch_model, cache, config, device, full=False, should_stop=lambda: stopped
            )
            append_metric(
                metrics_path,
                {
                    **evaluation,
                    "event": "validation",
                    "epoch": epoch_index + 1,
                    "step": step,
                    "tokens": cursor * length,
                    "evaluation_tokens": evaluation["tokens"],
                },
            )
            logger.info(
                "Epoch {} validation loss {:.5f} | perplexity {}",
                epoch_index + 1,
                evaluation["loss"],
                evaluation["perplexity"],
            )
            save_epoch = epoch_index + 1 in selected_epochs
            if save_epoch:
                save_weights(
                    epoch_model,
                    output / f"epoch-{epoch_index + 1:03d}.safetensors",
                )
            if isinstance(model, PackedLlama) and save_epoch:
                for worker in range(num_models):
                    directory = output / f"node-{worker:03d}"
                    directory.mkdir(exist_ok=True)
                    path = directory / f"epoch-{epoch_index + 1:03d}.safetensors"
                    temporary = path.with_suffix(".tmp")
                    save_file(
                        {
                            key: value.cpu().contiguous()
                            for key, value in model.local_state_dict(worker).items()
                        },
                        str(temporary),
                    )
                    temporary.replace(path)
            if evaluation["loss"] < best_loss:
                best_loss, best_epoch = evaluation["loss"], epoch_index + 1
                atomic_json(
                    output / "best.json",
                    dict(
                        epoch=best_epoch,
                        loss=best_loss,
                        weights=(f"epoch-{best_epoch:03d}.safetensors" if save_epoch else None),
                    ),
                )
            completed_epochs = epoch_index + 1
            if save_epoch and config.training.save_epoch_training_state:
                checkpoint(f"epoch-{completed_epochs:03d}.pt", epoch=True)
            if stopped:
                result = dict(status="interrupted", step=step, tokens=cursor * length)
                atomic_json(output / "status.json", result)
                return result
            # Exclude evaluation/checkpoint time from the next throughput window.
            window_start = time.monotonic()
        checkpoint("final.pt")
        training_elapsed_seconds = time.monotonic() - training_start
        loader.close()  # Release training buffers before allocating validation buffers.
        final_eval = evaluate(
            evaluation_model(), cache, config, device, full=True, should_stop=lambda: stopped
        )
        result = dict(
            status="complete",
            parameters=config.model.parameter_count,
            tokens=cursor * length,
            step=step,
            epochs=completed_epochs,
            best_subset_loss=best_loss,
            best_epoch=best_epoch,
            final_validation=final_eval,
            seconds_this_session=time.monotonic() - start,
            training_seconds=training_seconds,
            training_elapsed_seconds=training_elapsed_seconds,
            training_tokens_per_second=(cursor * length - session_initial_tokens) / training_seconds
            if training_seconds
            else 0.0,
            elapsed_tokens_per_second=(cursor * length - session_initial_tokens)
            / training_elapsed_seconds,
            peak_memory_bytes=torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            recipe_identity=identity,
        )
        if decentralized:
            result.update(
                num_models=num_models,
                total_parameters=model.parameter_count,
                tokens_per_model=cursor * length // num_models,
                topology=decentralized.topology,
                scheme=decentralized.scheme,
            )
        append_metric(
            metrics_path,
            {
                **final_eval,
                "event": "final_validation",
                "step": step,
                "tokens": cursor * length,
                "evaluation_tokens": final_eval["tokens"],
            },
        )
        atomic_json(output / "result.json", result)
        atomic_json(output / "status.json", result)
        logger.success("Completed training: full validation loss {:.5f}", final_eval["loss"])
        return result
    except InterruptedError:
        result = dict(status="interrupted", step=step, tokens=cursor * length)
        atomic_json(output / "status.json", result)
        return result
    except BaseException as exc:
        atomic_json(
            output / "status.json",
            dict(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                step=step,
                tokens=cursor * length,
            ),
        )
        raise
    finally:
        loader.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def evaluate_checkpoint(config: Config, checkpoint: Path, full: bool) -> EvaluationResult:
    device = setup_runtime(config)
    model = Llama(config.model, actual_backend(config)).to(device)
    if checkpoint.suffix == ".safetensors":
        model.load_state_dict(load_file(str(checkpoint)), strict=True)
    else:
        state = load_training_checkpoint(checkpoint)
        if "parameter_storage" in state["model"]:
            saved = Config.model_validate(state["config"])
            if saved.model != config.model or saved.decentralized is None:
                raise ValueError("checkpoint model configuration mismatch")
            with preserve_rng():
                packed = PackedLlama(
                    config.model, saved.decentralized.num_models, actual_backend(config)
                )
            packed.load_packed_state_dict(state["model"])
            packed.copy_average_to(model)
            del packed
        else:
            model.load_state_dict(state["model"], strict=True)
    cache = TokenCache(config.data.cache_dir)
    cache.validate_config(config)
    result = evaluate(model, cache, config, device, full)
    logger.info("Evaluation {}", result)
    return result
