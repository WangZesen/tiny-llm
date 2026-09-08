"""Token-budgeted training with resumable data position and independent evaluation."""

import math
import signal
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from safetensors.torch import load_file, save_file

from tiny_llm.config import Config, save_config
from tiny_llm.data import TokenCache, fingerprint, training_boundaries, validation_indices
from tiny_llm.model import Llama, token_losses
from tiny_llm.runtime import (
    actual_backend,
    append_metric,
    atomic_checkpoint,
    atomic_json,
    attention_kernels,
    autocast,
    environment,
    preserve_rng,
    restore_rng,
    rng_state,
    setup_logging,
    setup_runtime,
)


def make_optimizer(model: Llama, config: Config, device: torch.device):
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


def loss_function(model, config: Config, device: torch.device):
    def loss(inputs, targets):
        with autocast(config, device):
            return token_losses(model(inputs), targets).sum()

    if config.runtime.compile and not config.runtime.deterministic:
        return torch.compile(loss)
    return loss


@torch.no_grad()
def evaluate(
    model: Llama, cache: TokenCache, config: Config, device: torch.device, full: bool
) -> dict:
    was_training = model.training
    start = time.monotonic()
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    indices = validation_indices(cache, config, full)
    try:
        with preserve_rng():
            model.eval()
            for offset in range(0, len(indices), config.evaluation.batch_size):
                x, y = cache.batch(
                    "validation",
                    indices[offset : offset + config.evaluation.batch_size],
                    config.model.context_length,
                    device,
                )
                with autocast(config, device):
                    losses = token_losses(model(x), y)
                total_loss += losses.double().sum()
                count += int((y != -100).sum().item())
                if full and offset % (config.evaluation.batch_size * 100) == 0:
                    logger.info("Full validation: {}/{} blocks", offset, len(indices))
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("empty validation set")
    mean = total_loss.item() / count
    if not math.isfinite(mean):
        raise FloatingPointError("nonfinite validation loss")
    return dict(
        loss=mean,
        perplexity=math.exp(mean) if mean < 700 else None,
        tokens=count,
        seconds=time.monotonic() - start,
        split="full" if full else "subset",
        validation_complete=cache.manifest["validation_complete"] if full else False,
    )


def recipe_identity(config: Config, cache: TokenCache) -> str:
    value = config.model_dump(mode="json")
    for name in ("output_dir", "device", "cpu_threads"):
        value["runtime"].pop(name)
    value["data"].pop("cache_dir")
    value["cache_identity"] = cache.manifest["identity"]
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

    config = config.model_copy(deep=True)
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
    model = Llama(config.model, actual_backend(config)).to(device)
    kernels = attention_kernels(model, config, device)
    assert model.parameter_count == config.model.parameter_count
    boundaries = training_boundaries(config, model.parameter_count)
    blocks, length = boundaries[-1], config.model.context_length
    if blocks > cache.blocks("train", length):
        raise ValueError(f"cache too small: need {blocks * length + 1:,} training tokens")
    order = np.random.default_rng(config.runtime.seed).permutation(blocks)
    optimizer = make_optimizer(model, config, device)
    compute_loss = loss_function(model, config, device)
    identity = recipe_identity(config, cache)
    cursor, step, completed_epochs, best_loss = 0, 0, 0, float("inf")
    best_epoch = None
    if resume:
        # Resume checkpoints are trusted local pickle artifacts, not untrusted model downloads.
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state.get("version") != 1 or state["recipe_identity"] != identity:
            raise ValueError("resume checkpoint is incompatible with this recipe or token cache")
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
    save_config(config, output / "resolved.yaml")
    metadata = environment()
    metadata.update(
        parameters=model.parameter_count,
        non_embedding_parameters=model.parameter_count
        - config.model.vocab_size * config.model.width,
        cache_identity=cache.manifest["identity"],
        recipe_identity=identity,
        attention=actual_backend(config),
        attention_kernels=kernels,
        compiled=config.runtime.compile and not config.runtime.deterministic,
        realized_tokens=blocks * length,
        epochs=len(boundaries),
        deterministic=config.runtime.deterministic,
    )
    atomic_json(output / ("environment-resume.json" if resume else "environment.json"), metadata)
    logger.info(
        "Training {:,} parameters for {:,} targets in {} virtual epochs",
        model.parameter_count,
        blocks * length,
        len(boundaries),
    )
    metrics_path = output / "metrics.jsonl"
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        logger.warning(
            "Signal {} received; checkpointing after the current optimizer update", signum
        )

    previous_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}

    def checkpoint(name="latest.pt"):
        atomic_checkpoint(
            output / name,
            dict(
                version=1,
                recipe_identity=identity,
                config=config.model_dump(mode="json"),
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                rng=rng_state(),
                cursor=cursor,
                step=step,
                completed_epochs=completed_epochs,
                best_loss=best_loss,
                best_epoch=best_epoch,
            ),
        )

    start = time.monotonic()
    window_start, window_tokens, window_loss = start, 0, 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
        for epoch_index in range(completed_epochs, len(boundaries)):
            boundary = boundaries[epoch_index]
            while cursor < boundary:
                step_start = cursor
                step_blocks = min(config.training.batch_tokens // length, boundary - cursor)
                lr = learning_rate(config, (cursor + step_blocks) * length, blocks * length)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                step_loss = torch.zeros((), device=device)
                for offset in range(0, step_blocks, config.training.micro_batch_size):
                    count = min(config.training.micro_batch_size, step_blocks - offset)
                    x, y = cache.batch(
                        "train",
                        order[step_start + offset : step_start + offset + count],
                        length,
                        device,
                    )
                    summed_loss = compute_loss(x, y)
                    (summed_loss / (step_blocks * length)).backward()
                    step_loss += summed_loss.detach()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optimizer.grad_clip, error_if_nonfinite=True
                )
                if not torch.isfinite(step_loss).item():
                    raise FloatingPointError(f"nonfinite training loss at step {step}")
                optimizer.step()
                cursor += step_blocks
                step += 1
                window_loss += step_loss.item()
                window_tokens += step_blocks * length
                if step % config.training.log_every == 0 or cursor == boundary:
                    now = time.monotonic()
                    row = dict(
                        event="train",
                        step=step,
                        epoch=epoch_index + 1,
                        tokens=cursor * length,
                        loss=window_loss / window_tokens,
                        lr=lr,
                        grad_norm=grad_norm.item(),
                        tokens_per_second=window_tokens / (now - window_start),
                        peak_memory_bytes=torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else 0,
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
                    window_start, window_tokens, window_loss = now, 0, 0.0
                if step % config.training.checkpoint_every == 0 or stopped:
                    checkpoint()
                if stopped:
                    result = dict(status="interrupted", step=step, tokens=cursor * length)
                    atomic_json(output / "status.json", result)
                    return result
            evaluation = evaluate(model, cache, config, device, full=False)
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
                "Epoch {} validation loss {:.5f} | perplexity {:.3f}",
                epoch_index + 1,
                evaluation["loss"],
                evaluation["perplexity"],
            )
            save_weights(model, output / f"epoch-{epoch_index + 1:03d}.safetensors")
            if evaluation["loss"] < best_loss:
                best_loss, best_epoch = evaluation["loss"], epoch_index + 1
                atomic_json(
                    output / "best.json",
                    dict(
                        epoch=best_epoch,
                        loss=best_loss,
                        weights=f"epoch-{best_epoch:03d}.safetensors",
                    ),
                )
            completed_epochs = epoch_index + 1
            checkpoint()
            if stopped:
                result = dict(status="interrupted", step=step, tokens=cursor * length)
                atomic_json(output / "status.json", result)
                return result
            # Exclude evaluation/checkpoint time from the next throughput window.
            window_start = time.monotonic()
        checkpoint("final.pt")
        final_eval = evaluate(model, cache, config, device, full=True)
        result = dict(
            status="complete",
            parameters=model.parameter_count,
            tokens=cursor * length,
            step=step,
            epochs=completed_epochs,
            best_subset_loss=best_loss,
            best_epoch=best_epoch,
            final_validation=final_eval,
            seconds_this_session=time.monotonic() - start,
            peak_memory_bytes=torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            recipe_identity=identity,
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
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def evaluate_checkpoint(config: Config, checkpoint: Path, full: bool) -> dict:
    device = setup_runtime(config)
    model = Llama(config.model, actual_backend(config)).to(device)
    if checkpoint.suffix == ".safetensors":
        model.load_state_dict(load_file(str(checkpoint)), strict=True)
    else:
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=False)["model"], strict=True
        )
    cache = TokenCache(config.data.cache_dir)
    cache.validate_config(config)
    result = evaluate(model, cache, config, device, full)
    logger.info("Evaluation {}", result)
    return result
