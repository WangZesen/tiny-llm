#!/usr/bin/env python3
"""Allocation worker for the immutable native/GH200 comparison campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from gh200_campaign import activate, digest, read, sha, verify, write

if TYPE_CHECKING:
    from torch import Tensor


class NativeRunner:
    """Mirror production packed orchestration; call its actual synchronous update.

    This adapter does not alter any source in the frozen native package. The
    packed sequence is deliberately kept identical to train._train and covered
    by a comparison against the real trainer on CPU fixtures.
    """

    def __init__(self, config):
        import torch

        from tiny_llm.data import BufferedTokenLoader, TokenCache, training_boundaries
        from tiny_llm.model import Llama
        from tiny_llm.packed import PackedLlama
        from tiny_llm.runtime import actual_backend, setup_runtime
        from tiny_llm.train import adaptive_consensus_schedule, loss_function, make_optimizer

        self.torch = torch
        self.config = config
        self.device = setup_runtime(config)
        self.cache = TokenCache(config.data.cache_dir)
        self.cache.validate_config(config)
        self.n = config.decentralized.num_models if config.decentralized else 1
        self.model = (
            PackedLlama(config.model, self.n, actual_backend(config))
            if config.decentralized else Llama(config.model, actual_backend(config))
        ).to(self.device)
        self.optimizer = make_optimizer(self.model, config, self.device)
        self.loss = loss_function(self.model, config, self.device)
        self.boundaries = training_boundaries(config, config.model.parameter_count)
        self.schedule = adaptive_consensus_schedule(config, self.boundaries)
        self.length = config.model.context_length
        self.step_blocks = config.training.batch_tokens // self.length
        self.loader = BufferedTokenLoader(
            self.cache, "train", self.length, self.boundaries[-1],
            config.data.shuffle_group_size, seed=config.runtime.seed, prefetch=config.data.prefetch,
        )
        self.cursor = self.step = 0
        self.gradients: dict[str, Tensor] | None = None
        self.input_hashes = []

    def state(self, *, cpu=False):
        from tiny_llm.checkpoints import compact_cpu
        from tiny_llm.packed import PackedLlama

        model = self.model.packed_state_dict() if isinstance(self.model, PackedLlama) else self.model.state_dict()
        state = dict(model=model, optimizer=self.optimizer.state_dict())
        return compact_cpu(state) if cpu else copy.deepcopy(state)

    def restore(self, state):
        from tiny_llm.packed import PackedLlama

        if isinstance(self.model, PackedLlama):
            self.model.load_packed_state_dict(state["model"])
        else:
            self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    def update(self, inspect=False):
        from unittest.mock import patch

        from torch.optim import AdamW

        from tiny_llm.model import Llama
        from tiny_llm.packed import PackedLlama
        from tiny_llm.packed_optimizer import PackedAdamW
        from tiny_llm.train import learning_rate, optimizer_update

        torch, config = self.torch, self.config
        lr = learning_rate(config, (self.cursor + self.step_blocks) * self.length,
                           self.boundaries[-1] * self.length)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        def remember():
            gradients = {}
            for name, parameter in self.model.named_parameters():
                if parameter.grad is None:
                    raise AssertionError(f"missing dense gradient: {name}")
                gradients[name] = parameter.grad.detach().clone()
            self.gradients = gradients

        def next_batch(count, device):
            x, y = self.loader.next_batch(count, device)
            if inspect:
                self.input_hashes.append(hashlib.sha256(
                    x.cpu().numpy().tobytes() + y.cpu().numpy().tobytes()).hexdigest())
            return x, y

        if config.decentralized:
            assert isinstance(self.model, PackedLlama)
            assert isinstance(self.optimizer, PackedAdamW)
            self.optimizer.zero_grad(set_to_none=True)
            local_batch = self.step_blocks // self.n
            x, y = next_batch(self.step_blocks, self.device)
            x, y = (t.view(local_batch, self.n, self.length).transpose(0, 1).contiguous()
                    for t in (x, y))
            means = self.loss(x, y)
            means.sum().backward()
            value = means.detach().sum() * (local_batch * self.length)
            if inspect:
                remember()
            norms = self.optimizer.clip_grad_norm_(config.optimizer.grad_clip)
            if not torch.isfinite(value).item():
                raise FloatingPointError("nonfinite training loss")
            gamma = self.schedule.gamma(self.step, lr) if self.schedule else 1.0
            if config.decentralized.scheme == "atc":
                self.optimizer.step()
            self.model.mix_(config.decentralized.topology, self.step, gamma=gamma)
            if config.decentralized.scheme == "awc":
                self.optimizer.step()
            local_losses = means.detach()
        else:
            assert isinstance(self.model, Llama)
            assert isinstance(self.optimizer, AdamW)
            if inspect:
                from tiny_llm.runtime import clip_grad_norm_

                def inspected_clip(parameters, maximum):
                    remember()
                    return clip_grad_norm_(parameters, maximum)

                with patch("tiny_llm.train.clip_grad_norm_", inspected_clip):
                    value, norm = optimizer_update(self.model, self.optimizer, self.loss,
                                                  next_batch, config, self.device, self.step_blocks)
            else:
                value, norm = optimizer_update(self.model, self.optimizer, self.loss,
                                              next_batch, config, self.device, self.step_blocks)
            norms = norm.reshape(1)
            local_losses = (value / config.training.batch_tokens).reshape(1)
        self.cursor += self.step_blocks
        self.step += 1
        return value, norms, local_losses, lr

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def close(self):
        self.loader.close()

    def tensors(self):
        from tiny_llm.packed import PackedLlama
        from tiny_llm.packed_optimizer import PackedAdamW

        packed = isinstance(self.model, PackedLlama)
        if self.gradients is None:
            raise RuntimeError("gradient inspection must be requested before exporting tensors")
        result = []
        for worker in range(self.n):
            parameters = {name: p[worker] if packed else p for name, p in self.model.named_parameters()}
            grads = {name: g[worker] if packed else g for name, g in self.gradients.items()}
            if packed:
                assert isinstance(self.optimizer, PackedAdamW)
                optimizer = self.optimizer.optimizers[worker]
                local_params = self.optimizer.local_parameters[worker]
            else:
                assert isinstance(self.optimizer, self.torch.optim.AdamW)
                optimizer = self.optimizer
                local_params = list(self.model.parameters())
            moments = {}
            counters = []
            for (name, _), p in zip(parameters.items(), local_params, strict=True):
                state = optimizer.state[p]
                for key in ("exp_avg", "exp_avg_sq"):
                    moments[f"{name}/{key}"] = state[key]
                counters.append(int(state["step"].item()))
            result.append(dict(parameters=parameters, gradients=grads, moments=moments,
                               counters=counters))
        return result


def tensor_metrics(left, right):
    by_parameter = {}
    error = scale = maximum = 0.0
    for name in left:
        b = right[name].detach().double()
        a = left[name].detach().to(device=b.device, dtype=b.dtype)
        delta = a - b
        e = float(delta.square().sum().item())
        s = float(a.square().sum().item())
        absolute = float(delta.abs().max().item())
        relative = math.sqrt(e / max(s, 1e-30))
        by_parameter[name] = dict(relative_l2=relative, maximum_absolute=absolute)
        error += e
        scale += s
        maximum = max(maximum, absolute)
    if not all(math.isfinite(v) for v in (error, scale, maximum)):
        raise FloatingPointError("nonfinite comparison statistics")
    return dict(relative_l2=math.sqrt(error / max(scale, 1e-30)),
                maximum_absolute=maximum, parameters=by_parameter)


def benchmark(config, protocol, output):
    import torch

    from tiny_llm.runtime import append_metric, environment

    start = time.monotonic()
    runner = NativeRunner(config)
    metadata = environment()
    clip_counts = torch.zeros(runner.n, dtype=torch.int64, device=runner.device)
    log_loss = 0.0
    log_tokens = 0
    metrics_path = output / "metrics.jsonl"
    log_start = time.monotonic()

    def update():
        nonlocal log_loss, log_tokens, log_start
        value, norms, losses, lr = runner.update()
        # Preserve current production per-step synchronization and diagnostics.
        log_loss += value.item()
        log_tokens += config.training.batch_tokens
        if config.optimizer.grad_clip is not None:
            clip_counts.add_(norms > config.optimizer.grad_clip)
        if runner.step % config.training.log_every == 0 or runner.cursor in runner.boundaries:
            runner.synchronize()
            now = time.monotonic()
            row = dict(event="train", step=runner.step, tokens=runner.cursor * runner.length,
                       loss=log_loss / log_tokens, lr=lr, grad_norm=norms.max().item(),
                       local_losses=losses.tolist(), local_grad_norms=norms.tolist(),
                       tokens_per_second=log_tokens / (now - log_start))
            append_metric(metrics_path, row)
            print(json.dumps(row), flush=True)
            log_loss = 0.0
            log_tokens = 0
            log_start = time.monotonic()

    try:
        for _ in range(protocol["warmup"]):
            update()
        runner.synchronize()
        startup = time.monotonic() - start
        estimate_start = time.monotonic()
        for _ in range(20):
            update()
        runner.synchronize()
        step_time = (time.monotonic() - estimate_start) / 20
        # Use a fixed count in all scored windows, rounded to logging cadence.
        cadence = config.training.log_every
        steps = max(cadence, math.ceil(1.2 * protocol["minimum_window_seconds"] / step_time / cadence) * cadence)
        torch.cuda.reset_peak_memory_stats()
        windows = []
        while len(windows) < protocol["windows"]:
            runner.synchronize()
            began = time.monotonic()
            for _ in range(steps):
                update()
            runner.synchronize()
            seconds = time.monotonic() - began
            if seconds < protocol["minimum_window_seconds"]:
                # Real executed work stays reflected in counters; never score a short window.
                steps = math.ceil(steps * 1.2 * protocol["minimum_window_seconds"] / seconds / cadence) * cadence
                windows.clear()
                continue
            windows.append(dict(steps=steps, seconds=seconds,
                                tokens_per_second=steps * config.training.batch_tokens / seconds))
        return dict(status="ok", environment=metadata, startup_seconds=startup,
                    elapsed_seconds=time.monotonic() - start, windows=windows,
                    tokens_per_second=statistics.median(w["tokens_per_second"] for w in windows),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    steps=runner.step, cursor=runner.cursor, clipping_counts=clip_counts.tolist())
    finally:
        runner.close()


def numerical(config, protocol, output):
    import torch

    from tiny_llm.runtime import environment

    left = NativeRunner(config)
    right = NativeRunner(config)
    # Both constructors reset the same seed; assert this instead of assuming it.
    for (a_name, a), (b_name, b) in zip(left.model.named_parameters(), right.model.named_parameters(), strict=True):
        if a_name != b_name or not torch.equal(a, b):
            raise AssertionError("native/native initial states differ")
    records = []
    try:
        for index in range(protocol["local_updates"]):
            before = [
                {name: p.detach().clone() for name, p in runner.model.named_parameters()}
                for runner in (left, right)
            ]
            a = left.update(inspect=True)
            b = right.update(inspect=True)
            workers = []
            for worker, (x, y) in enumerate(zip(left.tensors(), right.tensors(), strict=True)):
                def updates(old, values, worker_index=worker):
                    return {name: p - (old[name][worker_index] if config.decentralized else old[name])
                            for name, p in values.items()}
                workers.append(dict(
                    worker=worker, parameters=tensor_metrics(x["parameters"], y["parameters"]),
                    gradients=tensor_metrics(x["gradients"], y["gradients"]),
                    moments=tensor_metrics(x["moments"], y["moments"]),
                    updates=tensor_metrics(updates(before[0], x["parameters"]),
                                           updates(before[1], y["parameters"])),
                    left_counters=x["counters"], right_counters=y["counters"],
                ))
                if any(step != index + 1 for step in x["counters"] + y["counters"]):
                    raise AssertionError("Adam counters did not advance once")
            if left.input_hashes != right.input_hashes:
                raise AssertionError("native/native input ordering differs")
            records.append(dict(step=index + 1, left_losses=a[2].tolist(), right_losses=b[2].tolist(),
                                left_norms=a[1].tolist(), right_norms=b[1].tolist(), workers=workers))
            write(output / "updates.json", records)
        return dict(status="ok", kind="native_native_independent_control", environment=environment(),
                    input_hashes=left.input_hashes, updates_file="updates.json",
                    updates_sha256=sha(output / "updates.json"), steps=len(records))
    finally:
        left.close()
        right.close()


def numerical_serialized(config, protocol, output):
    """Independent trajectories sharing one GPU executor, with states offloaded.

    Used only when simultaneously resident native controls exceed device memory.
    This compares replay nondeterminism under one compiled graph, whereas the
    ordinary control also compares independent compilation instances. Both are
    identified explicitly in results; neither is a candidate precision gate.
    """
    from tiny_llm.checkpoints import compact_cpu
    from tiny_llm.data import BufferedTokenLoader
    from tiny_llm.runtime import environment

    runner = NativeRunner(config)
    loaders = [runner.loader, BufferedTokenLoader(
        runner.cache, "train", runner.length, runner.boundaries[-1],
        config.data.shuffle_group_size, seed=config.runtime.seed, prefetch=config.data.prefetch,
    )]
    initial = runner.state(cpu=True)
    states = [initial, initial]
    del initial
    hashes = [[], []]
    records = []
    reference = reference_updates = reference_losses = reference_norms = None
    try:
        for index in range(protocol["local_updates"]):
            for side in (0, 1):
                runner.restore(states[side])
                runner.loader = loaders[side]
                runner.step, runner.cursor = index, index * runner.step_blocks
                runner.input_hashes = hashes[side]
                runner.gradients = None
                before = {name: p.detach().to("cpu", copy=True)
                          for name, p in runner.model.named_parameters()}
                _, norms, losses, _ = runner.update(inspect=True)
                tensors = runner.tensors()
                states[side] = runner.state(cpu=True)
                if side == 0:
                    reference = cast(list[dict[str, Any]], compact_cpu(tensors))
                    reference_losses, reference_norms = losses.tolist(), norms.tolist()
                    reference_updates = [
                        {name: p - (before[name][worker] if config.decentralized else before[name])
                         for name, p in values["parameters"].items()}
                        for worker, values in enumerate(reference)
                    ]
                else:
                    assert reference is not None and reference_updates is not None
                    workers = []
                    for worker, (a, b) in enumerate(zip(reference, tensors, strict=True)):
                        updates = {
                            name: p - (before[name][worker] if config.decentralized else before[name]).to(p.device)
                            for name, p in b["parameters"].items()
                        }
                        workers.append(dict(
                            worker=worker,
                            **{field: tensor_metrics(a[field], b[field])
                               for field in ("parameters", "gradients", "moments")},
                            updates=tensor_metrics(reference_updates[worker], updates),
                            left_counters=a["counters"], right_counters=b["counters"],
                        ))
                        if any(s != index + 1 for s in a["counters"] + b["counters"]):
                            raise AssertionError("serialized control counters differ")
                    if hashes[0] != hashes[1]:
                        raise AssertionError("serialized control input order differs")
                    records.append(dict(step=index + 1, left_losses=reference_losses,
                                        right_losses=losses.tolist(), left_norms=reference_norms,
                                        right_norms=norms.tolist(), workers=workers))
                    write(output / "updates.json", records)
                    reference = reference_updates = None
                del before, tensors
        return dict(status="ok", kind="native_native_serialized_control", environment=environment(),
                    input_hashes=hashes[0], updates_file="updates.json",
                    updates_sha256=sha(output / "updates.json"), steps=len(records))
    finally:
        for loader in loaders:
            loader.close()


def execute(root, task):
    manifest = verify(root)
    activate(manifest["native_source"])
    import torch

    from tiny_llm.config import Config, save_config
    from tiny_llm.runtime import environment

    kind = task["kind"]
    row = next(row for row in manifest["convergence" if kind == "convergence" else "cases"]
               if row["id"] == task["id"])
    output = root / "baseline" / kind / f"{row['id']}-r{task['repeat']}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise ValueError(f"refusing to overwrite a result: {output}")
    cfg = Config.model_validate(row["config"])
    cfg.runtime.output_dir = output / "training"
    result = dict(task=task, manifest_identity=manifest["identity"], config_identity=digest(row["config"]),
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                  slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"))
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
            raise RuntimeError("qualification requires an allocated GH200 sm_90 GPU")
        if "GH200" not in torch.cuda.get_device_name():
            raise RuntimeError("qualification requires GH200 hardware")
        save_config(cfg, output / "config.yaml")
        write(output / "environment.json", environment())
        if kind == "convergence":
            from tiny_llm.train import train
            trained = train(cfg)
            if trained["status"] != "complete":
                raise RuntimeError(f"training did not complete: {trained}")
            result.update(status="ok", training=trained)
        elif kind == "throughput":
            result.update(benchmark(cfg, manifest["protocol"], output))
        else:
            function = numerical_serialized if task.get("serialized") else numerical
            result.update(function(cfg, manifest["protocol"], output))
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write(output / "result.json", result)
        raise
    write(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--subindex", type=int)
    args = parser.parse_args()
    tasks = read(args.tasks)[args.index]
    if args.subindex is not None:
        execute(args.root, tasks[args.subindex])
    else:
        failures = []
        for index, task in enumerate(tasks):
            process = subprocess.run([
                sys.executable, str(Path(__file__).resolve()), "--root", str(args.root),
                "--tasks", str(args.tasks), "--index", str(args.index), "--subindex", str(index),
            ])
            if process.returncode:
                failures.append(task)
        if failures:
            raise RuntimeError(f"failed tasks: {failures}")


if __name__ == "__main__":
    main()
