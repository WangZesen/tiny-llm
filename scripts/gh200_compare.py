"""Frozen-source native/candidate pairs and independent convergence qualification.

Each pair runs in isolated child processes in the same GH200 allocation. Raw
five-update native traces are temporary; reproducible initial states, inputs,
per-parameter diagnostics and source identities are retained permanently.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

from gh200_campaign import activate, command, digest, now, read, sha, verify, write

ROOT = Path(__file__).resolve().parents[1]


class CandidateRunner:
    def __init__(self, config):
        import torch

        from tiny_llm.data import BufferedTokenLoader, TokenCache, training_boundaries
        from tiny_llm.gh200.engine import GH200Update
        from tiny_llm.runtime import setup_runtime
        from tiny_llm.train import adaptive_consensus_schedule

        self.config, self.torch = config, torch
        self.device = setup_runtime(config)
        self.cache = TokenCache(config.data.cache_dir)
        self.cache.validate_config(config)
        self.engine = GH200Update(config, self.device)
        self.model, self.optimizer = self.engine.model, self.engine.optimizer
        self.n = self.engine.workers
        self.boundaries = training_boundaries(config, config.model.parameter_count)
        self.schedule = adaptive_consensus_schedule(config, self.boundaries)
        self.length = config.model.context_length
        self.step_blocks = config.training.batch_tokens // self.length
        self.loader = BufferedTokenLoader(
            self.cache, "train", self.length, self.boundaries[-1], config.data.shuffle_group_size,
            seed=config.runtime.seed, prefetch=config.data.prefetch,
        )
        self.cursor = self.step = 0
        self.input_hashes = []
        self.engine.prepare()

    def update(self, inspect=False):
        from tiny_llm.train import learning_rate

        lr = learning_rate(self.config, (self.cursor + self.step_blocks) * self.length,
                           self.boundaries[-1] * self.length)
        gamma = self.schedule.gamma(self.step, lr) if self.schedule else 1.

        def next_batch(count, device):
            x, y = self.loader.next_batch(count, device)
            if inspect:
                self.input_hashes.append(hashlib.sha256(
                    x.cpu().numpy().tobytes() + y.cpu().numpy().tobytes()).hexdigest())
            return x, y

        self.engine.execute(next_batch, lr, gamma)
        self.step += 1
        self.cursor += self.step_blocks
        if inspect:
            self.engine.inspect(self.step)
        return self.optimizer.loss_sum, self.optimizer.norms, self.optimizer.losses, lr

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def close(self):
        self.loader.close()

    def tensors(self):
        result = []
        for worker in range(self.n):
            fields = {name: {} for name in ("parameters", "gradients", "moments")}
            for entry in self.model.layout:
                fields["parameters"][entry.name] = self.model.get_parameter(entry.name)[worker]
                fields["gradients"][entry.name] = self.model.parameter_view(
                    self.optimizer.gradients, entry)[worker]
                for key, storage in (("exp_avg", self.optimizer.first_moment_storage),
                                     ("exp_avg_sq", self.optimizer.second_moment_storage)):
                    fields["moments"][entry.name + "/" + key] = self.model.parameter_view(storage, entry)[worker]
            result.append(dict(**fields, counters=[int(self.optimizer.completed.item())]
                               * len(self.model.layout)))
        return result


def freeze(root):
    manifest = verify(root)
    complete = read(root / "baseline-complete.json")
    if not read(root / "baseline-summary.json")["baseline_complete"]:
        raise ValueError("complete native baselines before freezing a candidate")
    destination = root / "candidates" / uuid.uuid4().hex[:12]
    destination.mkdir(parents=True)
    for name in ("src", "scripts", "tests", "configs"):
        shutil.copytree(ROOT / name, destination / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, destination / name)
    files = {str(p.relative_to(destination)): sha(p) for p in destination.rglob("*") if p.is_file()}
    info = dict(created=now(), source_identity=digest(files), files=files,
                manifest_identity=manifest["identity"], baseline_complete=complete,
                acceptance_contract_sha256=sha(root / "acceptance/contract.json"))
    write(destination / "candidate.json", info)
    print(destination)


def verify_candidate(root, candidate):
    manifest = verify(root)
    info = read(candidate / "candidate.json")
    if (info["manifest_identity"] != manifest["identity"]
            or info["acceptance_contract_sha256"] != sha(root / "acceptance/contract.json")
            or info["source_identity"] != digest(info["files"])):
        raise ValueError("candidate protocol identity mismatch")
    for name, expected in info["files"].items():
        if sha(candidate / name) != expected:
            raise ValueError(f"candidate source changed: {name}")
    return manifest, info


def submit(root, candidate, kind, ids=None, repeats=None):
    manifest, info = verify_candidate(root, candidate)
    rows = manifest["convergence" if kind == "convergence" else "cases"]
    selected = [row for row in rows if ids is None or row["id"] in ids]
    if not selected or (ids is not None and set(ids) != {r["id"] for r in selected}):
        raise ValueError("unknown or empty workload selection")
    repeats = repeats if repeats is not None else range(3) if kind == "performance" else [0]
    tasks = [dict(id=row["id"], repeat=repeat, kind=kind)
             for repeat in repeats for row in selected]
    with (candidate / ".submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        claimed = {(t["id"], t["repeat"], t["kind"])
                   for p in (candidate / "submissions").glob("*/receipt.json")
                   for t in read(p)["tasks"]}
        if any((t["id"], t["repeat"], t["kind"]) in claimed for t in tasks):
            raise ValueError("tasks already claimed; use a new candidate for changed implementations")
        destination = candidate / "submissions" / f"{kind}-{uuid.uuid4().hex[:8]}"
        destination.mkdir(parents=True)
        # Numerical traces use substantial temporary disk; limit each job to one
        # case and array concurrency to four. Different repeats get distinct jobs.
        chunk = 4 if kind == "performance" else 1
        jobs = []
        for repeat in repeats:
            local = [task for task in tasks if task["repeat"] == repeat]
            jobs.extend(local[i:i + chunk] for i in range(0, len(local), chunk))
        write(destination / "tasks.json", jobs)
        script = "\n".join([
            "#!/bin/bash", "set -euo pipefail", "cd " + shlex.quote(str(candidate)),
            "export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false",
            'pair_cache=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/gh200-pair.XXXXXXXX")',
            'export TORCHINDUCTOR_CACHE_DIR="$pair_cache/inductor" TRITON_CACHE_DIR="$pair_cache/triton"',
            "trap 'rm -rf -- \"$pair_cache\"' EXIT",
            shlex.join([str(ROOT / ".venv-aarch64/bin/python"), str(candidate / "scripts/gh200_compare.py"),
                        "worker", "--root", str(root), "--candidate", str(candidate),
                        "--tasks", str(destination / "tasks.json")]) + ' --index "$SLURM_ARRAY_TASK_ID"', "",
        ])
        (destination / "job.sh").write_text(script)
        receipt = dict(created=now(), status="submitting", tasks=tasks,
                       source_identity=info["source_identity"])
        write(destination / "receipt.json", receipt)
        try:
            job = command([
                "sbatch", "--parsable", "--account=naiss2026-3-205-gpu", "--partition=gpu",
                "--gpus=nvidia_gh200_120gb:1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G",
                "--time=01:00:00", f"--array=0-{len(jobs)-1}%4", f"--job-name=gh200-{kind}",
                f"--output={destination}/slurm-%A_%a.log", str(destination / "job.sh"),
            ], timeout=60).strip().split(";")[0]
            receipt.update(status="submitted", job=job)
        except BaseException as exc:
            receipt.update(status="uncertain", error=repr(exc))
            write(destination / "receipt.json", receipt)
            raise
        write(destination / "receipt.json", receipt)
        print(json.dumps(receipt))


def parameters(runner):
    from tiny_llm.checkpoints import compact_cpu
    from tiny_llm.packed import PackedLlama

    if isinstance(runner.model, PackedLlama):
        return [compact_cpu(runner.model.local_state_dict(i)) for i in range(runner.n)]
    return [compact_cpu(runner.model.state_dict())]


def fingerprint_parameters(values):
    h = hashlib.sha256()
    for worker in values:
        for name, value in worker.items():
            h.update(name.encode())
            h.update(str(tuple(value.shape)).encode())
            h.update(value.numpy().tobytes())
    return h.hexdigest()


def timed(runner, protocol, output, count, candidate):
    import torch

    from tiny_llm.runtime import append_metric, environment

    cfg = runner.config
    log_loss, log_tokens = 0., 0
    clips = torch.zeros(runner.n, device=runner.device, dtype=torch.int64)
    log_start = time.monotonic()

    def update():
        nonlocal log_loss, log_tokens, log_start
        value, norms, losses, lr = runner.update()
        maximum_norm = norms.max()
        if not candidate:
            log_loss += value.item()
            if cfg.optimizer.grad_clip is not None:
                clips.add_(norms > cfg.optimizer.grad_clip)
        log_tokens += cfg.training.batch_tokens
        if runner.step % cfg.training.log_every == 0 or runner.cursor in runner.boundaries:
            if candidate:
                runner.engine.inspect(runner.step)
                log_loss = runner.optimizer.loss_sum.item()
            runner.synchronize()
            now_ = time.monotonic()
            row = dict(event="train", step=runner.step, tokens=runner.cursor * runner.length,
                       loss=log_loss / log_tokens, lr=lr, grad_norm=maximum_norm.item(),
                       local_losses=losses.tolist(), local_grad_norms=norms.tolist(),
                       tokens_per_second=log_tokens / (now_ - log_start))
            append_metric(output / "metrics.jsonl", row)
            print(json.dumps(row), flush=True)
            log_loss, log_tokens = 0., 0
            if candidate:
                runner.engine.reset_window()
            log_start = time.monotonic()

    warming = time.monotonic()
    for _ in range(protocol["warmup"]):
        update()
    runner.synchronize()
    warmup_seconds = time.monotonic() - warming
    torch.cuda.reset_peak_memory_stats()
    windows = []
    for _ in range(protocol["windows"]):
        runner.synchronize()
        start = time.monotonic()
        for _ in range(count):
            update()
        runner.synchronize()
        seconds = time.monotonic() - start
        windows.append(dict(steps=count, seconds=seconds,
                            tokens_per_second=count * cfg.training.batch_tokens / seconds))
    if candidate:
        runner.engine.inspect(runner.step)
        clips = runner.optimizer.clip_counts
    return dict(status="ok", environment=environment(), windows=windows, steps=runner.step,
                warmup_seconds=warmup_seconds,
                training_seconds=sum(window["seconds"] for window in windows),
                cursor=runner.cursor, clipping_counts=clips.tolist(),
                tokens_per_second=statistics.median(row["tokens_per_second"] for row in windows),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def numerical(runner, protocol, output, trace, candidate, shared):
    import torch
    from gh200_worker import tensor_metrics

    from tiny_llm.checkpoints import compact_cpu
    from tiny_llm.runtime import atomic_checkpoint

    initial = parameters(runner)
    identity = fingerprint_parameters(initial)
    shared.mkdir(parents=True, exist_ok=True)
    saved = shared / (identity + ".pt")
    with (shared / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not saved.exists():
            atomic_checkpoint(saved, dict(parameters=initial, config=runner.config.model_dump(mode="json")))
    if candidate and identity != read(trace / "initial.json")["identity"]:
        raise AssertionError("native/candidate initial states differ")
    if not candidate:
        write(trace / "initial.json", dict(identity=identity, path=str(saved)))
    del initial
    original_next = runner.loader.next_batch
    inputs = []

    def next_batch(count, device):
        values = original_next(count, device)
        inputs.append(tuple(value.cpu() for value in values))
        return values

    runner.loader.next_batch = next_batch
    records, passed = [], True
    for index in range(protocol["local_updates"]):
        before = parameters(runner)
        _, norms, losses, lr = runner.update(inspect=True)
        values = compact_cpu(runner.tensors())
        for worker, fields in enumerate(values):
            fields["updates"] = {name: value - before[worker][name]
                                 for name, value in fields["parameters"].items()}
        del before
        state = dict(values=values, norms=norms.tolist(), losses=losses.tolist(), lr=lr,
                     hashes=list(runner.input_hashes), step=runner.step, cursor=runner.cursor)
        path = trace / f"step-{index + 1}.pt"
        if not candidate:
            atomic_checkpoint(path, state)
            atomic_checkpoint(output / f"inputs-{index + 1}.pt", inputs)
        else:
            reference = torch.load(path, map_location="cpu", weights_only=False)
            if any(state[key] != reference[key] for key in ("hashes", "step", "cursor", "lr")):
                raise AssertionError("native/candidate data order, schedule or counters differ")
            loss_ok = all(abs(a - b) <= protocol["loss_atol"] + protocol["loss_rtol"] * abs(a)
                          for a, b in zip(reference["losses"], state["losses"], strict=True))
            workers = []
            for worker, (left, right) in enumerate(zip(reference["values"], values, strict=True)):
                metrics = {field: tensor_metrics(left[field], right[field])
                           for field in ("parameters", "gradients", "moments", "updates")}
                ok = (metrics["parameters"]["relative_l2"] < protocol["parameter_relative_l2"]
                      and metrics["gradients"]["relative_l2"] < protocol["gradient_relative_l2"]
                      and metrics["updates"]["relative_l2"] < protocol["update_relative_l2"])
                if left["counters"] != right["counters"] or any(s != index + 1 for s in right["counters"]):
                    raise AssertionError("per-parameter Adam counters differ")
                passed &= ok
                workers.append(dict(worker=worker, passed=ok, **metrics))
            passed &= loss_ok
            records.append(dict(step=index + 1, loss_passed=loss_ok, workers=workers,
                                native_losses=reference["losses"], candidate_losses=state["losses"],
                                native_norms=reference["norms"], candidate_norms=state["norms"]))
            write(output / "updates.json", records)
            del reference
            path.unlink()
        inputs.clear()
        del values, state
    return dict(status="ok", gate="pass" if passed else "not_qualified", initial_identity=identity,
                initial_path=str(saved), input_hashes=runner.input_hashes, updates=records)


def child(args):
    manifest, info = verify_candidate(args.root, args.candidate)
    native = args.backend == "native"
    activate(manifest["native_source"] if native else args.candidate / "src")
    import torch
    from gh200_worker import NativeRunner

    from tiny_llm.config import Config, save_config
    from tiny_llm.runtime import environment

    rows = manifest["convergence" if args.kind == "convergence" else "cases"]
    row = next(row for row in rows if row["id"] == args.case)
    cfg = Config.model_validate(row["config"])
    if not native:
        cfg.runtime.training_backend = "gh200"
    cfg.runtime.output_dir = args.output / "training"
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(source_identity=info["source_identity"], manifest_identity=manifest["identity"],
                  config_identity=digest(row["config"]), backend=args.backend,
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    runner = None
    began = time.monotonic()
    try:
        if not torch.cuda.is_available() or "GH200" not in torch.cuda.get_device_name():
            raise RuntimeError("qualification requires GH200 hardware")
        save_config(cfg, args.output / "config.yaml")
        write(args.output / "environment.json", environment())
        if args.kind == "convergence":
            from tiny_llm.train import train

            trained = train(cfg)
            result.update(status="ok" if trained["status"] == "complete" else "failed", training=trained)
        else:
            runner = NativeRunner(cfg) if native else CandidateRunner(cfg)
            result["preparation_seconds"] = time.monotonic() - began
            if args.kind == "performance":
                result.update(timed(runner, manifest["protocol"], args.output, args.count, not native))
            else:
                result.update(numerical(runner, manifest["protocol"], args.output, args.trace,
                                        not native, args.root / "initial-states"))
                baseline = read(args.root / "baseline/numerical" / f"{row['id']}-r0/result.json")
                if result["input_hashes"] != baseline["input_hashes"]:
                    raise AssertionError("comparison inputs differ from the frozen numerical baseline")
                result["baseline_input_order_verified"] = True
    except BaseException as exc:
        result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if runner is not None:
            runner.close()
        result["elapsed_seconds"] = time.monotonic() - began
        if args.kind == "performance" and result.get("status") == "ok":
            result["startup_seconds"] = result["preparation_seconds"] + result["warmup_seconds"]
        write(args.output / "result.json", result)


def worker(args):
    manifest, info = verify_candidate(args.root, args.candidate)
    tasks = read(args.tasks)[args.index]
    failed = []
    for task_index, task in enumerate(tasks):
        output = args.candidate / "results" / task["kind"] / f"{task['id']}-r{task['repeat']}"
        output.mkdir(parents=True, exist_ok=True)
        if (output / "result.json").exists():
            raise ValueError("refusing to overwrite a comparison result")
        result = dict(task=task, source_identity=info["source_identity"],
                      manifest_identity=manifest["identity"], allocation=os.environ["SLURM_JOB_ID"])

        def run(backend, destination, count=0, trace=None, task=task):
            argv = [sys.executable, str(Path(__file__).resolve()), "child", "--root", str(args.root),
                    "--candidate", str(args.candidate), "--kind", task["kind"], "--case", task["id"],
                    "--backend", backend, "--output", str(destination), "--count", str(count)]
            if trace:
                argv.extend(["--trace", str(trace)])
            subprocess.run(argv, check=True)
            return read(destination / "result.json")

        try:
            if task["kind"] == "convergence":
                result.update(run("candidate", output / "candidate"))
            elif task["kind"] == "numerical":
                trace = output / "temporary-trace"
                trace.mkdir()
                try:
                    run("native", output / "native", trace=trace)
                    result.update(run("candidate", output / "candidate", trace=trace))
                finally:
                    # These large tensors can be regenerated from frozen source,
                    # stored initial states and the retained input batches.
                    shutil.rmtree(trace)
            else:
                baseline = [read(args.root / "baseline/throughput" / f"{task['id']}-r{r}/result.json")
                            for r in range(3)]
                count = 2 * max(w["steps"] for row in baseline for w in row["windows"])
                order = ["native", "candidate"] if (task_index + task["repeat"]) % 2 == 0 else ["candidate", "native"]
                pairs = None
                for attempt in range(3):
                    pairs = {backend: run(backend, output / f"attempt-{attempt}" / backend, count=count)
                             for backend in order}
                    if all(w["seconds"] >= manifest["protocol"]["minimum_window_seconds"]
                           for pair in pairs.values() for w in pair["windows"]):
                        break
                    count *= 2
                else:
                    raise RuntimeError("paired timing windows remain shorter than protocol minimum")
                assert pairs is not None
                result.update(status="ok", order=order, count=count, pairs=pairs,
                              speedup=pairs["candidate"]["tokens_per_second"] / pairs["native"]["tokens_per_second"])
        except BaseException as exc:
            result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            failed.append(task)
        write(output / "result.json", result)
    if failed:
        raise RuntimeError(f"failed comparison tasks: {failed}")


def collect(root, candidate):
    manifest, info = verify_candidate(root, candidate)
    spec = importlib.util.spec_from_file_location("frozen_acceptance", root / "acceptance/gh200_campaign.py")
    assert spec is not None and spec.loader is not None
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    failures, invalid, pairs, numerical_cases = [], [], [], []
    for row in manifest["cases"]:
        for kind, repeats in (("numerical", [0]), ("performance", range(3))):
            for repeat in repeats:
                path = candidate / "results" / kind / f"{row['id']}-r{repeat}/result.json"
                if not path.exists():
                    continue
                result = read(path)
                if result["status"] != "ok":
                    failures.append(str(path))
                    continue
                if (result["source_identity"] != info["source_identity"]
                        or result["manifest_identity"] != manifest["identity"]):
                    invalid.append(str(path))
                    continue
                if kind == "numerical":
                    if (result["gate"] != "pass" or len(result["updates"]) != manifest["protocol"]["local_updates"]
                            or not Path(result["initial_path"]).exists()):
                        failures.append(str(path))
                    else:
                        numerical_cases.append(row["id"])
                else:
                    a, b = result["pairs"]["native"], result["pairs"]["candidate"]
                    if (a["slurm_job_id"] != b["slurm_job_id"] or a["slurm_job_id"] != result["allocation"]
                            or a["steps"] != b["steps"] or a["cursor"] != b["cursor"]
                            or any(side["config_identity"] != digest(row["config"]) for side in (a, b))
                            or any(len(side["windows"]) != manifest["protocol"]["windows"] for side in (a, b))
                            or any(w["seconds"] < manifest["protocol"]["minimum_window_seconds"]
                                   for side in (a, b) for w in side["windows"])):
                        invalid.append(str(path))
                        continue
                    decentralized = row["config"]["decentralized"]
                    mode = f"packed{decentralized['num_models']}" if decentralized else "sync"
                    pairs.append(dict(id=row["id"], repeat=repeat, category=row["category"],
                                      mode=mode, allocation=result["allocation"],
                                      native=a["tokens_per_second"], candidate=b["tokens_per_second"]))
    convergence = {}
    for row in manifest["convergence"]:
        path = candidate / "results/convergence" / f"{row['id']}-r0/result.json"
        if not path.exists():
            continue
        result = read(path)
        if result["status"] != "ok":
            failures.append(str(path))
            continue
        baseline = read(root / "baseline/convergence" / f"{row['id']}-r0/result.json")
        a, b = baseline["training"], result["training"]
        if (a["tokens"] != b["tokens"] or a["step"] != b["step"] or a["epochs"] != b["epochs"]
                or a["final_validation"]["tokens"] != b["final_validation"]["tokens"]
                or result["source_identity"] != info["source_identity"]
                or result["config_identity"] != digest(row["config"])):
            invalid.append(str(path))
            continue
        key = row["recipe"]
        recipe = convergence.setdefault(key, dict(native={}, candidate={}))
        seed = row["config"]["runtime"]["seed"]
        recipe["native"][seed] = a["final_validation"]["loss"]
        recipe["candidate"][seed] = b["final_validation"]["loss"]
    summary = dict(source_identity=info["source_identity"], manifest_identity=manifest["identity"],
                   performance=gate.performance_gate(pairs, [row["id"] for row in manifest["cases"]]),
                   numerical_complete=len(numerical_cases) == len(manifest["cases"]),
                   numerical_passed=len(numerical_cases), performance_pairs=len(pairs),
                   convergence={key: gate.convergence_gate(**values) for key, values in convergence.items()},
                   failures=failures, invalid=invalid)
    # Interface tests are independent required evidence, reviewed before retirement.
    summary["measurement_gates_passed"] = (not failures and not invalid and summary["numerical_complete"]
                                           and summary["performance"]["status"] == "pass"
                                           and len(summary["convergence"]) == 5
                                           and all(row["status"] == "pass" for row in summary["convergence"].values()))
    write(candidate / "qualification-summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "submit", "worker", "child", "collect"])
    parser.add_argument("--root", type=Path, default=ROOT / "runs/gh200-integration-20261004")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--kind", choices=["performance", "numerical", "convergence"])
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--repeats", nargs="+", type=int)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--case")
    parser.add_argument("--backend", choices=["native", "candidate"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--count", type=int, default=0)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.candidate:
        args.candidate = args.candidate.resolve()
    if args.action == "freeze":
        freeze(args.root)
    elif args.action == "submit":
        submit(args.root, args.candidate, args.kind, args.ids, args.repeats)
    elif args.action == "worker":
        worker(args)
    elif args.action == "collect":
        collect(args.root, args.candidate)
    else:
        child(args)


if __name__ == "__main__":
    main()
