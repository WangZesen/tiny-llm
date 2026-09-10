"""Offline, full-epoch gradient-noise curvature measurements.

All derivatives use one fixed checkpoint, token-mean cross entropy, and reference
attention. Gradient microbatches come from training; curvature batches are independent.
"""

import hashlib
import json
import math
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from safetensors.torch import load_file
from torch.func import functional_call, grad, jvp
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import functional as F

from tiny_llm.checkpoints import RETENTION_FIELDS, load_training_checkpoint
from tiny_llm.config import Config, load_config
from tiny_llm.data import BufferedTokenLoader, TokenCache, fingerprint, training_boundaries
from tiny_llm.model import Llama, token_losses
from tiny_llm.packed import PackedLlama
from tiny_llm.runtime import atomic_json, environment, preserve_rng


@dataclass(frozen=True)
class AnalysisOptions:
    data_case: str = "both"
    noise_samples: int | str = 32
    random_samples: int = 32
    seed: int = 42
    device: str | None = None
    dtype: str = "float32"
    tf32: bool = True
    amp: bool = False
    compile_hvp: bool = True
    hvp_batch_size: int | None = None
    consensus: bool = True

    def __post_init__(self):
        if self.data_case not in ("seen", "unseen", "both"):
            raise ValueError("data_case must be seen, unseen, or both")
        if self.noise_samples != "all" and (
            not isinstance(self.noise_samples, int) or self.noise_samples <= 0
        ):
            raise ValueError("noise_samples must be positive or 'all'")
        if self.random_samples <= 0 or not 0 <= self.seed < 2**32:
            raise ValueError("random_samples must be positive and seed must be in [0, 2**32)")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if self.amp and self.dtype != "float32":
            raise ValueError("BF16 AMP requires FP32 parameters (--dtype float32)")
        if self.hvp_batch_size is not None and self.hvp_batch_size <= 0:
            raise ValueError("hvp_batch_size must be positive")


class AutocastModel(torch.nn.Module):
    def __init__(self, base, enabled=True):
        super().__init__()
        self.base, self.enabled = base, enabled

    def forward(self, x):
        # A cached cast could belong to another functional parameter/tangent
        # input. Traced higher derivatives must see every cast explicitly.
        with torch.autocast(
            x.device.type, dtype=torch.bfloat16, enabled=self.enabled, cache_enabled=False
        ):
            return self.base(x)

    @property
    def parameter_count(self):
        return self.base.parameter_count


class QuadraticKernel:
    """Compile Pearlmutter's jvp(grad(loss)) and the scalar reduction in one graph.

    Parameters and directions remain inputs, so one compilation serves all
    checkpoints with the same architecture. No dense Hessian is materialized.
    """

    def __init__(self, model, batch_size, *, compile=True, backend="inductor"):
        if batch_size <= 0:
            raise ValueError("HVP batch size must be positive")
        self.model, self.batch_size, self.compiled = model, batch_size, compile
        names = tuple(name for name, _ in model.named_parameters())

        def loss(parameters, x, y):
            weights = dict(zip(names, parameters, strict=True))
            return token_losses(functional_call(model, weights, (x,)), y).sum(dtype=torch.float64)

        def quadratic(parameters, direction, x, y):
            def objective(p):
                return loss(p, x, y)

            hvp = jvp(grad(objective), (parameters,), (direction,))[1]
            return sum((a.double() * b.double()).sum() for a, b in zip(hvp, direction, strict=True))

        self.eager = quadratic
        self.backend = backend
        self.function = None if compile else quadratic

    def __call__(self, x, y, direction):
        if x.shape[0] > self.batch_size:
            raise ValueError("batch exceeds configured HVP batch size")
        # Padding keeps the last batch in the same compiled graph. Ignored targets
        # contribute exactly zero to both directional derivatives.
        if self.compiled and x.shape[0] < self.batch_size:
            padding = self.batch_size - x.shape[0]
            x, y = F.pad(x, (0, 0, 0, padding)), F.pad(y, (0, 0, 0, padding), value=-100)
        parameters = tuple(p.detach() for p in self.model.parameters())
        direction = tuple(v.detach() for v in direction)
        if self.function is None:
            # Trace the AD transforms into ordinary ATen operations first. This
            # avoids nested-JVP fake-tensor aliasing failures in Dynamo and gives
            # Inductor the entire derivative graph, with no compiled backward to
            # differentiate later. Fake tracing avoids allocating epoch-sized
            # activations before the compiler can fuse and release intermediates.
            graph = make_fx(self.eager, tracing_mode="fake", _allow_non_fake_inputs=True)(
                parameters, direction, x, y
            )
            # Forward AD can create broadcast zero tangents whose fake strides
            # overlap. Materialize those fresh zeros before generated copy_ ops.
            for node in list(graph.graph.nodes):
                if node.target == torch.ops.aten._new_zeros_with_same_feature_meta.default:
                    with graph.graph.inserting_after(node):
                        contiguous = graph.graph.call_function(
                            torch.ops.aten.clone.default,
                            (node,),
                            {"memory_format": torch.contiguous_format},
                        )
                    node.replace_all_uses_with(contiguous)
                    contiguous.args = (node,)
            graph.graph.eliminate_dead_code()
            graph.recompile()
            self.function = torch.compile(
                graph, fullgraph=True, dynamic=False, backend=self.backend
            )
        return self.function(parameters, direction, x, y).detach()


@contextmanager
def analysis_runtime(config: Config, options: AnalysisOptions):
    """Do not invoke training's legacy matmul-precision/AMP setup."""
    device = torch.device(options.device or config.runtime.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("analysis supports CPU and CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --device cpu")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    previous = (
        torch.backends.fp32_precision,
        torch.backends.cuda.matmul.fp32_precision,
        torch.get_num_threads(),
    )
    previous_device = torch.cuda.current_device() if device.type == "cuda" else None
    try:
        with preserve_rng():
            torch.set_num_threads(config.runtime.cpu_threads)
            if device.type == "cuda":
                torch.cuda.set_device(device)
                if options.amp and not torch.cuda.is_bf16_supported():
                    raise ValueError("BF16 AMP is unsupported on this GPU")
            supported = device.type == "cuda" and torch.cuda.is_tf32_supported()
            effective = options.tf32 and options.dtype == "float32" and supported
            torch.backends.fp32_precision = "ieee"
            torch.backends.cuda.matmul.fp32_precision = "tf32" if effective else "ieee"
            torch.manual_seed(options.seed)
            with torch.autocast(device.type, enabled=False):
                yield (
                    device,
                    dict(
                        dtype=options.dtype,
                        tf32_requested=options.tf32,
                        tf32_supported=bool(supported),
                        tf32_effective=bool(effective),
                        matmul_precision=torch.backends.cuda.matmul.fp32_precision,
                        attention_backend="reference",
                        amp=options.amp,
                        amp_dtype="bfloat16" if options.amp else None,
                        compiled=options.compile_hvp and device.type == "cuda",
                    ),
                )
    finally:
        torch.backends.fp32_precision = previous[0]
        torch.backends.cuda.matmul.fp32_precision = previous[1]
        torch.set_num_threads(previous[2])
        if previous_device is not None:
            torch.cuda.set_device(previous_device)


class _CacheRange:
    """A read-only block-aligned view; the training loader itself is unchanged."""

    def __init__(self, cache, start, blocks, length):
        self.cache, self.start, self.count = cache, start * length, blocks * length + 1
        self.manifest = cache.manifest
        self.offsets = {"train": [0, self.count]}

    def blocks(self, split, length, partial=False):
        assert split == "train"
        return (self.count - 1) // length

    def read_into(self, split, start, stop, destination):
        if not 0 <= start < stop <= self.count:
            raise ValueError("read outside analysis range")
        self.cache.read_into(split, self.start + start, self.start + stop, destination)


@dataclass(frozen=True)
class NoiseBatch:
    cursor: int
    blocks: int
    worker: int = 0
    workers: int = 1

    @property
    def local_blocks(self):
        return self.blocks // self.workers


class EpochData:
    def __init__(self, cache: TokenCache, config: Config, case: str):
        self.cache, self.config, self.case = cache, config, case
        length = config.model.context_length
        boundaries = training_boundaries(config, config.model.parameter_count)
        self.training_blocks = boundaries[-1]
        self.blocks = boundaries[0]
        self.start = 0
        self.seed = config.runtime.seed
        if case == "unseen":
            nominal_tokens = (
                config.training.epoch_tokens
                or config.model.parameter_count * config.training.epoch_tokens_per_parameter
            )
            self.blocks = math.ceil(nominal_tokens / length)
            self.start = self.training_blocks
            self.seed = int(np.random.SeedSequence([config.runtime.seed, 2]).generate_state(1)[0])
        elif case != "seen":
            raise ValueError("unknown data case")
        required = self.training_blocks if case == "seen" else self.start + self.blocks
        if required > cache.blocks("train", length):
            raise ValueError(
                f"cache too small for {case}: need {required * length + 1} tokens. "
                "Prepare a new cache with tiny-llm prepare --config RUN/resolved.yaml "
                f"--set data.cache_dir=NEW_CACHE --set data.prepare_train_tokens={required * length}"
            )
        workers = config.decentralized.num_models if config.decentralized else 1
        if self.blocks % workers:
            raise ValueError("analysis epoch must contain an equal number of blocks per worker")
        self.samples = []
        batch_blocks = config.training.batch_tokens // length
        for cursor in range(0, self.blocks, batch_blocks):
            count = min(batch_blocks, self.blocks - cursor)
            self.samples.extend(NoiseBatch(cursor, count, i, workers) for i in range(workers))

    @property
    def tokens(self):
        return self.blocks * self.config.model.context_length

    def loader(self, cursor=0):
        cfg = self.config
        cache = self.cache
        budget = self.training_blocks
        if self.case == "unseen":
            cache = _CacheRange(cache, self.start, self.blocks, cfg.model.context_length)
            budget = self.blocks
        return BufferedTokenLoader(
            cache,
            "train",
            cfg.model.context_length,
            budget,
            cfg.data.buffer_size_mib,
            seed=self.seed,
            prefetch=cfg.data.prefetch,
            cursor=cursor,
        )

    def batches(self, device, sample: NoiseBatch | None = None, *, batch_size=None):
        if sample is not None and batch_size is not None:
            raise ValueError("noise samples must retain training microbatches")
        micro = batch_size or self.config.training.micro_batch_size
        if sample is None:
            with self.loader() as loader:
                for offset in range(0, self.blocks, micro):
                    yield loader.next_batch(min(micro, self.blocks - offset), device)
        else:
            with self.loader(sample.cursor) as loader:
                # Fetch interleaved rows in groups that produce at most one local microbatch.
                for offset in range(0, sample.local_blocks, micro):
                    count = min(micro, sample.local_blocks - offset)
                    x, y = loader.next_batch(count * sample.workers, device)
                    yield x[sample.worker :: sample.workers], y[sample.worker :: sample.workers]

    def metadata(self):
        with self.loader() as loader:
            return dict(
                case=self.case,
                loader=loader.identity,
                physical_start_block=self.start,
                blocks=self.blocks,
                tokens=self.tokens,
                available_noise_batches=len(self.samples),
                configured_total_epochs=len(
                    training_boundaries(self.config, self.config.model.parameter_count)
                ),
            )


def tensor_dot(left, right):
    return sum((a.double() * b.double()).sum() for a, b in zip(left, right, strict=True))


def gradient(model, batches, tokens):
    parameters = tuple(model.parameters())
    result = tuple(torch.zeros_like(p) for p in parameters)
    count = 0
    for x, y in batches:
        count += int((y != -100).sum().item())
        loss = token_losses(model(x), y).sum() / tokens
        values = torch.autograd.grad(loss, parameters)
        with torch.no_grad():
            for total, value in zip(result, values, strict=True):
                total.add_(value)
        del loss, values
    if count != tokens:
        raise ValueError(f"gradient token count mismatch: {count} != {tokens}")
    if not all(torch.isfinite(g).all().item() for g in result):
        raise FloatingPointError("nonfinite analysis gradient")
    return result


def hessian_quadratic(model, batches, tokens, direction, *, kernel=None):
    parameters = tuple(model.parameters())
    direction = tuple(v.detach() for v in direction)
    total = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    count = 0
    for index, (x, y) in enumerate(batches):
        count += int((y != -100).sum().item())
        if kernel is None:
            loss = token_losses(model(x), y).sum() / tokens
            first = torch.autograd.grad(loss, parameters, create_graph=True)
            directional = sum(
                (g * v).sum(dtype=torch.float64) for g, v in zip(first, direction, strict=True)
            )
            hvp = torch.autograd.grad(directional, parameters)
            total += tensor_dot(hvp, direction).detach()
            del loss, first, directional, hvp
        else:
            total += kernel(x, y, direction) / tokens
        if (index + 1) % 50 == 0:
            logger.info("HVP progress: {}/{} tokens", count, tokens)
    if count != tokens:
        raise ValueError(f"HVP token count mismatch: {count} != {tokens}")
    value = total.item()
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite Hessian quadratic form")
    return value


def rademacher(parameters, seed, index):
    generator = torch.Generator(device=parameters[0].device)
    # Each vector has its own seed, independent of model initialization and data iteration.
    generator.manual_seed(int(np.random.SeedSequence([seed, 3, index]).generate_state(1)[0]))
    # Draw in a fixed integer dtype: CUDA Bernoulli consumes a different RNG
    # stream for float32 and float64, which would confound precision comparisons.
    return tuple(
        torch.randint(0, 2, p.shape, device=p.device, dtype=torch.int8, generator=generator)
        .to(p.dtype)
        .mul_(2)
        .sub_(1)
        for p in parameters
    )


def summarize(noise, random, dimension, mean_norm):
    q = math.fsum(row["quadratic"] for row in noise)
    squared = math.fsum(row["noise_norm_squared"] for row in noise)
    return dict(
        noise_samples=len(noise),
        random_samples=len(random),
        dimension=dimension,
        noise_alignment=q / len(noise),
        normalized_noise_alignment=q / squared if squared else None,
        average_noise_norm=math.fsum(math.sqrt(row["noise_norm_squared"]) for row in noise)
        / len(noise),
        average_batch_gradient_norm=math.fsum(row["gradient_norm"] for row in noise) / len(noise),
        mean_gradient_norm=mean_norm,
        normalized_random_alignment=math.fsum(row["quadratic"] for row in random)
        / (dimension * len(random)),
    )


def consensus_statistics(rows):
    quadratic = math.fsum(row["quadratic"] for row in rows)
    squared = math.fsum(row["norm_squared"] for row in rows)
    return dict(
        consensus_workers=len(rows),
        consensus_alignment=quadratic / len(rows),
        normalized_consensus_alignment=quadratic / squared if squared else None,
        average_consensus_norm=math.fsum(row["norm"] for row in rows) / len(rows),
    )


def consensus_directions(model, workers):
    base = model.base if isinstance(model, AutocastModel) else model
    for state in workers:
        yield tuple(
            (state[name].to(device=p.device, dtype=p.dtype) - p).detach()
            for name, p in base.named_parameters()
        )


def measure_consensus(model, data, options, device, workers, kernel=None):
    rows = []
    directions = consensus_directions(model, workers)
    for index in range(len(workers)):
        start = time.monotonic()
        direction = next(directions)
        squared = tensor_dot(direction, direction).item()
        logger.info("{}: consensus direction {}/{}", data.case, index + 1, len(workers))
        hvp_start = time.monotonic()
        quadratic = (
            hessian_quadratic(
                model,
                data.batches(device, batch_size=options.hvp_batch_size),
                data.tokens,
                direction,
                kernel=kernel,
            )
            if squared
            else 0.0
        )
        rows.append(
            dict(
                worker=index,
                norm_squared=squared,
                norm=math.sqrt(squared),
                quadratic=quadratic,
                normalized_alignment=quadratic / squared if squared else None,
                hvp_seconds=time.monotonic() - hvp_start,
                total_seconds=time.monotonic() - start,
            )
        )
        del direction
    return rows


def validate_consensus(result, checkpoint):
    """Require complete, finite worker scalars and their derived statistics."""
    count = len(checkpoint.get("workers", []))
    rows = result.get("consensus", [])
    if len(rows) != count or [row["worker"] for row in rows] != list(range(count)):
        raise ValueError("incomplete consensus worker IDs/counts")
    if not count:
        return
    for row in rows:
        for field in ("norm_squared", "norm", "quadratic", "hvp_seconds", "total_seconds"):
            if not math.isfinite(row[field]) or (field != "quadratic" and row[field] < 0):
                raise ValueError("invalid consensus scalar")
        squared = row["norm_squared"]
        if not math.isclose(row["norm"], math.sqrt(squared), rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("inconsistent consensus norm")
        expected = row["quadratic"] / squared if squared else None
        if expected is not None and not math.isfinite(expected):
            raise ValueError("nonfinite normalized consensus alignment")
        if row["normalized_alignment"] != expected or (not squared and row["quadratic"] != 0):
            raise ValueError("inconsistent consensus alignment")
    if any(result["statistics"].get(k) != v for k, v in consensus_statistics(rows).items()):
        raise ValueError("inconsistent consensus statistics")


def measure(model, data: EpochData, options: AnalysisOptions, device, kernel=None, workers=()):
    started = time.monotonic()
    logger.info("{}: full-epoch mean gradient over {:,} tokens", data.case, data.tokens)
    mean = gradient(model, data.batches(device), data.tokens)
    mean_seconds = time.monotonic() - started
    mean_norm = math.sqrt(tensor_dot(mean, mean).item())
    count = (
        len(data.samples)
        if options.noise_samples == "all"
        else min(options.noise_samples, len(data.samples))
    )
    indices = sorted(
        np.random.default_rng(np.random.SeedSequence([options.seed, 4]))
        .choice(len(data.samples), count, replace=False)
        .tolist()
    )
    noise, random = [], []
    for i, index in enumerate(indices):
        sample = data.samples[index]
        logger.info("{}: noise direction {}/{} (batch {})", data.case, i + 1, count, index)
        start = time.monotonic()
        g = gradient(
            model,
            data.batches(device, sample),
            sample.local_blocks * data.config.model.context_length,
        )
        norm = math.sqrt(tensor_dot(g, g).item())
        v = tuple((a - b).detach() for a, b in zip(g, mean, strict=True))
        del g
        squared = tensor_dot(v, v).item()
        gradient_seconds = time.monotonic() - start
        start = time.monotonic()
        quadratic = hessian_quadratic(
            model,
            data.batches(device, batch_size=options.hvp_batch_size),
            data.tokens,
            v,
            kernel=kernel,
        )
        noise.append(
            dict(
                batch_index=index,
                batch=asdict(sample),
                quadratic=quadratic,
                noise_norm_squared=squared,
                gradient_norm=norm,
                gradient_seconds=gradient_seconds,
                hvp_seconds=time.monotonic() - start,
            )
        )
        del v
    del mean
    parameters = tuple(model.parameters())
    for i in range(options.random_samples):
        logger.info("{}: random direction {}/{}", data.case, i + 1, options.random_samples)
        start = time.monotonic()
        v = rademacher(parameters, options.seed, i)
        quadratic = hessian_quadratic(
            model,
            data.batches(device, batch_size=options.hvp_batch_size),
            data.tokens,
            v,
            kernel=kernel,
        )
        random.append(dict(index=i, quadratic=quadratic, hvp_seconds=time.monotonic() - start))
        del v
    consensus = measure_consensus(model, data, options, device, workers, kernel) if workers else []
    statistics = summarize(noise, random, model.parameter_count, mean_norm)
    if consensus:
        statistics.update(consensus_statistics(consensus))
    return dict(
        statistics=statistics,
        **(dict(consensus=consensus) if consensus else {}),
        noise=noise,
        random=random,
        timing=dict(mean_gradient_seconds=mean_seconds, total_seconds=time.monotonic() - started),
    )


def checkpoint_paths(run: Path, selected):
    run = run.resolve()
    if list(selected) == ["all"]:
        paths = sorted([*run.glob("*.safetensors"), *run.glob("*.pt")])
    else:
        paths = []
        for value in selected:
            path = Path(value)
            if not path.is_absolute():
                path = run / path
            paths.append(path.resolve())
    if not paths:
        raise ValueError("no checkpoints selected")
    for path in paths:
        if path.parent != run or path.suffix not in (".pt", ".safetensors") or not path.is_file():
            raise ValueError(f"expected a root checkpoint in {run}: {path}")
    return sorted(set(paths))


def result_key(checkpoint):
    return checkpoint.get("result_key", checkpoint["weights_hash"])


def weights_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_checkpoint(config, path, *, consensus=True, worker_states=None):
    model = Llama(config.model, "reference")
    packed = None
    boundaries = training_boundaries(config, config.model.parameter_count)
    if path.suffix == ".safetensors":
        match = re.fullmatch(r"epoch-(\d+)\.safetensors", path.name)
        if match is None or not 1 <= int(match[1]) <= len(boundaries):
            raise ValueError("safetensors checkpoint must use epoch-NNN.safetensors naming")
        epoch = int(match[1])
        cursor = boundaries[epoch - 1]
        batch_blocks = config.training.batch_tokens // config.model.context_length
        previous = 0
        step = 0
        for boundary in boundaries[:epoch]:
            step += math.ceil((boundary - previous) / batch_blocks)
            previous = boundary
        model.load_state_dict(load_file(str(path)), strict=True)
    else:
        # Trusted local training artifacts, as in train/evaluate.
        state = load_training_checkpoint(path)
        saved = Config.model_validate(state["config"])
        if (
            saved.model != config.model
            or saved.training.model_dump(exclude=RETENTION_FIELDS)
            != config.training.model_dump(exclude=RETENTION_FIELDS)
            or saved.decentralized != config.decentralized
        ):
            raise ValueError(f"checkpoint recipe does not match resolved.yaml: {path}")
        if "parameter_storage" in state["model"]:
            if config.decentralized is None:
                raise ValueError("packed checkpoint needs decentralized configuration")
            packed = PackedLlama(config.model, config.decentralized.num_models, "reference")
            packed.load_packed_state_dict(state["model"])
            packed.copy_average_to(model)
        else:
            model.load_state_dict(state["model"], strict=True)
        cursor, step, epoch = state["cursor"], state["step"], state["completed_epochs"]
    info = dict(
        path=str(path),
        name=path.name,
        weights_hash=weights_hash(model.state_dict()),
        tokens=cursor * config.model.context_length,
        step=step,
        epoch=epoch,
    )
    if consensus and config.decentralized is not None:
        parameters = dict(model.named_parameters())
        workers, provenance = [], []
        for index in range(config.decentralized.num_models):
            source = path if path.suffix == ".pt" else path.parent / f"node-{index:03d}" / path.name
            if path.suffix == ".pt":
                if "worker_files" in state:
                    source = path.parent / state["worker_files"][index]["path"]
                if packed is None:
                    raise ValueError(
                        f"checkpoint lacks packed worker weights: {path}; use --no-consensus"
                    )
                local = packed.local_state_dict(index)
            else:
                if not source.is_file():
                    raise ValueError(f"missing worker checkpoint: {source}; use --no-consensus")
                local = load_file(str(source))
            if set(local) != set(parameters) or any(
                local[name].shape != p.shape
                or not local[name].is_floating_point()
                or not torch.isfinite(local[name]).all()
                or not torch.isfinite(p).all()
                for name, p in parameters.items()
            ):
                raise ValueError(f"invalid worker parameters: {source} (worker {index})")
            workers.append(local)
            provenance.append(
                dict(worker=index, path=str(source), weights_hash=weights_hash(local))
            )
        for name, p in parameters.items():
            average = torch.stack([local[name].float() for local in workers]).mean(0)
            if not torch.allclose(p, average, rtol=1e-5, atol=1e-7):
                raise ValueError(f"root weights differ from worker average: {path} ({name})")
        info["workers"] = provenance
        info["result_key"] = fingerprint(
            dict(
                weights_hash=info["weights_hash"],
                workers=[row["weights_hash"] for row in provenance],
            )
        )
        if worker_states is not None:
            worker_states.extend(workers)
    return model.eval(), info


def analyze(
    run: Path,
    selected,
    output: Path,
    options: AnalysisOptions,
    *,
    plots=True,
    expected_checkpoints=None,
):
    import fcntl

    run, output = run.resolve(), output.resolve()
    config = load_config(run / "resolved.yaml")
    paths = checkpoint_paths(run, selected)
    cache = TokenCache(config.data.cache_dir)
    cache.validate_config(config)
    cache.verify()
    cases = ("seen", "unseen") if options.data_case == "both" else (options.data_case,)
    datasets = [EpochData(cache, config, case) for case in cases]
    # Validate every selected worker file and submission assignment before CUDA work.
    checkpoints = []
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config.runtime.cpu_threads)
        with preserve_rng():
            for path in paths:
                loaded, checkpoint = load_checkpoint(config, path, consensus=options.consensus)
                checkpoints.append(checkpoint)
                del loaded
    finally:
        torch.set_num_threads(previous_threads)
    if expected_checkpoints is not None and sorted(checkpoints, key=lambda r: r["path"]) != sorted(
        expected_checkpoints, key=lambda r: r["path"]
    ):
        raise ValueError("checkpoint hashes differ from submission assignments")
    output.mkdir(parents=True, exist_ok=True)
    with (
        (output / ".lock").open("a") as lock,
        analysis_runtime(config, options) as (device, policy),
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = environment()
        identity = fingerprint(
            dict(
                version=1,
                config=config.model_dump(mode="json"),
                options=asdict(options),
                policy=policy,
                data=[data.metadata() for data in datasets],
                source_hash=metadata["source_hash"],
                torch=metadata["versions"]["torch"],
                gpu=metadata["gpu"] if device.type == "cuda" else None,
            )
        )
        manifest_path = output / "manifest.json"
        manifest = dict(
            version=1,
            identity=identity,
            run=str(run),
            options=asdict(options),
            policy=policy,
            environment=metadata,
            checkpoints=[],
            micro_batch_size=config.training.micro_batch_size,
            hvp_batch_size=options.hvp_batch_size or config.training.micro_batch_size,
            effective_batch_tokens=config.training.batch_tokens,
            data=[data.metadata() for data in datasets],
        )
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old["identity"] != identity:
                raise ValueError(
                    "analysis settings/source differ from saved results; use a new output"
                )
            manifest = old
        atomic_json(manifest_path, manifest)
        base, model, kernel = None, None, None
        for path, expected in zip(paths, checkpoints, strict=True):
            workers = []
            loaded, checkpoint = load_checkpoint(
                config, path, consensus=options.consensus, worker_states=workers
            )
            if checkpoint != expected:
                raise ValueError(f"checkpoint changed after validation: {path}")
            if base is None:
                base = loaded.to(device=device, dtype=getattr(torch, options.dtype))
                model = base
                if options.amp:
                    model = AutocastModel(base).eval()
            else:
                base.load_state_dict(loaded.state_dict(), strict=True)
            del loaded
            entries = [entry for entry in manifest["checkpoints"] if entry["path"] != str(path)]
            entries.append(checkpoint)
            manifest["checkpoints"] = sorted(entries, key=lambda row: (row["tokens"], row["name"]))
            atomic_json(manifest_path, manifest)
            for data in datasets:
                destination = output / f"{result_key(checkpoint)}-{data.case}.json"
                if destination.exists():
                    saved = json.loads(destination.read_text())
                    if saved["identity"] != identity:
                        raise ValueError(f"incompatible saved result: {destination}")
                    if result_key(saved) != result_key(checkpoint):
                        raise ValueError(f"incompatible saved checkpoint: {destination}")
                    validate_consensus(saved, checkpoint)
                    logger.info("Reusing {} {} (verified identical weights)", path.name, data.case)
                    continue
                if kernel is None and policy["compiled"]:
                    batch_size = options.hvp_batch_size or config.training.micro_batch_size
                    kernel = QuadraticKernel(model, batch_size)
                    logger.info(
                        "Compiling complete Pearlmutter curvature graph, batch {}",
                        batch_size,
                    )
                    warmup = time.monotonic()
                    with data.loader() as loader:
                        x, y = loader.next_batch(min(batch_size, data.blocks), device)
                    v = rademacher(tuple(model.parameters()), options.seed, 0)
                    kernel(x, y, v).item()
                    kernel(x, y, v).item()
                    del x, y, v
                    manifest["hvp_warmup_seconds"] = time.monotonic() - warmup
                    atomic_json(manifest_path, manifest)
                logger.info(
                    "Analyzing {} {} | microbatch={} | TF32={}",
                    path.name,
                    data.case,
                    config.training.micro_batch_size,
                    policy["tf32_effective"],
                )
                result = measure(model, data, options, device, kernel, workers)
                validate_consensus(result, checkpoint)
                atomic_json(
                    destination,
                    dict(
                        identity=identity,
                        weights_hash=checkpoint["weights_hash"],
                        result_key=result_key(checkpoint),
                        case=data.case,
                        data=data.metadata(),
                        **result,
                    ),
                )
            if plots:
                from tiny_llm.analysis.plot import plot_analysis

                plot_analysis(output)
    return manifest
