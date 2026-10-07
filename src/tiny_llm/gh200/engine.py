"""Complete-update CUDA graphs, shared by production training and comparisons."""

from __future__ import annotations

import time

import torch

from tiny_llm.checkpoints import compact_cpu
from tiny_llm.config import Config
from tiny_llm.runtime import preserve_rng

from .loss import local_cross_entropy
from .model import GH200Model
from .optimizer import ArenaAdamW


class GH200Update:
    def __init__(self, config: Config, device: torch.device):
        runtime = config.runtime
        if (device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0)
                or "GH200" not in torch.cuda.get_device_name(device)):
            raise ValueError("GH200 training requires an NVIDIA GH200 GPU")
        if (runtime.deterministic or not runtime.amp or runtime.attention_backend != "sdpa"
                or not runtime.compile):
            raise ValueError("GH200 training requires compiled BF16 SDPA and nondeterministic execution")
        geometry = (config.model.width, config.model.heads, config.model.ffn_width)
        if geometry not in ((320, 5, 896), (512, 8, 1408), (640, 10, 1792)):
            raise ValueError("GH200 training supports the 20M, 50M and 90M model widths")
        self.config, self.device = config, device
        self.workers = config.decentralized.num_models if config.decentralized else 1
        self.model = GH200Model(config.model, self.workers, device)
        self.optimizer = ArenaAdamW(self.model, config)
        self.blocks = config.training.batch_tokens // config.model.context_length
        self.length = config.model.context_length
        shape = (self.blocks, self.length)
        self.inputs = torch.zeros(shape, dtype=torch.int64, device=device)
        self.targets = torch.zeros_like(self.inputs)
        self.graph = None
        self.preparation_seconds = 0.

        def loss(x, y):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return local_cross_entropy(self.model(x), y,
                                           valid_mean=config.decentralized is not None)

        self.compute_loss = torch.compile(loss, dynamic=False,
                                          options={"triton.cudagraphs": False})

    def _update(self):
        self.optimizer.zero_grad()
        if self.config.decentralized:
            local = self.blocks // self.workers
            x, y = (tensor.view(local, self.workers, self.length).transpose(0, 1).contiguous()
                    for tensor in (self.inputs, self.targets))
            means = self.compute_loss(x, y)
            means.sum().backward()
        else:
            means = torch.zeros(1, device=self.device)
            for offset in range(0, self.blocks, self.config.training.micro_batch_size):
                count = min(self.config.training.micro_batch_size, self.blocks - offset)
                x, y = (tensor[offset:offset + count].unsqueeze(0)
                        for tensor in (self.inputs, self.targets))
                weighted = self.compute_loss(x, y) * (count / self.blocks)
                weighted.sum().backward()
                means.add_(weighted.detach())
        self.optimizer.step(means.detach())

    def _persistent(self):
        optimizer = self.optimizer
        return dict(parameters=self.model.parameter_storage,
                    first_moment=optimizer.first_moment_storage,
                    second_moment=optimizer.second_moment_storage,
                    lr=optimizer.lr, gamma=optimizer.gamma, finite=optimizer.finite,
                    completed=optimizer.completed, clips=optimizer.clip_counts,
                    loss_sum=optimizer.loss_sum, losses=optimizer.losses, norms=optimizer.norms)

    def prepare(self):
        """Warm/capture without consuming data or changing any committed state/RNG."""
        if self.graph is not None:
            return
        start = time.monotonic()
        saved = compact_cpu(self._persistent())
        with preserve_rng():
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            try:
                with torch.cuda.stream(stream):
                    self.optimizer.lr.zero_()
                    for _ in range(3):
                        self._update()
                    self.optimizer.inspect()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        self._update()
                    self.optimizer.finalize_capture()
                torch.cuda.current_stream(self.device).wait_stream(stream)
                self.graph = graph
            finally:
                stream.synchronize()
                for name, tensor in self._persistent().items():
                    tensor.copy_(saved[name])
                torch.cuda.synchronize(self.device)
        self.preparation_seconds = time.monotonic() - start

    def execute(self, next_batch, lr: float, gamma: float = 1.):
        """Transfer the exact production batches, then enqueue one complete update."""
        if self.graph is None:
            raise RuntimeError("prepare the complete update before execution")
        if self.config.decentralized:
            x, y = next_batch(self.blocks, self.device)
            self.inputs.copy_(x)
            self.targets.copy_(y)
        else:
            for offset in range(0, self.blocks, self.config.training.micro_batch_size):
                count = min(self.config.training.micro_batch_size, self.blocks - offset)
                x, y = next_batch(count, self.device)
                self.inputs[offset:offset + count].copy_(x)
                self.targets[offset:offset + count].copy_(y)
        self.optimizer.lr.fill_(lr)
        self.optimizer.gamma.fill_(gamma)
        self.graph.replay()

    def inspect(self, expected_step=None):
        return self.optimizer.inspect(expected_step)

    def device_metrics(self):
        return dict(loss_sum=self.optimizer.loss_sum, local_losses=self.optimizer.losses,
                    local_grad_norms=self.optimizer.norms, clip_counts=self.optimizer.clip_counts,
                    completed=self.optimizer.completed)

    def reset_window(self):
        self.optimizer.loss_sum.zero_()

    def reset_epoch(self):
        self.optimizer.clip_counts.zero_()

    def canonical_state(self):
        self.inspect()
        return dict(model=self.model.packed_state_dict() if self.config.decentralized
                    else self.model.local_state_dict(0), optimizer=self.optimizer.state_dict())

    def load_canonical_state(self, state):
        if self.config.decentralized:
            self.model.load_packed_state_dict(state["model"])
        else:
            self.model.load_local_state_dict(0, state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
