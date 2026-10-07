# pyright: reportIncompatibleMethodOverride=false
"""One valid-token mean per independent worker, with a segmented CE backward."""

import torch

from .kernels import _ce_forward, jit, tl, triton


@jit
def _backward(X, Y, LSE, COUNTS, DOUT, DX, V: tl.constexpr, ROWS: tl.constexpr,
              BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    worker = row // ROWS
    j = tl.arange(0, BLOCK)
    x = tl.load(X + row * V + j, j < V, other=-float("inf")).to(tl.float32)
    target = tl.load(Y + row)
    p = tl.exp(x - tl.load(LSE + row))
    grad = (p - (j == target).to(tl.float32)) * tl.load(DOUT + worker) / tl.load(COUNTS + worker)
    grad = tl.where(target == -100, 0.0, grad)
    tl.store(DX + row * V + j, grad, j < V)


class _LocalCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, valid_mean):
        n, vocab = logits.shape[0], logits.shape[-1]
        rows = targets.numel()
        losses = torch.empty(rows, device=logits.device, dtype=torch.float32)
        lse = torch.empty_like(losses)
        _ce_forward[(rows,)](logits, targets, losses, lse, vocab,
                            triton.next_power_of_2(vocab), num_warps=32)
        counts = ((targets.reshape(n, -1) != -100).sum(1).clamp_min(1).float() if valid_mean else
                  torch.full((n,), rows // n, device=logits.device, dtype=torch.float32))
        ctx.save_for_backward(logits, targets, lse, counts)
        return losses.view(n, -1).sum(1) / counts

    @staticmethod
    def backward(ctx, dout):
        logits, targets, lse, counts = ctx.saved_tensors
        dx = torch.empty_like(logits)
        vocab = logits.shape[-1]
        _backward[(targets.numel(),)](
            logits, targets, lse, counts, dout.contiguous(), dx, vocab,
            targets.numel() // logits.shape[0], triton.next_power_of_2(vocab), num_warps=32,
        )
        return dx, None, None


def local_cross_entropy(logits, targets, *, valid_mean=True):
    if logits.ndim != 4 or logits.shape[:-1] != targets.shape:
        raise ValueError("expected [workers,batch,context,vocabulary] logits and matching targets")
    if not logits.is_contiguous() or not targets.is_contiguous():
        raise ValueError("GH200 cross entropy requires contiguous logits and targets")
    return _LocalCE.apply(logits, targets, valid_mean)
