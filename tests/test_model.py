import pytest
import torch
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

from tiny_llm.config import ModelConfig
from tiny_llm.model import Llama, token_losses


def test_causality(tiny_config):
    model = Llama(tiny_config.model, "reference").double()
    x = torch.tensor([[1, 2, 3, 4]])
    y = torch.tensor([[1, 2, 8, 9]])
    torch.testing.assert_close(model(x)[:, :2], model(y)[:, :2], rtol=0, atol=0)


def test_fp64_backend_parity(tiny_config):
    torch.manual_seed(42)
    ref = Llama(tiny_config.model, "reference").double()
    fast = Llama(tiny_config.model, "sdpa").double()
    fast.load_state_dict(ref.state_dict(), strict=True)
    x, y = torch.randint(17, (2, 4)), torch.randint(17, (2, 4))
    with sdpa_kernel(SDPBackend.MATH):
        a, b = ref(x), fast(x)
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
        la, lb = token_losses(a, y).mean(), token_losses(b, y).mean()
        torch.testing.assert_close(la, lb, rtol=1e-10, atol=1e-12)
        ga = torch.autograd.grad(la, tuple(ref.parameters()))
        gb = torch.autograd.grad(lb, tuple(fast.parameters()))
        for left, right in zip(ga, gb, strict=True):
            torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-11)


def test_second_derivatives_and_hvp():
    torch.manual_seed(7)
    model = Llama(
        ModelConfig(vocab_size=5, layers=1, width=4, heads=1, ffn_width=4, context_length=3),
        "reference",
    ).double()
    names, values = zip(*model.named_parameters(), strict=True)
    x, y = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])

    def loss(*parameters):
        logits = functional_call(model, dict(zip(names, parameters, strict=True)), (x,))
        return token_losses(logits, y).mean()

    assert torch.autograd.gradcheck(loss, values, fast_mode=True, atol=1e-5, rtol=1e-3)
    assert torch.autograd.gradgradcheck(loss, values, fast_mode=True, atol=1e-4, rtol=1e-3)
    directions = tuple(torch.randn_like(p) * 0.01 for p in values)
    gradient = torch.autograd.grad(loss(*values), values, create_graph=True)
    hvp = torch.autograd.grad(
        sum((g * v).sum() for g, v in zip(gradient, directions, strict=True)), values
    )
    epsilon = 1e-4
    plus = tuple(
        (p + epsilon * v).detach().requires_grad_() for p, v in zip(values, directions, strict=True)
    )
    minus = tuple(
        (p - epsilon * v).detach().requires_grad_() for p, v in zip(values, directions, strict=True)
    )
    gp = torch.autograd.grad(loss(*plus), plus)
    gm = torch.autograd.grad(loss(*minus), minus)
    for actual, left, right in zip(hvp, gp, gm, strict=True):
        torch.testing.assert_close(actual, (left - right) / (2 * epsilon), atol=1e-5, rtol=1e-3)


def test_overfit_and_padding(tiny_config):
    torch.manual_seed(1)
    model = Llama(tiny_config.model, "reference")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    x = torch.tensor([[1, 2, 3, 4]])
    y = torch.tensor([[2, 3, 4, -100]])
    initial = token_losses(model(x), y).sum().item() / 3
    for _ in range(50):
        optimizer.zero_grad()
        loss = token_losses(model(x), y).sum() / 3
        loss.backward()
        optimizer.step()
    assert loss.item() < initial * 0.1
    assert token_losses(model(x), y)[0, -1] == 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_backend_parity():
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(False)
    config = ModelConfig(
        vocab_size=128, layers=3, width=64, heads=2, ffn_width=176, context_length=32
    )
    ref, fast = Llama(config, "reference").cuda(), Llama(config, "sdpa").cuda()
    fast.load_state_dict(ref.state_dict(), strict=True)
    x, y = torch.randint(128, (2, 32), device="cuda"), torch.randint(128, (2, 32), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a, b = ref(x), fast(x)
        la, lb = token_losses(a, y).mean(), token_losses(b, y).mean()

    def relative(a, b):
        return (a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-12)

    assert relative(a, b) < 0.02
    torch.testing.assert_close(la, lb, rtol=0.002, atol=0.002)
    ga = torch.cat([p.flatten() for p in torch.autograd.grad(la, tuple(ref.parameters()))])
    gb = torch.cat([p.flatten() for p in torch.autograd.grad(lb, tuple(fast.parameters()))])
    assert relative(ga, gb) < 0.05
