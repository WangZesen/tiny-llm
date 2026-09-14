from typing import cast

import pytest
import torch
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

from tiny_llm.config import Config, ModelConfig
from tiny_llm.model import Attention, Llama, token_losses


def test_causality(tiny_config: Config):
    model = Llama(tiny_config.model, "reference").double()
    x = torch.tensor([[1, 2, 3, 4]])
    y = torch.tensor([[1, 2, 8, 9]])
    torch.testing.assert_close(model(x)[:, :2], model(y)[:, :2], rtol=0, atol=0)


def test_fp64_backend_parity(tiny_config: Config):
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
        cast(torch.Tensor, sum((g * v).sum() for g, v in zip(gradient, directions, strict=True))),
        values,
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


def test_overfit_and_padding(tiny_config: Config):
    torch.manual_seed(1)
    model = Llama(tiny_config.model, "reference")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    x = torch.tensor([[1, 2, 3, 4]])
    y = torch.tensor([[2, 3, 4, -100]])
    loss = token_losses(model(x), y).sum() / 3
    initial = loss.item()
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


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("packed", [False, True])
def test_rotary_cache_dtype_device_and_state_dict(device, packed):
    from tiny_llm.model import rotary
    from tiny_llm.packed import PackedLlama

    config = ModelConfig(
        vocab_size=17, layers=1, width=64, heads=2, ffn_width=128, context_length=32
    )
    model = (PackedLlama(config, 4) if packed else Llama(config)).to(device)
    canonical = set(model.state_dict())
    assert not any("rope" in key for key in canonical)
    for dtype in (torch.float32, torch.bfloat16, torch.float32, torch.float64):
        model.to(dtype=dtype)
        attention = model.blocks[0].attention
        assert isinstance(attention, Attention)
        assert attention._rope_cos.dtype == torch.float32
        for length in (1, 7, 32):
            x = torch.randn(2, 2, length, 32, dtype=dtype, device=device, requires_grad=True)
            actual = attention._rotary(x)
            expected = rotary(x, config.rope_theta)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            a = torch.autograd.grad(actual.square().sum(), x, create_graph=True)[0]
            b = torch.autograd.grad(expected.square().sum(), x, create_graph=True)[0]
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert set(model.state_dict()) == canonical
    reference = (
        (PackedLlama(config, 4, "reference") if packed else Llama(config, "reference"))
        .to(device)
        .double()
    )
    reference.load_state_dict(model.state_dict(), strict=True)
    x = torch.ones((4, 2, 7) if packed else (2, 7), device=device, dtype=torch.long)
    with sdpa_kernel(SDPBackend.MATH):
        torch.testing.assert_close(model(x), reference(x), rtol=1e-10, atol=1e-12)
