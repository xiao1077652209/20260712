import torch
import pytest

from model import JACSCM, SSCJAM, make_lawd_adamw


def test_shapes_and_optimizer_groups():
    attention = SSCJAM(32)
    x = torch.randn(4, 32, 57)
    assert attention(x).shape == x.shape
    model = JACSCM()
    assert model(torch.randn(4, 114)).shape == (4, 10)
    optimizer = make_lawd_adamw(model, 1e-3)
    assert [g["lr"] for g in optimizer.param_groups] == pytest.approx([1.2e-3, 1e-3, 0.9e-3])
    assert [g["weight_decay"] for g in optimizer.param_groups] == [1e-4, 1e-4, 5e-4]
