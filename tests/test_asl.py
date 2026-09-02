import pytest
import torch

from baseline.asl import AsymmetricLoss


def test_asymmetric_loss_is_finite_and_differentiable():
    logits = torch.tensor([[0.2, -0.3]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])
    loss = AsymmetricLoss()(logits, targets)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_asymmetric_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="matching shapes"):
        AsymmetricLoss()(torch.zeros(1, 2), torch.zeros(1, 1))
