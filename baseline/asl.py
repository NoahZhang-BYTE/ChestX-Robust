from __future__ import annotations

import torch
from torch import nn


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label classification (B3 only)."""

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05) -> None:
        super().__init__()
        if gamma_neg < 0 or gamma_pos < 0:
            raise ValueError("gamma values must be non-negative")
        if not 0 <= clip < 1:
            raise ValueError("clip must be in [0, 1)")
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must have matching shapes")
        if logits.ndim < 2:
            raise ValueError("logits and targets must have at least two dimensions")
        targets = targets.to(dtype=logits.dtype)
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        loss = targets * torch.log(xs_pos.clamp_min(torch.finfo(logits.dtype).tiny))
        loss += (1.0 - targets) * torch.log(xs_neg.clamp_min(torch.finfo(logits.dtype).tiny))
        if self.gamma_neg or self.gamma_pos:
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss *= torch.pow(1.0 - pt, one_sided_gamma)
        return -loss.mean()
