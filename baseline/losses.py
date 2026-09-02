from __future__ import annotations

import torch
from torch import nn


def compute_capped_pos_weight(
    positive_counts: torch.Tensor,
    negative_counts: torch.Tensor,
    cap: float = 20.0,
) -> torch.Tensor:
    positive_counts = torch.as_tensor(positive_counts, dtype=torch.float32)
    negative_counts = torch.as_tensor(negative_counts, dtype=torch.float32)
    if positive_counts.shape != negative_counts.shape or positive_counts.ndim != 1:
        raise ValueError("positive_counts and negative_counts must be matching 1D tensors")
    if cap <= 0 or not torch.isfinite(torch.tensor(cap)):
        raise ValueError("cap must be positive and finite")
    if (positive_counts <= 0).any():
        raise ValueError("positive counts must be greater than zero")
    if (negative_counts < 0).any() or not torch.isfinite(negative_counts).all():
        raise ValueError("negative counts must be finite and non-negative")
    return torch.clamp(negative_counts / positive_counts, max=float(cap))


def compute_sqrt_capped_pos_weight(
    positive_counts: torch.Tensor,
    negative_counts: torch.Tensor,
    cap: float = 20.0,
) -> torch.Tensor:
    """Return min(sqrt(negative / positive), cap) for each label."""
    positive_counts = torch.as_tensor(positive_counts, dtype=torch.float32)
    negative_counts = torch.as_tensor(negative_counts, dtype=torch.float32)
    if positive_counts.shape != negative_counts.shape or positive_counts.ndim != 1:
        raise ValueError("positive_counts and negative_counts must be matching 1D tensors")
    if cap <= 0 or not torch.isfinite(torch.tensor(cap)):
        raise ValueError("cap must be positive and finite")
    if (positive_counts <= 0).any():
        raise ValueError("positive counts must be greater than zero")
    if (negative_counts < 0).any() or not torch.isfinite(negative_counts).all():
        raise ValueError("negative counts must be finite and non-negative")
    return torch.clamp(torch.sqrt(negative_counts / positive_counts), max=float(cap))


def build_loss(
    name: str = "bce",
    positive_counts: torch.Tensor | None = None,
    negative_counts: torch.Tensor | None = None,
    cap: float = 20.0,
) -> nn.BCEWithLogitsLoss:
    normalized = name.lower()
    if normalized == "bce":
        return nn.BCEWithLogitsLoss()
    if normalized not in {"bce_pos_weight", "bce_sqrt_pos_weight"}:
        raise ValueError(
            "Unknown loss name; choose 'bce', 'bce_pos_weight', or 'bce_sqrt_pos_weight'"
        )
    if positive_counts is None or negative_counts is None:
        raise ValueError(f"{normalized} requires positive and negative counts")
    if normalized == "bce_pos_weight":
        weights = compute_capped_pos_weight(positive_counts, negative_counts, cap)
    else:
        weights = compute_sqrt_capped_pos_weight(positive_counts, negative_counts, cap)
    return nn.BCEWithLogitsLoss(pos_weight=weights)
