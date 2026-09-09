"""Canonical training API.

This module intentionally re-exports the stable engine helpers so legacy
scripts can migrate without maintaining a second training loop.
"""
from .engine import (
    build_training_criterion,
    evaluate,
    fit,
    load_training_checkpoint,
    save_checkpoint,
    save_history,
    train_one_epoch,
)

__all__ = [
    "build_training_criterion", "evaluate", "fit", "load_training_checkpoint",
    "save_checkpoint", "save_history", "train_one_epoch",
]
