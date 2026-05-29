"""Checkpoint I/O for FR-UNet ensemble weights.

Training was done outside this repository. Use ``save_checkpoint`` when
saving new weights after external retraining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def plainize(value: Any) -> Any:
    """Recursively convert legacy objects (e.g. bunch.Bunch) to plain Python types."""
    if hasattr(value, "toDict"):
        return plainize(value.toDict())
    if isinstance(value, dict):
        return {k: plainize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plainize(v) for v in value]
    if isinstance(value, tuple):
        return tuple(plainize(v) for v in value)
    return value


def convert_legacy_checkpoint(ckpt: dict) -> dict:
    """Return a bunch-free copy of a legacy checkpoint with identical keys."""
    return {key: plainize(value) for key, value in ckpt.items()}


def save_checkpoint(
    path: str | Path,
    *,
    state_dict: dict,
    arch: str = "FR_UNet",
    epoch: int | None = None,
    optimizer: dict | None = None,
    config: dict | Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a checkpoint without bunch-dependent objects.

    Preserves the legacy key layout when ``optimizer`` and ``config`` are given.
    All values are plainized before saving so ``torch.load`` works without
    legacy dependencies.

    Args:
        path: Output ``.pth`` path.
        state_dict: Model weights from ``model.state_dict()``.
        arch: Model architecture name.
        epoch: Optional training epoch.
        optimizer: Optional optimizer state dict.
        config: Optional training config (dict or legacy bunch-like object).
        extra: Optional additional plain fields.
    """
    checkpoint: dict[str, Any] = {
        "arch": arch,
        "epoch": epoch,
        "state_dict": state_dict,
    }
    if optimizer is not None:
        checkpoint["optimizer"] = plainize(optimizer)
    if config is not None:
        checkpoint["config"] = plainize(config)
    if extra:
        checkpoint.update(plainize(extra))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict:
    """Load a checkpoint saved by ``save_checkpoint`` or ``get_ensemble``."""
    return torch.load(path, map_location=device, weights_only=False)
