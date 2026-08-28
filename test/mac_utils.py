"""Mac / device helpers for bond-deal training scripts."""

from __future__ import annotations

import torch


def training_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def apply_mac_training_defaults(train_cfg) -> None:
    """Disable mixed precision and heavy dataloader opts on Mac CPU/MPS."""
    device = training_device_name()
    if device in ("cpu", "mps"):
        train_cfg.fp16 = False
        train_cfg.bf16 = False
        train_cfg.num_workers = 0
        train_cfg.pin_memory = False
    print(f"[device] training on: {device}")
