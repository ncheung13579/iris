"""
Shared utility functions for the IRIS project.

This module provides seeding, device management, and checkpoint I/O
that all other modules and notebooks depend on.

Author: Nathan Cheung (ncheung3@my.yorku.ca)
York University | CSSD 2221 | Winter 2026
"""

import random
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


# ===========================================================================
# Project paths — all relative to the project root
# ===========================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"


# ===========================================================================
# Reproducibility
# ===========================================================================

def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.

    This must be called at the start of every notebook and script.
    It seeds Python's random module, NumPy, PyTorch (CPU and CUDA),
    and sets CUDA to deterministic mode where possible.

    Args:
        seed: The random seed. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic mode trades some speed for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ===========================================================================
# Device management
# ===========================================================================

def get_device() -> torch.device:
    """
    Get the best available device (CUDA GPU if available, otherwise CPU).

    Returns:
        torch.device: The device to use for computation.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU available)")
    return device


# ===========================================================================
# Checkpoint I/O
# ===========================================================================

def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    epoch: Optional[int] = None,
) -> None:
    """
    Save a training checkpoint with model state, optimizer state, and metadata.

    Args:
        path: File path to save to (should end in .pt).
        model: The model whose weights to save.
        optimizer: Optional optimizer whose state to save (for resuming training).
        config: Optional dict of hyperparameters and configuration.
        metrics: Optional dict of training metrics at time of save.
        epoch: Optional epoch number.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config or {},
        "metrics": metrics or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint and restore model/optimizer state.

    Args:
        path: File path to load from.
        model: The model to load weights into.
        optimizer: Optional optimizer to restore state into.
        device: Device to map tensors to. Defaults to CPU.

    Returns:
        Dict containing config, metrics, and epoch from the checkpoint.
    """
    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded: {path}")
    if "metrics" in checkpoint:
        for k, v in checkpoint["metrics"].items():
            print(f"  {k}: {v}")

    return {
        "config": checkpoint.get("config", {}),
        "metrics": checkpoint.get("metrics", {}),
        "epoch": checkpoint.get("epoch", None),
    }
