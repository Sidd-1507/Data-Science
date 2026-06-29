"""
utils.py
────────
Shared utilities: reproducibility, device setup, logging, parameter counting.
"""

import os
import random
import torch
import torch.nn as nn
import numpy as np


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Fix all random seeds for full reproducibility.

    Covers Python's ``random``, NumPy, PyTorch (CPU & CUDA), and cuDNN.

    Args:
        seed : Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device(verbose: bool = True) -> torch.device:
    """
    Return the best available device (CUDA → MPS → CPU) and optionally print info.

    Args:
        verbose : If True, print device info to stdout.

    Returns:
        ``torch.device`` object.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose:
        sep = "=" * 60
        print(sep)
        print(f"  PyTorch Version : {torch.__version__}")
        print(f"  Device Selected : {device}")
        if device.type == "cuda":
            print(f"  GPU Name        : {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version    : {torch.version.cuda}")
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU Memory      : {mem:.1f} GB")
        print(sep)

    return device


# ── Model Helpers ──────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Count total and trainable parameters in a model.

    Args:
        model : Any ``nn.Module``.

    Returns:
        Tuple ``(total_params, trainable_params)``.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(model: nn.Module, path: str, extra: dict | None = None) -> None:
    """
    Save model state dict (and optional metadata) to disk.

    Args:
        model : Model to save.
        path  : File path (e.g. ``"best_model.pth"``).
        extra : Optional dict of extra values to store (e.g. epoch, loss).
    """
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, path: str, device: torch.device | None = None) -> dict:
    """
    Load model weights from a checkpoint file.

    Args:
        model  : Model whose weights will be updated in-place.
        path   : Path to ``.pth`` checkpoint file.
        device : Target device. Defaults to CPU if None.

    Returns:
        Full checkpoint dict (may contain extra metadata).
    """
    map_loc  = device if device else torch.device("cpu")
    payload  = torch.load(path, map_location=map_loc)
    model.load_state_dict(payload["state_dict"])
    return payload


# ── Formatting ─────────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """Convert elapsed seconds to a human-readable string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"
