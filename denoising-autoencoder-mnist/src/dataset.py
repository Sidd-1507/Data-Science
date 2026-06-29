"""
dataset.py
──────────
MNIST data loading with train / validation / test split.

Usage
─────
  from src.dataset import get_dataloaders
  train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms


def get_dataloaders(
    batch_size:  int   = 128,
    valid_split: float = 0.15,
    data_root:   str   = "./data",
    num_workers: int   = 0,
    seed:        int   = 42,
    pin_memory:  bool  = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Download MNIST and return train, validation, and test DataLoaders.

    The training set (60 000 samples) is split into:
      - Train      : ``(1 - valid_split) × 60 000`` samples
      - Validation : ``valid_split       × 60 000`` samples
    The test set (10 000 samples) is kept separate and untouched.

    Args:
        batch_size  : Number of samples per batch.
        valid_split : Fraction of training data to reserve for validation.
        data_root   : Directory where MNIST is downloaded / cached.
        num_workers : Subprocesses for data loading (0 = main process).
        seed        : Random seed for reproducible split.
        pin_memory  : Pin CPU memory for faster GPU transfer.

    Returns:
        (train_loader, val_loader, test_loader)

    Example::

        >>> train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
        >>> images, labels = next(iter(train_loader))
        >>> print(images.shape)   # [128, 1, 28, 28]
    """
    # ── Transform: only ToTensor (noise applied dynamically in training loop) ──
    transform = transforms.Compose([
        transforms.ToTensor(),   # uint8 [H,W] → float32 [1,28,28] in [0,1]
    ])

    # ── Download ──────────────────────────────────────────────────────────────
    train_dataset = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    # ── Train / Validation Split ──────────────────────────────────────────────
    rng     = np.random.default_rng(seed)
    indices = np.arange(len(train_dataset))
    rng.shuffle(indices)

    split      = int(np.floor(valid_split * len(train_dataset)))
    val_idx    = indices[:split].tolist()
    train_idx  = indices[split:].tolist()

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler   = SubsetRandomSampler(val_idx)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, **loader_kwargs
    )
    val_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=val_sampler, **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )

    return train_loader, val_loader, test_loader


def dataset_info(train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader) -> None:
    """Print a summary of dataset sizes and pixel statistics."""
    train_n = len(train_loader.sampler)
    val_n   = len(val_loader.sampler)
    test_n  = len(test_loader.dataset)

    print(f"{'─'*40}")
    print(f"  Training   samples : {train_n:,}")
    print(f"  Validation samples : {val_n:,}")
    print(f"  Test       samples : {test_n:,}")
    print(f"  Train batches      : {len(train_loader)}")
    print(f"  Val   batches      : {len(val_loader)}")
    print(f"  Test  batches      : {len(test_loader)}")
    print(f"{'─'*40}")

    # Pixel statistics (sampled from one epoch of train loader)
    all_pixels = torch.cat([img for img, _ in train_loader], dim=0)
    print(f"  Pixel min  : {all_pixels.min().item():.4f}")
    print(f"  Pixel max  : {all_pixels.max().item():.4f}")
    print(f"  Pixel mean : {all_pixels.mean().item():.4f}")
    print(f"  Pixel std  : {all_pixels.std().item():.4f}")
    print(f"{'─'*40}")


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
    dataset_info(train_loader, val_loader, test_loader)
