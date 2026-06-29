"""
train.py
────────
Training script for the Convolutional Denoising Autoencoder (CDAE).

Runs from the command line::

    python src/train.py

Or import and call ``train()`` programmatically.
"""

import time
import argparse

import numpy as np
import torch
import torch.nn as nn

from src.model   import DenoisingAutoencoder, count_parameters
from src.dataset import get_dataloaders
from src.noise   import add_gaussian_noise
from src.utils   import set_seed, get_device, save_checkpoint, format_time


# ── Default Hyperparameters ────────────────────────────────────────────────────

DEFAULTS = dict(
    seed           = 42,
    batch_size     = 128,
    valid_split    = 0.15,
    epochs         = 30,
    lr             = 1e-3,
    weight_decay   = 1e-5,
    noise_factor   = 0.4,
    lr_patience    = 5,
    lr_factor      = 0.5,
    early_stop_pat = 10,
    base_channels  = 32,
    checkpoint     = "best_denoising_ae.pth",
    num_workers    = 0,
)


# ── One Epoch ──────────────────────────────────────────────────────────────────

def train_epoch(
    model:        nn.Module,
    loader:       torch.utils.data.DataLoader,
    optimizer:    torch.optim.Optimizer,
    criterion:    nn.Module,
    noise_factor: float,
    device:       torch.device,
) -> float:
    """
    Run one full training epoch.

    Args:
        model        : CDAE model.
        loader       : Training DataLoader.
        optimizer    : Optimiser instance.
        criterion    : Loss function (MSELoss).
        noise_factor : Gaussian noise standard deviation.
        device       : Compute device.

    Returns:
        Average training loss over all samples.
    """
    model.train()
    total_loss = 0.0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        noisy  = add_gaussian_noise(images, noise_factor)

        optimizer.zero_grad()
        recon = model(noisy)
        loss  = criterion(recon, images)   # ← compare to CLEAN target
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.sampler)


@torch.no_grad()
def validate_epoch(
    model:        nn.Module,
    loader:       torch.utils.data.DataLoader,
    criterion:    nn.Module,
    noise_factor: float,
    device:       torch.device,
) -> float:
    """
    Run one full validation epoch (no gradient computation).

    Returns:
        Average validation loss over all samples.
    """
    model.eval()
    total_loss = 0.0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        noisy  = add_gaussian_noise(images, noise_factor)
        recon  = model(noisy)
        loss   = criterion(recon, images)
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.sampler)


# ── Main Training Loop ─────────────────────────────────────────────────────────

def train(cfg: dict) -> tuple[nn.Module, dict]:
    """
    Full training run with early stopping and LR scheduling.

    Args:
        cfg : Configuration dict (see ``DEFAULTS`` for keys).

    Returns:
        Tuple of ``(best_model, history)`` where ``history`` contains
        lists of ``train_loss``, ``val_loss``, and ``lr`` per epoch.
    """
    set_seed(cfg["seed"])
    device = get_device(verbose=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        batch_size  = cfg["batch_size"],
        valid_split = cfg["valid_split"],
        num_workers = cfg["num_workers"],
        seed        = cfg["seed"],
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DenoisingAutoencoder(base_channels=cfg["base_channels"]).to(device)
    total_p, trainable_p = count_parameters(model)
    print(f"\nModel  : DenoisingAutoencoder(base_channels={cfg['base_channels']})")
    print(f"Params : {total_p:,} total  |  {trainable_p:,} trainable\n")

    # ── Optimisation ──────────────────────────────────────────────────────────
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode      = "min",
        factor    = cfg["lr_factor"],
        patience  = cfg["lr_patience"],
        verbose   = True,
    )

    # ── Loop ──────────────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss      = float("inf")
    epochs_no_improve  = 0
    start_time         = time.time()

    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>10} | {'LR':>10} | Status")
    print("─" * 65)

    for epoch in range(1, cfg["epochs"] + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion,
                             cfg["noise_factor"], device)
        v_loss = validate_epoch(model, val_loader, criterion,
                                cfg["noise_factor"], device)

        scheduler.step(v_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["lr"].append(current_lr)

        # Checkpoint & early stopping
        if v_loss < best_val_loss:
            best_val_loss     = v_loss
            epochs_no_improve = 0
            save_checkpoint(
                model,
                cfg["checkpoint"],
                extra={"epoch": epoch, "val_loss": v_loss, "cfg": cfg},
            )
            status = "✅ Saved"
        else:
            epochs_no_improve += 1
            status = f"⏳ {epochs_no_improve}/{cfg['early_stop_pat']}"

        print(f"{epoch:>6} | {t_loss:>12.6f} | {v_loss:>10.6f} | "
              f"{current_lr:>10.6f} | {status}")

        if epochs_no_improve >= cfg["early_stop_pat"]:
            print(f"\n⛔  Early stopping at epoch {epoch}.")
            break

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {format_time(elapsed)}")
    print(f"Best Validation Loss : {best_val_loss:.6f}")
    print(f"Checkpoint saved to  : {cfg['checkpoint']}")

    # Reload best weights
    checkpoint = torch.load(cfg["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, history


# ── CLI Entry Point ────────────────────────────────────────────────────────────

def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Train Convolutional Denoising Autoencoder on MNIST"
    )
    parser.add_argument("--seed",           type=int,   default=DEFAULTS["seed"])
    parser.add_argument("--batch_size",     type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--epochs",         type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--lr",             type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight_decay",   type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--noise_factor",   type=float, default=DEFAULTS["noise_factor"])
    parser.add_argument("--lr_patience",    type=int,   default=DEFAULTS["lr_patience"])
    parser.add_argument("--early_stop_pat", type=int,   default=DEFAULTS["early_stop_pat"])
    parser.add_argument("--base_channels",  type=int,   default=DEFAULTS["base_channels"])
    parser.add_argument("--checkpoint",     type=str,   default=DEFAULTS["checkpoint"])
    parser.add_argument("--num_workers",    type=int,   default=DEFAULTS["num_workers"])
    parser.add_argument("--valid_split",    type=float, default=DEFAULTS["valid_split"])
    parser.add_argument("--lr_factor",      type=float, default=DEFAULTS["lr_factor"])
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    cfg = parse_args()
    model, history = train(cfg)
