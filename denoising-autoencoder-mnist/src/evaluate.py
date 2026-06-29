"""
evaluate.py
───────────
Evaluation utilities for the Convolutional Denoising Autoencoder.

Metrics
───────
  • MSE  — Mean Squared Error
  • PSNR — Peak Signal-to-Noise Ratio  (dB, higher is better)
  • SSIM — Structural Similarity Index  (0–1, higher is better)

Visualisations
──────────────
  • Side-by-side clean / noisy / denoised grid
  • Noise robustness curve (PSNR vs. σ)
  • t-SNE latent-space scatter plot (coloured by digit class)
  • Training loss & LR schedule curves

CLI::

    python src/evaluate.py --checkpoint best_denoising_ae.pth
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.model   import DenoisingAutoencoder
from src.dataset import get_dataloaders
from src.noise   import add_gaussian_noise
from src.utils   import get_device, load_checkpoint


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_psnr(clean: torch.Tensor, recon: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio in decibels.

    PSNR = 10 · log₁₀(MAX² / MSE)

    Args:
        clean   : Ground-truth image batch.
        recon   : Reconstructed image batch.
        max_val : Maximum pixel value (1.0 for normalised images).

    Returns:
        PSNR in dB (higher is better).  Returns ``inf`` if MSE == 0.
    """
    mse = F.mse_loss(recon, clean).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val ** 2 / mse)


def compute_ssim(clean: torch.Tensor, recon: torch.Tensor) -> float:
    """
    Simplified single-scale SSIM approximation (no external libraries required).

    Computed over a 3×3 average-pooling window. For exact per-image SSIM,
    consider ``pip install pytorch-msssim``.

    Args:
        clean : Ground-truth images  [B, C, H, W].
        recon : Reconstructed images [B, C, H, W].

    Returns:
        Mean SSIM across the batch (higher is better, ∈ [0, 1]).
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    mu1 = F.avg_pool2d(clean, 3, 1, 1)
    mu2 = F.avg_pool2d(recon, 3, 1, 1)

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sig1_sq = F.avg_pool2d(clean * clean, 3, 1, 1) - mu1_sq
    sig2_sq = F.avg_pool2d(recon * recon, 3, 1, 1) - mu2_sq
    sig12   = F.avg_pool2d(clean * recon, 3, 1, 1) - mu1_mu2

    ssim_map = (
        (2 * mu1_mu2 + C1) * (2 * sig12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2)
    )
    return ssim_map.mean().item()


@torch.no_grad()
def evaluate_testset(
    model:        torch.nn.Module,
    loader:       torch.utils.data.DataLoader,
    noise_factor: float,
    device:       torch.device,
) -> dict:
    """
    Compute MSE, PSNR, and SSIM on the full test set.

    Args:
        model        : Trained CDAE model (in eval mode).
        loader       : Test DataLoader.
        noise_factor : Gaussian noise σ to apply.
        device       : Compute device.

    Returns:
        Dict with keys ``"MSE"``, ``"PSNR"``, ``"SSIM"``, each mapping to
        ``{"Noisy": float, "Denoised": float}``.
    """
    model.eval()
    mse_noisy,  mse_recon  = [], []
    psnr_noisy, psnr_recon = [], []
    ssim_noisy, ssim_recon = [], []

    for images, _ in loader:
        images = images.to(device)
        noisy  = add_gaussian_noise(images, noise_factor)
        recon  = model(noisy)

        mse_noisy.append( F.mse_loss(noisy, images).item())
        mse_recon.append( F.mse_loss(recon, images).item())
        psnr_noisy.append(compute_psnr(images, noisy))
        psnr_recon.append(compute_psnr(images, recon))
        ssim_noisy.append(compute_ssim(images, noisy))
        ssim_recon.append(compute_ssim(images, recon))

    return {
        "MSE":  {"Noisy": np.mean(mse_noisy),  "Denoised": np.mean(mse_recon)},
        "PSNR": {"Noisy": np.mean(psnr_noisy), "Denoised": np.mean(psnr_recon)},
        "SSIM": {"Noisy": np.mean(ssim_noisy), "Denoised": np.mean(ssim_recon)},
    }


def print_metrics(metrics: dict, noise_factor: float) -> None:
    """Pretty-print evaluation metrics to stdout."""
    print(f"\n{'='*54}")
    print(f"  Evaluation  (noise σ = {noise_factor})")
    print(f"{'='*54}")
    print(f"  {'Metric':<8} | {'Noisy':>10} | {'Denoised':>10} | {'Gain':>8}")
    print(f"  {'-'*50}")
    for metric, vals in metrics.items():
        n, d = vals["Noisy"], vals["Denoised"]
        if metric == "MSE":
            gain = f"{(1 - d/n)*100:+.1f}%"
        else:
            gain = f"{(d/n - 1)*100:+.1f}%"
        print(f"  {metric:<8} | {n:>10.4f} | {d:>10.4f} | {gain:>8}")
    print(f"{'='*54}\n")


# ── Visualisations ─────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_denoising_grid(
    model:        torch.nn.Module,
    loader:       torch.utils.data.DataLoader,
    noise_factor: float,
    device:       torch.device,
    n_images:     int = 10,
) -> None:
    """
    Display a 3-row grid: Clean / Noisy / Denoised.
    """
    model.eval()
    images, labels = next(iter(loader))
    images = images[:n_images].to(device)
    labels = labels[:n_images]

    noisy = add_gaussian_noise(images, noise_factor)
    recon = model(noisy)

    images, noisy, recon = images.cpu(), noisy.cpu(), recon.cpu()

    fig = plt.figure(figsize=(n_images * 1.5, 5.5))
    fig.suptitle(f"Denoising Results  |  σ = {noise_factor}", fontsize=13, y=1.01)

    rows = [("Clean\n(Target)", images),
            ("Noisy\n(Input)",  noisy),
            ("Denoised\n(Output)", recon)]

    for r, (label, data) in enumerate(rows):
        for c in range(n_images):
            ax = fig.add_subplot(3, n_images, r * n_images + c + 1)
            ax.imshow(data[c].squeeze(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(label, fontsize=9, rotation=90)
            if r == 0:
                ax.set_title(str(labels[c].item()), fontsize=9, color="navy")

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_robustness_curve(
    model:         torch.nn.Module,
    loader:        torch.utils.data.DataLoader,
    device:        torch.device,
    noise_factors: list[float] | None = None,
) -> None:
    """
    Plot PSNR vs. noise level for noisy input and denoised output.
    """
    if noise_factors is None:
        noise_factors = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

    model.eval()
    images, _ = next(iter(loader))
    images    = images[:64].to(device)

    psnr_noisy, psnr_recon = [], []

    for nf in noise_factors:
        noisy = add_gaussian_noise(images, nf)
        recon = model(noisy)
        psnr_noisy.append(compute_psnr(images, noisy))
        psnr_recon.append(compute_psnr(images, recon))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(noise_factors, psnr_noisy, label="Noisy PSNR",    color="#e74c3c",
            lw=2, marker="o", ms=5)
    ax.plot(noise_factors, psnr_recon, label="Denoised PSNR", color="#27ae60",
            lw=2, marker="s", ms=5)
    ax.fill_between(noise_factors, psnr_noisy, psnr_recon,
                    alpha=0.12, color="green", label="PSNR Gain")
    ax.axvline(0.4, color="navy", ls=":", lw=1.5, label="Training σ = 0.4")
    ax.set_xlabel("Noise Factor (σ)", fontsize=12)
    ax.set_ylabel("PSNR (dB)",        fontsize=12)
    ax.set_title("Noise Robustness — PSNR vs. σ", fontsize=13)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


@torch.no_grad()
def plot_tsne(
    model:     torch.nn.Module,
    loader:    torch.utils.data.DataLoader,
    device:    torch.device,
    n_samples: int = 2000,
    seed:      int = 42,
) -> None:
    """
    Extract latent codes and visualise with t-SNE, coloured by digit class.
    """
    model.eval()
    codes_list, labels_list = [], []
    total = 0

    for images, labels in loader:
        if total >= n_samples:
            break
        codes = model.encode(images.to(device))
        codes = codes.view(codes.size(0), -1).cpu().numpy()
        codes_list.append(codes)
        labels_list.append(labels.numpy())
        total += len(images)

    codes  = np.concatenate(codes_list,  axis=0)[:n_samples]
    labels = np.concatenate(labels_list, axis=0)[:n_samples]

    codes_scaled = StandardScaler().fit_transform(codes)
    print(f"Running t-SNE on {len(codes)} samples…")
    embedded = TSNE(n_components=2, perplexity=40, n_iter=1000,
                    random_state=seed, init="pca",
                    learning_rate="auto").fit_transform(codes_scaled)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(embedded[:, 0], embedded[:, 1],
                    c=labels, cmap="tab10", s=12, alpha=0.7)
    cbar = plt.colorbar(sc, ax=ax, ticks=range(10))
    cbar.set_ticklabels([str(i) for i in range(10)])
    cbar.set_label("Digit Class", fontsize=11)
    ax.set_title("t-SNE of Latent Space (coloured by digit — no labels used!)", fontsize=12)
    ax.set_xlabel("t-SNE Dim 1"); ax.set_ylabel("t-SNE Dim 2")
    ax.grid(alpha=0.2)
    plt.tight_layout(); plt.show()


def plot_training_history(history: dict) -> None:
    """Plot train/val loss and LR schedule from a training history dict."""
    n = len(history["train_loss"])
    x = range(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.plot(x, history["train_loss"], label="Train", color="steelblue",  lw=2)
    ax.plot(x, history["val_loss"],   label="Val",   color="orangered",   lw=2, ls="--")
    best = np.argmin(history["val_loss"]) + 1
    ax.axvline(best, color="green", ls=":", lw=1.5, label=f"Best ({best})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Train / Val Loss"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(x, history["lr"], color="purple", lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR (log)")
    ax.set_title("Learning Rate Schedule"); ax.grid(alpha=0.3)

    plt.tight_layout(); plt.show()


# ── CLI Entry Point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CDAE on MNIST test set")
    parser.add_argument("--checkpoint",    type=str,   default="best_denoising_ae.pth")
    parser.add_argument("--noise_factor",  type=float, default=0.4)
    parser.add_argument("--batch_size",    type=int,   default=128)
    parser.add_argument("--base_channels", type=int,   default=32)
    parser.add_argument("--n_images",      type=int,   default=10)
    parser.add_argument("--tsne",          action="store_true", help="Run t-SNE visualisation")
    args = parser.parse_args()

    device = get_device(verbose=True)

    # Load model
    model = DenoisingAutoencoder(base_channels=args.base_channels).to(device)
    ckpt  = load_checkpoint(model, args.checkpoint, device)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    _, _, test_loader = get_dataloaders(batch_size=args.batch_size)

    # Metrics
    metrics = evaluate_testset(model, test_loader, args.noise_factor, device)
    print_metrics(metrics, args.noise_factor)

    # Visuals
    plot_denoising_grid(model, test_loader, args.noise_factor, device, args.n_images)
    plot_robustness_curve(model, test_loader, device)

    if args.tsne:
        plot_tsne(model, test_loader, device)


if __name__ == "__main__":
    main()
