"""
noise.py
────────
Noise injection utilities for the denoising autoencoder.

Supported noise types:
  • Gaussian (Additive White Gaussian Noise — AWGN)
  • Salt-and-Pepper (impulse noise)

All functions operate on PyTorch tensors and return values clamped to [0, 1].
"""

import torch


def add_gaussian_noise(
    images: torch.Tensor,
    noise_factor: float = 0.4,
) -> torch.Tensor:
    """
    Corrupt images with Additive White Gaussian Noise (AWGN).

    Each pixel is independently perturbed by a sample from N(0, noise_factor²).
    The output is clamped to [0, 1] so it remains a valid image.

    Args:
        images       : Clean image batch  [B, C, H, W],  values in [0, 1].
        noise_factor : Standard deviation of the Gaussian noise.
                       Typical values: 0.1 (mild) → 0.4 (moderate) → 0.8 (heavy).

    Returns:
        Noisy image batch  [B, C, H, W],  values in [0, 1].

    Example::

        >>> clean = torch.rand(8, 1, 28, 28)
        >>> noisy = add_gaussian_noise(clean, noise_factor=0.4)
        >>> assert noisy.shape == clean.shape
    """
    if noise_factor <= 0:
        return images.clone()

    noise = torch.randn_like(images) * noise_factor
    return (images + noise).clamp(0.0, 1.0)


def add_salt_pepper_noise(
    images: torch.Tensor,
    amount: float = 0.05,
) -> torch.Tensor:
    """
    Corrupt images with Salt-and-Pepper (impulse) noise.

    Randomly sets `amount/2` proportion of pixels to 1 (salt / white)
    and `amount/2` proportion to 0 (pepper / black), simulating sensor errors.

    Args:
        images : Clean image batch  [B, C, H, W],  values in [0, 1].
        amount : Total proportion of pixels to corrupt.
                 E.g., 0.05 → 5% of pixels are set to 0 or 1.

    Returns:
        Noisy image batch  [B, C, H, W],  values in [0, 1].

    Example::

        >>> clean = torch.rand(8, 1, 28, 28)
        >>> noisy = add_salt_pepper_noise(clean, amount=0.05)
        >>> assert noisy.shape == clean.shape
    """
    noisy    = images.clone()
    n_pixels = images.numel()
    n_salt   = int(amount / 2 * n_pixels)
    n_pepper = int(amount / 2 * n_pixels)

    # Salt  → white pixels (value = 1)
    salt_coords = [torch.randint(0, s, (n_salt,)) for s in images.shape]
    noisy[salt_coords[0], salt_coords[1], salt_coords[2], salt_coords[3]] = 1.0

    # Pepper → black pixels (value = 0)
    pepper_coords = [torch.randint(0, s, (n_pepper,)) for s in images.shape]
    noisy[pepper_coords[0], pepper_coords[1], pepper_coords[2], pepper_coords[3]] = 0.0

    return noisy


def add_mixed_noise(
    images: torch.Tensor,
    gaussian_factor: float = 0.2,
    sp_amount: float = 0.02,
) -> torch.Tensor:
    """
    Apply Gaussian and Salt-and-Pepper noise sequentially.

    Useful for robustness training on a harder, combined corruption.

    Args:
        images          : Clean image batch  [B, C, H, W].
        gaussian_factor : Gaussian noise σ.
        sp_amount       : Salt-and-pepper pixel proportion.

    Returns:
        Doubly-corrupted image batch  [B, C, H, W].
    """
    noisy = add_gaussian_noise(images, gaussian_factor)
    noisy = add_salt_pepper_noise(noisy, sp_amount)
    return noisy


# ── Noise factory (convenience) ────────────────────────────────────────────────

NOISE_FUNCTIONS = {
    "gaussian":      add_gaussian_noise,
    "salt_pepper":   add_salt_pepper_noise,
    "mixed":         add_mixed_noise,
}


def apply_noise(images: torch.Tensor, noise_type: str = "gaussian", **kwargs) -> torch.Tensor:
    """
    Apply a named noise function to a batch of images.

    Args:
        images     : Clean image batch.
        noise_type : One of ``"gaussian"``, ``"salt_pepper"``, ``"mixed"``.
        **kwargs   : Forwarded to the chosen noise function.

    Returns:
        Noisy image batch.

    Example::

        >>> noisy = apply_noise(clean, noise_type="gaussian", noise_factor=0.4)
    """
    if noise_type not in NOISE_FUNCTIONS:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. "
            f"Choose from: {list(NOISE_FUNCTIONS.keys())}"
        )
    return NOISE_FUNCTIONS[noise_type](images, **kwargs)
