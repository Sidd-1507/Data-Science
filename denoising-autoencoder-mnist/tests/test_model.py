"""
test_model.py
─────────────
Unit tests for model shapes, metric sanity, and noise functions.

Run with:
    pytest tests/test_model.py -v
"""

import torch
import pytest
from src.model    import DenoisingAutoencoder, count_parameters
from src.noise    import add_gaussian_noise, add_salt_pepper_noise
from src.evaluate import compute_psnr, compute_ssim


# ── Model ──────────────────────────────────────────────────────────────────────

def test_forward_shape():
    model = DenoisingAutoencoder(base_channels=32)
    x     = torch.randn(4, 1, 28, 28)
    out   = model(x)
    assert out.shape == x.shape

def test_output_range():
    model = DenoisingAutoencoder(base_channels=32)
    x     = torch.rand(4, 1, 28, 28)
    out   = model(x)
    assert out.min() >= 0.0
    assert out.max() <= 1.0

def test_encode_shape():
    model = DenoisingAutoencoder(base_channels=32)
    x     = torch.rand(4, 1, 28, 28)
    z     = model.encode(x)
    assert z.shape == (4, 128, 7, 7)

def test_parameter_count():
    model = DenoisingAutoencoder(base_channels=32)
    total, trainable = count_parameters(model)
    assert total == trainable
    assert total > 0

def test_different_base_channels():
    for bc in [16, 32, 64]:
        model = DenoisingAutoencoder(base_channels=bc)
        out   = model(torch.rand(2, 1, 28, 28))
        assert out.shape == (2, 1, 28, 28)

# ── Noise ──────────────────────────────────────────────────────────────────────

def test_gaussian_noise_shape():
    x = torch.rand(8, 1, 28, 28)
    assert add_gaussian_noise(x, 0.4).shape == x.shape

def test_gaussian_noise_clamped():
    x     = torch.rand(8, 1, 28, 28)
    noisy = add_gaussian_noise(x, 2.0)
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0

def test_zero_noise():
    x = torch.rand(4, 1, 28, 28)
    assert torch.allclose(x, add_gaussian_noise(x, 0.0))

def test_salt_pepper_shape():
    x = torch.rand(4, 1, 28, 28)
    assert add_salt_pepper_noise(x, 0.05).shape == x.shape

# ── Metrics ────────────────────────────────────────────────────────────────────

def test_psnr_identical():
    x = torch.rand(4, 1, 28, 28)
    assert compute_psnr(x, x) == float("inf")

def test_psnr_decreases_with_noise():
    x = torch.rand(4, 1, 28, 28)
    assert compute_psnr(x, add_gaussian_noise(x, 0.1)) > compute_psnr(x, add_gaussian_noise(x, 0.5))

def test_ssim_identical():
    x = torch.rand(4, 1, 28, 28)
    assert abs(compute_ssim(x, x) - 1.0) < 0.01

def test_ssim_range():
    x     = torch.rand(4, 1, 28, 28)
    noisy = add_gaussian_noise(x, 0.4)
    assert 0.0 <= compute_ssim(x, noisy) <= 1.0
