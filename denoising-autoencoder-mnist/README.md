# 🖼️ Convolutional Denoising Autoencoder on MNIST

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen" />
  <img src="https://img.shields.io/badge/PSNR-21.66_dB-orange" />
  <img src="https://img.shields.io/badge/SSIM-0.89-blueviolet" />
</p>

<p align="center">
  A deep learning project that uses a <b>Convolutional Denoising Autoencoder (CDAE)</b> to reconstruct clean handwritten digit images from noisy inputs — trained entirely without labels.
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Results](#-results)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Key Features](#-key-features)
- [Techniques Used](#-techniques-used)
- [Extensions](#-extensions--future-work)
- [License](#-license)

---

## 🔍 Overview

A **Denoising Autoencoder** is trained to recover a clean image `x` from a corrupted version `x̃ = x + noise`. This forces the network to learn **robust, semantically meaningful representations** of the data — not just a pass-through mapping.

| | Input | Output |
|---|---|---|
| **Train** | Noisy MNIST image (σ=0.4) | Clean MNIST image |
| **Evaluate** | Noisy image (any σ) | Denoised reconstruction |

The model is **fully unsupervised** — it never sees class labels during training, yet the latent space organises itself into clean digit clusters (verified via t-SNE).

---

## 📊 Results

Trained for **10 epochs** on a Tesla T4 GPU. Best validation loss: **0.006917**.

| Metric | Noisy Input | Denoised Output | Improvement |
|--------|------------|-----------------|-------------|
| **MSE** | 0.0796 | **0.0068** | ↓ 91.5% |
| **PSNR** | 10.99 dB | **21.66 dB** | ↑ +10.67 dB |
| **SSIM** | 0.1729 | **0.8903** | ↑ +415% |

### Visual Results (σ = 0.4)

```
Row 1 [Clean]    : ⬜⬜⬛⬛⬜⬜  ← Ground truth
Row 2 [Noisy]    : 🔲🔲🔲🔲🔲🔲  ← Model input
Row 3 [Denoised] : ⬜⬜⬛⬛⬜⬜  ← Model output
```

> See the notebook for full visual output with all 10 test samples.

---

## 🏗️ Architecture

```
INPUT  [B,  1, 28, 28]
  │
  ▼  Conv(1→32, k=3, p=1) + BatchNorm + ReLU + MaxPool(2)
[B, 32, 14, 14]
  │
  ▼  Conv(32→64, k=3, p=1) + BatchNorm + ReLU + MaxPool(2)
[B, 64,  7,  7]
  │
  ▼  Conv(64→128, k=3, p=1) + BatchNorm + ReLU          ← BOTTLENECK
[B,128,  7,  7]  (6,272 dims)
  │
  ▼  Upsample(×2) + Conv(128→64) + BatchNorm + ReLU
[B, 64, 14, 14]
  │
  ▼  Upsample(×2) + Conv(64→32) + BatchNorm + ReLU
[B, 32, 28, 28]
  │
  ▼  Conv(32→1, k=3, p=1) + Sigmoid
OUTPUT [B,  1, 28, 28]
```

| Component | Details |
|-----------|---------|
| **Total Parameters** | 185,537 |
| **Bottleneck** | 128 × 7 × 7 = 6,272 dimensions |
| **Upsampling** | Bilinear (no checkerboard artifacts) |
| **Activation** | ReLU (hidden), Sigmoid (output) |
| **Normalisation** | BatchNorm2d after every conv |

---

## 📁 Project Structure

```
denoising-autoencoder-mnist/
│
├── 📓 notebooks/
│   └── Denoising_Autoencoder_MNIST.ipynb   # Full annotated notebook
│
├── 🐍 src/
│   ├── model.py          # CDAE architecture (ConvBlock, UpBlock, DenoisingAutoencoder)
│   ├── dataset.py        # Data loading, train/val/test split
│   ├── noise.py          # Noise injection utilities (Gaussian, Salt-and-Pepper)
│   ├── train.py          # Training loop, early stopping, checkpointing
│   ├── evaluate.py       # PSNR, SSIM, MSE metrics + visualisation
│   └── utils.py          # Seed fixing, device setup, helpers
│
├── 🧪 tests/
│   └── test_model.py     # Unit tests (forward pass, shape checks, metric sanity)
│
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/denoising-autoencoder-mnist.git
cd denoising-autoencoder-mnist

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, CUDA-capable GPU (optional but recommended).

---

## 🚀 Quick Start

```python
import torch
from src.model import DenoisingAutoencoder
from src.noise import add_gaussian_noise

# Load trained model
model = DenoisingAutoencoder(base_channels=32)
model.load_state_dict(torch.load('best_denoising_ae.pth', map_location='cpu'))
model.eval()

# Denoise a batch of images  [B, 1, 28, 28]  values in [0, 1]
with torch.no_grad():
    noisy_images = add_gaussian_noise(clean_images, noise_factor=0.4)
    denoised     = model(noisy_images)
```

---

## 🏋️ Training

```bash
python src/train.py
```

**Default hyperparameters** (edit inside `train.py`):

| Parameter | Value |
|-----------|-------|
| Epochs | 30 (early stop) |
| Batch size | 128 |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Noise factor (σ) | 0.4 |
| LR patience | 5 epochs |
| Early stop patience | 10 epochs |
| Val split | 15% |

Training produces:
- `best_denoising_ae.pth` — best model checkpoint
- Loss curve and LR schedule plots

---

## 📐 Evaluation

```bash
python src/evaluate.py --checkpoint best_denoising_ae.pth
```

Outputs:
- MSE, PSNR, SSIM on full test set
- Side-by-side visual: clean / noisy / denoised
- Noise robustness curve (PSNR vs σ)
- t-SNE latent space plot coloured by digit class

---

## ✨ Key Features

- ✅ **Pure PyTorch** — no external denoising libraries
- ✅ **Fully unsupervised** — no class labels used during training
- ✅ **Reproducible** — fixed seeds for all random sources
- ✅ **Industry metrics** — PSNR & SSIM (not just loss)
- ✅ **Early stopping** — automatic, with best checkpoint saving
- ✅ **Noise robustness study** — tested across σ ∈ [0.05, 1.0]
- ✅ **t-SNE analysis** — latent space cluster visualisation
- ✅ **Modular code** — clean `src/` split for easy extension

---

## 🔧 Techniques Used

| Technique | Purpose |
|-----------|---------|
| **BatchNorm2d** | Stabilises training, implicit regularisation |
| **Bilinear Upsampling** | Smooth decoder output, no checkerboard artifacts |
| **Gradient Clipping** (`max_norm=1.0`) | Prevents exploding gradients |
| **ReduceLROnPlateau** | Auto-anneals LR when val loss plateaus |
| **Early Stopping** | Halts training before overfitting |
| **Weight Decay** (Adam) | L2 regularisation without modifying loss |
| **Xavier / Kaiming Init** | Healthy gradient flow from epoch 1 |

---

## 🔭 Extensions & Future Work

| Extension | Description |
|-----------|-------------|
| 🔷 **VAE** | Variational Autoencoder for generative capabilities |
| 🔷 **U-Net Skip Connections** | Preserve fine spatial detail from encoder to decoder |
| 🔷 **Perceptual Loss** | Use VGG features instead of MSE for sharper results |
| 🔷 **Blind Denoising** | Train on mixed noise levels (DnCNN-style) |
| 🔷 **Real Images** | Extend to BSDS500 or DIV2K natural image datasets |
| 🔷 **Attention** | Channel/spatial attention for adaptive denoising |

---

## 📄 License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

## 🙏 Acknowledgements

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) — Yann LeCun et al.
- [PyTorch](https://pytorch.org/) — Facebook AI Research
- Vincent et al. (2010) — *Stacked Denoising Autoencoders* (original DAE paper)
