"""
model.py
────────
Convolutional Denoising Autoencoder (CDAE) for MNIST.

Architecture
────────────
  Encoder : 3 ConvBlocks with progressive downsampling (28×28 → 7×7)
  Decoder : 2 UpBlocks  with bilinear upsampling     (7×7  → 28×28)
  Output  : Sigmoid activation → pixel values in [0, 1]

Usage
─────
  from src.model import DenoisingAutoencoder
  model = DenoisingAutoencoder(base_channels=32)
  output = model(noisy_images)   # [B,1,28,28] → [B,1,28,28]
"""

import torch
import torch.nn as nn


# ── Building Blocks ────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Encoder building block: Conv2d → BatchNorm2d → ReLU [→ MaxPool2d].

    Args:
        in_ch  : Number of input channels.
        out_ch : Number of output channels.
        pool   : If True, apply MaxPool2d(2,2) to halve spatial dimensions.
    """

    def __init__(self, in_ch: int, out_ch: int, pool: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    Decoder building block: Bilinear Upsample → Conv2d → BatchNorm2d → ReLU.

    Bilinear upsampling avoids the checkerboard artifacts that transposed
    convolutions can produce.

    Args:
        in_ch  : Number of input channels.
        out_ch : Number of output channels.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Main Model ─────────────────────────────────────────────────────────────────

class DenoisingAutoencoder(nn.Module):
    """
    Convolutional Denoising Autoencoder (CDAE).

    Learns to reconstruct clean images from noisy inputs by mapping
    through a compact bottleneck representation.

    Architecture Summary (base_channels=32):
    ─────────────────────────────────────────
      Input      [B,   1, 28, 28]
      enc1       [B,  32, 14, 14]  ← Conv + BN + ReLU + MaxPool
      enc2       [B,  64,  7,  7]  ← Conv + BN + ReLU + MaxPool
      enc3/btl   [B, 128,  7,  7]  ← Conv + BN + ReLU (BOTTLENECK)
      dec1       [B,  64, 14, 14]  ← Upsample + Conv + BN + ReLU
      dec2       [B,  32, 28, 28]  ← Upsample + Conv + BN + ReLU
      Output     [B,   1, 28, 28]  ← Conv + Sigmoid

    Args:
        base_channels : Filters in the first conv layer (default=32).
                        Subsequent layers use 2× and 4× this value.

    Total Parameters (base_channels=32): 185,537
    """

    def __init__(self, base_channels: int = 32):
        super().__init__()
        bc = base_channels

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = ConvBlock(1,    bc,    pool=True)   # 28×28 → 14×14
        self.enc2 = ConvBlock(bc,   bc*2,  pool=True)   # 14×14 →  7×7
        self.enc3 = ConvBlock(bc*2, bc*4,  pool=False)  #  7×7  →  7×7 (bottleneck)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec1 = UpBlock(bc*4, bc*2)                 #  7×7  → 14×14
        self.dec2 = UpBlock(bc*2, bc)                   # 14×14 → 28×28

        self.out = nn.Sequential(
            nn.Conv2d(bc, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),                               # → [0, 1]
        )

    # ── Forward Passes ────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through the encoder and return the bottleneck feature map.

        Args:
            x : Noisy input images  [B, 1, 28, 28]

        Returns:
            Latent feature map      [B, base_channels*4, 7, 7]
        """
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct an image from a bottleneck feature map.

        Args:
            z : Latent feature map  [B, base_channels*4, 7, 7]

        Returns:
            Reconstructed image     [B, 1, 28, 28]
        """
        z = self.dec1(z)
        z = self.dec2(z)
        return self.out(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full encode → decode pass.

        Args:
            x : Noisy input images  [B, 1, 28, 28],  values in [0, 1]

        Returns:
            Denoised reconstructions [B, 1, 28, 28], values in [0, 1]
        """
        return self.decode(self.encode(x))


# ── Helpers ────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = DenoisingAutoencoder(base_channels=32)
    total, trainable = count_parameters(model)

    print(model)
    print(f"\nTotal Parameters     : {total:,}")
    print(f"Trainable Parameters : {trainable:,}")
    print(f"Bottleneck Shape     : [B, 128, 7, 7]  ({128*7*7:,} dims)")

    # Sanity check
    dummy = torch.randn(4, 1, 28, 28)
    out   = model(dummy)
    assert out.shape == dummy.shape, f"Shape mismatch: {out.shape} vs {dummy.shape}"
    assert out.min() >= 0.0 and out.max() <= 1.0, "Output outside [0,1]"
    print("\n✅ Sanity check passed.")
