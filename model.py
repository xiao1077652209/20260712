"""PyTorch reproduction of the JAC-SCM network described by Guan et al. (2026)."""

from __future__ import annotations

import torch
from torch import nn


class SSCJAM(nn.Module):
    """Spectral-Spatial-Channel Joint Attention Mechanism (Eqs. 9-14)."""

    def __init__(self, channels: int, reduction: int = 8, alpha: float = 0.7):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.alpha = alpha
        self.channel_mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Sequential(
            nn.Conv1d(channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=-1, keepdim=True)
        maximum = torch.amax(x, dim=-1, keepdim=True)
        channel_weight = torch.sigmoid(self.channel_mlp(avg) + self.channel_mlp(maximum))
        spatial_weight = self.spatial(x)
        return self.alpha * (x * channel_weight) + (1.0 - self.alpha) * (x * spatial_weight)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        # padding="same" matches the boundary-padding statement in the paper.
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True), nn.MaxPool1d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class JACSCM(nn.Module):
    """Improved LeNet-5 for 114-point NIR soybean spectra."""

    def __init__(self, input_length: int = 114, num_classes: int = 10,
                 attention_reduction: int = 8, attention_alpha: float = 0.7):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(1, 32, 7),
            SSCJAM(32, attention_reduction, attention_alpha),
            ConvBlock(32, 64, 4),
            ConvBlock(64, 128, 4),
        )
        with torch.no_grad():
            flattened = self.conv(torch.zeros(1, 1, input_length)).numel()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(flattened, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("Expected spectra shaped [batch, length] or [batch, 1, length]")
        return self.fc(self.conv(x))


def make_lawd_adamw(model: JACSCM, base_lr: float = 1e-3) -> torch.optim.AdamW:
    """LAWD-AdamW parameter groups from Eq. 16 and Section 5.2."""
    attention, convolution, fully_connected = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "conv.1" in name:
            attention.append(parameter)
        elif name.startswith("fc"):
            fully_connected.append(parameter)
        else:
            convolution.append(parameter)
    return torch.optim.AdamW([
        {"params": attention, "lr": 1.2 * base_lr, "weight_decay": 1e-4},
        {"params": convolution, "lr": base_lr, "weight_decay": 1e-4},
        {"params": fully_connected, "lr": 0.9 * base_lr, "weight_decay": 5e-4},
    ])
