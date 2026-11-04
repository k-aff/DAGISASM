from __future__ import annotations

import torch
from torch import nn


class FrequencyBranch(nn.Module):
    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.GELU(),
            )

        self.encoder = nn.Sequential(
            block(1, 16), block(16, 32), block(32, 64), block(64, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)
        self.out_dim = out_dim

    @staticmethod
    def spectrum(x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) normalised RGB -> (B,1,H,W) standardised log-amplitude."""
        gray = x.mean(dim=1, keepdim=True)
        f = torch.fft.fft2(gray, norm="ortho")
        amp = torch.log1p(torch.abs(f))
        amp = torch.fft.fftshift(amp, dim=(-2, -1))
        mu = amp.mean(dim=(-2, -1), keepdim=True)
        sd = amp.std(dim=(-2, -1), keepdim=True)
        return (amp - mu) / (sd + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.spectrum(x)).flatten(1)
        return self.proj(z)
