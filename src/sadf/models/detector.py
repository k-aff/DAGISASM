from __future__ import annotations

import timm
import torch
from torch import nn

from .freq import FrequencyBranch
from .grl import GradientReversal


class DeepfakeDetector(nn.Module):
    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freq_branch: bool = True,
        freq_dim: int = 128,
        embed_dim: int = 256,
        dropout: float = 0.1,
        n_groups: int = 0,
        adversarial: bool = False,
        adv_hidden: int = 128,
        img_size: int | None = None,
    ) -> None:
        super().__init__()
        try:  # ViT-family models accept a custom input resolution
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0,
                **({"img_size": img_size} if img_size else {}),
            )
        except TypeError:  # backbones without an img_size argument
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features

        self.freq = FrequencyBranch(freq_dim) if freq_branch else None
        fused_dim = feat_dim + (freq_dim if freq_branch else 0)

        self.fuse = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.forgery_head = nn.Linear(embed_dim, 1)

        self.adversarial = bool(adversarial) and n_groups >= 2
        if self.adversarial:
            self.grl = GradientReversal(0.0)
            self.demog_head = nn.Sequential(
                nn.Linear(embed_dim, adv_hidden),
                nn.GELU(),
                nn.Linear(adv_hidden, n_groups),
            )

    # ------------------------------------------------------------------
    def set_grl_lambda(self, value: float) -> None:
        if self.adversarial:
            self.grl.lambd = float(value)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        feats = [self.backbone(x)]
        if self.freq is not None:
            feats.append(self.freq(x))
        z = self.fuse(torch.cat(feats, dim=1))
        out: dict[str, torch.Tensor | None] = {
            "logits": self.forgery_head(z).squeeze(-1),
            "embedding": z,
            "demog_logits": None,
        }
        if self.adversarial:
            out["demog_logits"] = self.demog_head(self.grl(z))
        return out


def build_model(cfg, n_groups: int) -> DeepfakeDetector:
    return DeepfakeDetector(
        backbone=cfg.model.backbone,
        pretrained=bool(cfg.model.pretrained),
        freq_branch=bool(cfg.model.freq_branch),
        freq_dim=int(cfg.model.freq_dim),
        embed_dim=int(cfg.model.embed_dim),
        dropout=float(cfg.model.dropout),
        n_groups=n_groups,
        adversarial=bool(cfg.adversarial.enabled),
        adv_hidden=int(cfg.adversarial.head_hidden),
        img_size=int(cfg.image.size),
    )
