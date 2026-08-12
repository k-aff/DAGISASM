from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

UNKNOWN_GROUP = -1


def _cvar(losses: torch.Tensor, rho: torch.Tensor, alpha: float) -> torch.Tensor:
    return rho + F.relu(losses - rho).mean() / alpha


class MeanLoss(nn.Module):
    """Plain expected risk (baseline)."""

    def forward(self, per_sample: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        return per_sample.mean()


class CVaRLoss(nn.Module):
    """DAG-FDD: group-label-free distributionally robust objective."""

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.rho = nn.Parameter(torch.zeros(1))

    def forward(self, per_sample: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        return _cvar(per_sample, self.rho, self.alpha).squeeze()


class GroupCVaRLoss(nn.Module):
    """DAW-FDD: outer CVaR over per-group inner-CVaR losses.

    Samples with unknown group (-1) form their own bucket so they still
    contribute without contaminating labelled groups.
    """

    def __init__(
        self, n_groups: int, alpha_group: float = 0.5, alpha_sample: float = 0.8
    ) -> None:
        super().__init__()
        self.n_groups = int(n_groups)
        self.alpha_group = float(alpha_group)
        self.alpha_sample = float(alpha_sample)
        # slot n_groups is the unknown bucket
        self.rho_inner = nn.Parameter(torch.zeros(self.n_groups + 1))
        self.rho_outer = nn.Parameter(torch.zeros(1))

    def forward(self, per_sample: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        g = groups.clone()
        g[g == UNKNOWN_GROUP] = self.n_groups
        group_losses = []
        for gi in torch.unique(g):
            sel = g == gi
            group_losses.append(
                _cvar(per_sample[sel], self.rho_inner[gi], self.alpha_sample)
            )
        stacked = torch.stack(group_losses)
        if stacked.numel() == 1:
            return stacked.squeeze()
        return _cvar(stacked, self.rho_outer, self.alpha_group).squeeze()


def adversary_loss(
    demog_logits: torch.Tensor | None, groups: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy of the demographic adversary on known-group samples.

    The GRL sits upstream, so *minimising* this trains the adversary
    while the reversed gradient scrubs demographic cues from the shared
    embedding.
    """
    if demog_logits is None:
        return torch.zeros((), device=groups.device)
    known = groups != UNKNOWN_GROUP
    if known.sum() < 2:
        return torch.zeros((), device=groups.device)
    return F.cross_entropy(demog_logits[known], groups[known])


def soft_fpr_gap(
    logits: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor
) -> torch.Tensor:
    """Differentiable max-min gap of expected false-positive rates.

    For real samples (label 0), sigmoid(logit) is a soft "flagged fake"
    indicator; per-group means approximate group FPRs.
    """
    probs = torch.sigmoid(logits)
    real = labels < 0.5
    fprs = []
    for gi in torch.unique(groups):
        if gi == UNKNOWN_GROUP:
            continue
        sel = real & (groups == gi)
        if sel.sum() >= 2:
            fprs.append(probs[sel].mean())
    if len(fprs) < 2:
        return torch.zeros((), device=logits.device)
    stacked = torch.stack(fprs)
    return stacked.max() - stacked.min()


def build_aggregator(cfg, n_groups: int) -> nn.Module:
    kind = cfg.loss.type
    if kind == "bce":
        return MeanLoss()
    if kind == "dag_fdd":
        return CVaRLoss(alpha=float(cfg.loss.dag_alpha))
    if kind == "daw_fdd":
        return GroupCVaRLoss(
            n_groups=n_groups,
            alpha_group=float(cfg.loss.daw_alpha_group),
            alpha_sample=float(cfg.loss.daw_alpha_sample),
        )
    raise ValueError(f"Unknown loss.type '{kind}'")
