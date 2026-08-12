from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import DotDict
from ..evaluation.fairness import fairness_gaps, per_group_table
from ..evaluation.metrics import binary_metrics
from ..utils.common import ensure_dir, get_logger, save_json
from .losses import adversary_loss, build_aggregator, soft_fpr_gap

log = get_logger()


class Trainer:
    def __init__(
        self,
        cfg: DotDict,
        model: nn.Module,
        group_vocab: list[str],
        run_dir: str | Path,
        device: str,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.group_vocab = group_vocab
        self.run_dir = ensure_dir(run_dir)

        self.aggregator = build_aggregator(cfg, len(group_vocab)).to(device)
        params = list(self.model.parameters()) + list(self.aggregator.parameters())
        self.opt = torch.optim.AdamW(
            params, lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay)
        )
        amp_cfg = cfg.train.amp
        self.use_amp = (device == "cuda") if amp_cfg == "auto" else bool(amp_cfg)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    # ------------------------------------------------------------------
    def _schedule(self, total_steps: int, warmup_steps: int):
        def fn(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * t))

        return torch.optim.lr_scheduler.LambdaLR(self.opt, fn)

    def _grl_lambda(self, epoch: int) -> float:
        ramp = max(1, int(self.cfg.adversarial.grl_ramp_epochs))
        return min(1.0, (epoch + 1) / ramp)

    # ------------------------------------------------------------------
    def train_epoch(self, loader: DataLoader, epoch: int, sched) -> dict:
        self.model.train()
        if bool(self.cfg.adversarial.enabled):
            self.model.set_grl_lambda(self._grl_lambda(epoch))
        beta = float(self.cfg.adversarial.weight) if self.cfg.adversarial.enabled else 0.0
        w_fpr = float(self.cfg.fpr_penalty.weight) if self.cfg.fpr_penalty.enabled else 0.0

        sums = {"total": 0.0, "task": 0.0, "adv": 0.0, "fpr": 0.0}
        n_batches = 0
        for x, y, g, _ in loader:
            x, y, g = x.to(self.device), y.to(self.device), g.to(self.device)
            self.opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=self.use_amp):
                out = self.model(x)
                per_sample = nn.functional.binary_cross_entropy_with_logits(
                    out["logits"], y, reduction="none"
                )
                task = self.aggregator(per_sample, g)
                adv = adversary_loss(out["demog_logits"], g) if beta > 0 else per_sample.new_zeros(())
                fpr = soft_fpr_gap(out["logits"], y, g) if w_fpr > 0 else per_sample.new_zeros(())
                loss = task + beta * adv + w_fpr * fpr

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.train.grad_clip))
            self.scaler.step(self.opt)
            self.scaler.update()
            sched.step()

            sums["total"] += float(loss.detach())
            sums["task"] += float(task.detach())
            sums["adv"] += float(adv.detach())
            sums["fpr"] += float(fpr.detach())
            n_batches += 1
        return {k: v / max(1, n_batches) for k, v in sums.items()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, loader: DataLoader) -> pd.DataFrame:
        self.model.eval()
        scores, labels, groups, positions = [], [], [], []
        for x, y, g, idx in loader:
            out = self.model(x.to(self.device))
            scores.append(torch.sigmoid(out["logits"]).float().cpu())
            labels.append(y)
            groups.append(g)
            positions.append(idx)
        return pd.DataFrame(
            {
                "score": torch.cat(scores).numpy(),
                "label": torch.cat(labels).numpy().astype(int),
                "group_idx": torch.cat(groups).numpy(),
                "row": torch.cat(positions).numpy(),
            }
        )

    def _val_summary(self, preds: pd.DataFrame) -> dict:
        thr = float(self.cfg.train.threshold)
        overall = binary_metrics(preds["label"].values, preds["score"].values, thr)
        idx_to_group = {i: g for i, g in enumerate(self.group_vocab)}
        preds = preds.assign(group=preds["group_idx"].map(idx_to_group).fillna("unknown"))
        table = per_group_table(preds, "group", thr, min_n=2)
        gaps = fairness_gaps(table)
        return {"auc": overall["auc"], "acc": overall["accuracy"], **gaps}

    def _selection_score(self, val: dict) -> float:
        auc = val["auc"] if not np.isnan(val["auc"]) else 0.0
        if self.cfg.train.select_metric == "auc_minus_fprgap":
            gap = val.get("fpr_gap")
            return auc - (gap if gap is not None and not np.isnan(gap) else 0.0)
        return auc

    # ------------------------------------------------------------------
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        epochs = int(self.cfg.train.epochs)
        steps = epochs * max(1, len(train_loader))
        warmup = int(self.cfg.train.warmup_epochs) * max(1, len(train_loader))
        sched = self._schedule(steps, warmup)

        history, best, patience = [], -np.inf, 0
        for epoch in range(epochs):
            t0 = time.time()
            train_stats = self.train_epoch(train_loader, epoch, sched)
            val = self._val_summary(self.predict(val_loader))
            score = self._selection_score(val)
            history.append({"epoch": epoch, **train_stats, **{f"val_{k}": v for k, v in val.items()}})
            log.info(
                "epoch %02d | loss %.4f | val_auc %.4f | val_fpr_gap %s | sel %.4f | %.1fs",
                epoch, train_stats["total"], val["auc"],
                f"{val.get('fpr_gap'):.4f}" if val.get("fpr_gap") is not None else "n/a",
                score, time.time() - t0,
            )
            self._save("last.pt", epoch, val)
            if score > best:
                best, patience = score, 0
                self._save("best.pt", epoch, val)
            else:
                patience += 1
                if patience >= int(self.cfg.train.early_stop_patience):
                    log.info("Early stopping at epoch %d", epoch)
                    break
        save_json(history, self.run_dir / "history.json")
        return {"history": history, "best_selection_score": best}

    def _save(self, name: str, epoch: int, val: dict) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": self.cfg.to_dict(),
                "group_vocab": self.group_vocab,
                "epoch": epoch,
                "val": val,
            },
            self.run_dir / name,
        )
