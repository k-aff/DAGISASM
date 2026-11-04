#!/usr/bin/env python3
r"""Train and evaluate the SKEWED-data Swin control.

It copies every setting from the balanced baseline checkpoint (--like), so the
only thing that differs is the training data. Nothing in your existing
configs or runs is changed.

USAGE
-----
  python scripts\train_skewed_swin.py ^
      --manifest data\faces140k\manifests\manifest_skewed.csv ^
      --like     runs\models140k\swin_aug\baseline\best.pt ^
      --runs     runs\models140k\swin_skewed

  --check      print the settings and data counts, then stop (no training)
  --eval-only  skip training and just evaluate an existing best.pt
  --device     cpu | cuda (default: auto)

Writes:
  <runs>\baseline\best.pt, last.pt, history.json, run_meta.json
  <runs>\baseline\eval\   overall.json, gaps.json, per_group_skin_tone.csv, ...
                          (same files as the balanced eval, so
                           compare_dataset_effect.py can read both)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        for cand in (parent / "src", parent):
            if (cand / "sadf" / "__init__.py").exists():
                sys.path.insert(0, str(cand))
                return
    raise SystemExit("ERROR: could not find the 'sadf' package.")


_bootstrap()

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from sadf.config import _dotify, set_dotted  # noqa: E402
from sadf.utils.common import (  # noqa: E402
    ensure_dir, get_logger, resolve_device, save_json, seed_everything,
)

log = get_logger()


def _settings(cfg) -> dict:
    """The settings that must match the balanced baseline."""
    return {
        "backbone": cfg.model.backbone,
        "freq_branch": cfg.model.freq_branch,
        "image_size": cfg.image.size,
        "epochs": cfg.train.epochs,
        "batch_size": cfg.train.batch_size,
        "lr": cfg.train.lr,
        "early_stop_patience": cfg.train.early_stop_patience,
        "select_metric": cfg.train.select_metric,
        "loss": cfg.loss.type,
        "balanced_sampler": cfg.train.balanced_sampler,
        "adversary": cfg.adversarial.enabled,
        "seed": cfg.seed,
    }


def _load_manifest(path: Path, group_key: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "status" in df.columns:
        n0 = len(df)
        df = df[df["status"] == "ok"]
        if len(df) < n0:
            log.warning("%d rows dropped because status != ok", n0 - len(df))
    if group_key not in df.columns:
        raise SystemExit(f"Manifest has no '{group_key}' column.")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="manifest_skewed.csv")
    ap.add_argument("--like", required=True,
                    help="balanced BASELINE best.pt whose settings are copied")
    ap.add_argument("--runs", default="runs/models140k/swin_skewed")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    from sadf.data.dataset import make_loader
    from sadf.evaluation.report import generate_report
    from sadf.models.detector import build_model
    from sadf.training.trainer import Trainer

    device = resolve_device(args.device)
    like = Path(args.like)
    if not like.exists():
        raise SystemExit(f"Not found: {like}")
    ref = torch.load(like, map_location="cpu", weights_only=False)

    # ---- settings: copied from the balanced baseline, baseline recipe -------
    cfg = _dotify(ref["config"])        # exact copy, no recipe re-applied
    set_dotted(cfg, "paths.runs", str(Path(args.runs)))
    vocab = list(ref["group_vocab"])
    run_dir = ensure_dir(Path(args.runs) / "baseline")
    group_key = cfg.group.key

    s = _settings(cfg)
    log.info("Settings copied from %s:", like)
    for k, v in s.items():
        log.info("  %-20s %s", k, v)
    if s["loss"] != "bce" or s["balanced_sampler"] or s["adversary"]:
        raise SystemExit("--like is not a BASELINE checkpoint (needs bce loss, no "
                         "balanced sampler, no adversary). Point it at the "
                         "balanced baseline best.pt, not the fair one.")

    # ---- data ---------------------------------------------------------------
    df = _load_manifest(Path(args.manifest), group_key)
    train_df = df[df.split == "train"]
    val_df = df[df.split == "val"]
    test_df = df[df.split == "test"].reset_index(drop=True)
    log.info("Rows: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))
    log.info("TRAIN mix:\n%s",
             pd.crosstab(train_df[group_key], train_df["label"], margins=True).to_string())
    if "is_augmented" in train_df.columns and (train_df["is_augmented"] != 0).any():
        log.warning("TRAIN contains augmented rows; the control should be originals only.")
    if train_df.empty or val_df.empty or test_df.empty:
        raise SystemExit("A split is empty. Rebuild the manifest with build_skewed_baseline.py.")
    missing = [p for p in train_df["path"].head(20) if not Path(p).exists()]
    if missing:
        raise SystemExit(f"Image files not found, e.g. {missing[0]}. Run this from "
                         "the project folder, the same place you ran training before.")

    if args.check:
        log.info("--check: settings and data look usable. Nothing was trained.")
        return

    # ---- train --------------------------------------------------------------
    if not args.eval_only:
        seed_everything(int(cfg.seed))
        model = build_model(cfg, n_groups=len(vocab))
        trainer = Trainer(cfg, model, vocab, run_dir, device)
        log.info("=== TRAIN Swin skewed (baseline recipe) | device=%s ===", device)
        res = trainer.fit(
            make_loader(train_df, cfg, vocab, True,
                        balanced=bool(cfg.train.balanced_sampler)),
            make_loader(val_df, cfg, vocab, False),
        )
        save_json({"model": "swin_skewed", "backbone": cfg.model.backbone,
                   "recipe": "baseline", "like": str(like), "manifest": args.manifest,
                   "groups": vocab, "n_train": len(train_df),
                   "best_selection_score": res["best_selection_score"]},
                  run_dir / "run_meta.json")

    # ---- evaluate on the SAME test set as the balanced model -----------------
    best = run_dir / "best.pt"
    if not best.exists():
        raise SystemExit(f"No checkpoint at {best}. Train first.")
    ck = torch.load(best, map_location=device, weights_only=False)
    rcfg = _dotify(ck["config"])
    model = build_model(rcfg, n_groups=len(ck["group_vocab"]))
    model.load_state_dict(ck["model_state"])
    trainer = Trainer(rcfg, model, ck["group_vocab"], run_dir, device)
    preds = trainer.predict(make_loader(test_df, rcfg, ck["group_vocab"], False))
    scored = test_df.loc[preds["row"].values].reset_index(drop=True)
    scored["score"] = preds["score"].values
    generate_report(
        scored, run_dir / "eval", group_cols=[group_key],
        threshold=float(rcfg.train.threshold),
        min_group_size=int(rcfg.eval.min_group_size),
        n_boot=int(args.n_boot), seed=int(rcfg.seed),
        title="Swin-Tiny skewed (baseline) - test",
    )
    log.info("Done. Eval written to %s", run_dir / "eval")


if __name__ == "__main__":
    main()
