#!/usr/bin/env python3
r"""Cross-generator evaluation: score held-out sources with a trained detector.

RQ2. The model was trained only on 140k (FFHQ reals vs StyleGAN fakes). This
scores it on sources it has never seen:

    imdb     real celebrity photos   (all label 0)
    sd_wiki  Stable Diffusion faces  (all label 1)

Neither was in training, so nothing here needs retraining -- inference only.

TWO VIEWS, AND THE DIFFERENCE MATTERS
-------------------------------------
1. WITHIN-SOURCE (the clean one).
   Each source carries a single label, so AUC is undefined and the metric is
   an error rate:
       imdb    -> per-race FPR: real faces wrongly called fake
       sd_wiki -> per-race FNR: fakes missed
   Races are directly comparable inside a source. Comparing imdb's FPR to
   sd_wiki's FNR is NOT meaningful -- different sources, different task.

2. COMBINED (AUC view, caveat attached).
   Pooling both sources gives two classes, so AUC works. The absolute AUC is
   inflated because source predicts label perfectly -- a model could separate
   them on JPEG signature alone. But the GAP BETWEEN RACES stays interpretable:
   every race has the same source structure (all its reals from imdb, all its
   fakes from sd_wiki), so the confound is uniform across groups and cancels
   in the comparison. Report the gap, not the absolute number.

IDENTITY DUPLICATION
--------------------
IMDB-WIKI is celebrity-sourced: many images per person. Without a cap, a few
heavily-photographed actors dominate their race group and the "per-image"
error rate really measures those few faces. --cap-per-identity limits this.

USAGE
-----
  python scripts\eval_cross_generator.py ^
      --checkpoint runs\models140k\swin_aug\baseline\best.pt ^
      --manifest data\faces140k\manifests\pool_merged.csv ^
      --out runs\models140k\swin_aug\baseline\eval_cross

  # keep at most 5 images per celebrity
  ... --cap-per-identity 5
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _bootstrap() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        for cand in (parent / "src", parent):
            if (cand / "sadf" / "__init__.py").exists():
                sys.path.insert(0, str(cand))
                return
    raise ImportError("Could not locate the `sadf` package.")


_bootstrap()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from sadf.config import _dotify  # noqa: E402
from sadf.data.dataset import make_loader  # noqa: E402
from sadf.evaluation.report import generate_report  # noqa: E402
from sadf.models.detector import build_model  # noqa: E402
from sadf.training.trainer import Trainer  # noqa: E402
from sadf.utils.common import (  # noqa: E402
    ensure_dir, get_logger, resolve_device, save_json,
)

log = get_logger()

IDENTITY_RE = re.compile(r"(nm\d{5,})")


def add_identity(df: pd.DataFrame, pattern: re.Pattern) -> pd.DataFrame:
    """Recover the person ID from IMDB-WIKI style paths (…/nm0000123_…)."""
    out = df.copy()
    out["identity"] = out["path"].astype(str).str.extract(pattern, expand=False)
    # rows with no match get their own filename as a unique identity
    missing = out["identity"].isna()
    out.loc[missing, "identity"] = out.loc[missing, "path"].astype(str)
    return out


def cap_identities(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Keep at most `cap` images per identity, sampled without replacement."""
    return (
        df.sample(frac=1.0, random_state=seed)
        .groupby("identity", group_keys=False)
        .head(cap)
        .reset_index(drop=True)
    )


def score(df: pd.DataFrame, model, rcfg, vocab, run_dir: Path, device: str) -> pd.DataFrame:
    """Run inference. Mirrors eval_one() in run_models140k.py."""
    df = df.reset_index(drop=True)
    trainer = Trainer(rcfg, model, vocab, run_dir, device)
    preds = trainer.predict(make_loader(df, rcfg, vocab, False))
    scored = df.loc[preds["row"].values].reset_index(drop=True)
    scored["score"] = preds["score"].values
    return scored


def error_rate_table(
    scored: pd.DataFrame, group_col: str, threshold: float,
    n_boot: int, seed: int,
) -> pd.DataFrame:
    """Per-group error rate with a bootstrap CI. Works with a single label."""
    rng = np.random.default_rng(seed)
    rows = []
    for g, part in scored.groupby(group_col):
        y = part["label"].to_numpy()
        pred = (part["score"].to_numpy() >= threshold).astype(int)
        wrong = (pred != y).astype(float)

        # name the error for what it is, given the label present
        labels = set(np.unique(y).tolist())
        kind = "FPR" if labels == {0} else "FNR" if labels == {1} else "error"

        boot = [
            wrong[rng.integers(0, len(wrong), len(wrong))].mean()
            for _ in range(n_boot)
        ] if len(wrong) else []
        lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))

        rows.append({
            "group": g,
            "n": len(part),
            "n_identities": part["identity"].nunique() if "identity" in part else np.nan,
            "metric": kind,
            "rate": wrong.mean() if len(wrong) else np.nan,
            "ci_lo": lo,
            "ci_hi": hi,
        })
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True, help="pool manifest with a `source` column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="imdb,sd_wiki")
    ap.add_argument("--group-col", default=None,
                    help="group column in the manifest (auto: race, then skin_tone)")
    ap.add_argument("--races", default="Black,Indian,White")
    ap.add_argument("--cap-per-identity", type=int, default=0,
                    help="max images per person (0 = no cap). Recommended for imdb.")
    ap.add_argument("--identity-regex", default=r"(nm\d{5,})")
    ap.add_argument("--max-per-source", type=int, default=0,
                    help="subsample each source for a faster run (0 = all)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = resolve_device(args.device)
    out_root = ensure_dir(args.out)

    # ---- model ------------------------------------------------------------
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"ERROR: checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    rcfg = _dotify(ckpt["config"])
    vocab = ckpt["group_vocab"]
    model = build_model(rcfg, n_groups=len(vocab))
    model.load_state_dict(ckpt["model_state"])
    threshold = args.threshold if args.threshold is not None else float(rcfg.train.threshold)
    log.info("Loaded %s | groups=%s | threshold=%.2f", ckpt_path, vocab, threshold)

    # ---- data -------------------------------------------------------------
    df = pd.read_csv(args.manifest, low_memory=False)
    if "source" not in df.columns:
        sys.exit(f"ERROR: manifest has no 'source' column. Columns: {list(df.columns)}")

    group_col = args.group_col
    if group_col is None:
        group_col = "race" if "race" in df.columns else "skin_tone"
    if group_col not in df.columns:
        sys.exit(f"ERROR: group column '{group_col}' not in manifest.")

    # the model's loader reads cfg.group.key (skin_tone); make sure it holds
    # the same values the checkpoint's vocab was built from
    df["skin_tone"] = df[group_col]

    races = [r.strip() for r in args.races.split(",")]
    df = df[df[group_col].isin(races)].copy()
    if "status" in df.columns:
        df = df[df["status"] == "ok"]

    df = add_identity(df, re.compile(args.identity_regex))

    summary: dict = {"checkpoint": str(ckpt_path), "threshold": threshold,
                     "group_col": group_col, "within_source": {}}
    combined_parts = []

    # ---- within-source, one report each ------------------------------------
    for src in [s.strip() for s in args.sources.split(",")]:
        part = df[df["source"] == src].copy()
        if part.empty:
            log.warning("No rows for source '%s' -- skipping", src)
            continue

        n_before, ids_before = len(part), part["identity"].nunique()
        if args.cap_per_identity > 0:
            part = cap_identities(part, args.cap_per_identity, args.seed)
        if args.max_per_source > 0 and len(part) > args.max_per_source:
            part = part.sample(args.max_per_source, random_state=args.seed)

        log.info("%s: %d rows / %d identities -> %d rows / %d identities",
                 src, n_before, ids_before, len(part), part["identity"].nunique())
        log.info("%s: labels present = %s", src, sorted(part["label"].unique().tolist()))

        scored = score(part, model, rcfg, vocab, out_root, device)
        combined_parts.append(scored)

        sdir = ensure_dir(out_root / src)
        scored.to_csv(sdir / "scores.csv", index=False)

        tbl = error_rate_table(scored, "skin_tone", threshold, args.n_boot, args.seed)
        tbl.to_csv(sdir / "per_group_error.csv", index=False)

        spread = float(tbl["rate"].max() - tbl["rate"].min())
        summary["within_source"][src] = {
            "n": int(len(scored)),
            "n_identities": int(scored["identity"].nunique()),
            "metric": tbl["metric"].iloc[0],
            "overall_rate": float((
                (scored["score"] >= threshold).astype(int) != scored["label"]
            ).mean()),
            "per_group": tbl.to_dict(orient="records"),
            "gap": spread,
        }

        print(f"\n{'=' * 70}\n{src}  ({tbl['metric'].iloc[0]} by race)\n{'=' * 70}")
        print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"gap (max-min): {spread:.4f}")

    # ---- combined: AUC view -------------------------------------------------
    if len(combined_parts) >= 2:
        comb = pd.concat(combined_parts, ignore_index=True)
        if comb["label"].nunique() >= 2:
            cdir = ensure_dir(out_root / "combined")
            generate_report(
                comb, cdir, group_cols=["skin_tone"], threshold=threshold,
                min_group_size=30, n_boot=args.n_boot, seed=args.seed,
                title="Cross-generator (imdb reals + sd_wiki fakes) — AUC view",
            )
            print(f"\n{'=' * 70}\ncombined AUC view -> {cdir}\n{'=' * 70}")
            print("NOTE: absolute AUC here is inflated -- source predicts label.")
            print("      The per-race GAP is the interpretable number, because")
            print("      every race has the same source structure.")

    save_json(summary, out_root / "cross_generator_summary.json")
    log.info("Wrote %s", out_root / "cross_generator_summary.json")


if __name__ == "__main__":
    main()
