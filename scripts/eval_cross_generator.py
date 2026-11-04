#!/usr/bin/env python3
r"""Score a trained detector on the cross-source/cross-generator test.

The test sources come from AI-Face and were never used in training:

    imdb     IMDb-WIKI real faces      (all label 0) -> per-race FPR
    sd_wiki  Stable Diffusion fakes    (all label 1) -> per-race FNR

Each source holds one label, so the error rate is computed within each source
and races are compared only inside a source. The combined/ folder pools both
sources; its AUC is not meaningful because the source alone predicts the label,
and its accuracy depends on the 6 499 real / 14 177 fake mix of this set.

IMDb-WIKI has many photos of the same celebrity, so --cap-per-identity keeps
at most N images per IMDb person ID (5 in the report).

    python scripts/eval_cross_generator.py \
        --checkpoint runs/models140k/swin_aug/baseline/best.pt \
        --manifest data/faces140k/manifests/pool_merged.csv \
        --out runs/models140k/swin_aug/baseline/eval_cross \
        --cap-per-identity 5
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
    """Read the IMDb person ID from paths such as .../nm0000123_rm..._face1.jpg."""
    out = df.copy()
    out["identity"] = out["path"].astype(str).str.extract(pattern, expand=False)
    # rows without an ID (the Stable Diffusion images) count as their own identity
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
    """Run inference in the same way as eval_one() in run_models140k.py."""
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

    df = pd.read_csv(args.manifest, low_memory=False)
    if "source" not in df.columns:
        sys.exit(f"ERROR: manifest has no 'source' column. Columns: {list(df.columns)}")

    group_col = args.group_col
    if group_col is None:
        group_col = "race" if "race" in df.columns else "skin_tone"
    if group_col not in df.columns:
        sys.exit(f"ERROR: group column '{group_col}' not in manifest.")

    # the loader groups by cfg.group.key (skin_tone), so copy the race labels into it
    df["skin_tone"] = df[group_col]

    races = [r.strip() for r in args.races.split(",")]
    df = df[df[group_col].isin(races)].copy()
    if "status" in df.columns:
        df = df[df["status"] == "ok"]

    df = add_identity(df, re.compile(args.identity_regex))

    summary: dict = {"checkpoint": str(ckpt_path), "threshold": threshold,
                     "group_col": group_col, "within_source": {}}
    combined_parts = []

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

    # pooled view over both sources (descriptive only, see module docstring)
    if len(combined_parts) >= 2:
        comb = pd.concat(combined_parts, ignore_index=True)
        if comb["label"].nunique() >= 2:
            cdir = ensure_dir(out_root / "combined")
            generate_report(
                comb, cdir, group_cols=["skin_tone"], threshold=threshold,
                min_group_size=30, n_boot=args.n_boot, seed=args.seed,
                title="Cross-source/cross-generator test, both sources pooled",
            )
            log.info("Pooled view -> %s", cdir)

    save_json(summary, out_root / "cross_generator_summary.json")
    log.info("Wrote %s", out_root / "cross_generator_summary.json")


if __name__ == "__main__":
    main()
