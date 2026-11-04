#!/usr/bin/env python3
r"""Build a SKEWED-baseline training manifest as a control for the balanced one.

WHY
---
To attribute any fairness change to *the dataset*, only the dataset may change.
This builds a control training set with the corpus's NATURAL race mix
(about 90% White once Other is dropped), while holding everything else fixed:

  * same architecture and recipe (train it with train_skewed_swin.py)
  * same number of ORIGINAL training images as the balanced run (13,912)
  * the SAME validation and test rows as the balanced manifest (copied verbatim)

Only the race mix of the TRAIN split differs:

    skewed-trained model   vs   balanced-trained model
    evaluated on the identical balanced test set

USAGE
-----
  python scripts\build_skewed_baseline.py ^
      --pool      data\faces140k\manifests\raw_race_labels.csv ^
      --balanced  data\faces140k\manifests\manifest_aug.csv ^
      --out       data\faces140k\manifests\manifest_skewed.csv

  Use raw_race_labels.csv (140k only). Do NOT use pool_merged.csv: it contains
  the imdb and Stable Diffusion images that make up the cross-generator test.

Then train:

  python scripts\train_skewed_swin.py ^
      --manifest data\faces140k\manifests\manifest_skewed.csv ^
      --like     runs\models140k\swin_aug\baseline\best.pt ^
      --runs     runs\models140k\swin_skewed

NOTES
-----
* Size matching counts ORIGINAL balanced training rows only (is_augmented == 0).
  Use --match-total to match the 24,000 rows including augmented copies.
* The natural mix is measured on the whole 140k pool (Black/Indian/White),
  before val/test rows are removed, so it reflects the corpus itself.
* Real and fake are kept 50/50 within each race so label balance is not a
  second thing that changes.
* If the pool cannot fill a cell, the script says so and caps honestly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RACES = ["Black", "Indian", "White"]
CROSS_GEN_HINTS = ("imdb", "stablediffusion", "stable_diffusion", "sd_wiki")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _norm(p) -> str:
    """Path key that ignores slash direction and letter case (Windows)."""
    return str(p).replace("\\", "/").strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True,
                    help="full FairFace-labelled 140k pool (raw_race_labels.csv)")
    ap.add_argument("--balanced", required=True,
                    help="the balanced manifest whose val/test we reuse")
    ap.add_argument("--out", required=True)
    ap.add_argument("--race-col", default="race")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--match-total", action="store_true",
                    help="match the balanced TRAIN row count including augmented "
                         "rows (default: match originals only)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for p in (args.pool, args.balanced):
        if not Path(p).exists():
            raise SystemExit(f"Not found: {p}")

    pool = pd.read_csv(args.pool)
    bal = pd.read_csv(args.balanced)
    rng = np.random.default_rng(args.seed)

    # ---- guard: the pool must not contain the cross-generator sources -------
    lowered = pool["path"].astype(str).str.lower()
    bad = lowered.apply(lambda s: any(h in s for h in CROSS_GEN_HINTS))
    if bad.any():
        raise SystemExit(
            f"The pool has {int(bad.sum())} imdb / Stable Diffusion rows. Those are "
            "the cross-generator TEST sources and must never be trained on.\n"
            "Use data\\faces140k\\manifests\\raw_race_labels.csv as --pool.")

    # ---- race column names in the two files --------------------------------
    rc = args.race_col
    if rc not in pool.columns:
        if "skin_tone" in pool.columns:
            rc = "skin_tone"
        else:
            raise SystemExit(f"Pool has no '{args.race_col}' or 'skin_tone' column.")
    bal_rc = args.race_col if args.race_col in bal.columns else "skin_tone"
    if bal_rc not in bal.columns:
        raise SystemExit("Balanced manifest has no race/skin_tone column.")

    # ---- 1. the val/test we must reuse verbatim -----------------------------
    held = bal[bal[args.split_col].isin(["val", "test"])].copy()
    if held.empty:
        raise SystemExit("Balanced manifest has no val/test rows to reuse.")
    held_keys = set(held["path"].map(_norm))
    _log(f"Reusing val/test from balanced manifest: "
         f"{(held[args.split_col] == 'val').sum()} val, "
         f"{(held[args.split_col] == 'test').sum()} test")

    # ---- 2. how big should TRAIN be? ----------------------------------------
    bal_train = bal[bal[args.split_col] == "train"]
    if "is_augmented" in bal_train.columns and not args.match_total:
        target_n = int((bal_train["is_augmented"] == 0).sum())
        _log(f"Matching ORIGINAL balanced train size: {target_n} "
             f"(balanced train incl. augmented = {len(bal_train)})")
    else:
        target_n = len(bal_train)
        _log(f"Matching TOTAL balanced train size: {target_n}")

    # ---- 3. natural race mix of the 140k pool (before removing val/test) ----
    usable = pool[pool[rc].isin(RACES)].copy()
    if "status" in usable.columns:
        usable = usable[usable["status"] == "ok"]
    nat = usable[rc].value_counts(normalize=True)
    _log("\nNatural race mix of the pool (Black/Indian/White only):")
    for r in RACES:
        _log(f"  {r:7s} {nat.get(r, 0) * 100:5.1f}%")

    # ---- 4. candidate training pool: everything not held out ----------------
    keys = usable["path"].map(_norm)
    cand = usable[~keys.isin(held_keys)].copy()
    removed = len(usable) - len(cand)
    _log(f"\nCandidate pool: {len(cand)} rows "
         f"({removed} removed because they are in val/test)")
    if removed == 0:
        raise SystemExit(
            "No val/test row was found in the pool, so the paths are written "
            "differently in the two files. Stopping, because otherwise test images "
            "could end up in training. Compare the 'path' column of both CSVs.")
    if removed < len(held):
        _log(f"WARNING: only {removed} of {len(held)} val/test rows matched the "
             "pool. Check that the rest are not in the pool under another path.")

    # ---- 5. per-cell targets at the natural mix, 50/50 real/fake ------------
    labels = sorted(cand[args.label_col].unique())
    want = {(r, lab): int(target_n * nat.get(r, 0) / len(labels))
            for r in RACES for lab in labels}
    # hand the rounding remainder to the largest group so the total is exact
    big = max(RACES, key=lambda r: nat.get(r, 0))
    short = target_n - sum(want.values())
    for i in range(short):
        want[(big, labels[i % len(labels)])] += 1

    picks, shortfalls = [], []
    for (r, lab), w in want.items():
        sub = cand[(cand[rc] == r) & (cand[args.label_col] == lab)]
        take = min(w, len(sub))
        if take < w:
            shortfalls.append((r, lab, w, take))
        if take:
            picks.append(sub.iloc[rng.permutation(len(sub))[:take]])

    if not picks:
        raise SystemExit("Nothing selected. Check the pool's race/label columns.")
    train = pd.concat(picks, ignore_index=True)
    train[args.split_col] = "train"

    if shortfalls:
        _log("\nWARNING: the pool could not fill these cells (capped honestly):")
        for r, lab, w, got in shortfalls:
            _log(f"  {r} label={lab}: wanted {w}, got {got}")

    # ---- 6. assemble, aligning columns with the balanced manifest ------------
    train[bal_rc] = train[rc]
    train["status"] = "ok"            # the trainer drops rows that are not "ok"
    train["is_augmented"] = 0
    if "source_id" in held.columns:
        train["source_id"] = train["path"]
    for col in held.columns:
        if col not in train.columns:
            train[col] = "n/a"
    out = pd.concat([train[held.columns], held], ignore_index=True)

    # final leak check on the assembled file
    tr_keys = set(out.loc[out[args.split_col] == "train", "path"].map(_norm))
    assert not (tr_keys & held_keys), "val/test image found in TRAIN"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # ---- 7. report ----------------------------------------------------------
    _log(f"\nWrote {args.out}  ({len(out)} rows)")
    _log("\nTRAIN composition (skewed control):")
    tr = out[out[args.split_col] == "train"]
    _log(pd.crosstab(tr[bal_rc], tr[args.label_col], margins=True).to_string())
    _log(f"\n  train={len(tr)}  (target was {target_n})")
    _log("\nTEST composition (identical to the balanced run, do not change):")
    te = out[out[args.split_col] == "test"]
    _log(pd.crosstab(te[bal_rc], te[args.label_col]).to_string())

    _log("\nNEXT: train with the SAME settings as the balanced baseline:")
    _log("  python scripts\\train_skewed_swin.py --manifest " + args.out +
         " --like runs\\models140k\\swin_aug\\baseline\\best.pt"
         " --runs runs\\models140k\\swin_skewed")


if __name__ == "__main__":
    main()