#!/usr/bin/env python3
r"""Select a demographically-balanced subset from a race-labelled manifest.

After label_race.py has written race labels into the manifest (in the
skin_tone column), this picks a balanced subset: within each split it caps
every (race x label) cell to the size of the SMALLEST such cell, so the
training set has equal representation across race groups AND equal real/fake
within each group. The cap is the honest ceiling — set by the scarcest group.

USAGE:
  python scripts\balance_by_group.py ^
      --manifest data\faces140k\manifests\manifest.csv ^
      --out      data\faces140k\manifests\manifest_balanced.csv

  Then train on the balanced manifest by pointing the config's manifest path
  at it (or copy it over manifest.csv).

OPTIONS:
  --group-col skin_tone   which column holds the group (default skin_tone,
                          which label_race.py overwrites with race)
  --min-per-cell 50       skip groups whose smallest cell is below this (too
                          few to be meaningful); they're dropped with a warning
  --drop-other            exclude the 'Other' race bucket from the balanced set
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
    print("ERROR: could not find the 'sadf' package."); sys.exit(1)


_bootstrap()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sadf.utils.common import get_logger  # noqa: E402

log = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-col", default="skin_tone")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--min-per-cell", type=int, default=50)
    ap.add_argument("--drop-other", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if args.group_col not in df.columns:
        raise SystemExit(f"No '{args.group_col}' column. Run label_race.py first.")
    df = df[df.get("status", "ok") == "ok"] if "status" in df.columns else df

    if args.drop_other:
        df = df[df[args.group_col] != "Other"]
    df = df[df[args.group_col] != "unknown"]

    rng = np.random.default_rng(args.seed)
    groups = sorted(df[args.group_col].unique())
    labels = sorted(df[args.label_col].unique())
    splits = sorted(df[args.split_col].unique()) if args.split_col in df.columns else ["all"]

    log.info("Groups found: %s", groups)
    log.info("Pre-balance distribution (%s x %s):", args.group_col, args.label_col)
    log.info("\n%s", pd.crosstab(df[args.group_col], df[args.label_col]).to_string())

    kept_parts = []
    for split in splits:
        sub = df[df[args.split_col] == split] if args.split_col in df.columns else df
        # cell sizes for this split
        cells = {}
        for g in groups:
            for lab in labels:
                n = len(sub[(sub[args.group_col] == g) & (sub[args.label_col] == lab)])
                cells[(g, lab)] = n
        # drop groups whose smallest cell is too small to be meaningful
        usable_groups = [g for g in groups
                         if min(cells[(g, lab)] for lab in labels) >= args.min_per_cell]
        dropped = [g for g in groups if g not in usable_groups]
        if dropped:
            log.warning("[%s] dropping groups (too few samples, < %d/cell): %s",
                        split, args.min_per_cell, dropped)
        if not usable_groups:
            log.warning("[%s] no usable groups — skipping split.", split)
            continue
        # cap = smallest cell across all usable (group,label) pairs in this split
        cap = min(cells[(g, lab)] for g in usable_groups for lab in labels)
        log.info("[%s] balancing %d groups x %d labels at %d each (=%d rows).",
                 split, len(usable_groups), len(labels), cap, cap*len(usable_groups)*len(labels))
        for g in usable_groups:
            for lab in labels:
                pool = sub[(sub[args.group_col] == g) & (sub[args.label_col] == lab)]
                pick = pool.iloc[rng.permutation(len(pool))[:cap]]
                kept_parts.append(pick)

    if not kept_parts:
        raise SystemExit("Nothing kept — every group was below --min-per-cell. "
                         "Lower --min-per-cell or harvest a bigger pool.")
    balanced = pd.concat(kept_parts, ignore_index=True)
    balanced.to_csv(args.out, index=False)

    log.info("Wrote %s (%d rows)", args.out, len(balanced))
    log.info("Balanced distribution:")
    log.info("\n%s", pd.crosstab([balanced[args.split_col]] if args.split_col in balanced
             else [["all"]*len(balanced)], [balanced[args.group_col], balanced[args.label_col]]
             ).to_string() if args.split_col in balanced else
             pd.crosstab(balanced[args.group_col], balanced[args.label_col]).to_string())
    print(f"\nBalanced manifest -> {args.out}")
    print("Point your training config's manifest at this file (or copy over "
          "manifest.csv), update group.skin_tone_categories to the race groups, "
          "then retrain.")


if __name__ == "__main__":
    main()
