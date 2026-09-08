#!/usr/bin/env python3
r"""Split-then-augment builder for the 140k-only fairness experiment.

ORDER MATTERS. This script splits FIRST, then augments the TRAIN split only.
Test and val stay pristine originals. Augmenting before splitting would put
flipped copies of the same face in both train and test, which inflates
accuracy and invalidates every number downstream.

What it does
------------
1. Filters the pool manifest to source == 140k and the chosen races.
2. Carves balanced TEST and VAL cells (equal n per race x label).
3. Oversamples the remaining TRAIN cells to a common target using
   artifact-safe augmentation, writing new images to disk as PNG.
4. Writes a manifest with split / source_id / is_augmented / aug_op columns.
5. Asserts no source image appears in more than one split.

Artifact safety
---------------
Deepfake detectors key on generator fingerprints in the high-frequency and
compression domains. The usual augmentation set destroys them, so this script
deliberately uses ONLY:

    horizontal flip, rotation +/-10deg, translation <=10%, brightness/contrast +/-15%

and never applies blur, sharpening, noise, rescaling or JPEG recompression.
Augmented images are saved as PNG (lossless): re-saving as JPEG would add a
second compression generation that correlates with augmentation, and since
augmentation concentrates in minority cells, that would become a race-correlated
artifact -- the exact confound this project keeps tripping over.

Hue/saturation jitter is also excluded: it shifts apparent skin tone, which is
the grouping variable.

USAGE
-----
  # 1. See the plan without writing anything
  python scripts\build_augmented_140k.py --manifest data\faces140k\manifests\pool_merged.csv --dry-run

  # 2. Execute
  python scripts\build_augmented_140k.py --manifest data\faces140k\manifests\pool_merged.csv

  # Useful flags
  --target-per-cell 4000     images per race x label cell in TRAIN after augmentation
  --test-per-cell 400        held-out test images per cell (pristine)
  --val-per-cell 300         validation images per cell (pristine)
  --max-aug-factor 3.0       refuse to exceed this multiple of originals
  --no-match-aug-ratio       let White use all-original images (see note below)
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

# --------------------------------------------------------------------------
# column detection -- the manifests in this project have used both 'race' and
# 'skin_tone' for the group column, and 'path' / 'filepath' for the image.
# --------------------------------------------------------------------------

RACE_CANDIDATES = ["race", "skin_tone", "race_label", "group"]
PATH_CANDIDATES = ["path", "filepath", "file", "image_path", "img_path"]
LABEL_CANDIDATES = ["label", "y", "target", "is_fake"]
SOURCE_CANDIDATES = ["source", "dataset", "src"]


def detect(df: pd.DataFrame, candidates: list[str], what: str, override: str | None) -> str:
    if override:
        if override not in df.columns:
            sys.exit(f"ERROR: --{what}-col '{override}' not in manifest. Columns: {list(df.columns)}")
        return override
    for c in candidates:
        if c in df.columns:
            return c
    sys.exit(
        f"ERROR: could not find the {what} column. Tried {candidates}.\n"
        f"       Manifest columns are: {list(df.columns)}\n"
        f"       Pass it explicitly with --{what}-col"
    )


# --------------------------------------------------------------------------
# augmentation -- artifact-safe transforms only
# --------------------------------------------------------------------------


def seeded_rng(source_path: str, copy_index: int, global_seed: int) -> random.Random:
    """Deterministic per-(image, copy) RNG so runs are reproducible."""
    key = f"{source_path}|{copy_index}|{global_seed}".encode()
    digest = hashlib.sha256(key).hexdigest()[:16]
    return random.Random(int(digest, 16))


def augment(img: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    """Apply the safe transform set. Returns (image, human-readable op string)."""
    ops: list[str] = []

    # horizontal flip -- geometry only, no resampling, artifact-preserving
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        ops.append("hflip")

    # rotation +/-10 degrees, bicubic, reflect-style fill via expand=False
    angle = rng.uniform(-10.0, 10.0)
    if abs(angle) > 0.5:
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
        ops.append(f"rot{angle:+.1f}")

    # translation up to 10% of each dimension
    w, h = img.size
    dx = int(rng.uniform(-0.10, 0.10) * w)
    dy = int(rng.uniform(-0.10, 0.10) * h)
    if dx or dy:
        img = img.transform(
            (w, h), Image.AFFINE, (1, 0, -dx, 0, 1, -dy), resample=Image.BICUBIC
        )
        ops.append(f"shift{dx:+d},{dy:+d}")

    # brightness / contrast +/-15%
    b = rng.uniform(0.85, 1.15)
    img = ImageEnhance.Brightness(img).enhance(b)
    ops.append(f"bright{b:.2f}")

    c = rng.uniform(0.85, 1.15)
    img = ImageEnhance.Contrast(img).enhance(c)
    ops.append(f"contrast{c:.2f}")

    return img, "|".join(ops)


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def build_plan(
    df: pd.DataFrame,
    race_col: str,
    label_col: str,
    races: list[str],
    test_n: int,
    val_n: int,
    target: int,
    max_factor: float,
    match_ratio: bool,
) -> tuple[pd.DataFrame, int]:
    """Work out per-cell test/val/train/augment counts. Returns (plan, effective_target)."""
    rows = []
    for race in races:
        for label in (0, 1):
            avail = len(df[(df[race_col] == race) & (df[label_col] == label)])
            rows.append({"race": race, "label": label, "available": avail})
    plan = pd.DataFrame(rows)

    held = test_n + val_n
    plan["train_orig"] = plan["available"] - held

    if (plan["train_orig"] <= 0).any():
        bad = plan[plan["train_orig"] <= 0]
        sys.exit(
            f"ERROR: these cells cannot supply {held} held-out images:\n{bad}\n"
            f"       Lower --test-per-cell / --val-per-cell."
        )

    # cap the target at what the scarcest cell can reach within max_factor
    reachable = int((plan["train_orig"] * max_factor).min())
    effective = min(target, reachable)

    plan["train_final"] = effective
    # how many originals we actually draw from (subsample majority cells)
    if match_ratio == "global":
        # every cell uses the SAME augmentation ratio, so "was augmented" carries
        # no information about race OR label. Safest, but forces every cell down
        # to the scarcest cell's original count -- costs majority-class diversity.
        floor = plan["train_orig"].min()
        plan["orig_used"] = np.minimum(plan["train_orig"], floor)
    elif match_ratio == "within-race":
        # real and fake matched WITHIN each race, so augmentation can never act
        # as a label shortcut. Ratios still differ ACROSS races, which can make
        # some groups slightly harder -- note it in limitations.
        floors = plan.groupby("race")["train_orig"].transform("min")
        plan["orig_used"] = np.minimum(plan["train_orig"], np.minimum(floors, effective))
    else:  # "none"
        plan["orig_used"] = np.minimum(plan["train_orig"], effective)

    plan["n_augmented"] = (plan["train_final"] - plan["orig_used"]).clip(lower=0)
    plan["aug_factor"] = plan["train_final"] / plan["orig_used"]
    plan["test"] = test_n
    plan["val"] = val_n
    return plan, effective


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="pool manifest CSV (all sources)")
    ap.add_argument("--out-manifest", default="data/faces140k/manifests/manifest_aug.csv")
    ap.add_argument("--aug-dir", default="data/faces140k/augmented")
    ap.add_argument("--races", default="Black,Indian,White")
    ap.add_argument("--source-filter", default="140k", help="keep only this source; '' to disable")
    ap.add_argument("--test-per-cell", type=int, default=400)
    ap.add_argument("--val-per-cell", type=int, default=300)
    ap.add_argument("--target-per-cell", type=int, default=4000)
    ap.add_argument("--max-aug-factor", type=float, default=3.0)
    ap.add_argument("--aug-balance", dest="match_ratio", default="within-race",
                    choices=["global", "within-race", "none"],
                    help="how evenly augmentation is spread. 'global' = identical "
                         "ratio everywhere (safest, fewest unique images). "
                         "'within-race' = real/fake matched inside each race so "
                         "augmentation can't become a label shortcut (default). "
                         "'none' = maximise unique images, accept the correlation.")
    ap.add_argument("--race-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--path-col", default=None)
    ap.add_argument("--source-col", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = ap.parse_args()

    rng_master = np.random.default_rng(args.seed)

    df = pd.read_csv(args.manifest)
    print(f"loaded {len(df):,} rows from {args.manifest}")
    print(f"columns: {list(df.columns)}\n")

    race_col = detect(df, RACE_CANDIDATES, "race", args.race_col)
    label_col = detect(df, LABEL_CANDIDATES, "label", args.label_col)
    path_col = detect(df, PATH_CANDIDATES, "path", args.path_col)
    print(f"using columns -> race:'{race_col}'  label:'{label_col}'  path:'{path_col}'")

    # filter to one source so real and fake share a generation pipeline
    if args.source_filter:
        source_col = detect(df, SOURCE_CANDIDATES, "source", args.source_col)
        before = len(df)
        df = df[df[source_col] == args.source_filter].copy()
        print(f"source filter '{args.source_filter}': {before:,} -> {len(df):,} rows")

    races = [r.strip() for r in args.races.split(",")]
    df = df[df[race_col].isin(races)].copy()
    df[label_col] = df[label_col].astype(int)
    print(f"races {races}: {len(df):,} rows\n")

    plan, effective = build_plan(
        df, race_col, label_col, races,
        args.test_per_cell, args.val_per_cell,
        args.target_per_cell, args.max_aug_factor, args.match_ratio,
    )

    print("=" * 78)
    print("PLAN")
    print("=" * 78)
    print(plan.to_string(index=False))
    print()
    if effective < args.target_per_cell:
        print(f"NOTE: target lowered {args.target_per_cell} -> {effective} "
              f"(capped by --max-aug-factor {args.max_aug_factor})")
    n_train = int(plan["train_final"].sum())
    n_aug = int(plan["n_augmented"].sum())
    print(f"TRAIN {n_train:,}  ({n_aug:,} augmented, {n_train - n_aug:,} original)")
    print(f"VAL   {int(plan['val'].sum()):,}  (all original)")
    print(f"TEST  {int(plan['test'].sum()):,}  (all original, balanced)")
    print(f"aug-balance mode: {args.match_ratio}")
    print()

    if args.dry_run:
        print("dry run -- nothing written. Re-run without --dry-run to execute.")
        return

    aug_dir = Path(args.aug_dir)
    aug_dir.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict] = []
    n_written = 0
    n_failed = 0

    for _, prow in plan.iterrows():
        race, label = prow["race"], int(prow["label"])
        cell = df[(df[race_col] == race) & (df[label_col] == label)]
        idx = rng_master.permutation(len(cell))
        cell = cell.iloc[idx].reset_index(drop=True)

        t, v = int(prow["test"]), int(prow["val"])
        test_part = cell.iloc[:t]
        val_part = cell.iloc[t:t + v]
        train_pool = cell.iloc[t + v:]
        train_part = train_pool.iloc[:int(prow["orig_used"])]

        # ---- pristine splits, originals only -------------------------------
        for split, part in (("test", test_part), ("val", val_part), ("train", train_part)):
            for _, r in part.iterrows():
                out_rows.append({
                    "path": r[path_col],
                    "label": label,
                    "skin_tone": race,
                    "split": split,
                    "source_id": r[path_col],
                    "is_augmented": 0,
                    "aug_op": "",
                })

        # ---- augmented copies, TRAIN ONLY ----------------------------------
        need = int(prow["n_augmented"])
        if need <= 0:
            continue

        cell_dir = aug_dir / race / f"label{label}"
        cell_dir.mkdir(parents=True, exist_ok=True)

        made, copy_index = 0, 0
        srcs = train_part[path_col].tolist()
        while made < need:
            src = srcs[made % len(srcs)]
            copy_index = made // len(srcs)
            try:
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    rng = seeded_rng(str(src), copy_index, args.seed)
                    aug_im, ops = augment(im, rng)
                    stem = Path(src).stem
                    out_path = cell_dir / f"{stem}_aug{copy_index}.png"
                    aug_im.save(out_path, format="PNG", compress_level=1)
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                if n_failed <= 5:
                    print(f"  WARN could not augment {src}: {exc}")
                made += 1
                continue

            out_rows.append({
                "path": str(out_path),
                "label": label,
                "skin_tone": race,
                "split": "train",
                "source_id": src,
                "is_augmented": 1,
                "aug_op": ops,
            })
            made += 1
            n_written += 1
            if n_written % 500 == 0:
                print(f"  wrote {n_written:,} augmented images...")

    out = pd.DataFrame(out_rows)

    # ---- leakage assertion --------------------------------------------------
    per_source = out.groupby("source_id")["split"].nunique()
    leaked = per_source[per_source > 1]
    if len(leaked):
        print("\nFATAL: source images found in more than one split:")
        print(leaked.head(20))
        sys.exit("Aborting -- this would invalidate the results.")
    print("\nleakage check PASSED: every source image lives in exactly one split")

    # augmentation must never touch val/test
    assert out[out["split"] != "train"]["is_augmented"].sum() == 0, \
        "FATAL: augmented rows found outside train"
    print("purity check PASSED: val and test contain originals only")

    out_path = Path(args.out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(pd.crosstab([out["skin_tone"], out["label"]], out["split"]))
    print(f"\naugmented images written: {n_written:,}   failures: {n_failed}")
    print(f"manifest: {out_path}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
