#!/usr/bin/env python3
r"""Scan a folder of downloaded AI-Face Stable-Diffusion images and build a
manifest ready for FairFace labelling.

Use this for the DIVERSITY CHECK: it finds every image under the given folder
(recursing into numbered subfolders), writes a manifest, and you then run
label_race.py on it to see the race distribution before committing to the full
merge + retrain.

USAGE:
  python scripts\build_sd_manifest.py ^
      --dir  "data\raw\StableDiffusion1.5_wiki_processed_process" ^
      --out  "data\raw\sd_check_manifest.csv" ^
      --sample 2000

  --sample N   only keep a random N images (for a quick check). Omit to take all.

Then:
  python scripts\label_race.py --manifest data\raw\sd_check_manifest.csv ^
      --weights models\fairface\res34_fair_align_multi_7_20190809.pt

  -> read the race distribution it prints. Enough Black/Indian? proceed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="folder containing the SD images")
    ap.add_argument("--out", required=True, help="manifest CSV to write")
    ap.add_argument("--sample", type=int, default=None,
                    help="optional: keep only N random images for a quick check")
    ap.add_argument("--label", type=int, default=1,
                    help="1=fake (SD-generated), 0=real. Default 1.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists():
        raise SystemExit(f"Folder not found: {root}")

    imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
    print(f"Found {len(imgs)} images under {root}")
    if not imgs:
        raise SystemExit("No images found — check the --dir path and its contents.")

    if args.sample and len(imgs) > args.sample:
        rng = np.random.default_rng(args.seed)
        imgs = [imgs[i] for i in rng.permutation(len(imgs))[:args.sample]]
        print(f"Sampled {len(imgs)} for the check.")

    df = pd.DataFrame({
        "path": [str(p) for p in imgs],
        "label": args.label,
        "split": "test",      # arbitrary; the check only needs labelling to run
        "skin_tone": "unknown",
        "status": "ok",
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote manifest -> {args.out}  ({len(df)} rows)")
    print("\nNEXT: FairFace-label it to check diversity:")
    print(f"  python scripts\\label_race.py --manifest {args.out} \\")
    print("      --weights models\\fairface\\res34_fair_align_multi_7_20190809.pt")


if __name__ == "__main__":
    main()
