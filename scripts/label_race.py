#!/usr/bin/env python3
r"""Label face crops with FairFace race predictions — replacing the ITA proxy.

WHY: ITA measures pixel-brightness, which does not recover perceived race
(a light face in shadow and a dark face in bright light can share an ITA).
FairFace is a ResNet-34 trained on a race-balanced dataset specifically to
reduce subgroup error; its labels are human-perceived race categories, the
proper demographic axis this project needs.

NO DLIB REQUIRED. The official FairFace predict.py uses dlib only to detect
and align faces. Your crops are ALREADY face crops (from the harvest), so we
skip dlib entirely and run the classifier straight on them. Only torch is
needed — which you already have.

SETUP (one time):
  1. Download the pretrained model from the FairFace repo (github.com/dchen236/FairFace
     or joojs/fairface). Get the 7-race multitask model:
        res34_fair_align_multi_7_20190809.pt
     (Google Drive link in the repo README.) Put it anywhere, e.g. models\fairface\.

USAGE:
  python scripts\label_race.py ^
      --manifest data\faces140k\manifests\manifest.csv ^
      --weights  models\fairface\res34_fair_align_multi_7_20190809.pt

  This ADDS columns: race7 (fine 7-class), race (SA-coarse grouping), and
  OVERWRITES skin_tone with the race grouping so the existing eval/compare
  groups by race with no other changes. The original ITA values are preserved
  in a new skin_tone_ita column.

THEN re-evaluate your already-trained models (no retraining needed):
  python scripts\run_models140k.py --eval-all
  python scripts\run_models140k.py --compare
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
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from sadf.utils.common import get_logger  # noqa: E402

log = get_logger()

# FairFace 7-race label order (fixed by the pretrained model).
RACE7 = ["White", "Black", "Latino_Hispanic", "East Asian",
         "Southeast Asian", "Indian", "Middle Eastern"]

# Coarse grouping relevant to the SA context. Black African / White / Indian
# map directly; everything else (rare in this data) goes to 'Other'. The
# Coloured group is NOT recoverable from a race classifier — documented limit.
SA_MAP = {
    "White": "White",
    "Black": "Black",
    "Indian": "Indian",
    "Latino_Hispanic": "Other",
    "East Asian": "Other",
    "Southeast Asian": "Other",
    "Middle Eastern": "Other",
}


def load_fairface(weights: str, device: str) -> nn.Module:
    model = torchvision.models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 18)  # 7 race + 2 gender + 9 age
    state = torch.load(weights, map_location=device, weights_only=False)
    # The checkpoint may be a raw state_dict or wrapped — handle both.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="manifest.csv to relabel")
    ap.add_argument("--weights", required=True, help="FairFace .pt model")
    ap.add_argument("--out", default=None, help="output CSV (default: overwrite --manifest)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not Path(args.weights).exists():
        raise SystemExit(f"Weights not found: {args.weights}\nDownload the "
                         "FairFace 7-race model from the repo (see this file's header).")
    if not Path(args.manifest).exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    log.info("Loading FairFace model on %s ...", device)
    model = load_fairface(args.weights, device)

    # FairFace preprocessing: 224, ImageNet normalisation.
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    df = pd.read_csv(args.manifest)
    paths = df["path"].tolist()
    log.info("Labelling %d crops ...", len(paths))

    race7_out: list[str] = []
    batch_imgs, batch_idx = [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            logits = model(x)[:, :7]
            pred = torch.argmax(logits, dim=1).cpu().numpy()
        for j, p in zip(batch_idx, pred):
            results[j] = RACE7[int(p)]

    results: dict[int, str] = {}
    for i, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
            batch_imgs.append(tf(img)); batch_idx.append(i)
        except Exception:
            results[i] = "unknown"
        if len(batch_imgs) >= args.batch:
            flush(); batch_imgs, batch_idx = [], []
        if (i + 1) % 500 == 0:
            log.info("  %d/%d", i + 1, len(paths))
    flush()

    df["race7"] = [results.get(i, "unknown") for i in range(len(df))]
    df["race"] = df["race7"].map(lambda r: SA_MAP.get(r, "Other"))

    # Preserve ITA, then overwrite skin_tone with race so eval groups by race
    # with no other code changes.
    if "skin_tone_ita" not in df.columns:
        df["skin_tone_ita"] = df.get("skin_tone", pd.Series(["n/a"] * len(df)))
    df["skin_tone"] = df["race"]

    out = args.out or args.manifest
    df.to_csv(out, index=False)

    log.info("Wrote %s", out)
    log.info("Race distribution (race7):\n%s", df["race7"].value_counts().to_string())
    log.info("SA grouping (now in skin_tone col):\n%s",
             df["race"].value_counts().to_string())
    print("\nDone. The skin_tone column now holds race groups; ITA preserved as "
          "skin_tone_ita.\nRe-run eval to get fairness-by-race:\n"
          "  python scripts\\run_models140k.py --eval-all\n"
          "  python scripts\\run_models140k.py --compare")


if __name__ == "__main__":
    main()
