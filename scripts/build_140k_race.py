#!/usr/bin/env python3
r"""Label the RAW 140k images by FairFace race and build a balanced manifest.

140k (FFHQ) images are already face-centred and aligned, so we skip face
detection/harvesting entirely and run FairFace straight on the raw images.
This is far faster than harvesting: one classifier pass, no cropping.

Outputs:
  * a manifest with race labels + splits, balanced across race x label
  * sample folders (data\faces140k\_race_check\<race>\) with ~30 faces each,
    for you to EYEBALL before training — the verification gate.

USAGE — two phases so you can check before committing:

  PHASE 1 (label + sample, for inspection):
    python scripts\build_140k_race.py ^
        --root    data\raw\real_vs_fake\real-vs-fake ^
        --weights models\fairface\res34_fair_align_multi_7_20190809.pt ^
        --phase label

    Then OPEN data\faces140k\_race_check\<race>\ and confirm the faces in
    each folder actually match the race label.

  PHASE 2 (build the balanced manifest, after you've confirmed):
    python scripts\build_140k_race.py ^
        --root    data\raw\real_vs_fake\real-vs-fake ^
        --per-group 3000 ^
        --phase balance

  Then train:
    python scripts\run_models140k.py --train vit   (etc.)
"""
from __future__ import annotations

import argparse
import shutil
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

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RACE7 = ["White", "Black", "Latino_Hispanic", "East Asian",
         "Southeast Asian", "Indian", "Middle Eastern"]
SA_MAP = {"White": "White", "Black": "Black", "Indian": "Indian",
          "Latino_Hispanic": "Other", "East Asian": "Other",
          "Southeast Asian": "Other", "Middle Eastern": "Other"}

WORK = Path("data/faces140k")
LABELS_CSV = WORK / "manifests" / "raw_race_labels.csv"


def load_fairface(weights: str, device: str) -> nn.Module:
    model = torchvision.models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 18)
    state = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def _label_from_path(p: Path) -> int | None:
    parts = {x.lower() for x in p.parts}
    if "fake" in parts:
        return 1
    if "real" in parts:
        return 0
    return None


def _split_from_path(p: Path) -> str:
    parts = {x.lower() for x in p.parts}
    if "test" in parts:
        return "test"
    if "valid" in parts or "val" in parts:
        return "val"
    return "train"


def phase_label(root: Path, weights: str, device: str, sample_check: int) -> None:
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
    log.info("Found %d images under %s", len(imgs), root)
    if not imgs:
        raise SystemExit("No images found — check --root.")

    model = load_fairface(weights, device)
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rows = []
    batch_imgs, batch_meta = [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            pred = torch.argmax(model(x)[:, :7], dim=1).cpu().numpy()
        for meta, pr in zip(batch_meta, pred):
            r7 = RACE7[int(pr)]
            rows.append({**meta, "race7": r7, "race": SA_MAP[r7]})

    for i, p in enumerate(imgs):
        lab = _label_from_path(p)
        if lab is None:
            continue
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        batch_imgs.append(tf(img))
        batch_meta.append({"path": str(p), "label": lab, "split": _split_from_path(p)})
        if len(batch_imgs) >= 64:
            flush(); batch_imgs, batch_meta = [], []
        if (i + 1) % 2000 == 0:
            log.info("  %d/%d", i + 1, len(imgs))
    flush()

    df = pd.DataFrame(rows)
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LABELS_CSV, index=False)
    log.info("Wrote %s (%d labelled)", LABELS_CSV, len(df))
    log.info("Race distribution (race7):\n%s", df["race7"].value_counts().to_string())
    log.info("SA grouping:\n%s", df["race"].value_counts().to_string())

    # sample folders for manual inspection
    check = WORK / "_race_check"
    if check.exists():
        shutil.rmtree(check)
    rng = np.random.default_rng(0)
    for race in ["White", "Black", "Indian", "Other"]:
        sub = df[df["race"] == race]
        if sub.empty:
            continue
        d = check / race
        d.mkdir(parents=True, exist_ok=True)
        picks = sub.iloc[rng.permutation(len(sub))[:sample_check]]
        for j, (_, r) in enumerate(picks.iterrows()):
            try:
                shutil.copy(r["path"], d / f"{j}_{Path(r['path']).name}")
            except Exception:
                pass
    print(f"\n>>> VERIFICATION GATE <<<")
    print(f"Open {check}\\<race>\\ and confirm the faces match the race label.")
    print("If they look right, run PHASE 2 (--phase balance) to build the set.")
    print("If they look wrong, STOP — tell me and we adjust before training.")


def phase_balance(per_group: int, min_per_cell: int, drop_other: bool) -> None:
    if not LABELS_CSV.exists():
        raise SystemExit(f"{LABELS_CSV} missing — run --phase label first.")
    df = pd.read_csv(LABELS_CSV)
    if drop_other:
        df = df[df["race"] != "Other"]

    groups = [g for g in ["White", "Black", "Indian", "Other"] if g in set(df["race"])]
    labels = sorted(df["label"].unique())
    rng = np.random.default_rng(0)

    kept = []
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        cells = {(g, lab): len(sub[(sub["race"] == g) & (sub["label"] == lab)])
                 for g in groups for lab in labels}
        usable = [g for g in groups
                  if min(cells[(g, lab)] for lab in labels) >= min_per_cell]
        dropped = [g for g in groups if g not in usable]
        if dropped:
            log.warning("[%s] dropping (too few, <%d/cell): %s", split, min_per_cell, dropped)
        if not usable:
            continue
        cap = min(min(cells[(g, lab)] for lab in labels) for g in usable)
        cap = min(cap, per_group)
        log.info("[%s] %d groups x %d labels @ %d each", split, len(usable), len(labels), cap)
        for g in usable:
            for lab in labels:
                pool = sub[(sub["race"] == g) & (sub["label"] == lab)]
                kept.append(pool.iloc[rng.permutation(len(pool))[:cap]])

    bal = pd.concat(kept, ignore_index=True)
    # write in the manifest shape the trainer expects: status + skin_tone(=race)
    bal["status"] = "ok"
    bal["skin_tone_ita"] = "n/a"
    bal["skin_tone"] = bal["race"]  # eval groups by skin_tone -> race
    out = WORK / "manifests" / "manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    bal.to_csv(out, index=False)
    log.info("Wrote %s (%d rows)", out, len(bal))
    log.info("Balanced race x label:\n%s",
             pd.crosstab(bal["race"], bal["label"]).to_string())
    log.info("Per split:\n%s", bal["split"].value_counts().to_string())
    print(f"\nBalanced manifest -> {out}")
    print("Now train:  python scripts\\run_models140k.py --train vit  (then xception, swin)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="140k folder (contains train/test/valid)")
    ap.add_argument("--weights", help="FairFace .pt (needed for --phase label)")
    ap.add_argument("--phase", choices=["label", "balance"], required=True)
    ap.add_argument("--per-group", type=int, default=3000)
    ap.add_argument("--min-per-cell", type=int, default=50)
    ap.add_argument("--sample-check", type=int, default=30)
    ap.add_argument("--drop-other", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.phase == "label":
        if not args.root or not args.weights:
            raise SystemExit("--root and --weights required for --phase label")
        if not Path(args.weights).exists():
            raise SystemExit(f"Weights not found: {args.weights}")
        phase_label(Path(args.root), args.weights, device, args.sample_check)
    else:
        phase_balance(args.per_group, args.min_per_cell, args.drop_other)


if __name__ == "__main__":
    main()
