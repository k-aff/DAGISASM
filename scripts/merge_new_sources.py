#!/usr/bin/env python3
r"""Label the AI-Face test sources with FairFace and merge them with the 140k labels.

Adds the Stable Diffusion 1.5 fakes (label 1) and the IMDb-WIKI real faces
(label 0) to raw_race_labels.csv, using the same FairFace model and settings,
and tags every row with its source (140k, sd_wiki or imdb):

    python scripts/merge_new_sources.py \
        --pool     data/faces140k/manifests/raw_race_labels.csv \
        --sd-dir   data/raw/StableDiffusion1.5_wiki_processed_process \
        --imdb-dir data/raw/imdb_wiki \
        --weights  models/fairface/res34_fair_align_multi_7_20190809.pt \
        --out      data/faces140k/manifests/pool_merged.csv

The split column written here is not used later: build_augmented_140k.py
re-splits the 140k rows, and eval_cross_generator.py uses every sd_wiki and
imdb row.
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

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RACE7 = ["White", "Black", "Latino_Hispanic", "East Asian",
         "Southeast Asian", "Indian", "Middle Eastern"]
SA_MAP = {"White": "White", "Black": "Black", "Indian": "Indian",
          "Latino_Hispanic": "Other", "East Asian": "Other",
          "Southeast Asian": "Other", "Middle Eastern": "Other"}


def load_fairface(weights: str, device: str) -> nn.Module:
    m = torchvision.models.resnet34(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 18)  # 7 race + 2 gender + 9 age
    state = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    m.load_state_dict(state)
    m.to(device).eval()
    return m


def label_folder(root: Path, label: int, model, tf, device) -> pd.DataFrame:
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
    log.info("  %s: %d images (label=%d)", root.name, len(imgs), label)
    rows, batch, meta = [], [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        with torch.no_grad():
            pred = torch.argmax(model(x)[:, :7], dim=1).cpu().numpy()
        for m_, pr in zip(meta, pred):
            r7 = RACE7[int(pr)]
            rows.append({**m_, "race7": r7, "race": SA_MAP[r7],
                         "skin_tone": SA_MAP[r7], "skin_tone_ita": "n/a",
                         "label": label, "status": "ok"})

    for i, p in enumerate(imgs):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        batch.append(tf(img)); meta.append({"path": str(p)})
        if len(batch) >= 64:
            flush(); batch, meta = [], []
        if (i + 1) % 2000 == 0:
            log.info("    %d/%d", i + 1, len(imgs))
    flush()
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="existing 140k labeled pool CSV")
    ap.add_argument("--sd-dir", required=True, help="Stable-Diffusion fakes folder")
    ap.add_argument("--imdb-dir", required=True, help="imdb reals folder")
    ap.add_argument("--weights", required=True, help="FairFace .pt")
    ap.add_argument("--out", required=True, help="merged pool CSV to write")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    for pth in (args.pool, args.weights):
        if not Path(pth).exists():
            raise SystemExit(f"Not found: {pth}")

    log.info("Loading FairFace on %s ...", device)
    model = load_fairface(args.weights, device)
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    log.info("Labelling SD fakes ...")
    sd = label_folder(Path(args.sd_dir), 1, model, tf, device)   # fakes
    log.info("Labelling imdb reals ...")
    imdb = label_folder(Path(args.imdb_dir), 0, model, tf, device)  # reals

    pool = pd.read_csv(args.pool)
    keep = ["path", "label", "race", "race7", "skin_tone", "skin_tone_ita", "status"]
    for df in (pool, sd, imdb):
        for c in keep:
            if c not in df.columns:
                df[c] = "n/a" if c not in ("label",) else 0
    pool["source"] = pool.get("source", "140k")
    sd["source"] = "sd_wiki"
    imdb["source"] = "imdb"

    merged = pd.concat([pool[keep + ["source"]], sd[keep + ["source"]],
                        imdb[keep + ["source"]]], ignore_index=True)

    # 70/15/15 split within (race, label, source); kept so that every row has a
    # split value, the later steps do not rely on it
    rng = np.random.default_rng(0)
    merged["split"] = "train"
    for _, idx in merged.groupby(["race", "label", "source"]).groups.items():
        idx = np.array(list(idx)); rng.shuffle(idx)
        n = len(idx)
        n_va = max(1, round(n * 0.15)) if n >= 3 else 0
        n_te = max(1, round(n * 0.15)) if n >= 3 else 0
        if n_va + n_te >= n:
            n_va = n_te = max(0, (n - 1) // 3)
        merged.loc[idx[:n_va], "split"] = "val"
        merged.loc[idx[n_va:n_va + n_te], "split"] = "test"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    log.info("Wrote merged pool -> %s (%d rows)", args.out, len(merged))
    log.info("Race x label in merged pool:\n%s",
             pd.crosstab(merged["race"], merged["label"]).to_string())
    log.info("By source:\n%s", merged["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
