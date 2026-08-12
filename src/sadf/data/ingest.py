from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def infer_label(rel_path: str, rules: list[dict]) -> int | None:
    """Infer a label from path tokens; first matching rule wins.

    The path is split into alphanumeric tokens ("train/AI_images/x.jpg"
    -> ["train", "ai", "images", "x", "jpg"]). A pattern matches when it
    equals a token, or — for patterns of 4+ characters — appears inside
    one (so "fake" matches "deepfakes" but "ai" can never match "train").
    """
    tokens = [t for t in re.split(r"[^a-z0-9]+", rel_path.lower()) if t]
    for rule in rules:
        pat = str(rule["pattern"]).lower()
        for tok in tokens:
            if tok == pat or (len(pat) >= 4 and pat in tok):
                return int(rule["label"])
    return None


def scan_dataset(root: str | Path, rules: list[dict]) -> pd.DataFrame:
    """Return DataFrame(orig_path, rel_path, label) for all images under root."""
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = str(p.relative_to(root))
        rows.append(
            {"orig_path": str(p), "rel_path": rel, "label": infer_label(rel, rules)}
        )
    return pd.DataFrame(rows, columns=["orig_path", "rel_path", "label"])


def inspect_tree(root: str | Path, rules: list[dict], max_dirs: int = 40) -> str:
    """Human-readable summary of directory layout and label inference."""
    root = Path(root)
    df = scan_dataset(root, rules)
    if df.empty:
        return f"{root}: no images found."
    dir_counts = Counter(str(Path(r).parent) for r in df["rel_path"])
    lines = [f"{root} — {len(df)} images"]
    for d, n in sorted(dir_counts.items())[:max_dirs]:
        sample = df[df["rel_path"].str.startswith(d if d != "." else "")].iloc[0]
        lines.append(f"  {d or '.'}/  ({n} imgs, inferred label={sample['label']})")
    if len(dir_counts) > max_dirs:
        lines.append(f"  ... and {len(dir_counts) - max_dirs} more directories")
    lab = df["label"].value_counts(dropna=False).to_dict()
    lines.append(f"  label counts: {lab}")
    n_none = int(df["label"].isna().sum())
    if n_none:
        lines.append(
            f"  WARNING: {n_none} images matched no label rule — "
            "adjust label_rules in the config for this dataset."
        )
    return "\n".join(lines)
