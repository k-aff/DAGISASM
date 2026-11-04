"""Find out which saved cross-test results came from the current baseline checkpoint.

Run after scoring a sample with the current best.pt into runs/_check_current:

    python compare_scores.py
"""
from pathlib import Path

import pandas as pd

NEW = Path("runs/_check_current")
OLD = {
    "eval_cross (numbers in the report)": Path("runs/models140k/swin_aug/baseline/eval_cross"),
    "eval_cross_full": Path("runs/models140k/swin_aug/baseline/eval_cross_full"),
}

for src in ("imdb", "sd_wiki"):
    new = pd.read_csv(NEW / src / "scores.csv")[["path", "score"]]
    print(f"\n{src}: {len(new)} images scored with the current best.pt")
    for name, folder in OLD.items():
        f = folder / src / "scores.csv"
        if not f.exists():
            print(f"  {name}: file not found ({f})")
            continue
        old = pd.read_csv(f)[["path", "score"]]
        m = new.merge(old, on="path", suffixes=("_now", "_saved"))
        diff = (m["score_now"] - m["score_saved"]).abs()
        same = "SAME MODEL" if len(m) and diff.max() < 1e-4 else "different model"
        print(f"  {name}: {len(m)} matching images, largest score difference "
              f"{diff.max():.6f} -> {same}")