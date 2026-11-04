"""Check whether a retrained baseline matches the saved results.

    python compare_runs.py history  runs/models140k/swin_aug/baseline/history.json  runs/models140k/swin_aug/baseline_old/history.json
    python compare_runs.py scores   runs/models140k/swin_aug/baseline/eval/scores.csv runs/models140k/swin_aug/baseline_old/eval/scores.csv
"""
import json
import sys

import pandas as pd

kind, new_path, old_path = sys.argv[1:4]

if kind == "history":
    new, old = json.load(open(new_path)), json.load(open(old_path))
    for n, o in zip(new, old):
        print(f"epoch {n['epoch']}: val_auc now {n['val_auc']:.6f} saved {o['val_auc']:.6f} | "
              f"val_fpr_gap now {n['val_fpr_gap']:.4f} saved {o['val_fpr_gap']:.4f}")
    print(f"epochs now {len(new)}, saved {len(old)}")
else:
    new = pd.read_csv(new_path)[["path", "score"]]
    old = pd.read_csv(old_path)[["path", "score"]]
    m = new.merge(old, on="path", suffixes=("_now", "_saved"))
    diff = (m["score_now"] - m["score_saved"]).abs()
    flips = ((m["score_now"] >= 0.5) != (m["score_saved"] >= 0.5)).sum()
    print(f"{len(m)} matching images | largest score difference {diff.max():.6f} | "
          f"mean difference {diff.mean():.6f} | decisions that changed: {flips}")
    print("SAME MODEL" if diff.max() < 1e-4 else "not identical (see the numbers above)")