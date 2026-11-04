from __future__ import annotations

import numpy as np
from sklearn import metrics as skm


def binary_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> dict:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    preds = (scores >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    def safe(num: float, den: float) -> float:
        return float(num / den) if den > 0 else float("nan")

    auc = (
        float(skm.roc_auc_score(labels, scores))
        if len(np.unique(labels)) == 2
        else float("nan")
    )
    return {
        "n": int(len(labels)),
        "accuracy": safe(tp + tn, len(labels)),
        "balanced_accuracy": float(np.nanmean([safe(tp, tp + fn), safe(tn, tn + fp)])),
        "precision": safe(tp, tp + fp),
        "recall": safe(tp, tp + fn),          # = TPR
        "f1": safe(2 * tp, 2 * tp + fp + fn),
        "auc": auc,
        "fpr": safe(fp, fp + tn),
        "fnr": safe(fn, fn + tp),
        "positive_rate": safe(tp + fp, len(labels)),
    }
