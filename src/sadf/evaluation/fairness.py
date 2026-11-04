"""Per-group metric tables, max-minus-min gaps and bootstrap intervals for the gaps."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import binary_metrics

EXCLUDED_GROUPS = {"unknown", "n/a", "nan", ""}


def per_group_table(
    df: pd.DataFrame,
    group_col: str,
    threshold: float = 0.5,
    min_n: int = 30,
) -> pd.DataFrame:
    """One row of binary metrics per group; small_n marks groups below min_n."""
    rows = []
    for group, sub in df.groupby(group_col):
        if str(group) in EXCLUDED_GROUPS:
            continue
        m = binary_metrics(sub["label"].values, sub["score"].values, threshold)
        rows.append({"group": group, "small_n": m["n"] < min_n, **m})
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def _gap(series: pd.Series) -> float:
    vals = series.dropna().values
    return float(vals.max() - vals.min()) if len(vals) >= 2 else float("nan")


def fairness_gaps(table: pd.DataFrame, include_small: bool = False) -> dict:
    if table.empty:
        return {
            "fpr_gap": float("nan"), "tpr_gap": float("nan"),
            "auc_gap": float("nan"), "equalized_odds_gap": float("nan"),
            "demographic_parity_gap": float("nan"),
            "worst_group_auc": float("nan"), "n_groups": 0,
        }
    t = table if include_small else table[~table["small_n"]]
    if t.empty:
        t = table  # fall back rather than report nothing
    fpr_gap = _gap(t["fpr"])
    tpr_gap = _gap(t["recall"])
    finite = [v for v in (fpr_gap, tpr_gap) if not np.isnan(v)]
    return {
        "fpr_gap": fpr_gap,
        "tpr_gap": tpr_gap,
        "auc_gap": _gap(t["auc"]),
        "equalized_odds_gap": float(max(finite)) if finite else float("nan"),
        "demographic_parity_gap": _gap(t["positive_rate"]),
        "worst_group_auc": float(np.nanmin(t["auc"].values)) if t["auc"].notna().any() else float("nan"),
        "n_groups": int(len(t)),
    }


def bootstrap_gap_cis(
    df: pd.DataFrame,
    group_col: str,
    threshold: float = 0.5,
    min_n: int = 30,
    n_boot: int = 200,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Percentile bootstrap CIs for fpr_gap and auc_gap."""
    rng = np.random.default_rng(seed)
    fprs, aucs = [], []
    n = len(df)
    for _ in range(int(n_boot)):
        sample = df.iloc[rng.integers(0, n, n)]
        gaps = fairness_gaps(per_group_table(sample, group_col, threshold, min_n))
        fprs.append(gaps["fpr_gap"])
        aucs.append(gaps["auc_gap"])
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100

    def interval(vals: list[float]) -> list[float]:
        arr = np.array([v for v in vals if not np.isnan(v)])
        if arr.size == 0:
            return [float("nan"), float("nan")]
        return [float(np.percentile(arr, lo)), float(np.percentile(arr, hi))]

    return {"fpr_gap_ci": interval(fprs), "auc_gap_ci": interval(aucs), "n_boot": int(n_boot)}
