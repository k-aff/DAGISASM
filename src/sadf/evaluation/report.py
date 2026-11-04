"""Evaluation report: overall metrics, per-group audits, gaps, plots, markdown."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics as skm

from ..utils.common import ensure_dir, get_logger, save_json
from .fairness import bootstrap_gap_cis, fairness_gaps, per_group_table
from .metrics import binary_metrics

log = get_logger()


def _plot_group_bar(table: pd.DataFrame, col: str, title: str, path: Path) -> None:
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#b0b0b0" if s else "#4472c4" for s in table["small_n"]]
    ax.bar(table["group"].astype(str), table[col], color=colors)
    for i, (v, n) in enumerate(zip(table[col], table["n"])):
        if not np.isnan(v):
            ax.text(i, v, f"{v:.3f}\n(n={n})", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(col)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_roc(df: pd.DataFrame, path: Path) -> None:
    if df["label"].nunique() < 2:
        return
    fpr, tpr, _ = skm.roc_curve(df["label"], df["score"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#4472c4")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC (AUC={skm.roc_auc_score(df['label'], df['score']):.4f})")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_report(
    scored: pd.DataFrame,
    out_dir: str | Path,
    group_cols: list[str],
    threshold: float = 0.5,
    min_group_size: int = 30,
    n_boot: int = 200,
    seed: int = 42,
    title: str = "Evaluation report",
) -> dict:
    """`scored` needs columns: label, score, plus each group col present."""
    out = ensure_dir(out_dir)
    scored.to_csv(out / "scores.csv", index=False)

    overall = binary_metrics(scored["label"].values, scored["score"].values, threshold)
    save_json(overall, out / "overall.json")
    _plot_roc(scored, out / "roc.png")

    lines = [f"# {title}", "", f"Rows: {overall['n']} | threshold: {threshold}", ""]
    lines += ["## Overall detection performance", "",
              _dict_md(overall), ""]

    all_gaps: dict[str, dict] = {}
    for col in group_cols:
        if col not in scored.columns:
            continue
        informative = scored[~scored[col].astype(str).isin({"unknown", "n/a", "nan", ""})]
        if informative[col].nunique() < 2:
            log.info("Skipping fairness audit on '%s' (<2 informative groups)", col)
            continue
        table = per_group_table(scored, col, threshold, min_group_size)
        gaps = fairness_gaps(table)
        cis = bootstrap_gap_cis(
            informative, col, threshold, min_group_size, n_boot=n_boot, seed=seed
        )
        all_gaps[col] = {**gaps, **cis}
        table.to_csv(out / f"per_group_{col}.csv", index=False)
        # the skin_tone column holds the FairFace race label
        pretty = {"skin_tone": "race group", "race": "race group"}.get(col, col)
        _plot_group_bar(table, "fpr", f"False-positive rate by {pretty}", out / f"fpr_by_{col}.png")
        _plot_group_bar(table, "auc", f"AUC by {pretty}", out / f"auc_by_{col}.png")

        lines += [f"## Fairness audit — grouped by `{col}`", "",
                  table.to_markdown(index=False, floatfmt=".4f"), "",
                  "Gaps (groups with n >= "
                  f"{min_group_size}; grey bars in plots are small-n groups):", "",
                  _dict_md({k: v for k, v in all_gaps[col].items() if k != "n_boot"}), ""]

    save_json(all_gaps, out / "gaps.json")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written to %s", out)
    return {"overall": overall, "gaps": all_gaps}


def _dict_md(d: dict) -> str:
    rows = ["| metric | value |", "| --- | --- |"]
    for k, v in d.items():
        if isinstance(v, float):
            rows.append(f"| {k} | {v:.4f} |")
        else:
            rows.append(f"| {k} | {v} |")
    return "\n".join(rows)