
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DotDict
from ..utils.common import ensure_dir, get_logger

log = get_logger()


def stratified_split(df: pd.DataFrame, cfg: DotDict) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg.seed))
    fr_train = float(cfg.splits.train)
    fr_val = float(cfg.splits.val)
    keys = [k for k in cfg.splits.stratify_on if k in df.columns]

    out = df.copy()
    out["split"] = "train"
    ok = out["status"] == "ok"
    strata = out.loc[ok].groupby(keys, dropna=False).indices if keys else {"all": out.index[ok]}

    for _, idx in strata.items():
        idx = np.asarray(list(idx))
        rng.shuffle(idx)
        n = len(idx)
        if n == 1:
            continue  # stays train
        if n == 2:
            out.loc[idx[1], "split"] = "test"
            continue
        n_train = max(1, int(round(n * fr_train)))
        n_val = max(1, int(round(n * fr_val)))
        n_train = min(n_train, n - 2)  # guarantee >=1 val and >=1 test
        n_val = min(n_val, n - n_train - 1)
        out.loc[idx[n_train: n_train + n_val], "split"] = "val"
        out.loc[idx[n_train + n_val:], "split"] = "test"

    counts = out.loc[ok].groupby(["split", "label"]).size().unstack(fill_value=0)
    log.info("Split x label counts:\n%s", counts)
    return out


def build_final_manifest(df: pd.DataFrame, cfg: DotDict) -> pd.DataFrame:
    df = stratified_split(df, cfg)
    out_path = ensure_dir(cfg.paths.manifests) / "manifest.csv"
    df.to_csv(out_path, index=False)

    ok = df[df["status"] == "ok"]
    if "skin_tone" in ok:
        tab = (
            ok[ok["skin_tone"].isin(cfg.group.skin_tone_categories)]
            .groupby(["skin_tone", "label"]).size().unstack(fill_value=0)
        )
        log.info("Skin-tone x label table (audit view):\n%s", tab)
    log.info("Wrote %s (%d ok rows)", out_path, len(ok))
    return df
