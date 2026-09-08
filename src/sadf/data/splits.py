"""Split assignment for the manifest.

Two modes:

  * cfg.splits.use_existing = true
        The manifest already carries a `split` column and it is used as-is.
        Required when the training set contains augmented copies: any random
        re-split would put copies of the same source face in both train and
        test, inflating every metric downstream.

  * cfg.splits.use_existing = false (default, original behaviour)
        Stratified random split into train/val/test.

Also tolerates a manifest with no `status` column, in which case every row
is treated as usable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DotDict
from ..utils.common import ensure_dir, get_logger

log = get_logger()

VALID_SPLITS = {"train", "val", "test"}


def _cfg_get(cfg: DotDict, path: str, default):
    """Safe nested lookup: _cfg_get(cfg, 'splits.use_existing', False)."""
    node = cfg
    for part in path.split("."):
        try:
            node = node[part] if isinstance(node, dict) else getattr(node, part)
        except (KeyError, AttributeError, TypeError):
            return default
        if node is None:
            return default
    return node


def _ok_mask(df: pd.DataFrame) -> pd.Series:
    """Usable rows. Manifests without a `status` column are all usable."""
    if "status" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["status"] == "ok"


def use_existing_split(df: pd.DataFrame, cfg: DotDict) -> pd.DataFrame:
    """Validate and return the manifest's own split column."""
    if "split" not in df.columns:
        raise ValueError(
            "splits.use_existing is true but the manifest has no 'split' column. "
            "Either set splits.use_existing=false or point at a manifest that has one."
        )

    out = df.copy()
    bad = set(out["split"].astype(str).unique()) - VALID_SPLITS
    if bad:
        raise ValueError(f"manifest 'split' column has unexpected values: {sorted(bad)}")

    missing = VALID_SPLITS - set(out["split"].astype(str).unique())
    if missing:
        raise ValueError(f"manifest 'split' column is missing: {sorted(missing)}")

    # Leakage guard. If the manifest tracks provenance (augmented copies point
    # back at their original via source_id), no source may span two splits.
    if "source_id" in out.columns:
        spread = out.groupby("source_id")["split"].nunique()
        leaked = spread[spread > 1]
        if len(leaked):
            raise ValueError(
                f"{len(leaked)} source images appear in more than one split "
                f"(e.g. {list(leaked.index[:3])}). Training would be invalid."
            )
        log.info("Leakage check passed: %d unique sources, none spanning splits",
                 out["source_id"].nunique())

    ok = _ok_mask(out)
    counts = out.loc[ok].groupby(["split", "label"]).size().unstack(fill_value=0)
    log.info("Using existing split from manifest. Split x label counts:\n%s", counts)

    if "is_augmented" in out.columns:
        n_aug_outside = int(out.loc[out["split"] != "train", "is_augmented"].sum())
        if n_aug_outside:
            raise ValueError(
                f"{n_aug_outside} augmented rows found outside the train split. "
                "Val and test must contain originals only."
            )
        log.info("Augmented rows: %d (train only)", int(out["is_augmented"].sum()))

    return out


def stratified_split(df: pd.DataFrame, cfg: DotDict) -> pd.DataFrame:
    if bool(_cfg_get(cfg, "splits.use_existing", False)):
        return use_existing_split(df, cfg)

    rng = np.random.default_rng(int(cfg.seed))
    fr_train = float(cfg.splits.train)
    fr_val = float(cfg.splits.val)
    keys = [k for k in cfg.splits.stratify_on if k in df.columns]

    out = df.copy()
    out["split"] = "train"
    ok = _ok_mask(out)
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

    # Don't clobber an existing manifest when we were handed a ready-made split.
    out_name = str(_cfg_get(cfg, "paths.manifest_out", "manifest.csv"))
    out_path = ensure_dir(cfg.paths.manifests) / out_name
    df.to_csv(out_path, index=False)

    ok = df[_ok_mask(df)]
    if "skin_tone" in ok:
        cats = list(_cfg_get(cfg, "group.skin_tone_categories", []))
        subset = ok[ok["skin_tone"].isin(cats)] if cats else ok
        if len(subset) == 0 and cats:
            log.warning(
                "No rows match group.skin_tone_categories=%s. Values present: %s",
                cats, sorted(ok["skin_tone"].astype(str).unique())[:10],
            )
        tab = subset.groupby(["skin_tone", "label"]).size().unstack(fill_value=0)
        log.info("Skin-tone x label table (audit view):\n%s", tab)
    log.info("Wrote %s (%d ok rows)", out_path, len(ok))
    return df
