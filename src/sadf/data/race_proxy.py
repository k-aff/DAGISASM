
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..utils.common import get_logger

log = get_logger()

# DeepFace category -> SA-relevant proxy bucket. Documented limitation:
# 'coloured' is intentionally absent (not visually inferable).
DEEPFACE_TO_PROXY = {
    "black": "black_african_proxy",
    "indian": "indian_asian_proxy",
    "asian": "indian_asian_proxy",
    "white": "white_proxy",
    "middle eastern": "other_proxy",
    "latino hispanic": "other_proxy",
}


def annotate_with_deepface(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from deepface import DeepFace  # heavy optional dependency
    except ImportError as e:
        raise ImportError(
            "race-proxy annotation needs `pip install deepface` "
            "(pulls TensorFlow; see README optional extras)."
        ) from e

    out = df.copy()
    if "race_proxy" not in out:
        out["race_proxy"] = "unknown"
    todo = out[(out["status"] == "ok") & (out["face_found"])]
    for idx in tqdm(todo.index, desc="race proxy (DeepFace)"):
        try:
            res = DeepFace.analyze(
                img_path=out.at[idx, "path"],
                actions=["race"],
                enforce_detection=False,
                detector_backend="skip",  # images are already face crops
                silent=True,
            )
            dom = (res[0] if isinstance(res, list) else res)["dominant_race"]
            out.at[idx, "race_proxy"] = DEEPFACE_TO_PROXY.get(dom.lower(), "other_proxy")
        except Exception:  # noqa: BLE001 — annotation must never kill the run
            out.at[idx, "race_proxy"] = "unknown"
    log.info("race_proxy counts: %s", out["race_proxy"].value_counts().to_dict())
    return out


def merge_external_labels(df: pd.DataFrame, csv_path: str | Path) -> pd.DataFrame:
    ext = pd.read_csv(csv_path)
    if "path" not in ext or "race_proxy" not in ext:
        raise ValueError("External label CSV needs columns: path, race_proxy")
    out = df.copy()
    if "race_proxy" not in out:
        out["race_proxy"] = "unknown"
    mapping = dict(zip(ext["path"].astype(str), ext["race_proxy"].astype(str)))
    hits = out["path"].map(mapping)
    out.loc[hits.notna(), "race_proxy"] = hits[hits.notna()]
    log.info("Merged %d external race labels", int(hits.notna().sum()))
    return out
