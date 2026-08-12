"""Skin-tone estimation via the Individual Typology Angle (ITA).

ITA is a dermatology-standard, fully deterministic measure computed from
CIELAB values of skin pixels:  ITA = arctan((L* - 50) / b*) * 180 / pi.
It needs no pretrained model, which makes it a reproducible primary
fairness axis. Skin pixels are selected with a classic YCrCb chromatic
mask inside the (already face-cropped) image; if the mask is too sparse
we fall back to a central patch.

Standard ITA -> category thresholds (Chardon et al.):
  very_light > 55 >= light > 41 >= intermediate > 28 >= tan > 10
  >= brown > -30 >= dark
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import DotDict
from ..utils.common import get_logger

log = get_logger()

ITA_BINS = [
    (55.0, "very_light"),
    (41.0, "light"),
    (28.0, "intermediate"),
    (10.0, "tan"),
    (-30.0, "brown"),
]


def ita_to_category(ita: float) -> str:
    if np.isnan(ita):
        return "unknown"
    for threshold, name in ITA_BINS:
        if ita > threshold:
            return name
    return "dark"


def _skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Skin-pixel mask combining chrominance AND luminance gates.

    The classic YCrCb chrominance band (Cr 133-173, Cb 77-127) keys on
    hue. We add a light luminance gate Y in [15, 252] whose only job is to
    drop near-black (true background/deep shadow) and blown-out white
    pixels. The floor is deliberately LOW (15, not 40): dark skin has
    genuinely low luminance, and a high floor was rejecting real dark-skin
    pixels — making dark-skinned faces fail the min-pixel test and get
    dropped as 'unknown' entirely. Chrominance, not luminance, is what
    distinguishes skin from non-skin; luminance here only removes extremes.
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    mask = (
        (cr >= 133) & (cr <= 173)
        & (cb >= 77) & (cb <= 127)
        & (y >= 15) & (y <= 252)
    )
    # Drop isolated specks: erode then dilate (morphological opening).
    m = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    return m.astype(bool)


def _central_face_mask(bgr: np.ndarray) -> np.ndarray:
    """Skin mask restricted to the central face region.

    Even a tight face crop contains hair, background and clothing at the
    edges. Intersecting the skin mask with a central ellipse (where
    cheeks/nose/forehead sit) concentrates the estimate on actual facial
    skin and further suppresses background contamination.
    """
    h, w = bgr.shape[:2]
    ell = np.zeros((h, w), np.uint8)
    cv2.ellipse(ell, (w // 2, h // 2), (int(w * 0.30), int(h * 0.38)),
                0, 0, 360, 1, -1)
    return _skin_mask(bgr) & ell.astype(bool)


def _face_region_mask(bgr: np.ndarray, face_box: tuple | None) -> np.ndarray:
    """Skin mask concentrated on a TIGHT central patch of the face.

    Given the face box (fx, fy, fw, fh), we take a tight central patch —
    the middle 50% horizontally and the 40%-75% vertical band (nose and
    upper cheeks) — which sits well inside the hairline and above the
    jaw/beard. Excluding hair matters most for dark-haired subjects of any
    skin tone. We intersect with the skin mask; but see estimate_ita for
    the geometric fallback that measures this patch directly when the skin
    mask is sparse (which happens on very dark skin).
    """
    h, w = bgr.shape[:2]
    region = np.zeros((h, w), np.uint8)

    if face_box is not None:
        fx, fy, fw, fh = face_box
        cx0 = int(fx + fw * 0.25)
        cx1 = int(fx + fw * 0.75)
        cy0 = int(fy + fh * 0.40)
        cy1 = int(fy + fh * 0.75)
        cx0, cy0 = max(0, cx0), max(0, cy0)
        cx1, cy1 = min(w, cx1), min(h, cy1)
        if cx1 > cx0 and cy1 > cy0:
            region[cy0:cy1, cx0:cx1] = 1
    else:
        cv2.ellipse(region, (w // 2, h // 2), (int(w * 0.25), int(h * 0.30)),
                    0, 0, 360, 1, -1)

    return _skin_mask(bgr) & region.astype(bool)


def _central_patch_box(bgr: np.ndarray, face_box: tuple | None) -> tuple:
    """Return (x0,y0,x1,y1) of the tight central face patch for a raw read."""
    h, w = bgr.shape[:2]
    if face_box is not None:
        fx, fy, fw, fh = face_box
        return (max(0, int(fx + fw * 0.30)), max(0, int(fy + fh * 0.42)),
                min(w, int(fx + fw * 0.70)), min(h, int(fy + fh * 0.72)))
    return (int(w * 0.35), int(h * 0.38), int(w * 0.65), int(h * 0.68))


def estimate_ita(
    bgr: np.ndarray, face_box: tuple | None = None, min_pixels: int = 80
) -> tuple[float, str, float]:
    """Return (median ITA, category, skin_fraction) for a face crop.

    Strategy, in order of preference:
    1. Skin pixels within the tight central face patch (best: excludes hair
       and background, keeps only exposed facial skin).
    2. If too few skin pixels there (common on very dark skin, where the
       chrominance mask is less confident), measure the central patch
       DIRECTLY as a geometric region — no skin mask. This is the key fix
       for dark-skinned faces: rather than dropping them as 'unknown', we
       trust the face-box geometry and read the centre of the face, which
       is skin by construction.
    3. Only if even that patch is unreadable do we return 'unknown'.
    """
    total = bgr.shape[0] * bgr.shape[1]

    # 1) skin within the tight central patch
    mask = _face_region_mask(bgr, face_box)
    skin_fraction = float(mask.sum()) / max(1, total)

    if int(mask.sum()) >= min_pixels:
        return _ita_from_mask(bgr, mask) + (skin_fraction,)

    # 2) geometric central patch, measured directly (no skin mask).
    #    This rescues dark-skinned faces the chrominance mask under-selects.
    x0, y0, x1, y1 = _central_patch_box(bgr, face_box)
    if x1 > x0 and y1 > y0:
        patch = np.zeros(bgr.shape[:2], bool)
        patch[y0:y1, x0:x1] = True
        # Drop only extreme black/white within the patch (hair strands, glare).
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        yv = ycrcb[..., 0]
        patch &= (yv >= 12) & (yv <= 250)
        if int(patch.sum()) >= min_pixels:
            frac = float(patch.sum()) / max(1, total)
            return _ita_from_mask(bgr, patch) + (frac,)

    # 3) genuinely unreadable
    return math.nan, "unknown", skin_fraction


def _ita_from_mask(bgr: np.ndarray, mask: np.ndarray) -> tuple[float, str]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0][mask] * (100.0 / 255.0)
    b = lab[..., 2][mask] - 128.0
    denom = np.where(np.abs(b) < 1e-3, 1e-3, b)
    ita = np.degrees(np.arctan((L - 50.0) / denom))
    med = float(np.median(ita)) if ita.size else math.nan
    return med, ita_to_category(med)


def annotate_skin_tone(manifest: pd.DataFrame, cfg: DotDict) -> pd.DataFrame:
    """Add `ita` and `skin_tone` columns to rows from face-bearing datasets."""
    face_datasets = {d["name"] for d in cfg.datasets if d.get("has_faces", True)}
    df = manifest.copy()
    df["ita"] = np.nan
    df["skin_tone"] = "n/a"
    df["skin_fraction"] = np.nan

    todo = df[(df["status"] == "ok") & (df["dataset"].isin(face_datasets))]
    for idx in tqdm(todo.index, desc="skin tone (ITA)"):
        bgr = cv2.imread(df.at[idx, "path"], cv2.IMREAD_COLOR)
        if bgr is None:
            df.at[idx, "skin_tone"] = "unknown"
            continue
        ita, cat, sf = estimate_ita(bgr)
        df.at[idx, "ita"] = ita
        df.at[idx, "skin_tone"] = cat
        df.at[idx, "skin_fraction"] = sf

    counts = df.loc[df["skin_tone"] != "n/a", "skin_tone"].value_counts().to_dict()
    log.info("Skin-tone distribution: %s", counts)
    return df