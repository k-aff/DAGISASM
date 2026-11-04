"""Skin tone from the Individual Typology Angle (ITA), the method tried first.

ITA = arctan((L* - 50) / b*) * 180 / pi, computed from the CIELAB values of
skin pixels. Skin pixels are chosen with a YCrCb mask inside a central patch
of the face; if the mask is too sparse, the patch is measured directly.

Category thresholds (Chardon et al.):
  very_light > 55 >= light > 41 >= intermediate > 28 >= tan > 10
  >= brown > -30 >= dark
The six categories are merged into light, medium and dark by coarsen_tone().
"""
from __future__ import annotations

import math

import cv2
import numpy as np

ITA_BINS = [
    (55.0, "very_light"),
    (41.0, "light"),
    (28.0, "intermediate"),
    (10.0, "tan"),
    (-30.0, "brown"),
]


COARSE_MAP = {
    "very_light": "light", "light": "light",
    "intermediate": "medium", "tan": "medium",
    "brown": "dark", "dark": "dark",
}


def coarsen_tone(fine: str) -> str:
    return COARSE_MAP.get(fine, "unknown")


def ita_to_category(ita: float) -> str:
    if np.isnan(ita):
        return "unknown"
    for threshold, name in ITA_BINS:
        if ita > threshold:
            return name
    return "dark"


def _skin_mask(bgr: np.ndarray) -> np.ndarray:
    """YCrCb skin mask: Cr 133-173, Cb 77-127 and luminance Y 15-252.

    The luminance floor is 15 rather than 40 because a floor of 40 rejected
    many real skin pixels on dark-skinned faces.
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    mask = (
        (cr >= 133) & (cr <= 173)
        & (cb >= 77) & (cb <= 127)
        & (y >= 15) & (y <= 252)
    )
    # morphological opening removes isolated pixels
    m = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    return m.astype(bool)


def _face_region_mask(bgr: np.ndarray, face_box: tuple | None) -> np.ndarray:
    """Skin mask inside the central patch of the face box.

    The patch is the middle 50% of the box width and the band from 40% to
    75% of its height (nose and upper cheeks), which avoids most hair and
    beards.
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
    """(x0, y0, x1, y1) of the central patch measured by the fallback."""
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

    1. Use the skin pixels in the central patch of the face.
    2. If fewer than min_pixels are found (mostly on very dark skin),
       measure the central patch directly without the skin mask.
    3. Otherwise return 'unknown'.
    """
    total = bgr.shape[0] * bgr.shape[1]

    mask = _face_region_mask(bgr, face_box)
    skin_fraction = float(mask.sum()) / max(1, total)

    if int(mask.sum()) >= min_pixels:
        return _ita_from_mask(bgr, mask) + (skin_fraction,)

    x0, y0, x1, y1 = _central_patch_box(bgr, face_box)
    if x1 > x0 and y1 > y0:
        patch = np.zeros(bgr.shape[:2], bool)
        patch[y0:y1, x0:x1] = True
        # drop only near-black and near-white pixels (hair strands, glare)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        yv = ycrcb[..., 0]
        patch &= (yv >= 12) & (yv <= 250)
        if int(patch.sum()) >= min_pixels:
            frac = float(patch.sum()) / max(1, total)
            return _ita_from_mask(bgr, patch) + (frac,)

    return math.nan, "unknown", skin_fraction


def _ita_from_mask(bgr: np.ndarray, mask: np.ndarray) -> tuple[float, str]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0][mask] * (100.0 / 255.0)
    b = lab[..., 2][mask] - 128.0
    denom = np.where(np.abs(b) < 1e-3, 1e-3, b)
    ita = np.degrees(np.arctan((L - 50.0) / denom))
    med = float(np.median(ita)) if ita.size else math.nan
    return med, ita_to_category(med)
