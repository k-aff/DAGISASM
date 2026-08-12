
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ..config import DotDict
from ..utils.common import ensure_dir, get_logger
from .faces import FaceCropper
from .ingest import scan_dataset

log = get_logger()


def _phash(pil_img: Image.Image, hash_size: int) -> str:
    import imagehash

    return str(imagehash.phash(pil_img, hash_size=hash_size))


def _out_name(orig_path: str) -> str:
    return hashlib.sha1(orig_path.encode()).hexdigest()[:16] + ".jpg"


def process_dataset(dcfg: dict, cfg: DotDict) -> pd.DataFrame:
    name = dcfg["name"]
    raw_root = Path(cfg.paths.raw) / name
    if not raw_root.exists():
        log.warning("Raw folder missing for %s (%s) — skipping.", name, raw_root)
        return pd.DataFrame()

    scanned = scan_dataset(raw_root, dcfg["label_rules"])
    log.info("%s: %d images discovered", name, len(scanned))

    size = int(cfg.image.size)
    quality = int(cfg.image.jpeg_quality)
    has_faces = bool(dcfg.get("has_faces", True))
    require_face = bool(cfg.face.require_face) and has_faces
    cropper = (
        FaceCropper(cfg.face.detector, cfg.face.min_face, cfg.face.margin)
        if has_faces
        else FaceCropper("none")
    )
    dedup_on = bool(cfg.dedup.enabled)
    seen_hashes: dict[tuple[int, str], str] = {}

    out_dir = ensure_dir(Path(cfg.paths.processed) / name)
    rows = []
    for rec in tqdm(scanned.itertuples(index=False), total=len(scanned), desc=name):
        row = {
            "dataset": name,
            "orig_path": rec.orig_path,
            "label": rec.label,
            "path": "",
            "face_found": False,
            "phash": "",
            "status": "ok",
        }
        if rec.label is None or (isinstance(rec.label, float) and np.isnan(rec.label)):
            row["status"] = "unlabeled"
            rows.append(row)
            continue
        row["label"] = int(rec.label)

        bgr = cv2.imread(rec.orig_path, cv2.IMREAD_COLOR)
        if bgr is None:
            row["status"] = "read_error"
            rows.append(row)
            continue

        crop, found = cropper.crop(bgr)
        row["face_found"] = bool(found)
        if crop is None:
            if require_face:
                row["status"] = "no_face"
                rows.append(row)
                continue
            crop = bgr  # keep full frame when faces are optional

        interp = cv2.INTER_AREA if crop.shape[0] >= size else cv2.INTER_CUBIC
        resized = cv2.resize(crop, (size, size), interpolation=interp)
        pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        if dedup_on:
            h = _phash(pil, int(cfg.dedup.hash_size))
            row["phash"] = h
            key = (row["label"], h)
            if key in seen_hashes:
                row["status"] = "duplicate"
                rows.append(row)
                continue
            seen_hashes[key] = rec.orig_path

        out_path = out_dir / str(row["label"]) / _out_name(rec.orig_path)
        ensure_dir(out_path.parent)
        pil.save(out_path, "JPEG", quality=quality)
        row["path"] = str(out_path)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        log.info("%s status counts: %s", name, df["status"].value_counts().to_dict())
    return df


def run_preprocessing(cfg: DotDict) -> pd.DataFrame:
    frames = [process_dataset(d, cfg) for d in cfg.datasets]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("No datasets processed — did you run 01_download.py?")
    df = pd.concat(frames, ignore_index=True)
    out = ensure_dir(cfg.paths.manifests) / "processed_manifest.csv"
    df.to_csv(out, index=False)
    log.info("Wrote %s (%d rows, %d ok)", out, len(df), int((df.status == "ok").sum()))
    return df
