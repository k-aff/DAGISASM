"""Build the pilot subset: N real + N fake images of PEOPLE only.

People gate
-----------
An image enters the pilot only if the configured face detector finds a
face (proposal scope: deepfake *face* detection). Everything else —
objects, scenery, memes without faces — is discarded, for both classes.

Sampling (two phases, one shuffled stream)
------------------------------------------
Phase 1 fills tone-balanced cells: target_per_label / |tone buckets|
images per (label, tone). Faces that arrive after their cell is full are
remembered as *reserve* candidates instead of being thrown away.
Phase 2 tops labels up to target_per_label from the reserve, always
taking from the currently least-represented tone first. Result: exactly
the requested totals whenever the pool allows, as tone-balanced as the
data permits — and the residual imbalance is quantified in the summary,
because that imbalance is itself evidence of the representation problem
this study investigates.

Scope note: the sources carry no nationality/population-group metadata,
so this is a TONE-DIVERSE face subset standing in for SA demographic
diversity, not an SA sample. Say it that way to your supervisor.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
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
from .skin_tone import estimate_ita

log = get_logger()

COARSE_MAP = {
    "very_light": "light", "light": "light",
    "intermediate": "medium", "tan": "medium",
    "brown": "dark", "dark": "dark",
}


def coarsen_tone(fine: str) -> str:
    return COARSE_MAP.get(fine, "unknown")


def _out_name(orig_path: str) -> str:
    return hashlib.sha1(orig_path.encode()).hexdigest()[:16] + ".jpg"


class _Saver:
    """Crop -> resize -> dedup -> JPEG, shared by both phases."""

    def __init__(self, cfg: DotDict, out_root: Path) -> None:
        self.size = int(cfg.image.size)
        self.quality = int(cfg.image.jpeg_quality)
        self.dedup = bool(cfg.dedup.enabled)
        self.hash_size = int(cfg.dedup.hash_size)
        self.seen: set[str] = set()
        self.out_root = out_root

    def save(self, crop_bgr: np.ndarray, orig_path: str, label: int) -> str | None:
        interp = cv2.INTER_AREA if crop_bgr.shape[0] >= self.size else cv2.INTER_CUBIC
        resized = cv2.resize(crop_bgr, (self.size, self.size), interpolation=interp)
        pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        if self.dedup:
            import imagehash

            h = str(imagehash.phash(pil, hash_size=self.hash_size))
            if h in self.seen:
                return None
            self.seen.add(h)
        out = self.out_root / str(label) / _out_name(orig_path)
        ensure_dir(out.parent)
        pil.save(out, "JPEG", quality=self.quality)
        return str(out)


def harvest(cfg: DotDict) -> pd.DataFrame:
    tones = (list(cfg.group.skin_tone_categories)
             if str(cfg.pilot.tone_scheme) == "fine"
             else list(cfg.pilot.coarse_categories))
    fine_mode = str(cfg.pilot.tone_scheme) == "fine"
    target = int(cfg.pilot.target_per_label)
    cell_cap = int(np.ceil(target / len(tones)))

    usable = [d for d in cfg.datasets
              if d.get("has_faces", True) and d.get("use_in_training", True)
              and (Path(cfg.paths.raw) / d["name"]).exists()]
    if not usable:
        raise RuntimeError(
            "No usable raw datasets found under data/raw/. Download first, "
            "then run scripts/10_inspect_raw.py to verify label inference."
        )
    for d in cfg.datasets:
        if d not in usable:
            log.info("Excluded from pilot: %s (no faces / not downloaded / "
                     "not for training).", d["name"])

    # One combined shuffled stream across usable datasets.
    pools = []
    for d in usable:
        scanned = scan_dataset(Path(cfg.paths.raw) / d["name"], d["label_rules"])
        n_unlab = int(scanned["label"].isna().sum())
        scanned = scanned[scanned["label"].notna()]
        log.info("%s: %d labelled images (%d unlabelled skipped)",
                 d["name"], len(scanned), n_unlab)
        scanned = scanned.assign(dataset=d["name"])
        pools.append(scanned)
    pool = pd.concat(pools, ignore_index=True)
    rng = np.random.default_rng(int(cfg.seed))
    pool = pool.iloc[rng.permutation(len(pool))].reset_index(drop=True)
    max_scan = min(int(cfg.pilot.max_scan_per_dataset) * len(usable), len(pool))

    cropper = FaceCropper(cfg.face.detector, cfg.face.min_face, cfg.face.margin)
    saver = _Saver(cfg, ensure_dir(Path(cfg.paths.processed)))
    cell: dict[tuple[int, str], int] = {(l, t): 0 for l in (0, 1) for t in tones}
    label_total = {0: 0, 1: 0}
    reserve: dict[int, list[dict]] = {0: [], 1: []}
    stats: dict[str, int] = defaultdict(int)
    rows: list[dict] = []

    def keep(rec_row: dict, crop: np.ndarray) -> bool:
        path = saver.save(crop, rec_row["orig_path"], rec_row["label"])
        if path is None:
            stats["duplicate"] += 1
            return False
        rows.append({**rec_row, "path": path, "status": "ok"})
        cell[(rec_row["label"], rec_row["skin_tone"])] += 1
        label_total[rec_row["label"]] += 1
        return True

    # ---------------- phase 1: tone-balanced fill ----------------------
    bar = tqdm(range(max_scan), desc="pilot harvest (phase 1)")
    for i in bar:
        if all(v >= target for v in label_total.values()):
            break
        stats["scanned"] += 1
        rec = pool.iloc[i]
        label = int(rec["label"])
        if label_total[label] >= target:
            continue

        bgr = cv2.imread(rec["orig_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            stats["read_error"] += 1
            continue
        crop, found, face_box = cropper.crop_with_box(bgr)
        if crop is None:
            stats["no_person"] += 1        # the people-only gate
            continue
        stats["faces_found"] += 1

        ita, fine_tone, skin_frac = estimate_ita(crop, face_box=face_box)
        tone = fine_tone if fine_mode else coarsen_tone(fine_tone)
        if fine_tone == "unknown" or tone not in tones:
            stats["tone_unknown"] += 1
            continue

        rec_row = {
            "dataset": rec["dataset"], "orig_path": rec["orig_path"],
            "label": label, "face_found": bool(found), "ita": float(ita),
            "skin_tone_fine": fine_tone, "skin_tone": tone,
        }
        if cell[(label, tone)] < cell_cap:
            keep(rec_row, crop)
        else:
            reserve[label].append(rec_row)  # metadata only; re-read in phase 2
            stats["reserved"] += 1
        bar.set_postfix(real=label_total[0], fake=label_total[1])

    # ---------------- phase 2: top-up from reserve ---------------------
    for label in (0, 1):
        if label_total[label] >= target or not reserve[label]:
            continue
        by_tone: dict[str, list[dict]] = defaultdict(list)
        for r in reserve[label]:
            by_tone[r["skin_tone"]].append(r)
        pbar = tqdm(total=target - label_total[label],
                    desc=f"pilot top-up (label {label})")
        while label_total[label] < target:
            # always draw from the least-represented tone that still has reserve
            avail = [t for t in tones if by_tone[t]]
            if not avail:
                break
            tone = min(avail, key=lambda t: cell[(label, t)])
            r = by_tone[tone].pop(0)
            bgr = cv2.imread(r["orig_path"], cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            crop, _ = cropper.crop(bgr)
            if crop is None:
                continue
            if keep(r, crop):
                stats["topped_up"] += 1
                pbar.update(1)
        pbar.close()

    df = pd.DataFrame(rows)
    log.info("Harvest: %d real, %d fake | %s",
             label_total[0], label_total[1], dict(stats))
    for label in (0, 1):
        if label_total[label] < target:
            log.warning(
                "Label %d reached only %d/%d — the face pool is exhausted. "
                "Report this shortfall; it is a data finding.",
                label, label_total[label], target,
            )
    out = ensure_dir(cfg.paths.manifests) / "pilot_manifest.csv"
    df.to_csv(out, index=False)
    log.info("Wrote %s", out)
    return df


def coverage_report(df: pd.DataFrame, cfg: DotDict) -> str:
    tones = (list(cfg.group.skin_tone_categories)
             if str(cfg.pilot.tone_scheme) == "fine"
             else list(cfg.pilot.coarse_categories))
    target = int(cfg.pilot.target_per_label)

    table = (df.groupby(["skin_tone", "label"]).size()
             .unstack(fill_value=0).reindex(tones, fill_value=0))
    for lab in (0, 1):
        if lab not in table.columns:
            table[lab] = 0
    table = table[[0, 1]].rename(columns={0: "real (0)", 1: "fake (1)"})
    table["total"] = table.sum(axis=1)
    per_ds = df.groupby(["dataset", "label"]).size().unstack(fill_value=0)

    n_real = int((df["label"] == 0).sum())
    n_fake = int((df["label"] == 1).sum())
    lines = [
        "# Pilot subset — coverage summary", "",
        f"People-only images kept: **{n_real} real / {n_fake} fake** "
        f"(target {target}/{target}) | tone scheme: `{cfg.pilot.tone_scheme}`", "",
        "## Skin tone (ITA proxy) x label", "",
        table.to_markdown(), "",
        "## Source dataset x label", "",
        per_ds.to_markdown(), "",
    ]
    if n_real < target or n_fake < target:
        lines += [
            "## Shortfall (a finding, not a failure)", "",
            f"The pool of detectable faces was exhausted before reaching "
            f"{target}/{target}. Options: lower `face.min_face`, switch to "
            "`face.detector: mtcnn` (higher recall), raise "
            "`pilot.max_scan_per_dataset`, or report the reduced N.", "",
        ]
    imb = table["total"].max() - table["total"].min()
    if imb > 0.2 * table["total"].max():
        lines += [
            "## Tone imbalance", "",
            "Even after balanced sampling, tone buckets are uneven — the "
            "source pool under-represents some tones. This is exactly the "
            "representation problem motivating the study; quantify it in "
            "the write-up rather than hiding it.", "",
        ]
    lines += [
        "## Scope caveat (state this to your supervisor)", "",
        "These sources carry no nationality or population-group metadata. "
        "This is a **tone-diverse face subset**, standing in for South "
        "African demographic diversity pending SA-representative data — "
        "not an SA sample. Skin tone is an ITA measurement of the image, "
        "not a self-identified identity; the Coloured population group is "
        "not visually inferable at all.", "",
    ]
    text = "\n".join(lines)
    out = Path(cfg.paths.manifests) / str(cfg.pilot.report_name)
    ensure_dir(out.parent)
    out.write_text(text, encoding="utf-8")
    log.info("Wrote %s", out)
    return text
