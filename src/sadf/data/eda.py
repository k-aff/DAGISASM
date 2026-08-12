"""Exploratory data analysis over the raw pool — the pre-commitment step.

This runs BEFORE harvesting. It answers the questions that determine
whether the planned run is even feasible, and at what scale:

* How many images per dataset, and what is the real/fake balance?
* What fraction contain a detectable face (the people-only yield)?
* How are faces distributed across skin-tone groups — and, crucially,
  how large is the *scarcest* (tone x label) cell? That number caps a
  balanced run.
* What are the image dimensions, modes, and how many files are corrupt
  or unreadable?

To stay fast on 100k+ pools it works on a random SAMPLE per dataset
(configurable) and extrapolates counts. Nothing is written to the
processed tree; outputs are a report and plots only.
"""
from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

from tqdm import tqdm

from ..config import DotDict
from ..utils.common import ensure_dir, get_logger, save_json
from .faces import FaceCropper
from .ingest import scan_dataset
from .skin_tone import estimate_ita
from .pilot import coarsen_tone

log = get_logger()


def _sample_indices(n: int, sample: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if sample <= 0 or sample >= n:
        return np.arange(n)
    return rng.choice(n, size=sample, replace=False)


def analyse_dataset(dcfg: dict, cfg: DotDict, sample: int) -> dict:
    """Sample one raw dataset and gather descriptive statistics."""
    name = dcfg["name"]
    root = Path(cfg.paths.raw) / name
    if not root.exists():
        log.warning("%s: %s not present — skipping.", name, root)
        return {}

    scanned = scan_dataset(root, dcfg["label_rules"])
    n_total = len(scanned)
    n_unlabeled = int(scanned["label"].isna().sum())
    label_counts_full = scanned["label"].value_counts(dropna=False).to_dict()

    idx = _sample_indices(n_total, sample, int(cfg.seed))
    sub = scanned.iloc[idx].reset_index(drop=True)
    has_faces = bool(dcfg.get("has_faces", True))
    cropper = FaceCropper(cfg.face.detector, cfg.face.min_face, cfg.face.margin) \
        if has_faces else None

    widths, heights = [], []
    modes: Counter = Counter()
    n_read_err = 0
    face_hits = 0
    # (label -> tone -> count) among sampled images with a detected face
    tone_by_label: dict[int, Counter] = defaultdict(Counter)

    for rec in tqdm(sub.itertuples(index=False), total=len(sub),
                    desc=f"EDA {name}"):
        bgr = cv2.imread(rec.orig_path, cv2.IMREAD_COLOR)
        if bgr is None:
            n_read_err += 1
            continue
        h, w = bgr.shape[:2]
        widths.append(w)
        heights.append(h)
        modes[f"{bgr.shape[2] if bgr.ndim == 3 else 1}ch"] += 1

        if not has_faces or rec.label is None:
            continue
        crop, found, face_box = cropper.crop_with_box(bgr)
        if crop is None:
            continue
        face_hits += 1
        _ita, fine, _sf = estimate_ita(crop, face_box=face_box)
        tone = fine if str(cfg.pilot.tone_scheme) == "fine" else coarsen_tone(fine)
        tone_by_label[int(rec.label)][tone] += 1

    n_sampled = len(sub)
    face_rate = face_hits / max(1, n_sampled) if has_faces else 0.0

    # Extrapolate per-(label,tone) face yield to the full dataset.
    est_faces_by_cell: dict[str, int] = {}
    if has_faces and face_hits > 0:
        scale = n_total / max(1, n_sampled)
        for label, ctr in tone_by_label.items():
            for tone, c in ctr.items():
                est_faces_by_cell[f"label{label}/{tone}"] = int(round(c * scale))

    return {
        "name": name,
        "has_faces": has_faces,
        "n_total": n_total,
        "n_unlabeled": n_unlabeled,
        "label_counts_full": {str(k): int(v) for k, v in label_counts_full.items()},
        "n_sampled": n_sampled,
        "n_read_error_in_sample": n_read_err,
        "face_rate_in_sample": round(face_rate, 4),
        "est_total_faces": int(round(face_rate * n_total)) if has_faces else 0,
        "tone_by_label_sampled": {str(k): dict(v) for k, v in tone_by_label.items()},
        "est_faces_by_cell": est_faces_by_cell,
        "width_stats": _dist_stats(widths),
        "height_stats": _dist_stats(heights),
        "channel_modes": dict(modes),
    }


def _dist_stats(vals: list[int]) -> dict:
    if not vals:
        return {}
    a = np.array(vals)
    return {
        "min": int(a.min()), "p25": int(np.percentile(a, 25)),
        "median": int(np.median(a)), "p75": int(np.percentile(a, 75)),
        "max": int(a.max()), "mean": round(float(a.mean()), 1),
    }


# ----------------------------------------------------------------------
def _plot_class_balance(results: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r["name"] for r in results]
    reals = [r["label_counts_full"].get("0", 0) for r in results]
    fakes = [r["label_counts_full"].get("1", 0) for r in results]
    x = np.arange(len(names))
    ax.bar(x - 0.2, reals, 0.4, label="real (0)", color="#4472c4")
    ax.bar(x + 0.2, fakes, 0.4, label="fake (1)", color="#c00000")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("images"); ax.set_title("Class balance per dataset")
    ax.legend(); plt.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def _plot_face_yield(results: list[dict], out: Path) -> None:
    face_ds = [r for r in results if r["has_faces"]]
    if not face_ds:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r["name"] for r in face_ds]
    rates = [r["face_rate_in_sample"] * 100 for r in face_ds]
    ax.bar(names, rates, color="#548235")
    for i, r in enumerate(rates):
        ax.text(i, r, f"{r:.1f}%", ha="center", va="bottom")
    ax.set_ylabel("% images with a detected face")
    ax.set_title(f"People-only yield (detector={face_ds[0].get('_det','haar')})")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)


def _plot_tone_cells(results: list[dict], cfg: DotDict, out: Path) -> None:
    """Estimated usable faces per (tone x label), summed across datasets."""
    tones = (list(cfg.group.skin_tone_categories)
             if str(cfg.pilot.tone_scheme) == "fine"
             else list(cfg.pilot.coarse_categories))
    agg: dict[tuple[str, int], int] = defaultdict(int)
    for r in results:
        for cell, est in r.get("est_faces_by_cell", {}).items():
            lab_part, tone = cell.split("/")
            label = int(lab_part.replace("label", ""))
            agg[(tone, label)] += est
    if not agg:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(tones))
    reals = [agg.get((t, 0), 0) for t in tones]
    fakes = [agg.get((t, 1), 0) for t in tones]
    ax.bar(x - 0.2, reals, 0.4, label="real (0)", color="#4472c4")
    ax.bar(x + 0.2, fakes, 0.4, label="fake (1)", color="#c00000")
    ax.set_xticks(x); ax.set_xticklabels(tones)
    ax.set_ylabel("estimated usable faces")
    ax.set_title("Estimated face yield per skin-tone x label cell")
    ax.legend(); plt.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------
def run_eda(cfg: DotDict, sample: int = 3000) -> dict:
    out_dir = ensure_dir(Path(cfg.paths.manifests).parent / "eda")
    results = [analyse_dataset(d, cfg, sample) for d in cfg.datasets]
    results = [r for r in results if r]
    for r in results:
        r["_det"] = cfg.face.detector

    _plot_class_balance(results, out_dir / "class_balance.png")
    _plot_face_yield(results, out_dir / "face_yield.png")
    _plot_tone_cells(results, cfg, out_dir / "tone_cells.png")
    save_json(results, out_dir / "eda.json")

    report = _render_report(results, cfg, sample)
    (out_dir / "eda_report.md").write_text(report, encoding="utf-8")
    log.info("EDA written to %s", out_dir)
    print("\n" + report)
    return {"results": results, "out_dir": str(out_dir)}


def _render_report(results: list[dict], cfg: DotDict, sample: int) -> str:
    tones = (list(cfg.group.skin_tone_categories)
             if str(cfg.pilot.tone_scheme) == "fine"
             else list(cfg.pilot.coarse_categories))
    L = ["# Exploratory data analysis — raw pool", "",
         f"Face detector: `{cfg.face.detector}` | tone scheme: "
         f"`{cfg.pilot.tone_scheme}` | sample/dataset: {sample}", ""]

    # ---- per-dataset overview ----
    L += ["## Per-dataset overview", "",
          "| dataset | faces? | total | real | fake | unlabeled | "
          "face rate | est. faces |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lc = r["label_counts_full"]
        L.append(
            f"| {r['name']} | {'yes' if r['has_faces'] else 'NO'} "
            f"| {r['n_total']} | {lc.get('0', 0)} | {lc.get('1', 0)} "
            f"| {r.get('n_unlabeled', 0)} "
            f"| {r['face_rate_in_sample']*100:.1f}% "
            f"| {r['est_total_faces'] if r['has_faces'] else '—'} |")
    L.append("")

    # ---- the decision table: usable faces per cell ----
    agg: dict[tuple[str, int], int] = defaultdict(int)
    for r in results:
        for cell, est in r.get("est_faces_by_cell", {}).items():
            lab_part, tone = cell.split("/")
            agg[(tone, int(lab_part.replace("label", "")))] += est
    if agg:
        L += ["## Estimated usable faces per (skin tone x label)", "",
              "Extrapolated from the sample. **The scarcest cell caps a "
              "tone-balanced run.**", "",
              "| tone | real (0) | fake (1) | min of the two |",
              "|---|---|---|---|"]
        scarcest = None
        for t in tones:
            r0, r1 = agg.get((t, 0), 0), agg.get((t, 1), 0)
            m = min(r0, r1)
            scarcest = m if scarcest is None else min(scarcest, m)
            L.append(f"| {t} | {r0} | {r1} | {m} |")
        L += ["",
              f"**Scarcest cell ≈ {scarcest} images.** A fully tone-balanced "
              f"run can hold about **{scarcest} per cell** "
              f"(~{scarcest*len(tones)} per label) before that group runs out. "
              "Beyond this you must either accept tone imbalance "
              "(rely on the balanced sampler + fairness loss) or stop at the "
              "balanced cap.", ""]

    # ---- image geometry & health ----
    L += ["## Image dimensions & health (sampled)", "",
          "| dataset | width med (min–max) | height med (min–max) | "
          "read errors | channels |", "|---|---|---|---|---|"]
    for r in results:
        ws, hs = r.get("width_stats", {}), r.get("height_stats", {})
        if not ws:
            continue
        L.append(
            f"| {r['name']} | {ws['median']} ({ws['min']}–{ws['max']}) "
            f"| {hs['median']} ({hs['min']}–{hs['max']}) "
            f"| {r['n_read_error_in_sample']}/{r['n_sampled']} "
            f"| {r.get('channel_modes', {})} |")
    L.append("")

    # ---- flags ----
    flags = []
    for r in results:
        if r.get("n_unlabeled", 0) > 0:
            flags.append(f"`{r['name']}`: {r['n_unlabeled']} unlabelled images "
                         "— check label_rules with scripts/10_inspect_raw.py.")
        if r["has_faces"] and r["face_rate_in_sample"] < 0.3:
            flags.append(f"`{r['name']}`: low face yield "
                         f"({r['face_rate_in_sample']*100:.0f}%) — many non-face "
                         "images, or Haar missing faces (try mtcnn).")
        if r.get("n_read_error_in_sample", 0) > 0:
            flags.append(f"`{r['name']}`: {r['n_read_error_in_sample']} unreadable "
                         "files in the sample — corrupt images present.")
        ws = r.get("width_stats", {})
        if ws and ws.get("median", 999) < 64:
            flags.append(f"`{r['name']}`: tiny images (median {ws['median']}px) "
                         "— upscaling to 224 will be heavy; may not be faces.")
    if flags:
        L += ["## Flags", ""] + [f"- {f}" for f in flags] + [""]

    L += ["## Plots", "",
          "- `class_balance.png` — real vs fake per dataset",
          "- `face_yield.png` — people-only yield per dataset",
          "- `tone_cells.png` — estimated faces per tone × label", ""]
    return "\n".join(L)
