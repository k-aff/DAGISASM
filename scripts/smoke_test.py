#!/usr/bin/env python3
"""End-to-end smoke test on synthetic data (no downloads, CPU, ~2 min).

Builds a tiny fake "Kaggle" dataset with light- and dark-skin-toned
synthetic face patches, then drives every pipeline stage exactly as the
real scripts do: preprocess -> annotate -> split -> train (baseline and
fair recipes, tiny ViT, no pretrained weights) -> evaluate -> compare.
Asserts the expected artefacts exist. Run it after any code change.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sadf.config import _dotify, apply_recipe, load_config  # noqa: E402
from sadf.data.dataset import build_group_vocab, make_loader  # noqa: E402
from sadf.data.preprocess import run_preprocessing  # noqa: E402
from sadf.data.skin_tone import annotate_skin_tone  # noqa: E402
from sadf.data.splits import build_final_manifest  # noqa: E402
from sadf.evaluation.report import generate_report  # noqa: E402
from sadf.models.detector import build_model  # noqa: E402
from sadf.training.trainer import Trainer  # noqa: E402
from sadf.utils.common import get_logger, seed_everything  # noqa: E402

log = get_logger("smoke")
RNG = np.random.default_rng(0)


def synth_image(path: Path, skin: str, fake: bool, size: int = 96) -> None:
    """Skin-toned base + noise; 'fake' images get a high-freq checkerboard."""
    base = (222, 184, 155) if skin == "light" else (86, 55, 36)
    img = np.tile(np.array(base, np.uint8), (size, size, 1)).astype(np.int16)
    img += RNG.integers(-14, 14, img.shape)
    if fake:
        yy, xx = np.mgrid[0:size, 0:size]
        img += (((xx + yy) % 2) * 26 - 13)[..., None]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path, "JPEG", quality=95)


def build_synthetic_raw(raw: Path, n_per_cell: int = 24) -> None:
    for skin in ("light", "dark"):
        for label, sub in ((0, "real"), (1, "fake")):
            for i in range(n_per_cell):
                synth_image(raw / "toyset" / sub / f"{skin}_{i}.jpg", skin, bool(label))


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="sadf_smoke_"))
    log.info("Workspace: %s", work)
    try:
        cfg = load_config(REPO / "config" / "default.yaml")
        cfg["paths"] = {
            "raw": str(work / "raw"), "processed": str(work / "processed"),
            "manifests": str(work / "manifests"), "runs": str(work / "runs"),
        }
        cfg["datasets"] = [{
            "name": "toyset", "kaggle": "n/a", "has_faces": True,
            "use_in_training": True, "use_in_fairness": True,
            "label_rules": [{"pattern": "fake", "label": 1},
                            {"pattern": "real", "label": 0}],
        }]
        cfg["face"]["detector"] = "none"        # synthetic patches, skip detection
        cfg["model"].update({"backbone": "vit_tiny_patch16_224", "pretrained": False})
        cfg["image"]["size"] = 96
        cfg["train"].update({"epochs": 2, "batch_size": 16, "num_workers": 0,
                             "early_stop_patience": 5})
        cfg["eval"].update({"bootstrap": 20, "min_group_size": 5})
        cfg["group"]["min_group_size"] = 5
        cfg = _dotify(cfg.to_dict())
        seed_everything(int(cfg["seed"]))

        build_synthetic_raw(Path(cfg.paths.raw))

        # 02 preprocess -> 03 annotate -> 04 split ------------------------
        df = run_preprocessing(cfg)
        assert (df["status"] == "ok").sum() > 60, "preprocessing dropped too much"
        df = annotate_skin_tone(df, cfg)
        tones = set(df.loc[df.status == "ok", "skin_tone"])
        assert len(tones & {"very_light", "light", "intermediate", "tan",
                            "brown", "dark"}) >= 2, f"ITA failed: {tones}"
        df["race_proxy"] = "unknown"
        df = build_final_manifest(df, cfg)

        ok = df[df["status"] == "ok"]
        eval_dirs = []
        for recipe in ("baseline", "fair"):
            rcfg = apply_recipe(cfg, recipe)
            vocab = build_group_vocab(ok[ok.split == "train"], rcfg.group.key,
                                      list(rcfg.group.skin_tone_categories))
            model = build_model(rcfg, n_groups=len(vocab))
            run_dir = Path(rcfg.paths.runs) / recipe
            trainer = Trainer(rcfg, model, vocab, run_dir, "cpu")
            trainer.fit(
                make_loader(ok[ok.split == "train"], rcfg, vocab, True,
                            balanced=bool(rcfg.train.balanced_sampler)),
                make_loader(ok[ok.split == "val"], rcfg, vocab, False),
            )
            assert (run_dir / "best.pt").exists()

            test = ok[ok.split == "test"].reset_index(drop=True)
            preds = trainer.predict(make_loader(test, rcfg, vocab, False))
            scored = test.loc[preds["row"].values].reset_index(drop=True)
            scored["score"] = preds["score"].values
            out = run_dir / "eval"
            generate_report(scored, out, ["skin_tone"], min_group_size=5,
                            n_boot=20, title=f"smoke-{recipe}")
            for artefact in ("overall.json", "gaps.json", "report.md",
                             "per_group_skin_tone.csv"):
                assert (out / artefact).exists(), f"missing {artefact}"
            eval_dirs.append(out)

        # 07 compare -------------------------------------------------------
        import subprocess
        res = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "07_compare.py"),
             str(eval_dirs[0]), str(eval_dirs[1]),
             "--names", "baseline", "fair",
             "--out", str(work / "compare.md")],
            capture_output=True, text=True, check=True,
        )
        assert "Fairness gaps" in res.stdout
        log.info("SMOKE TEST PASSED — full pipeline runs end to end.")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
