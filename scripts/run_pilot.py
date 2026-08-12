#!/usr/bin/env python3
"""Run the complete pilot: data -> model -> bias audit -> comparison.

    python scripts/run_pilot.py                       # everything
    python scripts/run_pilot.py --stage harvest       # one stage
    python scripts/run_pilot.py --stage train,eval    # a subset, in order

Stages
------
harvest  500 real + 500 fake PEOPLE-ONLY images, tone-balanced (phase 1)
         with top-up (phase 2). Writes data/pilot/processed + manifest +
         pilot_summary.md.
split    Stratified 70/15/15 over (dataset, label, skin_tone).
train    Baseline recipe (plain BCE ViT — the RQ1 audit subject) and
         fair recipe (DAW-FDD + balanced sampler + adversarial
         disentanglement — RQ2), 3 epochs each per config.
eval     Test-split evaluation of both checkpoints: overall metrics,
         per-tone tables, FPR-parity gap and dAUC with bootstrap CIs,
         plots, report.md per run.
report   Side-by-side baseline-vs-fair table -> runs/pilot/compare.md.

Show your supervisor: data/pilot/manifests/pilot_summary.md,
runs/pilot/baseline/eval/report.md, runs/pilot/fair/eval/report.md,
runs/pilot/compare.md.

Prereq: run scripts/10_inspect_raw.py --config config/pilot.yaml first
and confirm label inference on your actual folders.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        for cand in (parent / "src", parent):
            if (cand / "sadf" / "__init__.py").exists():
                sys.path.insert(0, str(cand))
                return
    raise ImportError("Could not locate the `sadf` package.")


_bootstrap()

import pandas as pd  # noqa: E402

from sadf.config import apply_overrides, apply_recipe, load_config  # noqa: E402
from sadf.utils.common import (  # noqa: E402
    ensure_dir, get_logger, resolve_device, save_json, seed_everything,
)

log = get_logger()
ALL_STAGES = ["harvest", "split", "train", "eval", "report"]


# ----------------------------------------------------------------------
def stage_harvest(cfg) -> None:
    from sadf.data.pilot import coverage_report, harvest

    df = harvest(cfg)
    print()
    print(coverage_report(df, cfg))


def stage_split(cfg) -> None:
    from sadf.data.splits import build_final_manifest

    src = Path(cfg.paths.manifests) / "pilot_manifest.csv"
    df = pd.read_csv(src)
    df["race_proxy"] = "unknown"     # optional axis; not annotated in the pilot
    build_final_manifest(df, cfg)    # -> manifests/manifest.csv with `split`


def _load_split_manifest(cfg) -> pd.DataFrame:
    path = Path(cfg.paths.manifests) / "manifest.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing — run the split stage first.")
    df = pd.read_csv(path)
    return df[df["status"] == "ok"]


def stage_train(cfg, device: str) -> None:
    from sadf.data.dataset import build_group_vocab, make_loader
    from sadf.models.detector import build_model
    from sadf.training.trainer import Trainer

    base = _load_split_manifest(cfg)
    for recipe in ("baseline", "fair"):
        rcfg = apply_recipe(cfg, recipe)
        run_dir = ensure_dir(Path(rcfg.paths.runs) / recipe)
        train_df = base[base["split"] == "train"]
        val_df = base[base["split"] == "val"]
        vocab = build_group_vocab(train_df, rcfg.group.key,
                                  list(rcfg.group.skin_tone_categories))
        log.info("== TRAIN %s == device=%s groups=%s train=%d val=%d epochs=%s",
                 recipe, device, vocab, len(train_df), len(val_df),
                 rcfg.train.epochs)
        model = build_model(rcfg, n_groups=len(vocab))
        trainer = Trainer(rcfg, model, vocab, run_dir, device)
        result = trainer.fit(
            make_loader(train_df, rcfg, vocab, train=True,
                        balanced=bool(rcfg.train.balanced_sampler)),
            make_loader(val_df, rcfg, vocab, train=False),
        )
        save_json({"recipe": recipe, "groups": vocab,
                   "best_selection_score": result["best_selection_score"],
                   "config": rcfg.to_dict()}, run_dir / "run_meta.json")


def stage_eval(cfg, device: str) -> None:
    import torch

    from sadf.config import _dotify
    from sadf.data.dataset import make_loader
    from sadf.evaluation.report import generate_report
    from sadf.models.detector import build_model
    from sadf.training.trainer import Trainer

    base = _load_split_manifest(cfg)
    test = base[base["split"] == "test"].reset_index(drop=True)
    for recipe in ("baseline", "fair"):
        ckpt_path = Path(cfg.paths.runs) / recipe / "best.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"{ckpt_path} missing — run the train stage first.")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        rcfg = _dotify(ckpt["config"])
        vocab = ckpt["group_vocab"]
        model = build_model(rcfg, n_groups=len(vocab))
        model.load_state_dict(ckpt["model_state"])

        trainer = Trainer(rcfg, model, vocab, Path(cfg.paths.runs) / recipe, device)
        preds = trainer.predict(make_loader(test, rcfg, vocab, train=False))
        scored = test.loc[preds["row"].values].reset_index(drop=True)
        scored["score"] = preds["score"].values

        generate_report(
            scored, Path(cfg.paths.runs) / recipe / "eval",
            group_cols=["skin_tone", "dataset"],
            threshold=float(rcfg.train.threshold),
            min_group_size=int(cfg.eval.min_group_size),
            n_boot=int(cfg.eval.bootstrap), seed=int(cfg.seed),
            title=f"Pilot {recipe} — test split",
        )


def stage_report(cfg) -> None:
    import subprocess

    runs = Path(cfg.paths.runs)
    out = runs / "compare.md"
    script = Path(__file__).with_name("07_compare.py")
    subprocess.run(
        [sys.executable, str(script),
         str(runs / "baseline" / "eval"), str(runs / "fair" / "eval"),
         "--names", "baseline", "fair", "--group", "skin_tone",
         "--out", str(out)],
        check=True,
    )
    log.info("Pilot complete. Show: %s, %s, %s, %s",
             Path(cfg.paths.manifests) / str(cfg.pilot.report_name),
             runs / "baseline" / "eval" / "report.md",
             runs / "fair" / "eval" / "report.md", out)


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--stage", default="all",
                    help="all | comma list of: " + ",".join(ALL_STAGES))
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("-o", "--override", nargs="*", default=None,
                    help="Dotted overrides, e.g. pilot.target_per_label=100")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    seed_everything(int(cfg.seed))
    stages = ALL_STAGES if args.stage == "all" else [
        s.strip() for s in args.stage.split(",")]
    bad = set(stages) - set(ALL_STAGES)
    if bad:
        raise SystemExit(f"Unknown stage(s): {bad}. Valid: {ALL_STAGES}")

    device = resolve_device(args.device)
    for s in stages:
        log.info(">>> STAGE: %s", s)
        if s == "harvest":
            stage_harvest(cfg)
        elif s == "split":
            stage_split(cfg)
        elif s == "train":
            stage_train(cfg, device)
        elif s == "eval":
            stage_eval(cfg, device)
        elif s == "report":
            stage_report(cfg)


if __name__ == "__main__":
    main()
