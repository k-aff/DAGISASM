#!/usr/bin/env python3
"""Train and evaluate the detector on the 140k manifests.

Final models (balanced, augmented training set):

    python scripts/run_models140k.py --train swin --recipe baseline --config config/model140k_swin_aug.yaml
    python scripts/run_models140k.py --train swin --recipe fair     --config config/model140k_swin_aug.yaml
    python scripts/run_models140k.py --eval swin --recipe baseline  --config config/model140k_swin_aug.yaml
    python scripts/run_models140k.py --eval swin --recipe fair      --config config/model140k_swin_aug.yaml

Preliminary backbone comparison (manifest.csv from build_140k_race.py):

    python scripts/run_models140k.py --train vit
    python scripts/run_models140k.py --train xception
    python scripts/run_models140k.py --train swin
    python scripts/run_models140k.py --eval-all
    python scripts/run_models140k.py --compare

Checkpoints and evaluation reports are written under each config's paths.runs.
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

from sadf.config import apply_recipe, load_config  # noqa: E402
from sadf.utils.common import (  # noqa: E402
    ensure_dir, get_logger, resolve_device, save_json, seed_everything,
)

log = get_logger()

MODELS = {
    "vit": "config/model140k_vit.yaml",
    "xception": "config/model140k_xception.yaml",
    "swin": "config/model140k_swin.yaml",
}
PRETTY = {"vit": "ViT-Small", "xception": "Xception", "swin": "Swin-Tiny"}


def _cfg_get(cfg, path: str, default):
    """Safe nested config lookup: _cfg_get(cfg, 'paths.manifest_file', 'manifest.csv')."""
    node = cfg
    for part in path.split("."):
        try:
            node = node[part] if isinstance(node, dict) else getattr(node, part)
        except (KeyError, AttributeError, TypeError):
            return default
        if node is None:
            return default
    return node


def _load_split(cfg) -> pd.DataFrame:
    name = str(_cfg_get(cfg, "paths.manifest_file", "manifest.csv"))
    path = Path(cfg.paths.manifests) / name
    if not path.exists():
        raise SystemExit(f"{path} missing. Build the manifest first (see README).")
    df = pd.read_csv(path, low_memory=False)

    # manifest_aug.csv has no status column; every row in it is usable
    if "status" in df.columns:
        df = df[df["status"] == "ok"]

    if "split" not in df.columns:
        raise SystemExit(f"{path} has no 'split' column.")

    # augmented copies carry the source_id of their original, so no source
    # image may appear in more than one split
    if "source_id" in df.columns:
        spread = df.groupby("source_id")["split"].nunique()
        leaked = spread[spread > 1]
        if len(leaked):
            raise SystemExit(
                f"{len(leaked)} source images span multiple splits "
                f"(e.g. {list(leaked.index[:3])}). Training would be invalid."
            )
        log.info("Leakage check passed: %d unique sources across %d rows",
                 df["source_id"].nunique(), len(df))

    if "is_augmented" in df.columns:
        outside = int(df.loc[df["split"] != "train", "is_augmented"].sum())
        if outside:
            raise SystemExit(
                f"{outside} augmented rows outside the train split. "
                "Val and test must be originals only."
            )
        log.info("Augmented rows: %d (train only)", int(df["is_augmented"].sum()))

    log.info("Manifest %s -> %d rows", path, len(df))
    return df


def train_one(key: str, recipe: str, device: str) -> None:
    from sadf.data.dataset import build_group_vocab, make_loader
    from sadf.models.detector import build_model
    from sadf.training.trainer import Trainer

    cfg = apply_recipe(load_config(MODELS[key]), recipe)
    seed_everything(int(cfg.seed))
    base = _load_split(cfg)
    train_df, val_df = base[base.split == "train"], base[base.split == "val"]
    vocab = build_group_vocab(train_df, cfg.group.key,
                              list(cfg.group.skin_tone_categories))
    run_dir = ensure_dir(Path(cfg.paths.runs) / recipe)
    log.info("=== TRAIN %s (%s) | backbone=%s | device=%s | train=%d val=%d ===",
             PRETTY[key], recipe, cfg.model.backbone, device, len(train_df), len(val_df))
    model = build_model(cfg, n_groups=len(vocab))
    trainer = Trainer(cfg, model, vocab, run_dir, device)
    res = trainer.fit(
        make_loader(train_df, cfg, vocab, True, balanced=bool(cfg.train.balanced_sampler)),
        make_loader(val_df, cfg, vocab, False),
    )
    save_json({"model": key, "backbone": cfg.model.backbone, "recipe": recipe,
               "groups": vocab, "best_selection_score": res["best_selection_score"]},
              run_dir / "run_meta.json")
    log.info("Done %s -> %s", PRETTY[key], run_dir / "best.pt")


def eval_one(key: str, recipe: str, device: str) -> None:
    import torch

    from sadf.config import _dotify
    from sadf.data.dataset import make_loader
    from sadf.evaluation.report import generate_report
    from sadf.models.detector import build_model
    from sadf.training.trainer import Trainer

    cfg = load_config(MODELS[key])
    ckpt_path = Path(cfg.paths.runs) / recipe / "best.pt"
    if not ckpt_path.exists():
        log.warning("Skip %s: no checkpoint at %s", PRETTY[key], ckpt_path)
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    rcfg = _dotify(ckpt["config"])
    vocab = ckpt["group_vocab"]
    model = build_model(rcfg, n_groups=len(vocab))
    model.load_state_dict(ckpt["model_state"])

    base = _load_split(rcfg)
    test = base[base.split == "test"].reset_index(drop=True)
    trainer = Trainer(rcfg, model, vocab, Path(cfg.paths.runs) / recipe, device)
    preds = trainer.predict(make_loader(test, rcfg, vocab, False))
    scored = test.loc[preds["row"].values].reset_index(drop=True)
    scored["score"] = preds["score"].values
    generate_report(
        scored, Path(cfg.paths.runs) / recipe / "eval",
        group_cols=["skin_tone"], threshold=float(rcfg.train.threshold),
        min_group_size=int(cfg.eval.min_group_size),
        n_boot=int(cfg.eval.bootstrap), seed=int(cfg.seed),
        title=f"{PRETTY[key]} ({recipe}) — test",
    )
    log.info("Evaluated %s", PRETTY[key])


def compare(recipe: str) -> None:
    from sadf.utils.common import load_json

    rows, gap_rows = [], []
    for key in MODELS:
        cfg = load_config(MODELS[key])
        eval_dir = Path(cfg.paths.runs) / recipe / "eval"
        ov = eval_dir / "overall.json"
        gp = eval_dir / "gaps.json"
        if not ov.exists():
            log.warning("No eval for %s — run --eval-all first.", PRETTY[key])
            continue
        o = load_json(ov)
        rows.append((PRETTY[key], o))
        g = load_json(gp).get("skin_tone", {}) if gp.exists() else {}
        gap_rows.append((PRETTY[key], g))

    if not rows:
        raise SystemExit("Nothing to compare. Train and evaluate first.")

    lines = ["# Backbone comparison", "",
             f"Recipe: `{recipe}` | shared data, identical splits", "",
             "## Detection performance (test split)", "",
             "| model | accuracy | precision | recall | AUC | FPR |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, o in rows:
        lines.append(f"| {name} | {o['accuracy']:.4f} | {o['precision']:.4f} | "
                     f"{o['recall']:.4f} | {o['auc']:.4f} | {o['fpr']:.4f} |")

    lines += ["", "## Fairness across FairFace race groups", "",
              "| model | FPR gap | TPR gap | AUC gap | worst-group AUC |",
              "| --- | --- | --- | --- | --- |"]
    for name, g in gap_rows:
        if g:
            lines.append(f"| {name} | {g.get('fpr_gap', float('nan')):.4f} | "
                         f"{g.get('tpr_gap', float('nan')):.4f} | "
                         f"{g.get('auc_gap', float('nan')):.4f} | "
                         f"{g.get('worst_group_auc', float('nan')):.4f} |")
    lines.append("")

    out = ensure_dir("runs/models140k") / f"comparison_{recipe}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    log.info("Wrote %s", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", choices=list(MODELS), help="Train one model")
    ap.add_argument("--eval", choices=list(MODELS), help="Evaluate one model")
    ap.add_argument("--eval-all", action="store_true", help="Evaluate all trained models")
    ap.add_argument("--compare", action="store_true", help="Build the comparison table")
    ap.add_argument("--recipe", default="baseline", help="baseline | fair")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--config", default=None,
                    help="use this config instead of the default one for the model(s)")
    args = ap.parse_args()

    if args.config:
        if not Path(args.config).exists():
            raise SystemExit(f"--config {args.config} not found")
        targets = [k for k in (args.train, args.eval) if k] or list(MODELS)
        for k in targets:
            MODELS[k] = args.config
        log.info("Config override -> %s for: %s", args.config, ", ".join(targets))
    device = resolve_device(args.device)

    if args.train:
        train_one(args.train, args.recipe, device)
    if args.eval and not args.eval_all:
        eval_one(args.eval, args.recipe, device)
    if args.eval_all:
        for key in MODELS:
            eval_one(key, args.recipe, device)
    if args.compare:
        compare(args.recipe)
    if not (args.train or args.eval or args.eval_all or args.compare):
        ap.print_help()


if __name__ == "__main__":
    main()
