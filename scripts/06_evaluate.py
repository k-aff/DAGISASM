#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sadf.config import _dotify  # noqa: E402
from sadf.data.dataset import make_loader
from sadf.evaluation.report import generate_report
from sadf.models.detector import build_model
from sadf.training.trainer import Trainer
from sadf.utils.common import get_logger, resolve_device, seed_everything

log = get_logger()


def load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = _dotify(ckpt["config"])
    model = build_model(cfg, n_groups=len(ckpt["group_vocab"]))
    model.load_state_dict(ckpt["model_state"])
    return model, cfg, ckpt["group_vocab"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Restrict to these source datasets")
    ap.add_argument("--per-dataset", action="store_true",
                    help="Additionally write one sub-report per dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, cfg, vocab = load_checkpoint(args.checkpoint, device)
    seed_everything(int(cfg.seed))

    manifest = pd.read_csv(Path(cfg.paths.manifests) / "manifest.csv")
    df = manifest[manifest["status"] == "ok"]
    if args.split != "all":
        df = df[df["split"] == args.split]
    if args.datasets:
        df = df[df["dataset"].isin(args.datasets)]
    if df.empty:
        raise SystemExit("No rows match the requested split/dataset filter.")
    df = df.reset_index(drop=True)

    trainer = Trainer(cfg, model, vocab, args.out, device)
    loader = make_loader(df, cfg, vocab, train=False)
    preds = trainer.predict(loader)

    scored = df.loc[preds["row"].values].reset_index(drop=True)
    scored["score"] = preds["score"].values
    group_cols = ["skin_tone", "race_proxy", "dataset"]

    generate_report(
        scored, args.out, group_cols,
        threshold=float(cfg.train.threshold),
        min_group_size=int(cfg.eval.min_group_size),
        n_boot=int(cfg.eval.bootstrap), seed=int(cfg.seed),
        title=f"Evaluation — {Path(args.checkpoint).parent.name} / {args.split}",
    )

    if args.per_dataset:
        for name, sub in scored.groupby("dataset"):
            generate_report(
                sub.reset_index(drop=True), Path(args.out) / f"dataset_{name}",
                ["skin_tone", "race_proxy"],
                threshold=float(cfg.train.threshold),
                min_group_size=int(cfg.eval.min_group_size),
                n_boot=int(cfg.eval.bootstrap), seed=int(cfg.seed),
                title=f"Generalisation — {name} / {args.split}",
            )


if __name__ == "__main__":
    main()
