
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sadf.config import apply_overrides, apply_recipe, load_config
from sadf.data.dataset import build_group_vocab, make_loader
from sadf.models.detector import build_model
from sadf.training.trainer import Trainer
from sadf.utils.common import ensure_dir, get_logger, resolve_device, save_json, seed_everything

log = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--recipe", default=None, help="baseline | fair | (none)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--exclude-dataset", default=None,
                    help="Hold out one dataset for leave-one-dataset-out runs")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("-o", "--override", nargs="*", default=None,
                    help="Dotted config overrides key=value")
    args = ap.parse_args()

    cfg = apply_overrides(apply_recipe(load_config(args.config), args.recipe),
                          args.override)
    seed_everything(int(cfg.seed))
    device = resolve_device(args.device)

    run_name = args.run_name or (args.recipe or "run")
    if args.exclude_dataset:
        run_name += f"_lodo-{args.exclude_dataset}"
    run_dir = ensure_dir(Path(cfg.paths.runs) / run_name)

    # ------------------------------------------------------------------
    manifest = pd.read_csv(Path(cfg.paths.manifests) / "manifest.csv")
    trainable = {d["name"] for d in cfg.datasets if d.get("use_in_training", True)}
    df = manifest[(manifest["status"] == "ok") & manifest["dataset"].isin(trainable)]
    if args.exclude_dataset:
        df = df[df["dataset"] != args.exclude_dataset]
    if df.empty:
        raise SystemExit("No trainable rows in manifest — check earlier stages.")

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    vocab = build_group_vocab(train_df, cfg.group.key, list(cfg.group.skin_tone_categories))
    log.info("Device=%s | recipe=%s | groups=%s | train=%d val=%d",
             device, args.recipe, vocab, len(train_df), len(val_df))

    train_loader = make_loader(train_df, cfg, vocab, train=True,
                               balanced=bool(cfg.train.balanced_sampler))
    val_loader = make_loader(val_df, cfg, vocab, train=False)

    model = build_model(cfg, n_groups=len(vocab))
    trainer = Trainer(cfg, model, vocab, run_dir, device)
    result = trainer.fit(train_loader, val_loader)

    save_json({"recipe": args.recipe, "exclude_dataset": args.exclude_dataset,
               "groups": vocab, "best_selection_score": result["best_selection_score"],
               "config": cfg.to_dict()}, run_dir / "run_meta.json")
    log.info("Done. Best checkpoint: %s", run_dir / "best.pt")


if __name__ == "__main__":
    main()
