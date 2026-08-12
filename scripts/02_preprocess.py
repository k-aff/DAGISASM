#!/usr/bin/env python3
"""Face-crop, resize and deduplicate raw images -> processed_manifest.csv.

Run with --dry-run FIRST after downloading: it prints each dataset's
directory tree and the labels inferred by the config's label_rules so
you can verify (and adjust) them before spending time on processing.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sadf.config import load_config
from sadf.data.ingest import inspect_tree
from sadf.data.preprocess import run_preprocessing
from sadf.utils.common import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only inspect folder layouts and label inference")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))

    if args.dry_run:
        for d in cfg.datasets:
            root = Path(cfg.paths.raw) / d["name"]
            print()
            print(inspect_tree(root, d["label_rules"]) if root.exists()
                  else f"{root}: not downloaded yet.")
        return

    run_preprocessing(cfg)


if __name__ == "__main__":
    main()
