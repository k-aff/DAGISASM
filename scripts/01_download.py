#!/usr/bin/env python3
"""Download the supervisor-provided Kaggle datasets into data/raw/."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sadf.config import load_config
from sadf.data.download import download_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    download_all(cfg)


if __name__ == "__main__":
    main()
