
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sadf.config import load_config
from sadf.data.splits import build_final_manifest
from sadf.utils.common import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))

    df = pd.read_csv(Path(cfg.paths.manifests) / "annotated_manifest.csv")
    build_final_manifest(df, cfg)


if __name__ == "__main__":
    main()
