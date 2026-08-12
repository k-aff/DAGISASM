
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sadf.config import load_config
from sadf.data.race_proxy import annotate_with_deepface, merge_external_labels
from sadf.data.skin_tone import annotate_skin_tone
from sadf.utils.common import get_logger, seed_everything

log = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--race-model", choices=["none", "deepface"], default="none")
    ap.add_argument("--external-labels", default=None,
                    help="CSV with columns path,race_proxy (takes precedence)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))

    manifest_path = Path(cfg.paths.manifests) / "processed_manifest.csv"
    df = pd.read_csv(manifest_path)

    df = annotate_skin_tone(df, cfg)
    df["race_proxy"] = "unknown"
    if args.race_model == "deepface":
        df = annotate_with_deepface(df)
    if args.external_labels:
        df = merge_external_labels(df, args.external_labels)

    out = Path(cfg.paths.manifests) / "annotated_manifest.csv"
    df.to_csv(out, index=False)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
