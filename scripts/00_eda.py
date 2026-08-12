#!/usr/bin/env python3
"""Step 0 — Exploratory data analysis over the raw pool.

Run this BEFORE harvesting a full run. It samples each dataset and
reports class balance, people-only face yield, skin-tone distribution,
image dimensions and corrupt-file counts — then tells you the scarcest
(tone x label) cell, which caps a tone-balanced run.

    python scripts/00_eda.py --config config/pilot.yaml
    python scripts/00_eda.py --config config/pilot.yaml --sample 5000

--sample is images inspected PER dataset (random). Larger = slower but
more accurate face-yield and tone estimates. 0 = inspect everything.
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

from sadf.config import apply_overrides, load_config  # noqa: E402
from sadf.data.eda import run_eda  # noqa: E402
from sadf.utils.common import seed_everything  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--sample", type=int, default=3000,
                    help="Images inspected per dataset (0 = all)")
    ap.add_argument("-o", "--override", nargs="*", default=None)
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    seed_everything(int(cfg.seed))
    run_eda(cfg, sample=args.sample)


if __name__ == "__main__":
    main()
