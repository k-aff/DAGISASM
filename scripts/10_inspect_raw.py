#!/usr/bin/env python3
"""Report what is actually in data/raw/ and how labels would be inferred.

RUN THIS FIRST, before any harvesting. Kaggle datasets follow no shared
folder convention, so the label rules in the config are provisional
guesses until checked against the real tree.

    python scripts/10_inspect_raw.py --config config/pilot.yaml

Paste the output back if you want the label rules adjusted for you.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap() -> None:
    """Find the `sadf` package whether it sits in src/ or at the repo root."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        for cand in (parent / "src", parent):
            if (cand / "sadf" / "__init__.py").exists():
                sys.path.insert(0, str(cand))
                return
    raise ImportError("Could not locate the `sadf` package.")


_bootstrap()

from sadf.config import load_config  # noqa: E402
from sadf.data.ingest import inspect_tree, scan_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--max-dirs", type=int, default=40)
    args = ap.parse_args()
    cfg = load_config(args.config)

    raw_root = Path(cfg.paths.raw)
    print(f"\nRaw directory: {raw_root.resolve()}")
    if not raw_root.exists():
        print("  -> does not exist yet. Download first, or unzip datasets into it.")
        return

    print("\nTop-level contents:")
    for child in sorted(raw_root.iterdir()):
        kind = "dir " if child.is_dir() else "file"
        print(f"  [{kind}] {child.name}")

    for d in cfg.datasets:
        root = raw_root / d["name"]
        print("\n" + "=" * 72)
        print(f"DATASET: {d['name']}   (has_faces={d.get('has_faces', True)})")
        print("=" * 72)
        if not root.exists():
            print(f"  {root} not present — skipping.")
            continue

        print(inspect_tree(root, d["label_rules"], max_dirs=args.max_dirs))

        df = scan_dataset(root, d["label_rules"])
        if df.empty:
            continue
        print("\n  Sample relative paths:")
        for p in df["rel_path"].head(8):
            print(f"    {p}")
        unlabeled = df[df["label"].isna()]
        if not unlabeled.empty:
            print(f"\n  !! {len(unlabeled)} images matched NO rule. Examples:")
            for p in unlabeled["rel_path"].head(8):
                print(f"    {p}")
            print("  -> Add patterns to this dataset's label_rules in the config.")

    print("\nDone. Verify the inferred labels above before harvesting.\n")


if __name__ == "__main__":
    main()
