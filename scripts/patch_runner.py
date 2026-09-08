#!/usr/bin/env python3
r"""One-off patch for scripts/run_models140k.py.

Three changes, all needed before the augmented Swin run:

  1. --config FLAG
     MODELS hardcodes config/model140k_swin.yaml. Without an override there
     is no way to point the runner at model140k_swin_aug.yaml.

  2. CONFIGURABLE MANIFEST
     _load_split() hardcodes "manifest.csv". The augmented data lives in
     manifest_aug.csv. Now reads cfg.paths.manifest_file, defaulting to
     manifest.csv so existing configs are unaffected.

  3. TOLERATE A MISSING `status` COLUMN
     _load_split() does df[df["status"] == "ok"], which raises KeyError on
     manifest_aug.csv. Now filters only when the column exists.

Safe to run twice: it detects work already applied and skips it.

    python scripts\patch_runner.py
    python scripts\patch_runner.py --path scripts\run_models140k.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

OLD_LOAD = '''def _load_split(cfg) -> pd.DataFrame:
    path = Path(cfg.paths.manifests) / "manifest.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} missing. Run the shared harvest+split first:\\n"
            "  python scripts/run_pilot.py --config config/model_vit.yaml "
            "--stage harvest,split"
        )
    df = pd.read_csv(path)
    return df[df["status"] == "ok"]'''

NEW_LOAD = '''def _cfg_get(cfg, path: str, default):
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
    # Which manifest to train from. Defaults to manifest.csv so older configs
    # behave exactly as before; the augmented run sets manifest_file.
    name = str(_cfg_get(cfg, "paths.manifest_file", "manifest.csv"))
    path = Path(cfg.paths.manifests) / name
    if not path.exists():
        raise SystemExit(
            f"{path} missing. Run the shared harvest+split first:\\n"
            "  python scripts/run_pilot.py --config config/model_vit.yaml "
            "--stage harvest,split"
        )
    df = pd.read_csv(path, low_memory=False)

    # `status` comes from the harvester. Manifests built by
    # build_augmented_140k.py have no such column - every row is usable.
    if "status" in df.columns:
        df = df[df["status"] == "ok"]

    if "split" not in df.columns:
        raise SystemExit(f"{path} has no 'split' column.")

    # Leakage guard: augmented copies carry source_id pointing at their
    # original. No source image may appear in more than one split.
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
    return df'''

OLD_ARG = '''    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    args = ap.parse_args()'''

NEW_ARG = '''    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--config", default=None,
                    help="Override the config file for this run "
                         "(e.g. config/model140k_swin_aug.yaml)")
    args = ap.parse_args()

    if args.config:
        if not Path(args.config).exists():
            raise SystemExit(f"--config {args.config} not found")
        targets = [args.train] if args.train else list(MODELS)
        for k in targets:
            MODELS[k] = args.config
        log.info("Config override -> %s for: %s", args.config, ", ".join(targets))'''

EDITS = [
    ("manifest loader", OLD_LOAD, NEW_LOAD, "_cfg_get(cfg, \"paths.manifest_file\""),
    ("--config flag", OLD_ARG, NEW_ARG, 'ap.add_argument("--config"'),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="scripts/run_models140k.py")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        sys.exit(f"ERROR: {p} not found. Pass --path explicitly.")

    text = p.read_text(encoding="utf-8")
    original = text
    applied, skipped = [], []

    for name, old, new, marker in EDITS:
        if marker in text:
            skipped.append(f"{name} (already patched)")
            continue
        if old not in text:
            sys.exit(
                f"ERROR: could not locate the '{name}' block to replace.\n"
                f"       The file may differ from the expected version.\n"
                f"       Nothing has been written."
            )
        text = text.replace(old, new, 1)
        applied.append(name)

    if text == original:
        print("Nothing to do - all edits already present.")
        return

    backup = p.with_suffix(p.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(p, backup)
        print(f"backup -> {backup}")
    p.write_text(text, encoding="utf-8")

    for a in applied:
        print(f"  applied: {a}")
    for s in skipped:
        print(f"  skipped: {s}")

    import py_compile
    try:
        py_compile.compile(str(p), doraise=True)
        print(f"\nOK: {p} patched and compiles cleanly.")
    except py_compile.PyCompileError as exc:
        shutil.copy2(backup, p)
        sys.exit(f"\nSyntax error after patch - reverted from backup.\n{exc}")


if __name__ == "__main__":
    main()
