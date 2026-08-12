from __future__ import annotations

import os
from pathlib import Path

from ..config import DotDict
from ..utils.common import ensure_dir, get_logger

log = get_logger()


def _have_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def download_dataset(slug: str) -> Path:
    import kagglehub

    path = kagglehub.dataset_download(slug)
    return Path(path)


def link_into_raw(src: Path, raw_dir: Path, name: str) -> Path:
    dst = raw_dir / name
    if dst.is_symlink() or dst.exists():
        return dst
    try:
        dst.symlink_to(src, target_is_directory=True)
    except OSError:  # e.g. filesystems without symlink support
        import shutil

        log.warning("Symlink unsupported; copying %s -> %s (may be large)", src, dst)
        shutil.copytree(src, dst)
    return dst


def download_all(cfg: DotDict) -> dict[str, Path]:
    raw_dir = ensure_dir(cfg.paths.raw)
    if not _have_credentials():
        raise RuntimeError(
            "Kaggle credentials not found. Create an API token at "
            "kaggle.com -> Account -> Create New Token, then either place "
            "kaggle.json in ~/.kaggle/ or export KAGGLE_USERNAME and KAGGLE_KEY. "
            "Alternatively, download each dataset manually in the browser and "
            "unzip it into data/raw/<dataset-name>/ — the rest of the pipeline "
            "only needs the files to exist there."
        )
    resolved: dict[str, Path] = {}
    for d in cfg.datasets:
        log.info("Downloading %s (%s) ...", d["name"], d["kaggle"])
        src = download_dataset(d["kaggle"])
        dst = link_into_raw(src, raw_dir, d["name"])
        n_files = sum(1 for _ in dst.rglob("*") if _.is_file())
        log.info("  -> %s (%d files)", dst, n_files)
        resolved[d["name"]] = dst
    return resolved
