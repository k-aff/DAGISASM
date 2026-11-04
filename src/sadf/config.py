from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class DotDict(dict):
    """dict subclass allowing cfg.model.backbone style access."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:  # pragma: no cover
            raise AttributeError(key) from e

    __setattr__ = dict.__setitem__

    def to_dict(self) -> dict:
        return _undot(self)


def _dotify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return DotDict({k: _dotify(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_dotify(v) for v in obj]
    return obj


def _undot(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _undot(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_undot(v) for v in obj]
    return obj


def load_config(path: str | Path) -> DotDict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _dotify(raw)


def set_dotted(cfg: DotDict, dotted_key: str, value: Any) -> None:
    """Set cfg['a']['b']['c'] via 'a.b.c'."""
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = DotDict()
        node = node[k]
    node[keys[-1]] = value


def apply_recipe(cfg: DotDict, recipe_name: str | None) -> DotDict:
    """Apply a named recipe (dotted-key overrides) from cfg.recipes."""
    cfg = _dotify(copy.deepcopy(cfg.to_dict()))
    if not recipe_name:
        return cfg
    recipes = cfg.get("recipes", {}) or {}
    if recipe_name not in recipes:
        raise KeyError(
            f"Unknown recipe '{recipe_name}'. Available: {list(recipes)}"
        )
    for dotted, value in recipes[recipe_name].items():
        set_dotted(cfg, dotted, value)
    return cfg


def apply_overrides(cfg: DotDict, overrides: list[str] | None) -> DotDict:
    """Apply CLI overrides of the form key.subkey=value (YAML-parsed)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key=value")
        key, value = item.split("=", 1)
        set_dotted(cfg, key.strip(), yaml.safe_load(value))
    return cfg
