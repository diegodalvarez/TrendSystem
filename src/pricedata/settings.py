"""Find the repo root and load the YAML config files."""

from __future__ import annotations

from pathlib import Path

import yaml


def repo_root(start: Path | None = None) -> Path:
    """Walk up from this file (or `start`) to the folder that holds config/sources.yaml."""
    here = Path(start or __file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "config" / "sources.yaml").exists():
            return p
    raise FileNotFoundError("could not find config/sources.yaml above " + str(here))


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_sources(root: Path) -> dict:
    return load_yaml(root / "config" / "sources.yaml").get("sources", {}) or {}


def load_bbg_store_config(root: Path) -> dict:
    return load_yaml(root / "config" / "bbg_store.yaml")
