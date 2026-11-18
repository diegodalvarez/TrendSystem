"""Load config/sources.yaml and resolve paths.

Paths are found from the location of this file, never from os.getcwd(), so
code works no matter which folder a script or notebook is run from.

Quick check from a terminal:
    python -m trendsys.config
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

VALID_MODES = {"full", "names_only"}
VALID_VENDORS = {"bbg", "yahoo", "thetadata", "cboe", "firstratedata", "other"}


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Walk up from this file until a folder holding config/sources.yaml is found.

    Set the TRENDSYS_ROOT environment variable to override.
    """
    env = os.environ.get("TRENDSYS_ROOT")
    if env:
        return Path(env).resolve()

    here = (start or Path(__file__)).resolve()
    for folder in [here, *here.parents]:
        if (folder / "config" / "sources.yaml").is_file():
            return folder
    raise FileNotFoundError(
        "Could not find config/sources.yaml above {}. "
        "Set TRENDSYS_ROOT to the repo folder.".format(here)
    )


@dataclass
class Source:
    name: str
    root: Optional[Path]
    vendor: str
    mode: str = "full"
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def configured(self) -> bool:
        return self.root is not None

    @property
    def exists(self) -> bool:
        return self.root is not None and self.root.is_dir()


@dataclass
class Config:
    repo_root: Path
    data_root: Path
    catalog: Path
    old_data: Path
    sources: Dict[str, Source]

    def source(self, name: str) -> Source:
        if name not in self.sources:
            raise KeyError(
                "Unknown source '{}'. Known: {}".format(name, ", ".join(sorted(self.sources)))
            )
        return self.sources[name]

    def configured_sources(self) -> List[Source]:
        return [s for s in self.sources.values() if s.configured]


def _resolve(repo_root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (repo_root / p)


def load_config(path: Optional[Path] = None) -> Config:
    repo_root = find_repo_root() if path is None else Path(path).resolve().parents[1]
    cfg_path = path or (repo_root / "config" / "sources.yaml")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    paths = raw.get("paths", {}) or {}
    errors: List[str] = []
    sources: Dict[str, Source] = {}

    for name, spec in (raw.get("sources", {}) or {}).items():
        spec = spec or {}
        root_str = (spec.get("root") or "").strip()
        vendor = spec.get("vendor", "other")
        mode = spec.get("mode", "full")

        if vendor not in VALID_VENDORS:
            errors.append("{}: vendor '{}' not in {}".format(name, vendor, sorted(VALID_VENDORS)))
        if mode not in VALID_MODES:
            errors.append("{}: mode '{}' not in {}".format(name, mode, sorted(VALID_MODES)))

        sources[name] = Source(
            name=name,
            root=Path(root_str) if root_str else None,
            vendor=vendor,
            mode=mode,
            include=list(spec.get("include") or []),
            exclude=list(spec.get("exclude") or []),
            notes=spec.get("notes") or "",
        )

    if errors:
        raise ValueError("Problems in {}:\n  ".format(cfg_path) + "\n  ".join(errors))

    return Config(
        repo_root=repo_root,
        data_root=_resolve(repo_root, paths.get("data_root", "data")),
        catalog=_resolve(repo_root, paths.get("catalog", "data/catalog")),
        old_data=_resolve(repo_root, paths.get("old_data", "old_data")),
        sources=sources,
    )


def main() -> None:
    cfg = load_config()
    print("Repo root : {}".format(cfg.repo_root))
    print("Data root : {}".format(cfg.data_root))
    print("Catalog   : {}".format(cfg.catalog))
    print("Old data  : {}  (exists: {})".format(cfg.old_data, cfg.old_data.is_dir()))
    print()
    print("{:<26} {:<8} {:<11} {:<8} {}".format("source", "vendor", "mode", "status", "root"))
    for s in cfg.sources.values():
        if not s.configured:
            status = "empty"
        elif s.exists:
            status = "ok"
        else:
            status = "MISSING"
        print("{:<26} {:<8} {:<11} {:<8} {}".format(s.name, s.vendor, s.mode, status, s.root or ""))


if __name__ == "__main__":
    main()
