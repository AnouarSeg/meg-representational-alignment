"""Configuration loading and path resolution.

A thin, dependency-light wrapper around config/config.yaml. Paths in the YAML are
interpreted relative to the project root (the directory containing config/), so the
code works regardless of the current working directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the project root (two levels up from this file: src/thingsmeg/ -> root)."""
    return Path(__file__).resolve().parents[2]


@dataclass
class Config:
    """Parsed configuration with resolved absolute paths.

    Access raw nested values via `cfg.raw[...]`; common paths are exposed as
    attributes for convenience.
    """

    raw: dict[str, Any]
    root: Path
    path_raw: Path = field(init=False)
    path_derivatives: Path = field(init=False)
    path_results: Path = field(init=False)
    path_figures: Path = field(init=False)

    def __post_init__(self) -> None:
        p = self.raw["paths"]
        self.path_raw = (self.root / p["raw"]).resolve()
        self.path_derivatives = (self.root / p["derivatives"]).resolve()
        self.path_results = (self.root / p["results"]).resolve()
        self.path_figures = (self.root / p["figures"]).resolve()

    # convenience accessors -------------------------------------------------
    @property
    def openneuro_id(self) -> str:
        return self.raw["dataset"]["openneuro_id"]

    @property
    def random_seed(self) -> int:
        return int(self.raw["stats"]["random_seed"])

    # preprocessing
    @property
    def l_freq(self) -> float:
        return float(self.raw["preprocessing"]["l_freq"])

    @property
    def h_freq(self) -> float:
        return float(self.raw["preprocessing"]["h_freq"])

    @property
    def notch(self):
        return self.raw["preprocessing"]["notch"]

    @property
    def resample_sfreq(self) -> float:
        return float(self.raw["preprocessing"]["resample_sfreq"])

    @property
    def tmin(self) -> float:
        return float(self.raw["preprocessing"]["tmin"])

    @property
    def tmax(self) -> float:
        return float(self.raw["preprocessing"]["tmax"])

    @property
    def baseline(self) -> list:
        return self.raw["preprocessing"]["baseline"]

    @property
    def ica_n_components(self):
        return self.raw["preprocessing"]["ica"]["n_components"]

    @property
    def ica_method(self) -> str:
        return self.raw["preprocessing"]["ica"]["method"]

    def ensure_dirs(self) -> None:
        for d in (self.path_raw, self.path_derivatives, self.path_results, self.path_figures):
            d.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> Config:
    """Load config/config.yaml (or an explicit path) into a Config object."""
    root = project_root()
    cfg_path = Path(path) if path is not None else root / "config" / "config.yaml"
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, root=root)
