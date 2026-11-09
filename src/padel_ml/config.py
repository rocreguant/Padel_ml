"""Central configuration. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("PADEL_ML_DATA_DIR", "data")))

    # Scraping etiquette
    user_agent: str = field(
        default_factory=lambda: _env(
            "PADEL_ML_USER_AGENT", "padel_ml/0.1 (personal research project)"
        )
    )
    request_delay_s: float = field(default_factory=lambda: float(_env("PADEL_ML_DELAY", "2.0")))
    request_timeout_s: float = field(default_factory=lambda: float(_env("PADEL_ML_TIMEOUT", "30")))
    cache_ttl_hours: float = field(default_factory=lambda: float(_env("PADEL_ML_CACHE_TTL_H", "12")))

    # Betting defaults
    bankroll: float = field(default_factory=lambda: float(_env("PADEL_ML_BANKROLL", "100")))
    kelly_fraction: float = field(default_factory=lambda: float(_env("PADEL_ML_KELLY", "0.25")))
    min_edge: float = field(default_factory=lambda: float(_env("PADEL_ML_MIN_EDGE", "0.05")))
    max_stake_pct: float = field(default_factory=lambda: float(_env("PADEL_ML_MAX_STAKE", "0.05")))

    @property
    def db_path(self) -> Path:
        return Path(_env("PADEL_ML_DB", str(self.data_dir / "padel.db")))

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def report_dir(self) -> Path:
        return self.data_dir / "reports"


def get_settings() -> Settings:
    return Settings()
