from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from bayesian_arima_ts.paths import DEFAULT_CONFIG_PATH


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def configure_logging(cfg: dict[str, Any]) -> None:
    level_name = (cfg.get("logging") or {}).get("level", "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
