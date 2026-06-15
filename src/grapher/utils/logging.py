from __future__ import annotations

import logging
import os


def configure_logging(level: str | int | None = None) -> None:
    raw_level = level or os.environ.get("GRAPHER_LOG_LEVEL", "INFO")
    if isinstance(raw_level, str):
        resolved = getattr(logging, raw_level.upper(), logging.INFO)
    else:
        resolved = int(raw_level)
    logging.basicConfig(level=resolved, format="[%(levelname)s] %(message)s", force=True)
    logging.getLogger().setLevel(resolved)


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)
