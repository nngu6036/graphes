from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable

from grapher.utils.logging import get_logger

logger = get_logger(__name__)


def run_command(cmd: list[str], *, continue_on_error: bool = True) -> int:
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        msg = f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}"
        if continue_on_error:
            logger.error(msg)
        else:
            raise RuntimeError(msg)
    return int(proc.returncode)


def run_evaluation_step(name: str) -> None:
    logger.info("Running step: %s", name)
