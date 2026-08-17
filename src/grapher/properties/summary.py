"""Compatibility re-export for structural-summary definitions.

The wrapper keeps the public ``ORCA_EXEC`` mirror synchronized when callers use
``configure_orca_executable`` through the historical import path.
"""

from grapher.rewiring_mlp.properties import summary as _summary
from grapher.rewiring_mlp.properties.summary import *  # noqa: F403


def configure_orca_executable(executable=None, *, required: bool = True):
    result = _summary.configure_orca_executable(executable, required=required)
    globals()["ORCA_EXEC"] = _summary.ORCA_EXEC
    return result
