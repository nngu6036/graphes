"""Compatibility namespace for the relocated structural-summary module.

New code should import from :mod:`grapher.rewiring_mlp.properties`.  This shim
keeps legacy internal imports working while callers migrate.
"""

from grapher.rewiring_mlp.properties.summary import *  # noqa: F403
