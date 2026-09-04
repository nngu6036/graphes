"""Compatibility constants for the attached HOG-Diff release.

The supplied source imports these names from a top-level ``data`` module, but
that module is absent from the archive.  If a future HOG-Diff checkout contains
its own data.py, workers insert that checkout ahead of this directory and the
upstream module takes precedence.
"""

_GENERIC_DATASETS = ["community_small", "ego_small", "enzymes"]
_MOL_DATASETS = ["qm9", "zinc250k", "guacamol", "moses"]
