"""Compatibility loading for trusted NetworkX graph pickles.

NetworkX 3 caches report/core view objects directly on a graph after properties
such as ``degree`` or ``edges`` are accessed.  Some NetworkX 2 releases cannot
unpickle those cyclic objects because a view's ``__setstate__`` dereferences the
owner before the graph state (including ``_adj``) has been installed.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

_VIEW_MODULES = frozenset(
    {
        "networkx.classes.coreviews",
        "networkx.classes.reportviews",
    }
)
_CACHED_VIEW_KEYS = frozenset(
    {
        "adj",
        "degree",
        "edges",
        "in_degree",
        "in_edges",
        "nodes",
        "out_degree",
        "out_edges",
        "pred",
        "succ",
    }
)


class _DiscardedNetworkXView:
    """Inert placeholder removed from its owning graph after unpickling."""

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)


class _NetworkXCompatibilityUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module in _VIEW_MODULES and (name.endswith("View") or name.endswith("DataView")):
            return _DiscardedNetworkXView
        return super().find_class(module, name)


def _discard_graph_view_caches(value: Any) -> None:
    candidates = [value]
    visited: set[int] = set()
    while candidates:
        graph = candidates.pop()
        if id(graph) in visited:
            continue
        visited.add(id(graph))
        if isinstance(graph, dict):
            candidates.extend(graph.values())
            continue
        if isinstance(graph, (list, tuple)):
            candidates.extend(graph)
            continue
        state = getattr(graph, "__dict__", None)
        if not isinstance(state, dict) or "_adj" not in state or "_node" not in state:
            continue
        for key in _CACHED_VIEW_KEYS:
            state.pop(key, None)
        # This cache was introduced after the NetworkX versions used by some
        # attached baselines and is safe to recreate lazily when supported.
        state.pop("__networkx_cache__", None)


def load_trusted_networkx_pickle(path: str | Path) -> Any:
    """Load a trusted graph pickle without restoring version-specific views.

    Pickle is code-executing by design.  Callers must restrict this helper to
    GraphER-owned prepared dataset artifacts, never arbitrary uploaded files.
    """

    with Path(path).open("rb") as handle:
        value = _NetworkXCompatibilityUnpickler(handle).load()
    _discard_graph_view_caches(value)
    return value
