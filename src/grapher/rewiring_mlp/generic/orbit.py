from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from grapher.properties.summary import python_orbit_count_vector

TOPOLOGY_ORBIT_SUMMARY_WIDTH = 15


def orbit_summary_width(config: dict[str, Any] | None = None) -> int | None:
    values = dict(config or {})
    enabled = bool(values.get("orbit_summary", False))
    if not enabled:
        return None
    raw_width = values.get("orbit_width", TOPOLOGY_ORBIT_SUMMARY_WIDTH)
    if isinstance(raw_width, bool) or int(raw_width) != raw_width:
        raise ValueError("structure_summary_prediction.orbit_width must be an integer.")
    width = int(raw_width)
    if width != TOPOLOGY_ORBIT_SUMMARY_WIDTH:
        raise ValueError(
            "The minimal GraphER orbit summary uses the standard 15 ORCA node "
            "orbits for connected graphlets up to four nodes; orbit_width must be 15."
        )
    return width


def validate_orbit_summary(
    values: np.ndarray | list[float] | tuple[float, ...],
    *,
    width: int = TOPOLOGY_ORBIT_SUMMARY_WIDTH,
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != int(width):
        raise ValueError(
            f"Orbit summary must have width {int(width)}, got {vector.size}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("Orbit summary contains non-finite values.")
    if np.any(vector < -1.0e-12):
        raise ValueError("Orbit summary cannot contain negative counts.")
    return np.maximum(vector, 0.0)


def extract_orbit_summary(graph: nx.Graph) -> np.ndarray:
    """Return the evaluator-compatible 15-D mean per-node orbit descriptor.

    The pure-Python implementation is deliberately used here so training and
    candidate scoring do not launch an ORCA subprocess for every graph. Its
    coordinates follow the same ORCA 0--14 convention used by the standard
    GraphRNN/SPECTRE orbit evaluator.
    """

    return validate_orbit_summary(
        python_orbit_count_vector(nx.Graph(graph)),
        width=TOPOLOGY_ORBIT_SUMMARY_WIDTH,
    )


def orbit_summary_distance(
    current: np.ndarray | list[float] | tuple[float, ...],
    target: np.ndarray | list[float] | tuple[float, ...],
    *,
    distance: str = "log_rmse",
) -> float:
    """Distance between mean per-node orbit-count vectors.

    ``log_rmse`` is the default because orbit coordinates have very different
    count scales. It compares log1p(counts) so rare and common orbits can both
    influence a swap decision. ``raw_rmse`` is available for diagnostics.
    """

    a = validate_orbit_summary(current)
    b = validate_orbit_summary(target)
    mode = str(distance).lower()
    if mode in {"log_rmse", "log1p_rmse", "rmse_log1p"}:
        delta = np.log1p(a) - np.log1p(b)
    elif mode in {"raw_rmse", "rmse"}:
        delta = a - b
    else:
        raise ValueError(
            "orbit_guidance.distance must be 'log_rmse' or 'raw_rmse'."
        )
    return float(np.sqrt(np.mean(np.square(delta))))
