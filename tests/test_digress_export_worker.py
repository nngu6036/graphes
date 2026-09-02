from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKERS = ROOT / "src" / "grapher" / "models" / "digress" / "workers"
SPEC = importlib.util.spec_from_file_location(
    "digress_export_worker", WORKERS / "export.py"
)
assert SPEC is not None and SPEC.loader is not None
sys.path.insert(0, str(WORKERS))
try:
    WORKER = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(WORKER)
finally:
    sys.path.pop(0)


def _sample(*, nodes, edge_class: float):
    edges = np.zeros((len(nodes), len(nodes)), dtype=float)
    if len(nodes) >= 2:
        edges[0, 1] = edge_class
        edges[1, 0] = edge_class
    return np.asarray(nodes), edges


def test_zinc_export_packs_only_its_declared_categories() -> None:
    packed = WORKER._pack_samples(
        [_sample(nodes=range(9), edge_class=3)], dataset="zinc"
    )

    assert packed[1].tolist() == list(range(9))
    assert packed[3].tolist() == [[0, 1]]
    assert packed[4].tolist() == [3]


@pytest.mark.parametrize(
    "nodes, edge_class, message",
    [
        ([0.5], 0, "non-integral"),
        ([float("nan")], 0, "non-finite"),
        ([9], 0, "node class"),
        ([0, 1], -1, "edge class"),
        ([0, 1], 4, "edge class"),
    ],
)
def test_zinc_export_rejects_invalid_labels(nodes, edge_class, message) -> None:
    with pytest.raises(ValueError, match=message):
        WORKER._pack_samples(
            [_sample(nodes=nodes, edge_class=edge_class)], dataset="zinc"
        )


def test_qm9_export_preserves_aromatic_class_four() -> None:
    packed = WORKER._pack_samples(
        [_sample(nodes=[0, 1], edge_class=4)], dataset="qm9"
    )

    assert packed[4].tolist() == [4]
