from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from grapher.properties.summary import SummaryConfig
from grapher.utils.motifs import topology_graphlet_keys_by_size


@dataclass(frozen=True)
class TopologyGraphletBasis:
    """Complete, fixed coordinates for connected unlabelled graphlets."""

    keys_by_k: dict[str, tuple[str, ...]]
    connected_only: bool = True
    attributed: bool = False

    @classmethod
    def from_config(
        cls,
        config: SummaryConfig | dict[str, Any],
    ) -> "TopologyGraphletBasis":
        cfg = (
            config
            if isinstance(config, SummaryConfig)
            else SummaryConfig.from_dict(config or {})
        )
        if not cfg.graphlet_connected_only:
            raise ValueError(
                "The decoupled topology stage requires connected graphlets."
            )
        keys = topology_graphlet_keys_by_size(
            cfg.graphlet_k_min,
            cfg.graphlet_k_max,
            connected_only=True,
        )
        return cls(keys_by_k={key: tuple(value) for key, value in keys.items()})

    @classmethod
    def fit_from_graphs(
        cls,
        graphs: Sequence[Any],
        config: SummaryConfig | dict[str, Any],
        *,
        attributed: bool = False,
        **_unused: Any,
    ) -> "TopologyGraphletBasis":
        del graphs
        if attributed:
            raise ValueError("TopologyGraphletBasis cannot encode attributes.")
        return cls.from_config(config)

    @property
    def sizes(self) -> tuple[str, ...]:
        return tuple(sorted(self.keys_by_k, key=int))

    @property
    def width(self) -> int:
        return sum(len(self.keys_by_k[key]) for key in self.sizes)

    @property
    def slices(self) -> tuple[tuple[int, int], ...]:
        result: list[tuple[int, int]] = []
        start = 0
        for key in self.sizes:
            stop = start + len(self.keys_by_k[key])
            result.append((start, stop))
            start = stop
        return tuple(result)

    @property
    def simplex_width(self) -> int:
        """Width after appending one disconnected-subset bin per order."""

        return sum(len(self.keys_by_k[key]) + 1 for key in self.sizes)

    @property
    def simplex_slices(self) -> tuple[tuple[int, int], ...]:
        result: list[tuple[int, int]] = []
        start = 0
        for key in self.sizes:
            stop = start + len(self.keys_by_k[key]) + 1
            result.append((start, stop))
            start = stop
        return tuple(result)

    @property
    def simplex_block_widths(self) -> tuple[int, ...]:
        return tuple(len(self.keys_by_k[key]) + 1 for key in self.sizes)

    def flatten_history(self, history: dict[str, Any]) -> np.ndarray:
        return np.asarray(
            [
                float((history.get(key, {}) or {}).get(graphlet_key, 0.0))
                for key in self.sizes
                for graphlet_key in self.keys_by_k[key]
            ],
            dtype=np.float32,
        )

    def unflatten_history(
        self,
        values: Sequence[float],
    ) -> dict[str, dict[str, float]]:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size != self.width:
            raise ValueError(
                f"Expected {self.width} graphlet values, received {array.size}."
            )
        result: dict[str, dict[str, float]] = {}
        for key, (start, stop) in zip(self.sizes, self.slices):
            block = np.maximum(array[start:stop], 0.0)
            total = float(block.sum())
            if total > 0.0:
                block = block / total
            result[key] = {
                graphlet_key: float(value)
                for graphlet_key, value in zip(self.keys_by_k[key], block)
            }
        return result

    def flatten_mass(self, mass: dict[str, Any]) -> np.ndarray:
        return np.asarray(
            [float((mass or {}).get(key, 0.0)) for key in self.sizes],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "topology_graphlet_basis_v1",
            "keys_by_k": {
                key: list(values) for key, values in self.keys_by_k.items()
            },
            "connected_only": True,
            "attributed": False,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopologyGraphletBasis":
        if data.get("format") not in {None, "topology_graphlet_basis_v1"}:
            raise ValueError("Unsupported topology graphlet basis format.")
        if bool(data.get("attributed", False)):
            raise ValueError("A topology graphlet basis cannot be attributed.")
        if not bool(data.get("connected_only", True)):
            raise ValueError("A topology graphlet basis must be connected-only.")
        keys = {
            str(key): tuple(str(value) for value in values)
            for key, values in (data.get("keys_by_k", {}) or {}).items()
        }
        if not keys:
            raise ValueError("Topology graphlet basis contains no coordinates.")
        return cls(keys_by_k=keys)
