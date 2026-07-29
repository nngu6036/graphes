from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from grapher.construction.coarse import construct_coarse_graph
from grapher.properties.summary import SummaryConfig, extract_summary
from grapher.utils.motifs import (
    derive_k3_graphlet_distribution,
    topology_graphlet_keys_by_size,
)


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(values, dtype=np.float64).reshape(-1), 0.0)
    total = float(array.sum())
    if total <= 0.0:
        if array.size == 0:
            return array
        return np.full(array.size, 1.0 / float(array.size), dtype=np.float64)
    return array / total


def _pad(values: Any, width: int) -> np.ndarray:
    out = np.zeros(max(int(width), 0), dtype=np.float64)
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    out[: min(out.size, array.size)] = array[: out.size]
    return out


def _degree_histogram_from_sequence(
    sequence: list[int],
    width: int | None = None,
) -> np.ndarray:
    resolved_width = max(
        int(width or 0),
        max(sequence, default=0) + 1,
        1,
    )
    return _normalize(
        np.bincount(sequence, minlength=resolved_width).astype(np.float64)
    )


def _degree_only_summary(summary: dict[str, Any]) -> dict[str, Any]:
    sequence = sorted(
        [int(value) for value in summary.get("degree_sequence", [])],
        reverse=True,
    )
    n = int(summary.get("num_nodes", len(sequence)))
    if len(sequence) != n:
        raise ValueError(
            "Degree condition has inconsistent num_nodes and degree_sequence."
        )
    m = int(sum(sequence) // 2)
    # The sequence is the hard condition. Recompute all dependent fields so
    # stale caller-provided histograms, edge counts, or density cannot disagree.
    hist = _degree_histogram_from_sequence(sequence)
    return {
        "num_nodes": n,
        "num_edges": m,
        "degree_sequence": sequence,
        "degree_hist": hist,
        "density": float(2.0 * m / (n * (n - 1))) if n > 1 else 0.0,
    }


def validate_degree_condition(
    summary: dict[str, Any],
    *,
    require_connected: bool = True,
) -> dict[str, Any]:
    condition = _degree_only_summary(summary)
    sequence = condition["degree_sequence"]
    n = int(condition["num_nodes"])
    if any(value < 0 or value >= max(n, 1) for value in sequence):
        raise ValueError("Degree condition contains an out-of-range degree.")
    if sum(sequence) % 2:
        raise ValueError("Degree condition must have an even degree sum.")
    if not nx.is_graphical(sequence, method="eg"):
        raise ValueError("Degree condition is not graphical.")
    if require_connected and n > 1:
        if any(value <= 0 for value in sequence):
            raise ValueError(
                "A connected realization cannot contain a zero-degree node."
            )
        if sum(sequence) < 2 * (n - 1):
            raise ValueError(
                "Degree condition does not contain enough edges for connectivity."
            )
    return condition


def degree_wasserstein_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    normalize: bool = True,
) -> float:
    """One-dimensional Wasserstein-1 distance on integer degree support."""

    left = _normalize(np.asarray(left, dtype=np.float64))
    right = _normalize(np.asarray(right, dtype=np.float64))
    width = max(left.size, right.size, 1)
    delta = np.cumsum(_pad(left, width) - _pad(right, width))
    value = float(np.abs(delta[:-1]).sum()) if width > 1 else 0.0
    if normalize:
        value /= float(max(width - 1, 1))
    return value


def helmert_basis(dim: int) -> np.ndarray:
    """Return a ``dim x (dim - 1)`` orthonormal Helmert basis."""

    dim = int(dim)
    if dim <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    basis = np.zeros((dim, max(dim - 1, 0)), dtype=np.float64)
    for j in range(1, dim):
        basis[:j, j - 1] = 1.0 / math.sqrt(j * (j + 1))
        basis[j, j - 1] = -j / math.sqrt(j * (j + 1))
    return basis


def ilr_transform(probabilities: np.ndarray, *, epsilon: float) -> np.ndarray:
    probabilities = np.maximum(
        np.asarray(probabilities, dtype=np.float64).reshape(-1),
        0.0,
    )
    if probabilities.size <= 1:
        return np.zeros(0, dtype=np.float64)
    probabilities = probabilities + float(epsilon)
    probabilities /= float(probabilities.sum())
    return helmert_basis(probabilities.size).T @ np.log(probabilities)


def ilr_inverse(values: np.ndarray, *, dim: int) -> np.ndarray:
    dim = int(dim)
    if dim <= 0:
        return np.zeros(0, dtype=np.float64)
    if dim == 1:
        return np.ones(1, dtype=np.float64)
    logits = helmert_basis(dim) @ np.asarray(values, dtype=np.float64).reshape(-1)
    logits -= float(np.max(logits))
    probabilities = np.exp(np.clip(logits, -700.0, 0.0))
    return _normalize(probabilities)


def _ceil_div(numerator: int, denominator: int) -> int:
    return -((-int(numerator)) // int(denominator))


def triangle_count_bounds(degree_summary: dict[str, Any]) -> tuple[int, int]:
    """Necessary, but not always realizable, bounds for a fixed degree sequence."""

    condition = validate_degree_condition(
        degree_summary,
        require_connected=False,
    )
    sequence = condition["degree_sequence"]
    n = int(condition["num_nodes"])
    if n < 3:
        return 0, 0
    m = int(condition["num_edges"])
    wedges = int(sum(value * (value - 1) // 2 for value in sequence))
    triples = math.comb(n, 3)
    lower = max(0, _ceil_div(2 * wedges - m * (n - 2), 3))
    upper = min(
        wedges // 3,
        triples + wedges - m * (n - 2),
        triples,
    )
    if upper < lower:
        raise ValueError(
            "The degree condition yields an empty necessary triangle interval: "
            f"[{lower}, {upper}]."
        )
    return int(lower), int(upper)


def triangle_coordinate(
    summary: dict[str, Any],
    degree_summary: dict[str, Any],
    *,
    alpha: float,
) -> float:
    lower, upper = triangle_count_bounds(degree_summary)
    if upper <= lower:
        return 0.0
    n = max(int(degree_summary.get("num_nodes", 0)), 1)
    triangles = round(n * float(summary.get("triangle_count_norm", 0.0)))
    triangles = int(np.clip(triangles, lower, upper))
    probability = (
        triangles - lower + float(alpha)
    ) / float(upper - lower + 2.0 * float(alpha))
    probability = float(np.clip(probability, 1.0e-12, 1.0 - 1.0e-12))
    return float(math.log(probability) - math.log1p(-probability))


def triangle_count_from_coordinate(
    coordinate: float,
    degree_summary: dict[str, Any],
    *,
    alpha: float,
) -> int:
    lower, upper = triangle_count_bounds(degree_summary)
    if upper <= lower:
        return int(lower)
    clipped = float(np.clip(coordinate, -60.0, 60.0))
    probability = 1.0 / (1.0 + math.exp(-clipped))
    raw = (
        lower
        - float(alpha)
        + probability * (upper - lower + 2.0 * float(alpha))
    )
    return int(np.clip(round(raw), lower, upper))


def k3_connected_mass(
    degree_sequence: list[int],
    triangle_count: int,
) -> float:
    """Fraction of all induced triples that are connected."""

    n = len(degree_sequence)
    if n < 3:
        return 0.0
    # This call validates every induced three-node count, not only the
    # connected conditional distribution used below.
    derive_k3_graphlet_distribution(
        degree_sequence,
        triangle_count,
        connected_only=False,
    )
    wedges = sum(value * (value - 1) // 2 for value in degree_sequence)
    connected = wedges - 2 * int(triangle_count)
    return float(connected / math.comb(n, 3))


@dataclass(frozen=True)
class KernelResidualConfig:
    top_k: int = 10
    degree_wasserstein_weight: float = 1.0
    node_count_weight: float = 1.0
    edge_count_weight: float = 1.0
    normalize_wasserstein: bool = True
    bandwidth_multiplier: float = 1.0
    min_bandwidth: float = 1.0e-6
    pseudocount: float = 1.0e-6
    triangle_alpha: float = 0.5
    residual_scale: float = 1.0
    derive_k3_from_degree_and_triangle: bool = True
    seed: int = 0

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        *,
        seed: int = 0,
    ) -> KernelResidualConfig:
        data = dict(data or {})
        selection_path = data.get("selection_path")
        if selection_path:
            path = Path(str(selection_path))
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    selected = json.load(handle)
                selected = selected.get("selected_kernel", selected)
                data["top_k"] = int(selected.get("top_k", data.get("top_k", 10)))
                kernel_data = dict(data.get("kernel", {}) or {})
                kernel_data["bandwidth_multiplier"] = float(
                    selected.get(
                        "bandwidth_multiplier",
                        kernel_data.get("bandwidth_multiplier", 1.0),
                    )
                )
                data["kernel"] = kernel_data
        distance = data.get("distance", {}) or {}
        kernel = data.get("kernel", {}) or {}
        residual = data.get("residual", {}) or {}
        bandwidth_type = str(kernel.get("bandwidth", "adaptive_kth")).lower()
        if bandwidth_type not in {"adaptive_kth", "adaptive-kth"}:
            raise ValueError(
                "Only kernel.bandwidth=adaptive_kth is currently supported."
            )
        representation = str(residual.get("representation", "ilr")).lower()
        if representation != "ilr":
            raise ValueError(
                "Only residual.representation=ilr is currently supported."
            )
        return cls(
            top_k=int(data.get("top_k", 10)),
            degree_wasserstein_weight=float(
                distance.get("degree_wasserstein_weight", 1.0)
            ),
            node_count_weight=float(distance.get("node_count_weight", 1.0)),
            edge_count_weight=float(distance.get("edge_count_weight", 1.0)),
            normalize_wasserstein=bool(
                distance.get("normalize_wasserstein", True)
            ),
            bandwidth_multiplier=float(
                kernel.get("bandwidth_multiplier", 1.0)
            ),
            min_bandwidth=float(kernel.get("min_bandwidth", 1.0e-6)),
            pseudocount=float(residual.get("pseudocount", 1.0e-6)),
            triangle_alpha=float(residual.get("triangle_alpha", 0.5)),
            residual_scale=float(residual.get("residual_scale", 1.0)),
            derive_k3_from_degree_and_triangle=bool(
                residual.get("derive_k3_from_degree_and_triangle", True)
            ),
            seed=int(data.get("seed", seed)),
        )


@dataclass(frozen=True)
class _StructuralBlock:
    name: str
    kind: str
    width: int
    start: int
    stop: int
    graphlet_k: str | None = None


class StructuralResidualCodec:
    """Blockwise invertible codec for one joint structural-summary vector."""

    def __init__(
        self,
        summary_config: SummaryConfig,
        summaries: list[dict[str, Any]],
        config: KernelResidualConfig,
    ):
        self.summary_config = summary_config
        self.config = config
        self.graphlet_keys_by_k = (
            topology_graphlet_keys_by_size(
                summary_config.graphlet_k_min,
                summary_config.graphlet_k_max,
                connected_only=summary_config.graphlet_connected_only,
            )
            if summary_config.graphlet_history
            else {}
        )
        self.motif_dim = max(
            (
                np.asarray(item.get("motif_proxy", [])).size
                for item in summaries
            ),
            default=0,
        )
        self.orbit_dim = max(
            (
                np.asarray(item.get("orbit_count", [])).size
                for item in summaries
            ),
            default=0,
        )

        blocks: list[_StructuralBlock] = []
        position = 0

        def add(
            name: str,
            kind: str,
            width: int,
            *,
            graphlet_k: str | None = None,
        ) -> None:
            nonlocal position
            encoded_width = max(int(width) - 1, 0) if kind == "simplex" else int(width)
            blocks.append(
                _StructuralBlock(
                    name=name,
                    kind=kind,
                    width=int(width),
                    start=position,
                    stop=position + encoded_width,
                    graphlet_k=graphlet_k,
                )
            )
            position += encoded_width

        add("triangle", "triangle", 1)
        if summary_config.clustering_summary:
            add("clustering_hist", "simplex", summary_config.clustering_bins)
        if summary_config.spectral_summary:
            add("spectral_hist", "simplex", summary_config.spectral_bins)
        if summary_config.motif_proxy and self.motif_dim:
            add("motif_proxy", "positive", self.motif_dim)
        if summary_config.orbit_count and self.orbit_dim:
            add("orbit_count", "positive", self.orbit_dim)
        if summary_config.graphlet_history:
            for k in range(
                int(summary_config.graphlet_k_min),
                int(summary_config.graphlet_k_max) + 1,
            ):
                key = str(k)
                if (
                    k == 3
                    and config.derive_k3_from_degree_and_triangle
                ):
                    continue
                add(
                    f"graphlet_{key}",
                    "simplex",
                    len(self.graphlet_keys_by_k.get(key, [])),
                    graphlet_k=key,
                )
                add(
                    f"graphlet_connected_mass_{key}",
                    "unit_interval",
                    1,
                    graphlet_k=key,
                )
        self.blocks = blocks
        self.output_dim = int(position)

    def _simplex_values(
        self,
        summary: dict[str, Any],
        block: _StructuralBlock,
    ) -> np.ndarray:
        if block.graphlet_k is not None:
            history = summary.get("graphlet_history", {}) or {}
            values = history.get(block.graphlet_k, {}) or {}
            keys = self.graphlet_keys_by_k.get(block.graphlet_k, [])
            unknown = set(values).difference(keys)
            if unknown:
                raise ValueError(
                    "Graphlet history contains keys outside the fixed "
                    f"k={block.graphlet_k} basis: {sorted(unknown)!r}."
                )
            return np.asarray(
                [
                    float(values.get(key, 0.0))
                    for key in keys
                ],
                dtype=np.float64,
            )
        return _pad(summary.get(block.name, []), block.width)

    def encode(
        self,
        summary: dict[str, Any],
        degree_summary: dict[str, Any],
    ) -> np.ndarray:
        vector = np.zeros(self.output_dim, dtype=np.float64)
        for block in self.blocks:
            if (
                block.graphlet_k is not None
                and int(degree_summary.get("num_nodes", 0))
                < int(block.graphlet_k)
            ):
                impossible = (
                    summary.get("graphlet_history", {}) or {}
                ).get(block.graphlet_k, {}) or {}
                if impossible:
                    raise ValueError(
                        "Graphlet history cannot contain a nonempty "
                        f"k={block.graphlet_k} block when n < k."
                    )
                impossible_mass = float(
                    (
                        summary.get("graphlet_connected_mass", {}) or {}
                    ).get(block.graphlet_k, 0.0)
                )
                if impossible_mass != 0.0:
                    raise ValueError(
                        "Graphlet connected mass must be zero when n < k."
                    )
                continue
            if block.kind == "triangle":
                vector[block.start] = triangle_coordinate(
                    summary,
                    degree_summary,
                    alpha=self.config.triangle_alpha,
                )
            elif block.kind == "simplex":
                vector[block.start : block.stop] = ilr_transform(
                    self._simplex_values(summary, block),
                    epsilon=self.config.pseudocount,
                )
            elif block.kind == "positive":
                vector[block.start : block.stop] = np.log1p(
                    np.maximum(
                        _pad(summary.get(block.name, []), block.width),
                        0.0,
                    )
                )
            elif block.kind == "unit_interval":
                masses = summary.get("graphlet_connected_mass", {}) or {}
                value = float(masses.get(str(block.graphlet_k), 0.0))
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(
                        "Graphlet connected mass must be finite and in [0, 1]."
                    )
                probability = float(
                    np.clip(
                        value,
                        self.config.pseudocount,
                        1.0 - self.config.pseudocount,
                    )
                )
                vector[block.start] = (
                    math.log(probability) - math.log1p(-probability)
                )
            else:  # pragma: no cover - guarded by construction
                raise ValueError(f"Unknown structural block kind: {block.kind!r}.")
        return vector

    def decode(
        self,
        vector: np.ndarray,
        degree_summary: dict[str, Any],
        *,
        template: dict[str, Any],
    ) -> dict[str, Any]:
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vector.size != self.output_dim:
            raise ValueError(
                f"Expected structural vector of width {self.output_dim}, "
                f"got {vector.size}."
            )
        out = dict(template)
        out["graphlet_history"] = {
            str(k): dict(values)
            for k, values in (
                template.get("graphlet_history", {}) or {}
            ).items()
        }
        out["graphlet_connected_mass"] = {
            str(k): float(value)
            for k, value in (
                template.get("graphlet_connected_mass", {}) or {}
            ).items()
        }
        triangle_count = round(
            max(int(degree_summary.get("num_nodes", 0)), 1)
            * float(template.get("triangle_count_norm", 0.0))
        )
        for block in self.blocks:
            if (
                block.graphlet_k is not None
                and int(degree_summary.get("num_nodes", 0))
                < int(block.graphlet_k)
            ):
                out["graphlet_history"][block.graphlet_k] = {}
                out["graphlet_connected_mass"][block.graphlet_k] = 0.0
                continue
            if block.kind == "triangle":
                triangle_count = triangle_count_from_coordinate(
                    float(vector[block.start]),
                    degree_summary,
                    alpha=self.config.triangle_alpha,
                )
                n = max(int(degree_summary.get("num_nodes", 0)), 1)
                out["triangle_count_norm"] = float(triangle_count / n)
            elif block.kind == "simplex":
                probabilities = ilr_inverse(
                    vector[block.start : block.stop],
                    dim=block.width,
                )
                if block.graphlet_k is not None:
                    keys = self.graphlet_keys_by_k.get(block.graphlet_k, [])
                    out["graphlet_history"][block.graphlet_k] = {
                        key: float(probabilities[index])
                        for index, key in enumerate(keys)
                        if float(probabilities[index]) > 0.0
                    }
                else:
                    out[block.name] = probabilities
            elif block.kind == "positive":
                out[block.name] = np.maximum(
                    np.expm1(vector[block.start : block.stop]),
                    0.0,
                )
            elif block.kind == "unit_interval":
                clipped = float(np.clip(vector[block.start], -60.0, 60.0))
                out["graphlet_connected_mass"][str(block.graphlet_k)] = (
                    1.0 / (1.0 + math.exp(-clipped))
                )

        condition = _degree_only_summary(degree_summary)
        if (
            self.summary_config.graphlet_history
            and self.config.derive_k3_from_degree_and_triangle
            and self.summary_config.graphlet_k_min <= 3
            and self.summary_config.graphlet_k_max >= 3
        ):
            out["graphlet_history"]["3"] = derive_k3_graphlet_distribution(
                condition["degree_sequence"],
                triangle_count,
                connected_only=self.summary_config.graphlet_connected_only,
            )
            out["graphlet_connected_mass"]["3"] = k3_connected_mass(
                condition["degree_sequence"],
                triangle_count,
            )
        out.update(condition)
        return out


@dataclass
class _ResidualRecord:
    record_id: int
    condition: dict[str, Any]
    degree_hist: np.ndarray
    residual: np.ndarray


class KernelResidualSummarySampler:
    """Weighted top-k sampler over complete HH-relative summary residuals."""

    requires_source_graph = True

    def __init__(
        self,
        records: list[_ResidualRecord],
        codec: StructuralResidualCodec,
        summary_config: SummaryConfig,
        constructor_config: dict[str, Any],
        config: KernelResidualConfig,
    ):
        if not records:
            raise ValueError("Kernel residual sampler requires training records.")
        if config.top_k <= 0:
            raise ValueError("summary_generator.top_k must be positive.")
        numeric_parameters = {
            "degree_wasserstein_weight": config.degree_wasserstein_weight,
            "node_count_weight": config.node_count_weight,
            "edge_count_weight": config.edge_count_weight,
            "bandwidth_multiplier": config.bandwidth_multiplier,
            "min_bandwidth": config.min_bandwidth,
            "pseudocount": config.pseudocount,
            "triangle_alpha": config.triangle_alpha,
            "residual_scale": config.residual_scale,
        }
        if not all(math.isfinite(value) for value in numeric_parameters.values()):
            raise ValueError(
                "All kernel-residual numeric parameters must be finite."
            )
        if config.bandwidth_multiplier <= 0.0:
            raise ValueError("kernel.bandwidth_multiplier must be positive.")
        if config.min_bandwidth <= 0.0:
            raise ValueError("kernel.min_bandwidth must be positive.")
        if not 0.0 < config.pseudocount < 0.5:
            raise ValueError("residual.pseudocount must be in (0, 0.5).")
        if config.triangle_alpha <= 0.0:
            raise ValueError("residual.triangle_alpha must be positive.")
        if config.residual_scale < 0.0:
            raise ValueError("residual.residual_scale must be nonnegative.")
        if min(
            config.degree_wasserstein_weight,
            config.node_count_weight,
            config.edge_count_weight,
        ) < 0.0:
            raise ValueError("All conditional-distance weights must be nonnegative.")
        if (
            config.degree_wasserstein_weight
            + config.node_count_weight
            + config.edge_count_weight
            <= 0.0
        ):
            raise ValueError(
                "At least one conditional-distance weight must be positive."
            )
        self.records = records
        self.codec = codec
        self.summary_config = summary_config
        self.constructor_config = dict(constructor_config)
        self.config = config
        self.node_scale = max(
            max(int(item.condition["num_nodes"]) for item in records),
            1,
        )
        self.edge_scale = max(
            max(int(item.condition["num_edges"]) for item in records),
            1,
        )
        self.corpus_max_degree = max(
            (
                max(item.condition["degree_sequence"], default=0)
                for item in records
            ),
            default=0,
        )
        self.last_sample_metadata: dict[str, Any] | None = None
        self._source_summary_cache: dict[tuple[int, ...], dict[str, Any]] = {}

    @classmethod
    def fit(
        cls,
        graphs: list[nx.Graph],
        summary_config: SummaryConfig | dict[str, Any],
        constructor_config: dict[str, Any] | None,
        generator_config: dict[str, Any] | None,
        *,
        seed: int = 0,
        target_summaries: list[dict[str, Any]] | None = None,
    ) -> KernelResidualSummarySampler:
        if not graphs:
            raise ValueError("Cannot fit kernel residual sampler on no graphs.")
        cfg = (
            summary_config
            if isinstance(summary_config, SummaryConfig)
            else SummaryConfig.from_dict(summary_config or {}, graphs)
        )
        constructor = dict(constructor_config or {})
        kernel_cfg = KernelResidualConfig.from_dict(
            generator_config,
            seed=seed,
        )

        if target_summaries is not None:
            if len(target_summaries) != len(graphs):
                raise ValueError(
                    "target_summaries must align one-to-one with graphs."
                )
            targets = [dict(summary) for summary in target_summaries]
        else:
            targets = [extract_summary(graph, cfg) for graph in graphs]
        sources: list[dict[str, Any]] = []
        source_cache: dict[tuple[int, ...], dict[str, Any]] = {}
        for index, target in enumerate(targets):
            condition = validate_degree_condition(
                target,
                require_connected=bool(constructor.get("ensure_connected", True)),
            )
            key = tuple(condition["degree_sequence"])
            source = source_cache.get(key)
            if source is None:
                source_graph = construct_coarse_graph(
                    condition,
                    constructor,
                    np.random.default_rng(kernel_cfg.seed + index),
                )
                source = extract_summary(source_graph, cfg)
                source_cache[key] = source
            sources.append(source)

        codec = StructuralResidualCodec(
            cfg,
            targets + sources,
            kernel_cfg,
        )
        records: list[_ResidualRecord] = []
        for index, (target, source) in enumerate(zip(targets, sources)):
            condition = _degree_only_summary(target)
            residual = (
                codec.encode(target, condition)
                - codec.encode(source, condition)
            )
            records.append(
                _ResidualRecord(
                    record_id=index,
                    condition=condition,
                    degree_hist=_degree_histogram_from_sequence(
                        condition["degree_sequence"]
                    ),
                    residual=residual,
                )
            )
        return cls(
            records,
            codec,
            cfg,
            constructor,
            kernel_cfg,
        )

    @classmethod
    def from_config(
        cls,
        graphs: list[nx.Graph],
        summary_config: SummaryConfig,
        config: dict[str, Any],
        *,
        seed: int = 0,
        target_summaries: list[dict[str, Any]] | None = None,
    ) -> KernelResidualSummarySampler:
        return cls.fit(
            graphs,
            summary_config,
            config.get("constructor", {}) or {},
            config.get("summary_generator", {}) or {},
            seed=seed,
            target_summaries=target_summaries,
        )

    def _distances(self, condition: dict[str, Any]) -> np.ndarray:
        support = max(
            self.corpus_max_degree,
            max(condition["degree_sequence"], default=0),
        ) + 1
        query_hist = _degree_histogram_from_sequence(
            condition["degree_sequence"],
            support,
        )
        distances = np.zeros(len(self.records), dtype=np.float64)
        for index, record in enumerate(self.records):
            candidate_hist = _pad(record.degree_hist, support)
            distances[index] = (
                self.config.degree_wasserstein_weight
                * degree_wasserstein_distance(
                    query_hist,
                    candidate_hist,
                    normalize=self.config.normalize_wasserstein,
                )
                + self.config.node_count_weight
                * abs(
                    int(condition["num_nodes"])
                    - int(record.condition["num_nodes"])
                )
                / float(self.node_scale)
                + self.config.edge_count_weight
                * abs(
                    int(condition["num_edges"])
                    - int(record.condition["num_edges"])
                )
                / float(self.edge_scale)
            )
        return distances

    def _kernel_distribution(
        self,
        condition: dict[str, Any],
        *,
        top_k: int | None = None,
        bandwidth_multiplier: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        distances = self._distances(condition)
        if top_k is not None and int(top_k) <= 0:
            raise ValueError("top_k override must be positive.")
        if (
            bandwidth_multiplier is not None
            and (
                not math.isfinite(float(bandwidth_multiplier))
                or float(bandwidth_multiplier) <= 0.0
            )
        ):
            raise ValueError("bandwidth_multiplier override must be positive.")
        resolved_top_k = min(
            max(
                int(self.config.top_k if top_k is None else top_k),
                1,
            ),
            len(self.records),
        )
        order = np.lexsort(
            (
                np.asarray(
                    [record.record_id for record in self.records],
                    dtype=np.int64,
                ),
                distances,
            )
        )
        cutoff = float(distances[order[resolved_top_k - 1]])
        ordered_distances = distances[order]
        # Include every record tied at the kth boundary. This avoids arbitrary
        # corpus-order bias when many graphs have the same degree condition.
        selected = order[
            (ordered_distances < cutoff)
            | np.isclose(
                ordered_distances,
                cutoff,
                rtol=1.0e-12,
                atol=1.0e-15,
            )
        ]
        selected_distances = distances[selected]
        multiplier = float(
            self.config.bandwidth_multiplier
            if bandwidth_multiplier is None
            else bandwidth_multiplier
        )
        bandwidth = max(
            self.config.min_bandwidth,
            multiplier * float(selected_distances[-1]),
        )
        if np.allclose(selected_distances, 0.0):
            weights = np.full(
                selected.size,
                1.0 / float(selected.size),
                dtype=np.float64,
            )
        else:
            log_weights = -0.5 * (selected_distances / bandwidth) ** 2
            log_weights -= float(np.max(log_weights))
            weights = _normalize(np.exp(log_weights))
        return selected, selected_distances, weights, float(bandwidth)

    def sample(
        self,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        generator = (
            rng if rng is not None else np.random.default_rng(self.config.seed)
        )
        record = self.records[int(generator.integers(0, len(self.records)))]
        return self.sample_conditioned(record.condition, generator)

    def sample_with_source(
        self,
        constructor_config: dict[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[dict[str, Any], nx.Graph]:
        """Sample a target and return the exact source graph it conditions on."""

        generator = (
            rng if rng is not None else np.random.default_rng(self.config.seed)
        )
        record = self.records[int(generator.integers(0, len(self.records)))]
        constructor = dict(constructor_config or self.constructor_config)
        source_graph = construct_coarse_graph(
            record.condition,
            constructor,
            generator,
        )
        target = self.sample_conditioned(
            record.condition,
            generator,
            source_graph=source_graph,
        )
        return target, source_graph

    def sample_conditioned(
        self,
        degree_summary: dict[str, Any],
        rng: np.random.Generator | None = None,
        *,
        source_graph: nx.Graph | None = None,
        top_k: int | None = None,
        bandwidth_multiplier: float | None = None,
        return_metadata: bool = False,
    ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
        generator = (
            rng if rng is not None else np.random.default_rng(self.config.seed)
        )
        condition = validate_degree_condition(
            degree_summary,
            require_connected=bool(
                self.constructor_config.get("ensure_connected", True)
            ),
        )
        cache_key = tuple(condition["degree_sequence"])
        source_summary = None
        if source_graph is None:
            source_summary = self._source_summary_cache.get(cache_key)
            if source_summary is None:
                source_graph = construct_coarse_graph(
                    condition,
                    self.constructor_config,
                    generator,
                )
                source_summary = extract_summary(
                    source_graph,
                    self.summary_config,
                )
                self._source_summary_cache[cache_key] = source_summary
        else:
            actual_sequence = sorted(
                [int(degree) for _, degree in source_graph.degree()],
                reverse=True,
            )
            if actual_sequence != condition["degree_sequence"]:
                raise ValueError(
                    "source_graph does not realize the supplied degree condition."
                )
            source_summary = extract_summary(source_graph, self.summary_config)
        source_vector = self.codec.encode(source_summary, condition)

        selected, distances, weights, bandwidth = self._kernel_distribution(
            condition,
            top_k=top_k,
            bandwidth_multiplier=bandwidth_multiplier,
        )
        selected_position = int(generator.choice(selected.size, p=weights))
        donor_index = int(selected[selected_position])
        donor = self.records[donor_index]
        target_vector = (
            source_vector
            + self.config.residual_scale * donor.residual
        )
        target = self.codec.decode(
            target_vector,
            condition,
            template=source_summary,
        )
        metadata = {
            "donor_record_id": int(donor.record_id),
            "donor_distance": float(distances[selected_position]),
            "bandwidth": float(bandwidth),
            "top_k_record_ids": [
                int(self.records[int(index)].record_id)
                for index in selected
            ],
            "top_k_distances": distances.tolist(),
            "top_k_weights": weights.tolist(),
            "effective_neighbor_count": float(
                1.0 / max(float(np.sum(weights**2)), 1.0e-12)
            ),
        }
        self.last_sample_metadata = metadata
        if return_metadata:
            return target, metadata
        return target
