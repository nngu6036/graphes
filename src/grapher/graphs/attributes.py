from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np

DEFAULT_SCHEMA = {
    "enabled": "auto",
    "node_label_attr": "node_label",
    "node_feature_attr": "feats",
    "edge_label_attr": "edge_type",
    "edge_feature_attr": "edge_attr",
    "graph_label_attr": "graph_label",
    "node_label_aliases": ["label", "node_type", "type", "atom_type", "x_label"],
    "node_feature_aliases": ["feature", "features", "x", "node_features"],
    "edge_label_aliases": ["label", "edge_label", "bond_type", "type", "bond"],
    "edge_feature_aliases": ["feature", "features", "weight", "edge_features"],
    "graph_label_aliases": ["y", "label", "target"],
    "add_default_node_features": True,
    "add_default_node_labels": True,
    "add_default_edge_labels": True,
    "default_node_feature_dim": 1,
    "default_node_feature_value": 1.0,
    "generated_attribute_strategy": "empirical",
    "overwrite_generated_attributes": False,
}


def _as_bool_or_auto(value: Any) -> str | bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return "auto"
    s = str(value).strip().lower()
    if s in {"auto", "infer", "default"}:
        return "auto"
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return "auto"


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def normalize_schema(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_SCHEMA)
    if config:
        source = config.get("graph_attributes", config)
        if isinstance(source, dict):
            cfg.update({k: v for k, v in source.items() if v is not None})
    cfg["enabled"] = _as_bool_or_auto(cfg.get("enabled", "auto"))
    cfg["overwrite_generated_attributes"] = bool(cfg.get("overwrite_generated_attributes", False))
    cfg["generated_attribute_strategy"] = str(cfg.get("generated_attribute_strategy", "empirical")).lower()
    cfg["add_default_node_features"] = bool(cfg.get("add_default_node_features", True))
    cfg["add_default_node_labels"] = bool(cfg.get("add_default_node_labels", True))
    cfg["add_default_edge_labels"] = bool(cfg.get("add_default_edge_labels", True))
    cfg["default_node_feature_dim"] = int(cfg.get("default_node_feature_dim", 1) or 1)
    cfg["default_node_feature_value"] = float(cfg.get("default_node_feature_value", 1.0))
    for k in ("node_label_attr", "node_feature_attr", "edge_label_attr", "edge_feature_attr", "graph_label_attr"):
        cfg[k] = str(cfg.get(k) or DEFAULT_SCHEMA[k])
    alias_map = {
        "node_label_aliases": "node_label_attr",
        "node_feature_aliases": "node_feature_attr",
        "edge_label_aliases": "edge_label_attr",
        "edge_feature_aliases": "edge_feature_attr",
        "graph_label_aliases": "graph_label_attr",
    }
    for k, primary_key in alias_map.items():
        primary = cfg[primary_key]
        aliases = _as_list(cfg.get(k))
        cfg[k] = [a for a in aliases if a != primary]
    return cfg


def _to_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value.astype(np.float64, copy=False).reshape(-1)
    elif isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
        except Exception:
            return None
    elif isinstance(value, (int, float, np.integer, np.floating, bool)):
        arr = np.asarray([float(value)], dtype=np.float64)
    else:
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr.astype(np.float64)


def _value_key(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _get_first(data: dict[str, Any], primary: str, aliases: Sequence[str]) -> Any:
    if primary in data:
        return data[primary]
    for alias in aliases:
        if alias in data:
            return data[alias]
    return None


def graph_has_attributes(graph: nx.Graph, schema: dict[str, Any] | None = None) -> bool:
    schema = normalize_schema({"graph_attributes": schema or {}})
    if schema.get("enabled") is False:
        return False
    if _get_first(graph.graph, schema["graph_label_attr"], schema["graph_label_aliases"]) is not None:
        return True
    for _, data in graph.nodes(data=True):
        if _get_first(data, schema["node_label_attr"], schema["node_label_aliases"]) is not None:
            return True
        if _to_vector(_get_first(data, schema["node_feature_attr"], schema["node_feature_aliases"])) is not None:
            return True
    for _, _, data in graph.edges(data=True):
        if _get_first(data, schema["edge_label_attr"], schema["edge_label_aliases"]) is not None:
            return True
        if _to_vector(_get_first(data, schema["edge_feature_attr"], schema["edge_feature_aliases"])) is not None:
            return True
    return False


@dataclass
class AttributeStatistics:
    schema: dict[str, Any]
    node_label_values: list[str]
    node_label_probs: list[float]
    node_feature_dim: int
    node_feature_mean: list[float]
    node_feature_std: list[float]
    edge_label_values: list[str]
    edge_label_probs: list[float]
    edge_feature_dim: int
    edge_feature_mean: list[float]
    edge_feature_std: list[float]
    graph_label_values: list[str]
    graph_label_probs: list[float]
    num_graphs: int
    has_any_attributes: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AttributeStatistics":
        return cls(**payload)


def _probs_from_counter(counter: Counter[str]) -> tuple[list[str], list[float]]:
    if not counter:
        return [], []
    values = sorted(counter)
    counts = np.asarray([counter[v] for v in values], dtype=np.float64)
    probs = counts / max(float(counts.sum()), 1.0)
    return values, [float(x) for x in probs]


def _pad_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.zeros((0, 0), dtype=np.float64)
    dim = max(int(v.size) for v in vectors)
    arr = np.zeros((len(vectors), dim), dtype=np.float64)
    for i, v in enumerate(vectors):
        arr[i, : min(v.size, dim)] = v[:dim]
    return arr


def fit_attribute_statistics(graphs: Sequence[nx.Graph], schema: dict[str, Any] | None = None) -> AttributeStatistics:
    schema = normalize_schema({"graph_attributes": schema or {}})
    if schema.get("enabled") is False:
        return AttributeStatistics(schema, [], [], 0, [], [], [], [], 0, [], [], [], [], len(graphs), False)
    nl, nf = schema["node_label_attr"], schema["node_feature_attr"]
    el, ef = schema["edge_label_attr"], schema["edge_feature_attr"]
    gl = schema["graph_label_attr"]
    node_labels: Counter[str] = Counter()
    edge_labels: Counter[str] = Counter()
    graph_labels: Counter[str] = Counter()
    node_feats: list[np.ndarray] = []
    edge_feats: list[np.ndarray] = []
    for g in graphs:
        graph_label = _get_first(g.graph, gl, schema["graph_label_aliases"])
        if graph_label is not None:
            graph_labels[_value_key(graph_label)] += 1
        for _, data in g.nodes(data=True):
            node_label = _get_first(data, nl, schema["node_label_aliases"])
            if node_label is not None:
                node_labels[_value_key(node_label)] += 1
            v = _to_vector(_get_first(data, nf, schema["node_feature_aliases"]))
            if v is not None:
                node_feats.append(v)
        for _, _, data in g.edges(data=True):
            edge_label = _get_first(data, el, schema["edge_label_aliases"])
            if edge_label is not None:
                edge_labels[_value_key(edge_label)] += 1
            v = _to_vector(_get_first(data, ef, schema["edge_feature_aliases"]))
            if v is not None:
                edge_feats.append(v)
    node_label_values, node_label_probs = _probs_from_counter(node_labels)
    edge_label_values, edge_label_probs = _probs_from_counter(edge_labels)
    graph_label_values, graph_label_probs = _probs_from_counter(graph_labels)
    node_arr = _pad_vectors(node_feats)
    edge_arr = _pad_vectors(edge_feats)
    has_attrs = bool(node_label_values or edge_label_values or graph_label_values or node_arr.size or edge_arr.size)
    return AttributeStatistics(
        schema=schema,
        node_label_values=node_label_values,
        node_label_probs=node_label_probs,
        node_feature_dim=int(node_arr.shape[1]) if node_arr.ndim == 2 else 0,
        node_feature_mean=[float(x) for x in node_arr.mean(axis=0)] if node_arr.size else [],
        node_feature_std=[float(max(x, 1e-12)) for x in node_arr.std(axis=0, ddof=0)] if node_arr.size else [],
        edge_label_values=edge_label_values,
        edge_label_probs=edge_label_probs,
        edge_feature_dim=int(edge_arr.shape[1]) if edge_arr.ndim == 2 else 0,
        edge_feature_mean=[float(x) for x in edge_arr.mean(axis=0)] if edge_arr.size else [],
        edge_feature_std=[float(max(x, 1e-12)) for x in edge_arr.std(axis=0, ddof=0)] if edge_arr.size else [],
        graph_label_values=graph_label_values,
        graph_label_probs=graph_label_probs,
        num_graphs=len(graphs),
        has_any_attributes=has_attrs,
    )


def _canonical_label(raw: Any, values: Sequence[str], *, default: int = 0, offset: int = 0) -> int:
    if raw is None:
        return default
    key = _value_key(raw)
    lookup = {v: i + offset for i, v in enumerate(values)}
    if key in lookup:
        return int(lookup[key])
    try:
        val = int(raw)
        if offset == 0 and val >= 0:
            return val
        if offset == 1 and val >= 1:
            return val
    except Exception:
        pass
    return default


def _fixed_vector(value: Any, dim: int, *, default_value: float = 0.0) -> list[float]:
    if dim <= 0:
        return []
    v = _to_vector(value)
    out = np.full(dim, float(default_value), dtype=np.float64) if v is None else np.zeros(dim, dtype=np.float64)
    if v is not None:
        out[: min(dim, v.size)] = v[:dim]
    return [float(x) for x in out]


def canonicalize_graph_attributes(
    graphs: Sequence[nx.Graph],
    schema: dict[str, Any] | None = None,
    stats: AttributeStatistics | dict[str, Any] | None = None,
    *,
    copy: bool = True,
) -> tuple[list[nx.Graph], AttributeStatistics]:
    schema = normalize_schema({"graph_attributes": schema or {}})
    if isinstance(stats, dict):
        stats = AttributeStatistics.from_dict(stats)
    if stats is None:
        stats = fit_attribute_statistics(graphs, schema)
    if schema.get("enabled") is False:
        return [g.copy() for g in graphs] if copy else list(graphs), stats
    node_dim = int(stats.node_feature_dim or 0)
    if node_dim <= 0 and schema.get("add_default_node_features", True):
        node_dim = int(schema.get("default_node_feature_dim", 1) or 1)
    edge_dim = int(stats.edge_feature_dim or 0)
    nl, nf = schema["node_label_attr"], schema["node_feature_attr"]
    el, ef = schema["edge_label_attr"], schema["edge_feature_attr"]
    gl = schema["graph_label_attr"]
    out_graphs: list[nx.Graph] = []
    for g in graphs:
        h = g.copy() if copy else g
        graph_label = _get_first(h.graph, gl, schema["graph_label_aliases"])
        if graph_label is not None:
            h.graph[gl] = _canonical_label(graph_label, stats.graph_label_values, default=0, offset=0)
        for _, data in h.nodes(data=True):
            node_label = _get_first(data, nl, schema["node_label_aliases"])
            if node_label is not None or schema.get("add_default_node_labels", True):
                data[nl] = _canonical_label(node_label, stats.node_label_values, default=0, offset=0)
            feat = _get_first(data, nf, schema["node_feature_aliases"])
            if feat is not None or node_dim > 0:
                default_val = float(schema.get("default_node_feature_value", 1.0)) if feat is None else 0.0
                data[nf] = _fixed_vector(feat, node_dim, default_value=default_val)
        for _, _, data in h.edges(data=True):
            edge_label = _get_first(data, el, schema["edge_label_aliases"])
            if edge_label is not None or schema.get("add_default_edge_labels", True):
                data[el] = _canonical_label(edge_label, stats.edge_label_values, default=1, offset=1)
            feat = _get_first(data, ef, schema["edge_feature_aliases"])
            if feat is not None or edge_dim > 0:
                data[ef] = _fixed_vector(feat, edge_dim, default_value=0.0)
        out_graphs.append(h)
    return out_graphs, stats


def canonicalize_graphs(
    graphs: Sequence[nx.Graph],
    schema: dict[str, Any] | None = None,
    metadata: AttributeStatistics | dict[str, Any] | None = None,
    *,
    copy: bool = True,
) -> tuple[list[nx.Graph], AttributeStatistics]:
    return canonicalize_graph_attributes(graphs, schema=schema, stats=metadata, copy=copy)


def _sample_categorical(rng: np.random.Generator, values: Sequence[str], probs: Sequence[float]) -> str | None:
    if not values:
        return None
    p = np.asarray(probs, dtype=np.float64)
    if p.size != len(values) or p.sum() <= 0:
        p = np.ones(len(values), dtype=np.float64) / len(values)
    else:
        p = p / p.sum()
    return str(rng.choice(np.asarray(values, dtype=object), p=p))


def _sample_continuous(rng: np.random.Generator, mean: Sequence[float], std: Sequence[float]) -> list[float] | None:
    if not mean:
        return None
    mu = np.asarray(mean, dtype=np.float64)
    sd = np.asarray(std or [0.0] * len(mu), dtype=np.float64)
    if sd.size != mu.size:
        sd = np.zeros_like(mu)
    values = rng.normal(mu, np.maximum(sd, 1e-12))
    return [float(x) for x in values]


def apply_empirical_attributes(
    graphs: Sequence[nx.Graph],
    stats: AttributeStatistics | dict[str, Any],
    *,
    seed: int = 0,
    overwrite: bool | None = None,
) -> list[nx.Graph]:
    if isinstance(stats, dict):
        stats = AttributeStatistics.from_dict(stats)
    schema = normalize_schema({"graph_attributes": stats.schema})
    if overwrite is None:
        overwrite = bool(schema.get("overwrite_generated_attributes", False))
    if schema.get("enabled") is False:
        return [g.copy() for g in graphs]
    rng = np.random.default_rng(seed)
    nl, nf = schema["node_label_attr"], schema["node_feature_attr"]
    el, ef = schema["edge_label_attr"], schema["edge_feature_attr"]
    gl = schema["graph_label_attr"]
    node_dim = int(stats.node_feature_dim or (schema.get("default_node_feature_dim", 1) if schema.get("add_default_node_features", True) else 0))
    edge_dim = int(stats.edge_feature_dim or 0)
    out_graphs: list[nx.Graph] = []
    for g in graphs:
        h = g.copy()
        graph_label = _sample_categorical(rng, stats.graph_label_values, stats.graph_label_probs)
        if graph_label is not None and (overwrite or gl not in h.graph):
            h.graph[gl] = _canonical_label(graph_label, stats.graph_label_values, default=0, offset=0)
        for _, data in h.nodes(data=True):
            if overwrite or nl not in data:
                sampled = _sample_categorical(rng, stats.node_label_values, stats.node_label_probs)
                data[nl] = _canonical_label(sampled, stats.node_label_values, default=0, offset=0)
            if overwrite or nf not in data:
                feat = _sample_continuous(rng, stats.node_feature_mean, stats.node_feature_std)
                if feat is not None:
                    data[nf] = _fixed_vector(feat, len(feat), default_value=0.0)
                elif node_dim > 0:
                    data[nf] = [float(schema.get("default_node_feature_value", 1.0))] * node_dim
        for _, _, data in h.edges(data=True):
            if overwrite or el not in data:
                sampled = _sample_categorical(rng, stats.edge_label_values, stats.edge_label_probs)
                data[el] = _canonical_label(sampled, stats.edge_label_values, default=1, offset=1)
            if overwrite or ef not in data:
                feat = _sample_continuous(rng, stats.edge_feature_mean, stats.edge_feature_std)
                if feat is not None:
                    data[ef] = _fixed_vector(feat, len(feat), default_value=0.0)
                elif edge_dim > 0:
                    data[ef] = [0.0] * edge_dim
        out_graphs.append(h)
    return out_graphs


def attribute_coverage(graphs: Sequence[nx.Graph], schema: dict[str, Any] | None = None) -> dict[str, Any]:
    schema = normalize_schema({"graph_attributes": schema or {}})
    nl, nf = schema["node_label_attr"], schema["node_feature_attr"]
    el, ef = schema["edge_label_attr"], schema["edge_feature_attr"]
    gl = schema["graph_label_attr"]
    num_nodes = sum(g.number_of_nodes() for g in graphs)
    num_edges = sum(g.number_of_edges() for g in graphs)
    node_label = node_feat = edge_label = edge_feat = graph_label = 0
    for g in graphs:
        if _get_first(g.graph, gl, schema["graph_label_aliases"]) is not None:
            graph_label += 1
        for _, data in g.nodes(data=True):
            node_label += int(_get_first(data, nl, schema["node_label_aliases"]) is not None)
            node_feat += int(_to_vector(_get_first(data, nf, schema["node_feature_aliases"])) is not None)
        for _, _, data in g.edges(data=True):
            edge_label += int(_get_first(data, el, schema["edge_label_aliases"]) is not None)
            edge_feat += int(_to_vector(_get_first(data, ef, schema["edge_feature_aliases"])) is not None)
    return {
        "num_graphs": len(graphs),
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "node_label_coverage": float(node_label / num_nodes) if num_nodes else 0.0,
        "node_feature_coverage": float(node_feat / num_nodes) if num_nodes else 0.0,
        "edge_label_coverage": float(edge_label / num_edges) if num_edges else 0.0,
        "edge_feature_coverage": float(edge_feat / num_edges) if num_edges else 0.0,
        "graph_label_coverage": float(graph_label / len(graphs)) if graphs else 0.0,
        "has_any_attributes": any(graph_has_attributes(g, schema) for g in graphs),
    }


def _hist_from_counter(counter: Counter[str], values: Sequence[str]) -> np.ndarray:
    if not values:
        return np.zeros(0, dtype=np.float64)
    arr = np.asarray([counter.get(str(v), 0) for v in values], dtype=np.float64)
    s = arr.sum()
    return arr / s if s > 0 else arr


def _feature_summary_from_vectors(vectors: list[np.ndarray], dim: int | None = None) -> np.ndarray:
    if dim is None:
        dim = max((v.size for v in vectors), default=0)
    if dim <= 0:
        return np.zeros(1, dtype=np.float64)
    if not vectors:
        return np.zeros(1 + 2 * dim, dtype=np.float64)
    arr = np.zeros((len(vectors), dim), dtype=np.float64)
    for i, v in enumerate(vectors):
        arr[i, : min(dim, v.size)] = v[:dim]
    return np.concatenate([[1.0], arr.mean(axis=0), arr.std(axis=0, ddof=0)])


def attribute_descriptor_features(
    graphs: Sequence[nx.Graph],
    schema: dict[str, Any] | None = None,
    *,
    node_label_values: Sequence[str] | None = None,
    edge_label_values: Sequence[str] | None = None,
    graph_label_values: Sequence[str] | None = None,
    node_feature_dim: int | None = None,
    edge_feature_dim: int | None = None,
    include_continuous: bool = True,
) -> np.ndarray:
    schema = normalize_schema({"graph_attributes": schema or {}})
    if schema.get("enabled") is False:
        return np.zeros((len(graphs), 1), dtype=np.float64)
    if node_label_values is None or edge_label_values is None or graph_label_values is None or node_feature_dim is None or edge_feature_dim is None:
        stats = fit_attribute_statistics(graphs, schema)
        if node_label_values is None:
            node_label_values = stats.node_label_values
        if edge_label_values is None:
            edge_label_values = stats.edge_label_values
        if graph_label_values is None:
            graph_label_values = stats.graph_label_values
        if node_feature_dim is None:
            node_feature_dim = stats.node_feature_dim
        if edge_feature_dim is None:
            edge_feature_dim = stats.edge_feature_dim
    node_label_values = [str(v) for v in (node_label_values or [])]
    edge_label_values = [str(v) for v in (edge_label_values or [])]
    graph_label_values = [str(v) for v in (graph_label_values or [])]
    node_feature_dim = int(node_feature_dim or 0)
    edge_feature_dim = int(edge_feature_dim or 0)
    nl, nf = schema["node_label_attr"], schema["node_feature_attr"]
    el, ef = schema["edge_label_attr"], schema["edge_feature_attr"]
    gl = schema["graph_label_attr"]
    rows: list[np.ndarray] = []
    for g in graphs:
        nlc: Counter[str] = Counter()
        elc: Counter[str] = Counter()
        glc: Counter[str] = Counter()
        node_feats: list[np.ndarray] = []
        edge_feats: list[np.ndarray] = []
        graph_label = _get_first(g.graph, gl, schema["graph_label_aliases"])
        if graph_label is not None:
            glc[_value_key(graph_label)] += 1
        for _, data in g.nodes(data=True):
            node_label = _get_first(data, nl, schema["node_label_aliases"])
            if node_label is not None:
                nlc[_value_key(node_label)] += 1
            v = _to_vector(_get_first(data, nf, schema["node_feature_aliases"]))
            if v is not None:
                node_feats.append(v)
        for _, _, data in g.edges(data=True):
            edge_label = _get_first(data, el, schema["edge_label_aliases"])
            if edge_label is not None:
                elc[_value_key(edge_label)] += 1
            v = _to_vector(_get_first(data, ef, schema["edge_feature_aliases"]))
            if v is not None:
                edge_feats.append(v)
        parts = [
            _hist_from_counter(nlc, node_label_values),
            _hist_from_counter(elc, edge_label_values),
            _hist_from_counter(glc, graph_label_values),
        ]
        if include_continuous:
            parts.extend([
                _feature_summary_from_vectors(node_feats, node_feature_dim),
                _feature_summary_from_vectors(edge_feats, edge_feature_dim),
            ])
        row = np.concatenate([p.reshape(-1) for p in parts]) if parts else np.zeros(1, dtype=np.float64)
        if row.size == 0:
            row = np.zeros(1, dtype=np.float64)
        rows.append(row.astype(np.float64))
    if not rows:
        return np.zeros((0, 1), dtype=np.float64)
    width = max(r.size for r in rows)
    out = np.zeros((len(rows), width), dtype=np.float64)
    for i, r in enumerate(rows):
        out[i, : r.size] = r
    return out
