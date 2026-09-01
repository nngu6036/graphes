"""Fast, read-only statistics for prepared NetworkX graph datasets."""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import networkx as nx
import yaml

from grapher.data.builders import SPLIT_NAMES
from grapher.utils.io import load_pickle, load_yaml

_SAFE_DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ATOM_ATTRIBUTE_KEYS = ("atomic_num", "atomic_number", "atom_type", "z")
_BOND_ATTRIBUTE_KEYS = ("bond_type", "edge_type", "bond_order")


@dataclass(frozen=True)
class PreparedDataset:
    """A resolved directory containing the three prepared graph splits."""

    requested_name: str
    serialized_name: str
    directory: Path
    resolution: str
    config_path: Path | None = None


def _validate_dataset_name(value: str) -> str:
    name = str(value).strip()
    if name in {"", ".", ".."} or _SAFE_DATASET_NAME.fullmatch(name) is None:
        raise ValueError(
            "dataset must be one identifier containing only letters, digits, "
            "'.', '_', or '-'"
        )
    return name


def _split_paths(directory: Path) -> dict[str, Path]:
    return {split: directory / f"{split}.pkl" for split in SPLIT_NAMES}


def _has_all_splits(directory: Path) -> bool:
    return all(path.is_file() for path in _split_paths(directory).values())


def _available_prepared_datasets(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir() and _has_all_splits(child)
    )


def resolve_prepared_dataset(
    dataset: str,
    *,
    root: str | Path = "outputs/datasets",
    config_dir: str | Path = "configs/datasets",
) -> PreparedDataset:
    """Resolve a CLI dataset name without building or modifying any data.

    An exact prepared directory wins. Otherwise, a dataset config may map the
    requested name to its serialized name, for example ``community_small`` to
    ``sbm``.
    """

    requested_name = _validate_dataset_name(dataset)
    root_path = Path(root)
    direct_directory = root_path / requested_name
    if _has_all_splits(direct_directory):
        return PreparedDataset(
            requested_name=requested_name,
            serialized_name=requested_name,
            directory=direct_directory,
            resolution="direct",
        )

    candidates: list[tuple[str, Path]] = [(requested_name, direct_directory)]
    config_path = Path(config_dir) / f"{requested_name}.yaml"
    if config_path.is_file():
        try:
            config = load_yaml(config_path)
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(
                f"Could not load dataset config {config_path}: {exc}"
            ) from exc
        if not isinstance(config, Mapping):
            raise TypeError(f"Dataset config {config_path} must be a mapping.")
        serialized_name = _validate_dataset_name(
            str(config.get("name", requested_name))
        )
        alias_directory = root_path / serialized_name
        if serialized_name != requested_name:
            candidates.append((serialized_name, alias_directory))
        if _has_all_splits(alias_directory):
            return PreparedDataset(
                requested_name=requested_name,
                serialized_name=serialized_name,
                directory=alias_directory,
                resolution="config_alias",
                config_path=config_path,
            )

    missing = []
    seen: set[Path] = set()
    for _name, directory in candidates:
        for path in _split_paths(directory).values():
            if path not in seen and not path.is_file():
                missing.append(str(path))
            seen.add(path)
    available = _available_prepared_datasets(root_path)
    available_text = ", ".join(available) if available else "none"
    raise FileNotFoundError(
        f"Prepared dataset {requested_name!r} is incomplete or missing. "
        f"Missing split files: {missing}. Available prepared datasets under "
        f"{root_path}: {available_text}. Run the appropriate preparation "
        "script before reporting statistics."
    )


@dataclass
class _NumericAccumulator:
    count: int = 0
    total: float = 0.0
    total_squared: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def add(self, value: float) -> None:
        number = float(value)
        self.count += 1
        self.total += number
        self.total_squared += number * number
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)

    def merge(self, other: _NumericAccumulator) -> None:
        if other.count == 0:
            return
        self.count += other.count
        self.total += other.total
        self.total_squared += other.total_squared
        if self.minimum is None:
            self.minimum = other.minimum
        elif other.minimum is not None:
            self.minimum = min(self.minimum, other.minimum)
        if self.maximum is None:
            self.maximum = other.maximum
        elif other.maximum is not None:
            self.maximum = max(self.maximum, other.maximum)

    def as_dict(self, *, integral: bool = False) -> dict[str, int | float | None]:
        if self.count == 0:
            return {
                "count": 0,
                "total": 0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            }
        mean = self.total / self.count
        variance = max(0.0, self.total_squared / self.count - mean * mean)
        minimum: int | float = self.minimum or 0.0
        maximum: int | float = self.maximum or 0.0
        total: int | float = self.total
        if integral:
            minimum = int(minimum)
            maximum = int(maximum)
            total = int(total)
        return {
            "count": self.count,
            "total": total,
            "min": minimum,
            "max": maximum,
            "mean": float(mean),
            "std": float(math.sqrt(variance)),
        }


def _categorical_label(value: Any) -> str | None:
    if not isinstance(value, (str, int, float, bool, type(None))):
        item = getattr(value, "item", None)
        if callable(item) and getattr(value, "ndim", None) == 0:
            try:
                value = item()
            except (TypeError, ValueError):
                return None
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return format(value, ".12g")
    if isinstance(value, str):
        return value
    return None


def _category_sort_key(label: str) -> tuple[int, float, str]:
    try:
        numeric = float(label)
    except ValueError:
        return (1, 0.0, label)
    if not math.isfinite(numeric):
        return (1, 0.0, label)
    return (0, numeric, label)


def _sorted_category_counts(counts: Counter[str]) -> dict[str, int]:
    return {
        name: int(count)
        for name, count in sorted(
            counts.items(), key=lambda item: _category_sort_key(item[0])
        )
    }


@dataclass
class _AttributeAccumulator:
    max_categories: int = 50
    total_items: int = 0
    presence: Counter[str] = field(default_factory=Counter)
    categories: dict[str, Counter[str]] = field(default_factory=dict)
    non_categorical: set[str] = field(default_factory=set)

    def add(self, attributes: Mapping[str, Any]) -> None:
        self.total_items += 1
        for raw_key, value in attributes.items():
            key = str(raw_key)
            self.presence[key] += 1
            if key in self.non_categorical:
                continue
            label = _categorical_label(value)
            if label is None:
                self.non_categorical.add(key)
                self.categories.pop(key, None)
                continue
            counts = self.categories.setdefault(key, Counter())
            counts[label] += 1
            if len(counts) > self.max_categories:
                self.non_categorical.add(key)
                self.categories.pop(key, None)

    def merge(self, other: _AttributeAccumulator) -> None:
        self.total_items += other.total_items
        self.presence.update(other.presence)
        for key in other.non_categorical:
            self.non_categorical.add(key)
            self.categories.pop(key, None)
        for key, counts in other.categories.items():
            if key in self.non_categorical:
                continue
            combined = self.categories.setdefault(key, Counter())
            combined.update(counts)
            if len(combined) > self.max_categories:
                self.non_categorical.add(key)
                self.categories.pop(key, None)

    def as_dict(self) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for key in sorted(self.presence):
            present = int(self.presence[key])
            entry: dict[str, Any] = {
                "present": present,
                "missing": self.total_items - present,
                "coverage": (
                    float(present / self.total_items)
                    if self.total_items > 0
                    else None
                ),
            }
            counts = self.categories.get(key)
            entry["categorical_counts"] = (
                _sorted_category_counts(counts)
                if counts is not None
                else None
            )
            fields[key] = entry
        return {"total_items": self.total_items, "fields": fields}


def _first_categorical_value(
    attributes: Mapping[str, Any], keys: Sequence[str]
) -> tuple[str, str] | None:
    for key in keys:
        if key not in attributes:
            continue
        label = _categorical_label(attributes[key])
        if label is not None:
            return key, label
    return None


@dataclass
class GraphStatisticsAccumulator:
    """Incremental O(V + E) statistics for a graph collection."""

    num_graphs: int = 0
    node_count: _NumericAccumulator = field(default_factory=_NumericAccumulator)
    edge_count: _NumericAccumulator = field(default_factory=_NumericAccumulator)
    average_degree: _NumericAccumulator = field(default_factory=_NumericAccumulator)
    density: _NumericAccumulator = field(default_factory=_NumericAccumulator)
    component_count: _NumericAccumulator = field(default_factory=_NumericAccumulator)
    isolate_count: _NumericAccumulator = field(default_factory=_NumericAccumulator)
    degree: _NumericAccumulator = field(default_factory=_NumericAccumulator)
    degree_histogram: Counter[int] = field(default_factory=Counter)
    connected_graphs: int = 0
    empty_graphs: int = 0
    directed_graphs: int = 0
    multigraphs: int = 0
    graphs_without_self_loops: int = 0
    self_loops: int = 0
    isolated_nodes: int = 0
    node_attributes: _AttributeAccumulator = field(
        default_factory=_AttributeAccumulator
    )
    edge_attributes: _AttributeAccumulator = field(
        default_factory=_AttributeAccumulator
    )
    atom_type_counts: Counter[str] = field(default_factory=Counter)
    atom_attribute_keys: Counter[str] = field(default_factory=Counter)
    bond_type_counts: Counter[str] = field(default_factory=Counter)
    bond_attribute_keys: Counter[str] = field(default_factory=Counter)

    def add_graph(self, graph: nx.Graph) -> None:
        self.num_graphs += 1
        num_nodes = int(graph.number_of_nodes())
        num_edges = int(graph.number_of_edges())
        self.node_count.add(num_nodes)
        self.edge_count.add(num_edges)
        self.average_degree.add(2.0 * num_edges / num_nodes if num_nodes else 0.0)
        self.density.add(float(nx.density(graph)) if num_nodes > 1 else 0.0)

        if num_nodes == 0:
            components = 0
            connected = False
            self.empty_graphs += 1
        elif graph.is_directed():
            components = int(nx.number_weakly_connected_components(graph))
            connected = components == 1
        else:
            components = int(nx.number_connected_components(graph))
            connected = components == 1
        self.component_count.add(components)
        self.connected_graphs += int(connected)

        isolates = int(nx.number_of_isolates(graph))
        self.isolate_count.add(isolates)
        self.isolated_nodes += isolates
        self.directed_graphs += int(graph.is_directed())
        self.multigraphs += int(graph.is_multigraph())

        loops = int(nx.number_of_selfloops(graph))
        self.self_loops += loops
        self.graphs_without_self_loops += int(loops == 0)

        for _node, attributes in graph.nodes(data=True):
            self.node_attributes.add(attributes)
            atom = _first_categorical_value(attributes, _ATOM_ATTRIBUTE_KEYS)
            if atom is not None:
                key, value = atom
                self.atom_attribute_keys[key] += 1
                self.atom_type_counts[value] += 1

        for _left, _right, attributes in graph.edges(data=True):
            self.edge_attributes.add(attributes)
            bond = _first_categorical_value(attributes, _BOND_ATTRIBUTE_KEYS)
            if bond is not None:
                key, value = bond
                self.bond_attribute_keys[key] += 1
                self.bond_type_counts[value] += 1

        for _node, raw_degree in graph.degree():
            degree = int(raw_degree)
            self.degree.add(degree)
            self.degree_histogram[degree] += 1

    def merge(self, other: GraphStatisticsAccumulator) -> None:
        self.num_graphs += other.num_graphs
        for own, incoming in (
            (self.node_count, other.node_count),
            (self.edge_count, other.edge_count),
            (self.average_degree, other.average_degree),
            (self.density, other.density),
            (self.component_count, other.component_count),
            (self.isolate_count, other.isolate_count),
            (self.degree, other.degree),
        ):
            own.merge(incoming)
        self.degree_histogram.update(other.degree_histogram)
        self.connected_graphs += other.connected_graphs
        self.empty_graphs += other.empty_graphs
        self.directed_graphs += other.directed_graphs
        self.multigraphs += other.multigraphs
        self.graphs_without_self_loops += other.graphs_without_self_loops
        self.self_loops += other.self_loops
        self.isolated_nodes += other.isolated_nodes
        self.node_attributes.merge(other.node_attributes)
        self.edge_attributes.merge(other.edge_attributes)
        self.atom_type_counts.update(other.atom_type_counts)
        self.atom_attribute_keys.update(other.atom_attribute_keys)
        self.bond_type_counts.update(other.bond_type_counts)
        self.bond_attribute_keys.update(other.bond_attribute_keys)

    def as_dict(self) -> dict[str, Any]:
        node_summary = self.node_count.as_dict(integral=True)
        edge_summary = self.edge_count.as_dict(integral=True)
        result: dict[str, Any] = {
            "num_graphs": self.num_graphs,
            "total_nodes": int(self.node_count.total),
            "total_edges": int(self.edge_count.total),
            "node_count": node_summary,
            "edge_count": edge_summary,
            "min_nodes": node_summary["min"],
            "max_nodes": node_summary["max"],
            "mean_nodes": node_summary["mean"],
            "std_nodes": node_summary["std"],
            "min_edges": edge_summary["min"],
            "max_edges": edge_summary["max"],
            "mean_edges": edge_summary["mean"],
            "std_edges": edge_summary["std"],
            "degree": self.degree.as_dict(integral=True),
            "degree_histogram": {
                str(degree): int(count)
                for degree, count in sorted(self.degree_histogram.items())
            },
            "max_degree": (
                int(self.degree.maximum) if self.degree.maximum is not None else None
            ),
            "average_degree": self.average_degree.as_dict(),
            "density": self.density.as_dict(),
            "component_count": self.component_count.as_dict(integral=True),
            "isolate_count": self.isolate_count.as_dict(integral=True),
            "isolated_nodes": self.isolated_nodes,
            "connected_graphs": self.connected_graphs,
            "connected_rate": (
                float(self.connected_graphs / self.num_graphs)
                if self.num_graphs
                else None
            ),
            "empty_graphs": self.empty_graphs,
            "directed_graphs": self.directed_graphs,
            "multigraphs": self.multigraphs,
            "self_loops": self.self_loops,
            "zero_self_loop_rate": (
                float(self.graphs_without_self_loops / self.num_graphs)
                if self.num_graphs
                else None
            ),
            "node_attributes": self.node_attributes.as_dict(),
            "edge_attributes": self.edge_attributes.as_dict(),
        }
        num_atomic_nodes = sum(self.atom_attribute_keys.values())
        if num_atomic_nodes:
            num_bond_edges = sum(self.bond_attribute_keys.values())
            result["molecular_attributes"] = {
                "atom_attribute_coverage": (
                    float(num_atomic_nodes / self.node_attributes.total_items)
                    if self.node_attributes.total_items
                    else None
                ),
                "atom_attribute_keys": {
                    key: int(count)
                    for key, count in sorted(self.atom_attribute_keys.items())
                },
                "atom_type_counts": _sorted_category_counts(self.atom_type_counts),
                "bond_attribute_coverage": (
                    float(num_bond_edges / self.edge_attributes.total_items)
                    if self.edge_attributes.total_items
                    else None
                ),
                "bond_attribute_keys": {
                    key: int(count)
                    for key, count in sorted(self.bond_attribute_keys.items())
                },
                "bond_type_counts": _sorted_category_counts(self.bond_type_counts),
            }
        return result


def _load_graph_split(path: Path, split: str) -> Sequence[nx.Graph]:
    try:
        payload = load_pickle(path)
    except (OSError, EOFError, pickle.UnpicklingError, ImportError) as exc:
        raise ValueError(
            f"Could not load dataset split {split!r} from {path}: {exc}"
        ) from exc
    if not isinstance(payload, (list, tuple)):
        raise TypeError(
            f"Dataset split {split!r} in {path} must contain a list or tuple "
            f"of NetworkX graphs, found {type(payload).__name__}."
        )
    for index, graph in enumerate(payload):
        if not isinstance(graph, nx.Graph):
            raise TypeError(
                f"Dataset split {split!r} item {index} in {path} is not a "
                f"NetworkX graph: {type(graph).__name__}."
            )
    return payload


def compute_prepared_dataset_statistics(
    dataset: PreparedDataset,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Load one split at a time and compute split plus aggregate statistics."""

    overall = GraphStatisticsAccumulator()
    statistics: dict[str, dict[str, Any]] = {}
    split_sizes: dict[str, int] = {}
    for split, path in _split_paths(dataset.directory).items():
        if progress is not None:
            progress(f"Processing {split} split: {path}")
        graphs = _load_graph_split(path, split)
        accumulator = GraphStatisticsAccumulator()
        for graph in graphs:
            accumulator.add_graph(graph)
        split_sizes[split] = accumulator.num_graphs
        statistics[split] = accumulator.as_dict()
        overall.merge(accumulator)
        if progress is not None:
            progress(f"Processed {split}: {accumulator.num_graphs} graphs")
    statistics = {"all": overall.as_dict(), **statistics}
    return {
        "dataset": dataset.requested_name,
        "serialized_dataset": dataset.serialized_name,
        "dataset_directory": str(dataset.directory),
        "resolution": dataset.resolution,
        "config_path": (
            str(dataset.config_path) if dataset.config_path is not None else None
        ),
        "split_sizes": split_sizes,
        "statistics": statistics,
    }


def _format_number(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _format_range(summary: Mapping[str, Any]) -> str:
    if summary.get("mean") is None:
        return "n/a"
    return (
        f"{_format_number(summary.get('min'), digits=0)}/"
        f"{_format_number(summary.get('mean'), digits=2)}/"
        f"{_format_number(summary.get('max'), digits=0)}"
    )


def format_graph_statistics_table(statistics: Mapping[str, Mapping[str, Any]]) -> str:
    """Format the fast aggregate and split statistics as a compact table."""

    headers = (
        "split",
        "graphs",
        "nodes min/mean/max",
        "edges min/mean/max",
        "max degree",
        "graph avg degree",
        "mean density",
        "connected",
    )
    rows: list[tuple[str, ...]] = []
    for split in ("all", *SPLIT_NAMES):
        if split not in statistics:
            continue
        row = statistics[split]
        connected_rate = row.get("connected_rate")
        rows.append(
            (
                split,
                str(row.get("num_graphs", 0)),
                _format_range(row.get("node_count", {})),
                _format_range(row.get("edge_count", {})),
                _format_number(row.get("max_degree"), digits=0),
                _format_number(
                    (row.get("average_degree") or {}).get("mean"), digits=3
                ),
                _format_number((row.get("density") or {}).get("mean"), digits=4),
                (
                    f"{100.0 * float(connected_rate):.1f}%"
                    if connected_rate is not None
                    else "n/a"
                ),
            )
        )
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    header_line = "  ".join(
        header.ljust(width) for header, width in zip(headers, widths)
    )
    separator = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths))
        for row in rows
    ]
    return "\n".join((header_line, separator, *body))
