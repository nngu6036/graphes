"""Dependency-light neutral codec for molecular DeFoG artifacts.

The external DeFoG implementation represents atoms by zero-based class indices
and bonds by dense integer matrices whose value zero means ``no bond``.  The
GraphER project represents the same chemistry with semantic NetworkX
attributes (atomic numbers and bond types).  This module is the strict boundary
between those representations; it intentionally depends only on NumPy and
NetworkX, not on Torch, PyG, RDKit, or DeFoG.

Two representations are kept distinct in every neutral archive:

``source``
    A GraphER-prepared molecular graph.  ZINC source graphs may contain the
    explicit aromatic bond type 4.

``model``
    A graph in the native categorical support of the attached DeFoG model.
    Its ZINC implementation is kekulized and supports only bond types 1--3.

The codec never performs aromaticity perception or kekulization.  Those are
chemical transformations and require an explicit, separately audited RDKit
conversion.  In particular, a ZINC graph containing aromatic bond type 4 is
never silently accepted as a DeFoG-model graph.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import numpy as np

MOLECULAR_EXPORT_FORMAT = "grapher_defog_molecular_v1"
SOURCE_REPRESENTATION = "source"
MODEL_REPRESENTATION = "model"
MolecularRepresentation = Literal["source", "model"]

_BOND_ORDER = {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}
_UNSUPPORTED_NODE_ATTRIBUTES = (
    "chiral_tag",
    "chirality",
    "is_aromatic",
    "stereo",
)
_UNSUPPORTED_EDGE_ATTRIBUTES = (
    "bond_stereo",
    "stereo",
    "stereo_atoms",
)


@dataclass(frozen=True)
class MolecularCodecSpec:
    """Exact GraphER/DeFoG categorical mapping for one molecular benchmark."""

    dataset: str
    atom_class_to_atomic_number: tuple[int, ...]
    source_bond_types: frozenset[int]
    model_bond_types: frozenset[int]

    @property
    def atomic_number_to_atom_class(self) -> dict[int, int]:
        return {
            atomic_number: index
            for index, atomic_number in enumerate(self.atom_class_to_atomic_number)
        }

    def allowed_bond_types(
        self, representation: MolecularRepresentation
    ) -> frozenset[int]:
        _validate_representation(representation)
        if representation == SOURCE_REPRESENTATION:
            return self.source_bond_types
        return self.model_bond_types


MOLECULAR_CODEC_SPECS: dict[str, MolecularCodecSpec] = {
    "qm9": MolecularCodecSpec(
        dataset="qm9",
        atom_class_to_atomic_number=(6, 7, 8, 9),
        source_bond_types=frozenset({1, 2, 3, 4}),
        model_bond_types=frozenset({1, 2, 3, 4}),
    ),
    "zinc": MolecularCodecSpec(
        dataset="zinc",
        atom_class_to_atomic_number=(6, 7, 8, 9, 15, 16, 17, 35, 53),
        source_bond_types=frozenset({1, 2, 3, 4}),
        # The attached DeFoG ZINC loader kekulizes molecules and configures
        # aromatic=False.  Its dense edge labels are therefore 0--3 only.
        model_bond_types=frozenset({1, 2, 3}),
    ),
}


def molecular_codec_spec(dataset: str) -> MolecularCodecSpec:
    """Resolve a supported molecular dataset without accepting aliases."""

    normalized = str(dataset).strip().lower()
    try:
        return MOLECULAR_CODEC_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported DeFoG molecular dataset {dataset!r}; expected one of "
            f"{sorted(MOLECULAR_CODEC_SPECS)}."
        ) from exc


def _validate_representation(value: str) -> MolecularRepresentation:
    normalized = str(value).strip().lower()
    if normalized not in {SOURCE_REPRESENTATION, MODEL_REPRESENTATION}:
        raise ValueError("Molecular representation must be 'source' or 'model'.")
    return normalized  # type: ignore[return-value]


def _as_integer_array(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain numeric class labels.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite class labels.")
    rounded = np.rint(array)
    if not np.array_equal(array, rounded):
        raise ValueError(f"{name} must contain integral class labels.")
    return rounded.astype(np.int64, copy=False)


def _semantic_node_value(data: Mapping[str, Any], *, location: str) -> int:
    atomic_number = data.get("atomic_num")
    atom_type = data.get("atom_type")
    if atomic_number is None and atom_type is None:
        raise ValueError(f"{location} is missing atomic_num/atom_type.")
    if atomic_number is not None and atom_type is not None:
        if int(atomic_number) != int(atom_type):
            raise ValueError(
                f"{location} has inconsistent atomic_num={atomic_number!r} and "
                f"atom_type={atom_type!r}."
            )
    value = atomic_number if atomic_number is not None else atom_type
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} has a non-integral atom category.") from exc


def _reject_unrepresented_node_chemistry(
    data: Mapping[str, Any], *, location: str
) -> None:
    if "formal_charge" in data and int(data["formal_charge"]) != 0:
        raise ValueError(
            f"{location} has non-zero formal_charge, which the attached DeFoG "
            "QM9/ZINC categorical state does not represent."
        )
    for name in _UNSUPPORTED_NODE_ATTRIBUTES:
        value = data.get(name)
        if value not in (None, False, 0, "", "none", "unspecified"):
            raise ValueError(
                f"{location} has unsupported chemical attribute {name}={value!r}."
            )


def _semantic_bond_value(
    data: Mapping[str, Any],
    *,
    location: str,
) -> int:
    if "bond_type" not in data:
        raise ValueError(f"{location} is missing bond_type.")
    try:
        bond_type = int(data["bond_type"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} has a non-integral bond_type.") from exc
    if "bond_order" in data:
        try:
            observed_order = float(data["bond_order"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{location} has a non-numeric bond_order.") from exc
        expected_order = _BOND_ORDER.get(bond_type)
        if expected_order is None or not np.isclose(
            observed_order, expected_order, rtol=0.0, atol=1e-8
        ):
            raise ValueError(
                f"{location} has bond_type={bond_type} but "
                f"bond_order={observed_order}."
            )
    for name in _UNSUPPORTED_EDGE_ATTRIBUTES:
        value = data.get(name)
        if value not in (None, False, 0, "", "none", "unspecified"):
            raise ValueError(
                f"{location} has unsupported chemical attribute {name}={value!r}."
            )
    return bond_type


def _pack_semantic_graphs(
    semantic_graphs: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    dataset: str,
    representation: MolecularRepresentation,
) -> dict[str, np.ndarray]:
    if not semantic_graphs:
        raise ValueError("A molecular export must contain at least one graph.")
    node_ptr = [0]
    edge_ptr = [0]
    all_atoms: list[np.ndarray] = []
    all_endpoints: list[np.ndarray] = []
    all_bonds: list[np.ndarray] = []
    for atoms, endpoints, bonds in semantic_graphs:
        all_atoms.append(atoms.astype(np.int64, copy=False))
        all_endpoints.append(endpoints.astype(np.int64, copy=False).reshape(-1, 2))
        all_bonds.append(bonds.astype(np.int64, copy=False))
        node_ptr.append(node_ptr[-1] + int(atoms.size))
        edge_ptr.append(edge_ptr[-1] + int(bonds.size))
    return {
        "format": np.asarray(MOLECULAR_EXPORT_FORMAT),
        "dataset": np.asarray(dataset),
        "representation": np.asarray(representation),
        "node_ptr": np.asarray(node_ptr, dtype=np.int64),
        "node_atomic_numbers": np.concatenate(all_atoms),
        "edge_ptr": np.asarray(edge_ptr, dtype=np.int64),
        "edge_endpoints": (
            np.concatenate(all_endpoints, axis=0)
            if any(endpoints.size for endpoints in all_endpoints)
            else np.empty((0, 2), dtype=np.int64)
        ),
        "edge_bond_types": (
            np.concatenate(all_bonds)
            if any(bonds.size for bonds in all_bonds)
            else np.empty((0,), dtype=np.int64)
        ),
        # Sequential raw indices make omission or reordering observable.
        "raw_indices": np.arange(len(semantic_graphs), dtype=np.int64),
    }


def encode_molecular_graphs(
    graphs: Sequence[nx.Graph],
    *,
    dataset: str,
    representation: MolecularRepresentation = SOURCE_REPRESENTATION,
) -> dict[str, np.ndarray]:
    """Encode ordered GraphER NetworkX molecules into neutral semantic arrays.

    Node iteration order is retained as local node order.  Arbitrary original
    node identifiers are deliberately not serialized; decoded graphs use
    contiguous integer labels while preserving graph structure and isolates.
    """

    spec = molecular_codec_spec(dataset)
    representation = _validate_representation(representation)
    allowed_bonds = spec.allowed_bond_types(representation)
    semantic_graphs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    if not graphs:
        raise ValueError("A molecular export must contain at least one graph.")
    for graph_index, graph in enumerate(graphs):
        if not isinstance(graph, nx.Graph):
            raise TypeError(f"graph[{graph_index}] is not a NetworkX graph.")
        if graph.is_directed() or graph.is_multigraph():
            raise ValueError(
                f"graph[{graph_index}] must be a simple undirected graph."
            )
        if graph.number_of_nodes() <= 0:
            raise ValueError(f"graph[{graph_index}] has no nodes.")
        if nx.number_of_selfloops(graph):
            raise ValueError(f"graph[{graph_index}] contains a self-loop.")

        nodes = list(graph.nodes())
        node_to_local = {node: index for index, node in enumerate(nodes)}
        atomic_numbers: list[int] = []
        for local_index, node in enumerate(nodes):
            data = graph.nodes[node]
            location = f"graph[{graph_index}].node[{local_index}]"
            atomic_number = _semantic_node_value(data, location=location)
            _reject_unrepresented_node_chemistry(data, location=location)
            if atomic_number not in spec.atomic_number_to_atom_class:
                raise ValueError(
                    f"{location} has atomic number {atomic_number}, outside the "
                    f"{spec.dataset} vocabulary "
                    f"{spec.atom_class_to_atomic_number}."
                )
            atomic_numbers.append(atomic_number)

        endpoints: list[tuple[int, int]] = []
        bond_types: list[int] = []
        seen: set[tuple[int, int]] = set()
        for edge_index, (source, target, data) in enumerate(graph.edges(data=True)):
            u = node_to_local[source]
            v = node_to_local[target]
            edge = (u, v) if u < v else (v, u)
            if edge[0] == edge[1]:
                raise ValueError(f"graph[{graph_index}] contains a self-loop.")
            if edge in seen:
                raise ValueError(f"graph[{graph_index}] contains a duplicate edge.")
            seen.add(edge)
            location = f"graph[{graph_index}].edge[{edge_index}]"
            bond_type = _semantic_bond_value(data, location=location)
            if bond_type not in allowed_bonds:
                if (
                    spec.dataset == "zinc"
                    and representation == MODEL_REPRESENTATION
                    and bond_type == 4
                ):
                    raise ValueError(
                        "Aromatic ZINC bond type 4 is a source representation, "
                        "but the attached DeFoG ZINC model is kekulized and accepts "
                        "only bond types 1--3. Apply an explicit audited "
                        "kekulization step before model encoding."
                    )
                raise ValueError(
                    f"{location} has bond type {bond_type}, outside the "
                    f"{spec.dataset} {representation} vocabulary "
                    f"{sorted(allowed_bonds)}."
                )
            endpoints.append(edge)
            bond_types.append(bond_type)

        semantic_graphs.append(
            (
                np.asarray(atomic_numbers, dtype=np.int64),
                np.asarray(endpoints, dtype=np.int64).reshape(-1, 2),
                np.asarray(bond_types, dtype=np.int64),
            )
        )
    return _pack_semantic_graphs(
        semantic_graphs,
        dataset=spec.dataset,
        representation=representation,
    )


def encode_defog_molecular_samples(
    samples: Sequence[Any],
    *,
    dataset: str,
) -> dict[str, np.ndarray]:
    """Encode ordered raw DeFoG ``[X, E]`` samples into neutral arrays."""

    spec = molecular_codec_spec(dataset)
    if not samples:
        raise ValueError("DeFoG returned no molecular samples.")
    semantic_graphs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    allowed_edges = {0, *spec.model_bond_types}
    for graph_index, sample in enumerate(samples):
        if not isinstance(sample, (list, tuple)) or len(sample) != 2:
            raise TypeError(
                f"sample[{graph_index}] must be a two-item [X, E] pair."
            )
        atom_classes = _as_integer_array(
            sample[0], name=f"sample[{graph_index}].X"
        )
        edge_classes = _as_integer_array(
            sample[1], name=f"sample[{graph_index}].E"
        )
        if atom_classes.ndim != 1 or atom_classes.size <= 0:
            raise ValueError(
                f"sample[{graph_index}].X must have shape [N] with N > 0."
            )
        node_count = int(atom_classes.size)
        if edge_classes.shape != (node_count, node_count):
            raise ValueError(
                f"sample[{graph_index}].E has shape {edge_classes.shape}; "
                f"expected {(node_count, node_count)}."
            )
        if np.any(atom_classes < 0) or np.any(
            atom_classes >= len(spec.atom_class_to_atomic_number)
        ):
            raise ValueError(
                f"sample[{graph_index}].X contains a class outside "
                f"0--{len(spec.atom_class_to_atomic_number) - 1}."
            )
        if spec.dataset == "zinc" and np.any(edge_classes == 4):
            raise ValueError(
                f"sample[{graph_index}].E contains aromatic class 4, but the "
                "attached DeFoG ZINC model is kekulized and accepts only edge "
                "classes 0--3."
            )
        if not np.all(np.isin(edge_classes, tuple(sorted(allowed_edges)))):
            raise ValueError(
                f"sample[{graph_index}].E contains a class outside "
                f"{sorted(allowed_edges)}."
            )
        if not np.array_equal(edge_classes, edge_classes.T):
            raise ValueError(f"sample[{graph_index}].E is not symmetric.")
        if np.any(np.diag(edge_classes) != 0):
            raise ValueError(f"sample[{graph_index}].E contains a self-loop class.")

        endpoints = np.argwhere(np.triu(edge_classes > 0, k=1)).astype(
            np.int64, copy=False
        )
        bonds = (
            edge_classes[endpoints[:, 0], endpoints[:, 1]]
            if endpoints.size
            else np.empty((0,), dtype=np.int64)
        )
        atomic_numbers = np.asarray(
            [spec.atom_class_to_atomic_number[int(value)] for value in atom_classes],
            dtype=np.int64,
        )
        semantic_graphs.append((atomic_numbers, endpoints.reshape(-1, 2), bonds))
    return _pack_semantic_graphs(
        semantic_graphs,
        dataset=spec.dataset,
        representation=MODEL_REPRESENTATION,
    )


def _scalar_text(value: Any, *, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a scalar text field.")
    item = array.item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def decode_molecular_arrays(
    arrays: Mapping[str, Any],
    *,
    expected_dataset: str | None = None,
    expected_representation: MolecularRepresentation | None = None,
    expected_count: int | None = None,
) -> list[nx.Graph]:
    """Strictly validate neutral arrays and decode an ordered graph batch."""

    required = {
        "format",
        "dataset",
        "representation",
        "node_ptr",
        "node_atomic_numbers",
        "edge_ptr",
        "edge_endpoints",
        "edge_bond_types",
        "raw_indices",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"Molecular export is missing arrays: {missing}.")
    if _scalar_text(arrays["format"], name="format") != MOLECULAR_EXPORT_FORMAT:
        raise ValueError("Unsupported molecular export format.")
    dataset = _scalar_text(arrays["dataset"], name="dataset").lower()
    spec = molecular_codec_spec(dataset)
    if expected_dataset is not None and spec.dataset != str(expected_dataset).lower():
        raise ValueError(
            f"Molecular export dataset {spec.dataset!r} does not match expected "
            f"dataset {expected_dataset!r}."
        )
    representation = _validate_representation(
        _scalar_text(arrays["representation"], name="representation")
    )
    if expected_representation is not None:
        expected_representation = _validate_representation(expected_representation)
        if representation != expected_representation:
            raise ValueError(
                f"Molecular export representation {representation!r} does not "
                f"match expected representation {expected_representation!r}."
            )

    node_ptr = _as_integer_array(arrays["node_ptr"], name="node_ptr")
    atoms = _as_integer_array(
        arrays["node_atomic_numbers"], name="node_atomic_numbers"
    )
    edge_ptr = _as_integer_array(arrays["edge_ptr"], name="edge_ptr")
    endpoints = _as_integer_array(arrays["edge_endpoints"], name="edge_endpoints")
    bonds = _as_integer_array(arrays["edge_bond_types"], name="edge_bond_types")
    raw_indices = _as_integer_array(arrays["raw_indices"], name="raw_indices")

    if node_ptr.ndim != 1 or edge_ptr.ndim != 1 or node_ptr.size != edge_ptr.size:
        raise ValueError("node_ptr and edge_ptr must be aligned one-dimensional arrays.")
    graph_count = int(node_ptr.size - 1)
    if graph_count <= 0:
        raise ValueError("Molecular export contains no graphs.")
    if expected_count is not None and graph_count != int(expected_count):
        raise ValueError(
            f"Molecular export contains {graph_count} graphs; expected "
            f"{int(expected_count)}."
        )
    if node_ptr[0] != 0 or edge_ptr[0] != 0:
        raise ValueError("node_ptr and edge_ptr must start at zero.")
    if np.any(np.diff(node_ptr) <= 0) or np.any(np.diff(edge_ptr) < 0):
        raise ValueError("Pointer arrays are not monotone or contain an empty graph.")
    if node_ptr[-1] != atoms.size:
        raise ValueError("node_ptr does not cover node_atomic_numbers.")
    if endpoints.ndim != 2 or endpoints.shape[1:] != (2,):
        raise ValueError("edge_endpoints must have shape [M, 2].")
    if edge_ptr[-1] != endpoints.shape[0] or bonds.shape != (endpoints.shape[0],):
        raise ValueError("edge_ptr, edge_endpoints, and edge_bond_types disagree.")
    if raw_indices.shape != (graph_count,) or not np.array_equal(
        raw_indices, np.arange(graph_count, dtype=np.int64)
    ):
        raise ValueError(
            "raw_indices must be the sequential graph order 0..N-1; omission or "
            "reordering is not accepted."
        )

    allowed_atoms = set(spec.atom_class_to_atomic_number)
    allowed_bonds = spec.allowed_bond_types(representation)
    graphs: list[nx.Graph] = []
    for graph_index in range(graph_count):
        node_start = int(node_ptr[graph_index])
        node_stop = int(node_ptr[graph_index + 1])
        edge_start = int(edge_ptr[graph_index])
        edge_stop = int(edge_ptr[graph_index + 1])
        graph_atoms = atoms[node_start:node_stop]
        graph_endpoints = endpoints[edge_start:edge_stop]
        graph_bonds = bonds[edge_start:edge_stop]
        node_count = int(graph_atoms.size)
        if not set(int(value) for value in graph_atoms) <= allowed_atoms:
            raise ValueError(
                f"graph[{graph_index}] contains an atomic number outside the "
                f"{spec.dataset} vocabulary."
            )
        if not set(int(value) for value in graph_bonds) <= set(allowed_bonds):
            raise ValueError(
                f"graph[{graph_index}] contains a bond type outside the "
                f"{spec.dataset} {representation} vocabulary."
            )
        if graph_endpoints.size:
            if np.any(graph_endpoints < 0) or np.any(graph_endpoints >= node_count):
                raise ValueError(f"graph[{graph_index}] has an endpoint out of range.")
            if np.any(graph_endpoints[:, 0] >= graph_endpoints[:, 1]):
                raise ValueError(
                    f"graph[{graph_index}] edges must use canonical u < v order."
                )
            if np.unique(graph_endpoints, axis=0).shape[0] != graph_endpoints.shape[0]:
                raise ValueError(f"graph[{graph_index}] contains duplicate edges.")

        graph = nx.Graph()
        for node, atomic_number in enumerate(graph_atoms):
            value = int(atomic_number)
            graph.add_node(node, atomic_num=value, atom_type=value)
        if graph_endpoints.shape[0] != graph_bonds.shape[0]:
            raise ValueError(
                f"graph[{graph_index}] has mismatched edge endpoint and "
                "bond-type counts."
            )
        # ``zip(..., strict=True)`` was introduced in Python 3.10, while the
        # isolated DeFoG environment used by the wrapper may run Python 3.9.
        # Keep the same strictness guarantee with an explicit length check.
        for (source, target), bond_type in zip(graph_endpoints, graph_bonds):
            semantic_bond = int(bond_type)
            graph.add_edge(
                int(source),
                int(target),
                bond_type=semantic_bond,
                bond_order=_BOND_ORDER[semantic_bond],
            )
        graph.graph.update(
            {
                "base_generator": "defog",
                "defog_raw_index": int(raw_indices[graph_index]),
                "molecular_dataset": spec.dataset,
                "molecular_representation": representation,
            }
        )
        graphs.append(graph)
    return graphs


def save_molecular_export(path: str | Path, arrays: Mapping[str, Any]) -> Path:
    """Atomically save a neutral archive that is safe for ``allow_pickle=False``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **dict(arrays))
    os.replace(temporary, destination)
    return destination


def load_molecular_export(
    path: str | Path,
    *,
    expected_dataset: str | None = None,
    expected_representation: MolecularRepresentation | None = None,
    expected_count: int | None = None,
) -> list[nx.Graph]:
    """Load a neutral archive without allowing object-pickle payloads."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Missing molecular DeFoG export: {source}")
    with np.load(source, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    return decode_molecular_arrays(
        arrays,
        expected_dataset=expected_dataset,
        expected_representation=expected_representation,
        expected_count=expected_count,
    )


__all__ = [
    "MODEL_REPRESENTATION",
    "MOLECULAR_CODEC_SPECS",
    "MOLECULAR_EXPORT_FORMAT",
    "SOURCE_REPRESENTATION",
    "MolecularCodecSpec",
    "decode_molecular_arrays",
    "encode_defog_molecular_samples",
    "encode_molecular_graphs",
    "load_molecular_export",
    "molecular_codec_spec",
    "save_molecular_export",
]
