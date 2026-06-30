from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grapher.evaluation.data_io import load_dataset_splits
from grapher.evaluation.metrics import (
    clustering_histogram,
    degree_histogram,
    descriptor_matrix,
    mmd_gaussian_emd,
    mmd_rbf,
    motif_proxy_vector,
    spectral_histogram,
    structural_summary,
)
from grapher.generation.rewiring import (
    RewireAction,
    action_signature,
    degree_sequence,
    enumerate_rewire_actions,
    rewire_action,
)
from grapher.generation.validity import quality_metrics
from grapher.generation.molecular_rewiring import (
    EmpiricalBondTypePrior,
    MolecularRewireAction,
    apply_molecular_rewire,
    edge_type_value,
    enumerate_molecular_rewire_actions,
    node_type_value,
)
from grapher.evaluation.molecular_metrics import (
    compute_fcd,
    compute_nspdk_mmd,
    direct_molecular_conversions,
    molecular_novelty,
    molecular_uniqueness,
    valid_smiles,
    validity_without_correction,
)
from grapher.generation.validity import quality_metrics
from grapher.molecules.representation import (
    BOND_TYPE_TO_ORDER,
    canonicalize_molecular_graph,
    write_molecular_jsonl,
    write_molecular_sdf,
    write_smiles_file,
)
from grapher.utils.io import load_pickle, save_json, save_pickle
from grapher.utils.seed import set_seed


# -----------------------------------------------------------------------------
# I/O: DiGress sample loading
# -----------------------------------------------------------------------------


def _canonical_graph(graph: nx.Graph) -> nx.Graph:
    g = nx.Graph(graph)
    g.remove_edges_from(nx.selfloop_edges(g))
    return nx.convert_node_labels_to_integers(g, ordering="sorted")


def _graph_from_adjacency(adj: Any) -> nx.Graph:
    arr = np.asarray(adj)
    if arr.ndim == 3:
        # Common one-hot edge tensor layout: n x n x edge_classes. Treat class 0
        # as no-edge and any non-zero class as an edge.
        arr = np.argmax(arr, axis=-1)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected a square adjacency matrix, got shape={arr.shape}.")
    arr = np.asarray(arr)
    if arr.dtype != bool:
        arr = arr != 0
    np.fill_diagonal(arr, False)
    arr = np.logical_or(arr, arr.T)
    return _canonical_graph(nx.from_numpy_array(arr.astype(np.int8)))


def _graph_from_digress_tuple(item: Any) -> nx.Graph | None:
    """Convert a DiGress sample tuple/list to NetworkX when possible.

    DiGress generic samples usually look like ``(node_types, edge_types)`` where
    edge_types is an n x n integer matrix and 0 denotes no-edge.
    """

    if not isinstance(item, (tuple, list)) or len(item) < 2:
        return None
    edge_types = item[1]
    try:
        if hasattr(edge_types, "detach"):
            edge_types = edge_types.detach().cpu().numpy()
        return _graph_from_adjacency(edge_types)
    except Exception:
        return None



ATOM_SYMBOL_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53}


def _parse_decoder(raw: str | None, *, dataset: str, qm9_with_h: bool) -> list[int]:
    """Return atomic numbers for DiGress node-class ids."""

    if raw:
        values = [token.strip() for token in str(raw).replace(";", ",").split(",") if token.strip()]
        decoder: list[int] = []
        for value in values:
            if value.isdigit():
                decoder.append(int(value))
            else:
                if value not in ATOM_SYMBOL_TO_Z:
                    raise ValueError(f"Unknown atom symbol {value!r} in --digress-atom-decoder.")
                decoder.append(int(ATOM_SYMBOL_TO_Z[value]))
        if not decoder:
            raise ValueError("--digress-atom-decoder produced an empty decoder.")
        return decoder

    if str(dataset).lower() == "qm9":
        return [1, 6, 7, 8, 9] if bool(qm9_with_h) else [6, 7, 8, 9]
    # Conservative fallback for molecular datasets with QM9-style outputs.
    return [6, 7, 8, 9]


def _load_digress_txt_blocks(path: Path) -> list[tuple[list[int], np.ndarray]]:
    """Parse DiGress generated_samples*.txt into (X, E) hard arrays."""

    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[tuple[list[int], np.ndarray]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("N="):
            i += 1
            continue
        try:
            n = int(line.split("=", 1)[1])
        except ValueError:
            i += 1
            continue
        i += 1

        atom_types: list[int] = []
        while i < len(lines) and lines[i].strip() != "X:":
            i += 1
        if i < len(lines) and lines[i].strip() == "X:":
            i += 1
            while i < len(lines):
                raw = lines[i].strip()
                if raw == "E:":
                    break
                if raw:
                    atom_types.extend(int(float(tok)) for tok in raw.split())
                    if len(atom_types) >= n:
                        # DiGress writes all node labels on one line; keep the
                        # loop flexible in case a local fork wraps them.
                        pass
                i += 1
        atom_types = atom_types[:n]

        while i < len(lines) and lines[i].strip() != "E:":
            i += 1
        if i >= len(lines):
            break
        i += 1
        rows: list[list[int]] = []
        for _ in range(n):
            if i >= len(lines):
                break
            row = [int(float(tok)) for tok in lines[i].strip().split()] if lines[i].strip() else []
            rows.append(row[:n])
            i += 1
        if len(atom_types) == n and len(rows) == n and all(len(row) == n for row in rows):
            blocks.append((atom_types, np.asarray(rows, dtype=np.int64)))
    if not blocks:
        raise ValueError(f"No DiGress graph blocks could be parsed from {path}.")
    return blocks


def _molecular_graph_from_arrays(
    atom_types: Sequence[Any],
    edge_types: Any,
    *,
    atom_decoder: Sequence[int],
) -> nx.Graph:
    atoms = [int(x.item() if hasattr(x, "item") else x) for x in list(atom_types)]
    edges = edge_types.detach().cpu().numpy() if hasattr(edge_types, "detach") else np.asarray(edge_types)
    if edges.ndim == 3:
        edges = np.argmax(edges, axis=-1)
    if edges.ndim != 2 or edges.shape[0] != edges.shape[1]:
        raise ValueError(f"Expected square edge-type matrix for molecular sample, got shape={edges.shape}.")
    n = min(len(atoms), int(edges.shape[0]))
    g = nx.Graph()
    for node in range(n):
        cls = int(atoms[node])
        if cls < 0 or cls >= len(atom_decoder):
            raise ValueError(
                f"Atom class id {cls} is outside decoder of length {len(atom_decoder)}. "
                "Pass --qm9-with-h or --digress-atom-decoder."
            )
        z = int(atom_decoder[cls])
        g.add_node(
            int(node),
            atomic_number=z,
            z=z,
            atom_type=z,
            node_type=z,
            node_label=f"atomic_number={z}",
        )
    for u in range(n):
        for v in range(u + 1, n):
            bond = int(edges[u, v])
            if bond <= 0:
                continue
            # DiGress QM9 uses 0=no edge, 1=single, 2=double, 3=triple, 4=aromatic.
            bond_order = float(BOND_TYPE_TO_ORDER.get(bond, 1.0))
            g.add_edge(
                int(u),
                int(v),
                edge_type=bond,
                bond_type=bond,
                edge_attr=[float(bond)],
                bond_order=bond_order,
                is_aromatic=bool(bond == 4),
            )
    return canonicalize_molecular_graph(g)


def _molecular_graph_from_digress_tuple(item: Any, *, atom_decoder: Sequence[int]) -> nx.Graph | None:
    if not isinstance(item, (tuple, list)) or len(item) < 2:
        return None
    try:
        atom_types = item[0]
        edge_types = item[1]
        if hasattr(atom_types, "detach"):
            atom_types = atom_types.detach().cpu().reshape(-1).tolist()
        if isinstance(atom_types, np.ndarray):
            atom_types = atom_types.reshape(-1).tolist()
        return _molecular_graph_from_arrays(atom_types, edge_types, atom_decoder=atom_decoder)
    except Exception:
        return None


def _payload_to_molecular_graphs(payload: Any, *, atom_decoder: Sequence[int]) -> list[nx.Graph]:
    if isinstance(payload, nx.Graph):
        return [canonicalize_molecular_graph(payload)]
    if isinstance(payload, dict):
        for key in ("graphs", "samples", "generated_graphs", "networkx_graphs", "molecules"):
            if key in payload:
                return _payload_to_molecular_graphs(payload[key], atom_decoder=atom_decoder)
        graphs: list[nx.Graph] = []
        for value in payload.values():
            try:
                graphs.extend(_payload_to_molecular_graphs(value, atom_decoder=atom_decoder))
            except Exception:
                continue
        if graphs:
            return graphs
    if isinstance(payload, (list, tuple)):
        tuple_graph = _molecular_graph_from_digress_tuple(payload, atom_decoder=atom_decoder)
        if tuple_graph is not None:
            return [tuple_graph]
        graphs: list[nx.Graph] = []
        for item in payload:
            if isinstance(item, nx.Graph):
                graphs.append(canonicalize_molecular_graph(item))
                continue
            tuple_graph = _molecular_graph_from_digress_tuple(item, atom_decoder=atom_decoder)
            if tuple_graph is not None:
                graphs.append(tuple_graph)
                continue
            try:
                graphs.extend(_payload_to_molecular_graphs(item, atom_decoder=atom_decoder))
            except Exception:
                continue
        if graphs:
            return graphs
    raise TypeError(f"Could not convert payload of type {type(payload)} to molecular NetworkX graphs.")


def load_digress_molecular_graphs(
    path: str | Path,
    *,
    dataset: str,
    qm9_with_h: bool = False,
    atom_decoder: str | None = None,
) -> list[nx.Graph]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    decoder = _parse_decoder(atom_decoder, dataset=dataset, qm9_with_h=qm9_with_h)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [
            _molecular_graph_from_arrays(atoms, edges, atom_decoder=decoder)
            for atoms, edges in _load_digress_txt_blocks(path)
        ]
    if suffix == ".pkl" or suffix == ".pickle":
        return _payload_to_molecular_graphs(load_pickle(path), atom_decoder=decoder)
    if suffix == ".npy":
        return _payload_to_molecular_graphs(np.load(path, allow_pickle=True), atom_decoder=decoder)
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        graphs: list[nx.Graph] = []
        for key in sorted(data.files):
            graphs.extend(_payload_to_molecular_graphs(data[key], atom_decoder=decoder))
        return graphs
    if suffix in {".pt", ".pth"}:
        import torch
        return _payload_to_molecular_graphs(torch.load(path, map_location="cpu", weights_only=False), atom_decoder=decoder)
    raise ValueError(f"Unsupported molecular sample extension: {path.suffix}.")

def _load_digress_txt(path: Path) -> list[nx.Graph]:
    """Parse DiGress generated_samples*.txt files as featureless graphs."""

    return [_graph_from_adjacency(edges) for _, edges in _load_digress_txt_blocks(path)]


def _payload_to_graphs(payload: Any) -> list[nx.Graph]:
    if isinstance(payload, nx.Graph):
        return [_canonical_graph(payload)]

    if isinstance(payload, np.ndarray):
        if payload.ndim == 2:
            return [_graph_from_adjacency(payload)]
        if payload.ndim == 3:
            # Either a stack of adjacency matrices or a single one-hot adjacency.
            if payload.shape[0] == payload.shape[1]:
                return [_graph_from_adjacency(payload)]
            return [_graph_from_adjacency(payload[i]) for i in range(payload.shape[0])]
        if payload.ndim == 4:
            return [_graph_from_adjacency(payload[i]) for i in range(payload.shape[0])]

    if isinstance(payload, dict):
        for key in ("graphs", "samples", "generated_graphs", "networkx_graphs", "adjs", "adjacency"):
            if key in payload:
                return _payload_to_graphs(payload[key])
        graphs: list[nx.Graph] = []
        for value in payload.values():
            try:
                graphs.extend(_payload_to_graphs(value))
            except Exception:
                continue
        if graphs:
            return graphs

    if isinstance(payload, (list, tuple)):
        graphs: list[nx.Graph] = []
        for item in payload:
            if isinstance(item, nx.Graph):
                graphs.append(_canonical_graph(item))
                continue
            tuple_graph = _graph_from_digress_tuple(item)
            if tuple_graph is not None:
                graphs.append(tuple_graph)
                continue
            try:
                graphs.extend(_payload_to_graphs(item))
            except Exception:
                continue
        if graphs:
            return graphs

    raise TypeError(f"Could not convert payload of type {type(payload)} to NetworkX graphs.")


def load_digress_graphs(path: str | Path) -> list[nx.Graph]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _load_digress_txt(path)
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        graphs: list[nx.Graph] = []
        for key in sorted(data.files):
            graphs.extend(_payload_to_graphs(data[key]))
        return graphs
    if suffix == ".npy":
        return _payload_to_graphs(np.load(path, allow_pickle=True))
    if suffix in {".pkl", ".pickle"}:
        return _payload_to_graphs(load_pickle(path))
    if suffix in {".pt", ".pth"}:
        import torch

        return _payload_to_graphs(torch.load(path, map_location="cpu", weights_only=False))
    raise ValueError(f"Unsupported sample file extension: {path.suffix}. Use .txt, .npz, .npy, .pkl, or .pt.")


def default_digress_sample_path() -> Path | None:
    candidates = [
        ROOT / "baselines" / "DiGress" / "generated_adjs.npz",
        ROOT / "baselines" / "DiGress" / "generated_samples1.txt",
        ROOT / "generated_adjs.npz",
        ROOT / "generated_samples1.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


# -----------------------------------------------------------------------------
# Energy model for GraphER-as-optimizer
# -----------------------------------------------------------------------------


def _finite(value: float, default: float = 0.0) -> float:
    value = float(value)
    return value if math.isfinite(value) else float(default)


def _safe_average_square_clustering(graph: nx.Graph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    try:
        values = list(nx.square_clustering(graph).values())
        return _finite(float(np.mean(values)) if values else 0.0)
    except Exception:
        return 0.0


def _safe_assortativity(graph: nx.Graph) -> float:
    if graph.number_of_edges() == 0:
        return 0.0
    try:
        return _finite(nx.degree_assortativity_coefficient(graph))
    except Exception:
        return 0.0


def _safe_modularity(graph: nx.Graph) -> float:
    if graph.number_of_edges() == 0 or graph.number_of_nodes() == 0:
        return 0.0
    try:
        communities = list(nx.community.greedy_modularity_communities(graph))
        if not communities:
            return 0.0
        return _finite(nx.community.modularity(graph, communities))
    except Exception:
        return 0.0


def optimizer_feature_vector(
    graph: nx.Graph,
    *,
    include_spectral: bool = False,
    spectral_k: int = 8,
    include_modularity: bool = False,
) -> np.ndarray:
    """Small permutation-invariant topology vector used by the refiner.

    Degree and edge count are intentionally omitted because GraphER rewiring
    preserves them. The energy therefore focuses on higher-order topology.
    """

    g = _canonical_graph(graph)
    n = max(int(g.number_of_nodes()), 1)
    m = int(g.number_of_edges())
    triangles = sum(nx.triangles(g).values()) / 3.0 if n > 0 else 0.0
    wedges = sum(float(d * (d - 1) / 2.0) for _, d in g.degree())
    max_triangles = max(float(n * (n - 1) * (n - 2) / 6.0), 1.0)
    max_wedges = max(float(n * (n - 1) * (n - 2) / 2.0), 1.0)
    density = nx.density(g) if n > 1 else 0.0
    values: list[float] = [
        nx.average_clustering(g) if n > 0 else 0.0,
        nx.transitivity(g) if m > 0 else 0.0,
        triangles / max_triangles,
        wedges / max_wedges,
        _safe_average_square_clustering(g),
        _safe_assortativity(g),
        density,
    ]
    if include_modularity:
        values.append(_safe_modularity(g))
    if include_spectral:
        # Include a small number of sorted normalized-Laplacian eigenvalues. This
        # is more expensive, so it is disabled by default.
        try:
            adjacency = nx.to_numpy_array(g, dtype=np.float64)
            deg = adjacency.sum(axis=1)
            inv_sqrt = np.zeros_like(deg)
            inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
            lap = np.eye(adjacency.shape[0]) - np.diag(inv_sqrt) @ adjacency @ np.diag(inv_sqrt)
            eigs = np.sort(np.linalg.eigvalsh(lap))
        except Exception:
            eigs = np.zeros(0, dtype=np.float64)
        padded = np.zeros(int(spectral_k), dtype=np.float64)
        width = min(int(spectral_k), eigs.size)
        if width:
            padded[:width] = eigs[:width]
        values.extend(padded.tolist())
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


class FeatureEnergy:
    def __init__(
        self,
        train_graphs: Sequence[nx.Graph],
        *,
        target_mode: str,
        degree_match_weight: float,
        include_spectral: bool,
        spectral_k: int,
        include_modularity: bool,
        eps: float = 1e-6,
    ) -> None:
        self.target_mode = str(target_mode)
        self.degree_match_weight = float(degree_match_weight)
        self.include_spectral = bool(include_spectral)
        self.spectral_k = int(spectral_k)
        self.include_modularity = bool(include_modularity)
        self.train_graphs = [_canonical_graph(g) for g in train_graphs]
        self.train_features = np.vstack(
            [
                optimizer_feature_vector(
                    g,
                    include_spectral=self.include_spectral,
                    spectral_k=self.spectral_k,
                    include_modularity=self.include_modularity,
                )
                for g in self.train_graphs
            ]
        )
        self.mean = np.mean(self.train_features, axis=0)
        self.std = np.std(self.train_features, axis=0)
        self.std = np.where(self.std < eps, 1.0, self.std)
        self.max_degree = max([1] + [max((int(d) for _, d in g.degree()), default=0) for g in self.train_graphs])
        self.train_degree_hists = np.vstack([self._degree_hist(g) for g in self.train_graphs])
        self.train_nodes = np.asarray([g.number_of_nodes() for g in self.train_graphs], dtype=np.float64)

    def _degree_hist(self, graph: nx.Graph) -> np.ndarray:
        vals = [int(d) for _, d in graph.degree() if 0 <= int(d) <= self.max_degree]
        hist = np.bincount(vals, minlength=self.max_degree + 1).astype(np.float64)
        total = hist.sum()
        return hist / total if total > 0 else hist

    def target_for(self, graph: nx.Graph) -> tuple[np.ndarray, int | None]:
        if self.target_mode == "mean":
            return self.mean, None
        hist = self._degree_hist(graph)
        node_term = np.abs(self.train_nodes - float(graph.number_of_nodes())) / max(float(max(self.train_nodes.max(), 1.0)), 1.0)
        deg_term = np.sum(np.abs(self.train_degree_hists - hist.reshape(1, -1)), axis=1)
        score = node_term + self.degree_match_weight * deg_term
        idx = int(np.argmin(score))
        return self.train_features[idx], idx

    def graph_features(self, graph: nx.Graph) -> np.ndarray:
        return optimizer_feature_vector(
            graph,
            include_spectral=self.include_spectral,
            spectral_k=self.spectral_k,
            include_modularity=self.include_modularity,
        )

    def energy(self, graph: nx.Graph, target: np.ndarray) -> float:
        features = self.graph_features(graph)
        diff = (features - target) / self.std
        return _finite(float(np.dot(diff, diff)))




def _histogram_from_values(values: Sequence[int], vocabulary: Sequence[int]) -> np.ndarray:
    counts = {int(v): 0.0 for v in vocabulary}
    for raw in values:
        value = int(raw)
        if value in counts:
            counts[value] += 1.0
    arr = np.asarray([counts[int(v)] for v in vocabulary], dtype=np.float64)
    total = float(arr.sum())
    return arr / total if total > 0 else arr


def molecular_optimizer_feature_vector(
    graph: nx.Graph,
    *,
    prior: EmpiricalBondTypePrior,
    include_spectral: bool = False,
    spectral_k: int = 8,
    include_modularity: bool = False,
) -> np.ndarray:
    """Permutation-invariant molecular feature vector for energy-guided refinement.

    Atom labels are preserved by GraphER, but we include atom histograms for target
    matching. Bond histograms and valence summaries can change through typed swaps
    and therefore give the optimizer a chemistry-aware signal.
    """

    g = canonicalize_molecular_graph(graph)
    base = optimizer_feature_vector(
        g,
        include_spectral=include_spectral,
        spectral_k=spectral_k,
        include_modularity=include_modularity,
    )
    atom_hist = _histogram_from_values(
        [node_type_value(g, int(node)) for node in g.nodes()],
        prior.node_types,
    )
    bond_values = []
    bond_orders = []
    for u, v in g.edges():
        bond = edge_type_value(g, int(u), int(v))
        bond_values.append(int(bond))
        bond_orders.append(float(prior.bond_order(int(bond))))
    bond_hist = _histogram_from_values(bond_values, prior.edge_types)

    valence_ratios: list[float] = []
    unused_ratios: list[float] = []
    over_limit = 0
    for node in g.nodes():
        atom_type = node_type_value(g, int(node))
        limit = max(float(prior.max_valence(atom_type)), 1e-12)
        total = 0.0
        for nbr in g.neighbors(int(node)):
            total += float(prior.bond_order(edge_type_value(g, int(node), int(nbr))))
        valence_ratios.append(float(total / limit))
        unused_ratios.append(float(max(limit - total, 0.0) / limit))
        over_limit += int(total > limit + 1e-6)
    valence_arr = np.asarray(valence_ratios or [0.0], dtype=np.float64)
    unused_arr = np.asarray(unused_ratios or [0.0], dtype=np.float64)
    bond_order_arr = np.asarray(bond_orders or [0.0], dtype=np.float64)
    chem = np.asarray(
        [
            float(np.mean(valence_arr)),
            float(np.std(valence_arr)),
            float(np.max(valence_arr)),
            float(np.mean(unused_arr)),
            float(over_limit / max(g.number_of_nodes(), 1)),
            float(np.mean(bond_order_arr)),
            float(np.std(bond_order_arr)),
        ],
        dtype=np.float64,
    )
    return np.nan_to_num(np.concatenate([base, atom_hist, bond_hist, chem]), nan=0.0, posinf=0.0, neginf=0.0)


class MolecularFeatureEnergy:
    def __init__(
        self,
        train_graphs: Sequence[nx.Graph],
        *,
        target_mode: str,
        degree_match_weight: float,
        atom_match_weight: float,
        include_spectral: bool,
        spectral_k: int,
        include_modularity: bool,
        prior: EmpiricalBondTypePrior,
        eps: float = 1e-6,
    ) -> None:
        self.target_mode = str(target_mode)
        self.degree_match_weight = float(degree_match_weight)
        self.atom_match_weight = float(atom_match_weight)
        self.include_spectral = bool(include_spectral)
        self.spectral_k = int(spectral_k)
        self.include_modularity = bool(include_modularity)
        self.prior = prior
        self.train_graphs = [canonicalize_molecular_graph(g) for g in train_graphs]
        self.train_features = np.vstack([self.graph_features(g) for g in self.train_graphs])
        self.mean = np.mean(self.train_features, axis=0)
        self.std = np.std(self.train_features, axis=0)
        self.std = np.where(self.std < eps, 1.0, self.std)
        self.max_degree = max([1] + [max((int(d) for _, d in g.degree()), default=0) for g in self.train_graphs])
        self.train_degree_hists = np.vstack([self._degree_hist(g) for g in self.train_graphs])
        self.train_atom_hists = np.vstack([self._atom_hist(g) for g in self.train_graphs])
        self.train_nodes = np.asarray([g.number_of_nodes() for g in self.train_graphs], dtype=np.float64)

    def _degree_hist(self, graph: nx.Graph) -> np.ndarray:
        vals = [int(d) for _, d in graph.degree() if 0 <= int(d) <= self.max_degree]
        hist = np.bincount(vals, minlength=self.max_degree + 1).astype(np.float64)
        total = hist.sum()
        return hist / total if total > 0 else hist

    def _atom_hist(self, graph: nx.Graph) -> np.ndarray:
        return _histogram_from_values(
            [node_type_value(graph, int(node)) for node in graph.nodes()],
            self.prior.node_types,
        )

    def graph_features(self, graph: nx.Graph) -> np.ndarray:
        return molecular_optimizer_feature_vector(
            graph,
            prior=self.prior,
            include_spectral=self.include_spectral,
            spectral_k=self.spectral_k,
            include_modularity=self.include_modularity,
        )

    def target_for(self, graph: nx.Graph) -> tuple[np.ndarray, int | None]:
        if self.target_mode == "mean":
            return self.mean, None
        g = canonicalize_molecular_graph(graph)
        hist = self._degree_hist(g)
        atom_hist = self._atom_hist(g)
        node_term = np.abs(self.train_nodes - float(g.number_of_nodes())) / max(float(max(self.train_nodes.max(), 1.0)), 1.0)
        deg_term = np.sum(np.abs(self.train_degree_hists - hist.reshape(1, -1)), axis=1)
        atom_term = np.sum(np.abs(self.train_atom_hists - atom_hist.reshape(1, -1)), axis=1)
        score = node_term + self.degree_match_weight * deg_term + self.atom_match_weight * atom_term
        idx = int(np.argmin(score))
        return self.train_features[idx], idx

    def energy(self, graph: nx.Graph, target: np.ndarray) -> float:
        features = self.graph_features(graph)
        diff = (features - target) / self.std
        return _finite(float(np.dot(diff, diff)))

# -----------------------------------------------------------------------------
# Rewiring refiners
# -----------------------------------------------------------------------------


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def refine_graph_energy(
    graph: nx.Graph,
    *,
    energy_model: FeatureEnergy,
    target: np.ndarray,
    steps: int,
    candidate_budget: int,
    k_hop: int | None,
    ensure_connected: bool,
    seed: int,
    min_improvement: float,
) -> tuple[nx.Graph, dict[str, Any]]:
    rng_np = _rng(seed)
    rng_py = __import__("random").Random(int(seed))
    g = _canonical_graph(graph)
    initial_degree_sequence = degree_sequence(g)
    current_energy = energy_model.energy(g, target)
    accepted = 0
    evaluated = 0
    rejected_invalid = 0
    initial_rdkit_valid = _rdkit_valid_molecular_graph(g, isomeric_smiles=isomeric_smiles)
    history = [current_energy]

    for _step in range(int(steps)):
        actions = enumerate_rewire_actions(
            g,
            ensure_connected=ensure_connected,
            k_hop=k_hop,
            max_candidates=int(candidate_budget),
            rng=rng_py,
            shuffle=True,
        )
        if not actions:
            break
        best_action: RewireAction | None = None
        best_graph: nx.Graph | None = None
        best_energy = current_energy
        # Break ties randomly so repeated valid actions are not ordered by NetworkX.
        order = rng_np.permutation(len(actions))
        for idx in order:
            action = actions[int(idx)]
            out = rewire_action(g, action, ensure_connected=ensure_connected)
            if out is None:
                continue
            evaluated += 1
            candidate_graph = _canonical_graph(out[0])
            candidate_energy = energy_model.energy(candidate_graph, target)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_action = action
                best_graph = candidate_graph
        if best_action is None or best_graph is None or (current_energy - best_energy) <= float(min_improvement):
            break
        g = best_graph
        current_energy = best_energy
        accepted += 1
        history.append(current_energy)

    return g, {
        "accepted_steps": int(accepted),
        "evaluated_candidates": int(evaluated),
        "energy_initial": float(history[0]),
        "energy_final": float(history[-1]),
        "energy_delta": float(history[0] - history[-1]),
        "degree_preserved": degree_sequence(g) == initial_degree_sequence,
        "history": [float(v) for v in history],
    }


def random_rewire_graph(
    graph: nx.Graph,
    *,
    steps: int,
    candidate_budget: int,
    k_hop: int | None,
    ensure_connected: bool,
    seed: int,
) -> tuple[nx.Graph, dict[str, Any]]:
    rng_py = __import__("random").Random(int(seed))
    g = _canonical_graph(graph)
    initial_degree_sequence = degree_sequence(g)
    accepted = 0
    for _step in range(int(steps)):
        actions = enumerate_rewire_actions(
            g,
            ensure_connected=ensure_connected,
            k_hop=k_hop,
            max_candidates=int(candidate_budget),
            rng=rng_py,
            shuffle=True,
        )
        if not actions:
            break
        action = rng_py.choice(actions)
        out = rewire_action(g, action, ensure_connected=ensure_connected)
        if out is None:
            break
        g = _canonical_graph(out[0])
        accepted += 1
    return g, {
        "accepted_steps": int(accepted),
        "degree_preserved": degree_sequence(g) == initial_degree_sequence,
    }






def _rdkit_valid_molecular_graph(graph: nx.Graph, *, isomeric_smiles: bool = False) -> bool:
    """Return True when the hard molecular graph sanitizes in RDKit.

    Valence checks alone are not enough for QM9 because arbitrary rewiring can
    create aromaticity/kekulization failures. This helper is intentionally used
    only in conservative molecular-refinement mode because it is slower than the
    structural validators.
    """

    try:
        return bool(direct_molecular_conversions([graph], isomeric_smiles=bool(isomeric_smiles))[0].valid)
    except Exception:
        return False

def refine_molecular_graph_energy(
    graph: nx.Graph,
    *,
    energy_model: MolecularFeatureEnergy,
    target: np.ndarray,
    prior: EmpiricalBondTypePrior,
    steps: int,
    candidate_budget: int,
    k_hop: int | None,
    ensure_connected: bool,
    seed: int,
    min_improvement: float,
    proposals_per_edge: int,
    proposal_mode: str,
    allow_global_backoff: bool,
    reject_unseen_endpoint_pairs: bool,
    valence_tolerance: float,
    require_valid_candidates: bool = False,
    isomeric_smiles: bool = False,
) -> tuple[nx.Graph, dict[str, Any]]:
    rng_np = _rng(seed)
    rng_py = __import__("random").Random(int(seed))
    g = canonicalize_molecular_graph(graph)
    initial_degree_sequence = degree_sequence(g)
    initial_atom_types = [node_type_value(g, int(node)) for node in sorted(g.nodes())]
    current_energy = energy_model.energy(g, target)
    initial_rdkit_valid = _rdkit_valid_molecular_graph(g, isomeric_smiles=isomeric_smiles)
    rejected_invalid = 0
    accepted = 0
    evaluated = 0
    history = [current_energy]

    for _step in range(int(steps)):
        actions = enumerate_molecular_rewire_actions(
            g,
            prior,
            rng=rng_py,
            ensure_connected=ensure_connected,
            k_hop=k_hop,
            max_candidates=int(candidate_budget),
            proposals_per_edge=int(proposals_per_edge),
            proposal_mode=proposal_mode,
            allow_global_backoff=allow_global_backoff,
            reject_unseen_endpoint_pairs=reject_unseen_endpoint_pairs,
            valence_tolerance=float(valence_tolerance),
            shuffle=True,
        )
        if not actions:
            break
        best_action: MolecularRewireAction | None = None
        best_graph: nx.Graph | None = None
        best_energy = current_energy
        for idx in rng_np.permutation(len(actions)):
            action = actions[int(idx)]
            candidate_graph = apply_molecular_rewire(
                g,
                action,
                prior,
                ensure_connected=ensure_connected,
                valence_tolerance=float(valence_tolerance),
            )
            if candidate_graph is None:
                continue
            evaluated += 1
            candidate_graph = canonicalize_molecular_graph(candidate_graph)
            if require_valid_candidates and not _rdkit_valid_molecular_graph(
                candidate_graph, isomeric_smiles=isomeric_smiles
            ):
                rejected_invalid += 1
                continue
            candidate_energy = energy_model.energy(candidate_graph, target)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_action = action
                best_graph = candidate_graph
        if best_action is None or best_graph is None or (current_energy - best_energy) <= float(min_improvement):
            break
        g = best_graph
        current_energy = best_energy
        accepted += 1
        history.append(current_energy)

    final_atom_types = [node_type_value(g, int(node)) for node in sorted(g.nodes())]
    return g, {
        "accepted_steps": int(accepted),
        "evaluated_candidates": int(evaluated),
        "energy_initial": float(history[0]),
        "energy_final": float(history[-1]),
        "energy_delta": float(history[0] - history[-1]),
        "degree_preserved": degree_sequence(g) == initial_degree_sequence,
        "node_types_preserved": final_atom_types == initial_atom_types,
        "rdkit_valid_initial": bool(initial_rdkit_valid),
        "rdkit_valid_final": bool(_rdkit_valid_molecular_graph(g, isomeric_smiles=isomeric_smiles)),
        "rejected_invalid_candidates": int(rejected_invalid),
        "history": [float(v) for v in history],
    }


def random_molecular_rewire_graph(
    graph: nx.Graph,
    *,
    prior: EmpiricalBondTypePrior,
    steps: int,
    candidate_budget: int,
    k_hop: int | None,
    ensure_connected: bool,
    seed: int,
    proposals_per_edge: int,
    proposal_mode: str,
    allow_global_backoff: bool,
    reject_unseen_endpoint_pairs: bool,
    valence_tolerance: float,
    require_valid_candidates: bool = False,
    isomeric_smiles: bool = False,
) -> tuple[nx.Graph, dict[str, Any]]:
    rng_py = __import__("random").Random(int(seed))
    g = canonicalize_molecular_graph(graph)
    initial_degree_sequence = degree_sequence(g)
    initial_atom_types = [node_type_value(g, int(node)) for node in sorted(g.nodes())]
    initial_rdkit_valid = _rdkit_valid_molecular_graph(g, isomeric_smiles=isomeric_smiles)
    rejected_invalid = 0
    accepted = 0
    for _step in range(int(steps)):
        actions = enumerate_molecular_rewire_actions(
            g,
            prior,
            rng=rng_py,
            ensure_connected=ensure_connected,
            k_hop=k_hop,
            max_candidates=int(candidate_budget),
            proposals_per_edge=int(proposals_per_edge),
            proposal_mode=proposal_mode,
            allow_global_backoff=allow_global_backoff,
            reject_unseen_endpoint_pairs=reject_unseen_endpoint_pairs,
            valence_tolerance=float(valence_tolerance),
            shuffle=True,
        )
        if not actions:
            break
        action = rng_py.choice(actions)
        candidate = apply_molecular_rewire(
            g,
            action,
            prior,
            ensure_connected=ensure_connected,
            valence_tolerance=float(valence_tolerance),
        )
        if candidate is None:
            break
        candidate = canonicalize_molecular_graph(candidate)
        if require_valid_candidates and not _rdkit_valid_molecular_graph(
            candidate, isomeric_smiles=isomeric_smiles
        ):
            rejected_invalid += 1
            continue
        g = candidate
        accepted += 1
    final_atom_types = [node_type_value(g, int(node)) for node in sorted(g.nodes())]
    return g, {
        "accepted_steps": int(accepted),
        "degree_preserved": degree_sequence(g) == initial_degree_sequence,
        "node_types_preserved": final_atom_types == initial_atom_types,
        "rdkit_valid_initial": bool(initial_rdkit_valid),
        "rdkit_valid_final": bool(_rdkit_valid_molecular_graph(g, isomeric_smiles=isomeric_smiles)),
        "rejected_invalid_candidates": int(rejected_invalid),
    }

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def _metric_mmd(reference_graphs: Sequence[nx.Graph], graphs: Sequence[nx.Graph], *, degree_bins: int, clustering_bins: int, spectral_bins: int, sigma: float) -> dict[str, float]:
    max_degree = max(
        [int(degree_bins) - 1, 0]
        + [max((int(d) for _, d in g.degree()), default=0) for g in reference_graphs]
        + [max((int(d) for _, d in g.degree()), default=0) for g in graphs]
    )
    descriptor_specs = {
        "degree_mmd": (lambda g: degree_histogram(g, max_degree=max_degree), "emd"),
        "clustering_mmd": (lambda g: clustering_histogram(g, bins=clustering_bins), "emd"),
        "spectral_mmd": (lambda g: spectral_histogram(g, bins=spectral_bins), "emd"),
        "motif_proxy_mmd": (motif_proxy_vector, "rbf"),
        "structural_summary_mmd": (structural_summary, "rbf"),
    }
    results: dict[str, float] = {}
    for name, (fn, kind) in descriptor_specs.items():
        ref_desc = descriptor_matrix(reference_graphs, fn)
        gen_desc = descriptor_matrix(graphs, fn)
        results[name] = (
            mmd_gaussian_emd(ref_desc, gen_desc, sigma=sigma)
            if kind == "emd"
            else mmd_rbf(ref_desc, gen_desc, sigma=None)
        )
    return results


def _degree_preservation_rate(before: Sequence[nx.Graph], after: Sequence[nx.Graph]) -> float:
    if not before:
        return 0.0
    return float(np.mean([degree_sequence(g0) == degree_sequence(g1) for g0, g1 in zip(before, after)]))


def evaluate_sets(
    *,
    reference_graphs: Sequence[nx.Graph],
    train_graphs: Sequence[nx.Graph],
    base_graphs: Sequence[nx.Graph],
    random_graphs: Sequence[nx.Graph],
    refined_graphs: Sequence[nx.Graph],
    dataset: str,
    degree_bins: int,
    clustering_bins: int,
    spectral_bins: int,
    sigma: float,
) -> dict[str, Any]:
    sets = {
        "digress": list(base_graphs),
        "digress_random_rewire": list(random_graphs),
        "digress_grapher_optimizer": list(refined_graphs),
    }
    out: dict[str, Any] = {}
    for name, graphs in sets.items():
        payload: dict[str, Any] = {}
        payload.update(_metric_mmd(reference_graphs, graphs, degree_bins=degree_bins, clustering_bins=clustering_bins, spectral_bins=spectral_bins, sigma=sigma))
        payload.update(quality_metrics(graphs, reference_graphs=train_graphs, dataset=dataset))
        if name != "digress":
            payload["degree_preservation_from_digress_rate"] = _degree_preservation_rate(base_graphs, graphs)
        out[name] = payload
    return out




def _molecular_conversion_error_counts(conversions: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for conversion in conversions:
        if getattr(conversion, "valid", False):
            continue
        error = getattr(conversion, "error", None) or "unknown"
        key = str(error).split(":", 1)[0]
        counts[key] = counts.get(key, 0) + 1
    return counts


def _evaluate_one_molecular_set(
    *,
    graphs: Sequence[nx.Graph],
    reference_graphs: Sequence[nx.Graph],
    train_smiles: Sequence[str],
    reference_valid_smiles: Sequence[str],
    reference_valid_graphs: Sequence[nx.Graph],
    nspdk_backend: str,
    nspdk_complexity: int,
    skip_nspdk: bool,
    skip_fcd: bool,
    require_fcd: bool,
    fcd_device: str,
    fcd_n_jobs: int,
    fcd_batch_size: int,
    keep_explicit_hydrogens: bool,
    isomeric_smiles: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    conversions = direct_molecular_conversions(graphs, isomeric_smiles=isomeric_smiles)
    heavy_atoms_only = not bool(keep_explicit_hydrogens)
    gen_valid_smiles = valid_smiles(
        conversions,
        remove_explicit_hydrogens=heavy_atoms_only,
        isomeric_smiles=isomeric_smiles,
    )
    gen_valid_graphs = [g for g, c in zip(graphs, conversions) if c.valid]
    payload: dict[str, Any] = {}
    payload["validity_without_correction"] = validity_without_correction(conversions)
    payload["validity_rate"] = payload["validity_without_correction"]
    payload["num_valid_generated_molecules"] = int(sum(bool(c.valid) for c in conversions))
    payload["num_invalid_generated_molecules"] = int(len(graphs) - payload["num_valid_generated_molecules"])
    payload["uniqueness_rate"] = molecular_uniqueness(gen_valid_smiles)
    payload["novelty_rate"] = molecular_novelty(gen_valid_smiles, train_smiles)
    payload.update(quality_metrics(graphs, reference_graphs=[], dataset="qm9"))

    protocol: dict[str, Any] = {
        "num_valid_smiles": int(len(gen_valid_smiles)),
        "conversion_error_counts": _molecular_conversion_error_counts(conversions),
    }
    if skip_nspdk:
        payload["nspdk_mmd"] = None
        payload["nspdk_mmd_valid_only"] = None
        protocol["nspdk"] = {"status": "skipped_by_user"}
    else:
        nspdk_all, nspdk_protocol = compute_nspdk_mmd(
            reference_graphs,
            graphs,
            complexity=max(int(nspdk_complexity), 1),
            backend=nspdk_backend,
            heavy_atoms_only=heavy_atoms_only,
        )
        nspdk_valid, nspdk_valid_protocol = compute_nspdk_mmd(
            reference_valid_graphs,
            gen_valid_graphs,
            complexity=max(int(nspdk_complexity), 1),
            backend=nspdk_backend,
            heavy_atoms_only=heavy_atoms_only,
        )
        payload["nspdk_mmd"] = float(nspdk_all) if np.isfinite(nspdk_all) else None
        payload["nspdk_mmd_valid_only"] = float(nspdk_valid) if np.isfinite(nspdk_valid) else None
        protocol["nspdk"] = nspdk_protocol
        protocol["nspdk_valid_only"] = nspdk_valid_protocol

    if skip_fcd:
        payload["fcd"] = None
        protocol["fcd"] = {"status": "skipped_by_user"}
    else:
        fcd_score, fcd_status = compute_fcd(
            reference_valid_smiles,
            gen_valid_smiles,
            device=fcd_device,
            n_jobs=int(fcd_n_jobs),
            batch_size=int(fcd_batch_size),
        )
        payload["fcd"] = fcd_score
        protocol["fcd"] = fcd_status
        if require_fcd and fcd_score is None:
            raise RuntimeError("FCD was required but could not be computed: " + str(fcd_status))
    payload["fcd_num_valid_generated_molecules"] = int(len(gen_valid_smiles))
    payload["fcd_num_generated_molecules"] = int(len(graphs))
    return payload, protocol


def evaluate_molecular_sets(
    *,
    reference_graphs: Sequence[nx.Graph],
    train_graphs: Sequence[nx.Graph],
    base_graphs: Sequence[nx.Graph],
    random_graphs: Sequence[nx.Graph],
    refined_graphs: Sequence[nx.Graph],
    nspdk_backend: str,
    nspdk_complexity: int,
    skip_nspdk: bool,
    skip_fcd: bool,
    require_fcd: bool,
    fcd_device: str,
    fcd_n_jobs: int,
    fcd_batch_size: int,
    keep_explicit_hydrogens: bool,
    isomeric_smiles: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_conversions = direct_molecular_conversions(reference_graphs, isomeric_smiles=isomeric_smiles)
    train_conversions = direct_molecular_conversions(train_graphs, isomeric_smiles=isomeric_smiles)
    heavy_atoms_only = not bool(keep_explicit_hydrogens)
    reference_valid_smiles = valid_smiles(
        reference_conversions,
        remove_explicit_hydrogens=heavy_atoms_only,
        isomeric_smiles=isomeric_smiles,
    )
    train_smiles = valid_smiles(
        train_conversions,
        remove_explicit_hydrogens=heavy_atoms_only,
        isomeric_smiles=isomeric_smiles,
    )
    reference_valid_graphs = [g for g, c in zip(reference_graphs, reference_conversions) if c.valid]
    sets = {
        "digress": list(base_graphs),
        "digress_random_rewire": list(random_graphs),
        "digress_grapher_optimizer": list(refined_graphs),
    }
    out: dict[str, Any] = {}
    protocols: dict[str, Any] = {
        "reference_validity_without_correction": validity_without_correction(reference_conversions),
        "num_valid_reference_smiles": int(len(reference_valid_smiles)),
        "num_valid_train_smiles": int(len(train_smiles)),
        "keep_explicit_hydrogens": bool(keep_explicit_hydrogens),
        "isomeric_smiles": bool(isomeric_smiles),
    }
    for name, graphs in sets.items():
        payload, protocol = _evaluate_one_molecular_set(
            graphs=graphs,
            reference_graphs=reference_graphs,
            train_smiles=train_smiles,
            reference_valid_smiles=reference_valid_smiles,
            reference_valid_graphs=reference_valid_graphs,
            nspdk_backend=nspdk_backend,
            nspdk_complexity=nspdk_complexity,
            skip_nspdk=skip_nspdk,
            skip_fcd=skip_fcd,
            require_fcd=require_fcd,
            fcd_device=fcd_device,
            fcd_n_jobs=fcd_n_jobs,
            fcd_batch_size=fcd_batch_size,
            keep_explicit_hydrogens=keep_explicit_hydrogens,
            isomeric_smiles=isomeric_smiles,
        )
        if name != "digress":
            payload["degree_preservation_from_digress_rate"] = _degree_preservation_rate(base_graphs, graphs)
            payload["node_type_preservation_from_digress_rate"] = _node_type_preservation_rate(base_graphs, graphs)
        out[name] = payload
        protocols[name] = protocol
    return out, protocols


def _node_type_preservation_rate(before: Sequence[nx.Graph], after: Sequence[nx.Graph]) -> float:
    if not before:
        return 0.0
    flags = []
    for g0, g1 in zip(before, after):
        try:
            values0 = [node_type_value(g0, int(node)) for node in sorted(g0.nodes())]
            values1 = [node_type_value(g1, int(node)) for node in sorted(g1.nodes())]
            flags.append(values0 == values1)
        except Exception:
            flags.append(False)
    return float(np.mean(flags)) if flags else 0.0

def _subsample(items: Sequence[Any], max_items: int | None, seed: int) -> list[Any]:
    items = list(items)
    if max_items is None or int(max_items) <= 0 or len(items) <= int(max_items):
        return items
    rng = _rng(seed)
    idx = rng.choice(len(items), size=int(max_items), replace=False)
    return [items[int(i)] for i in idx]


def _filter_graphs(graphs: Sequence[nx.Graph], *, require_connected: bool, min_nodes: int = 4) -> tuple[list[nx.Graph], dict[str, int]]:
    kept: list[nx.Graph] = []
    dropped_empty = 0
    dropped_disconnected = 0
    dropped_too_small = 0
    for graph in graphs:
        g = _canonical_graph(graph)
        if g.number_of_nodes() < int(min_nodes):
            dropped_too_small += 1
            continue
        if g.number_of_edges() < 2:
            dropped_empty += 1
            continue
        if require_connected and not nx.is_connected(g):
            dropped_disconnected += 1
            continue
        kept.append(g)
    return kept, {
        "input_graphs": int(len(graphs)),
        "kept_graphs": int(len(kept)),
        "dropped_too_small": int(dropped_too_small),
        "dropped_too_few_edges": int(dropped_empty),
        "dropped_disconnected": int(dropped_disconnected),
    }



def _filter_molecular_graphs(
    graphs: Sequence[nx.Graph],
    *,
    require_connected: bool,
    min_nodes: int = 2,
) -> tuple[list[nx.Graph], dict[str, int]]:
    kept: list[nx.Graph] = []
    dropped_empty = 0
    dropped_disconnected = 0
    dropped_too_small = 0
    dropped_schema = 0
    for graph in graphs:
        try:
            g = canonicalize_molecular_graph(graph)
        except Exception:
            dropped_schema += 1
            continue
        if g.number_of_nodes() < int(min_nodes):
            dropped_too_small += 1
            continue
        if g.number_of_edges() < 2:
            dropped_empty += 1
            continue
        if require_connected and g.number_of_nodes() > 1 and not nx.is_connected(g):
            dropped_disconnected += 1
            continue
        kept.append(g)
    return kept, {
        "input_graphs": int(len(graphs)),
        "kept_graphs": int(len(kept)),
        "dropped_too_small": int(dropped_too_small),
        "dropped_too_few_edges": int(dropped_empty),
        "dropped_disconnected": int(dropped_disconnected),
        "dropped_schema": int(dropped_schema),
    }

def _print_compact_table(metrics: dict[str, Any], *, molecular: bool = False) -> None:
    if molecular:
        keys = [
            "validity_without_correction",
            "nspdk_mmd",
            "nspdk_mmd_valid_only",
            "fcd",
            "uniqueness_rate",
            "novelty_rate",
            "degree_preservation_from_digress_rate",
            "node_type_preservation_from_digress_rate",
        ]
    else:
        keys = [
            "degree_mmd",
            "clustering_mmd",
            "spectral_mmd",
            "motif_proxy_mmd",
            "connectedness_rate",
            "uniqueness_rate",
            "novelty_rate",
        ]
    names = ["digress", "digress_random_rewire", "digress_grapher_optimizer"]
    print("\nMetric summary")
    print("method".ljust(30) + " ".join(k.rjust(26) for k in keys))
    for name in names:
        row = metrics.get(name, {})
        values = []
        for key in keys:
            value = row.get(key)
            if value is None:
                values.append("None".rjust(26))
            elif isinstance(value, (float, int)):
                values.append(f"{float(value):26.6g}")
            else:
                values.append(str(value).rjust(26))
        print(name.ljust(30) + " ".join(values))


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------


def _experiment_is_molecular(args: argparse.Namespace) -> bool:
    mode = str(getattr(args, "mode", "auto")).strip().lower()
    if mode == "molecular":
        return True
    if mode == "generic":
        return False
    return str(args.dataset).lower() in {"qm9", "zinc"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    set_seed(int(args.seed), include_torch=False)
    is_molecular = _experiment_is_molecular(args)

    sample_path = Path(args.digress_sample_path) if args.digress_sample_path else default_digress_sample_path()
    if sample_path is None:
        raise FileNotFoundError(
            "No DiGress sample file was provided and no default generated_adjs.npz/generated_samples1.txt "
            "was found. Run DiGress sampling first or pass --digress-sample-path."
        )

    print(f"Loading dataset splits for {args.dataset!r}...")
    splits = load_dataset_splits(args.dataset, output_root=args.dataset_root, build_if_missing=True)
    train_graphs = _subsample(splits["train"], args.max_train_graphs, int(args.seed) + 1)
    reference_graphs = _subsample(splits[args.reference_split], args.max_reference_graphs, int(args.seed) + 2)

    require_connected = not bool(args.allow_disconnected)
    ensure_connected = not bool(args.allow_disconnected)
    k_hop = None if int(args.k_hop) < 0 else int(args.k_hop)

    print(f"Loading DiGress samples from {sample_path}...")
    if is_molecular:
        raw_digress_graphs = load_digress_molecular_graphs(
            sample_path,
            dataset=args.dataset,
            qm9_with_h=bool(args.qm9_with_h),
            atom_decoder=args.digress_atom_decoder,
        )
        digress_graphs, filter_info = _filter_molecular_graphs(
            raw_digress_graphs,
            require_connected=require_connected,
            min_nodes=int(args.min_nodes),
        )
    else:
        raw_digress_graphs = load_digress_graphs(sample_path)
        digress_graphs, filter_info = _filter_graphs(
            raw_digress_graphs,
            require_connected=require_connected,
            min_nodes=int(args.min_nodes),
        )
    digress_graphs = _subsample(digress_graphs, args.num_graphs, int(args.seed) + 3)
    if not digress_graphs:
        raise RuntimeError(
            "No usable DiGress graphs remained after filtering. Use --allow-disconnected "
            "or check the DiGress sample file/atom decoder."
        )

    print(f"Using {len(digress_graphs)} DiGress graphs; filter_info={filter_info}")
    refined_graphs: list[nx.Graph] = []
    random_graphs: list[nx.Graph] = []
    refine_traces: list[dict[str, Any]] = []
    random_traces: list[dict[str, Any]] = []
    target_indices: list[int | None] = []

    if is_molecular:
        print("Fitting molecular GraphER optimizer energy and empirical bond prior...")
        train_graphs = [canonicalize_molecular_graph(g) for g in train_graphs]
        reference_graphs = [canonicalize_molecular_graph(g) for g in reference_graphs]
        bond_prior = EmpiricalBondTypePrior.fit(
            train_graphs,
            smoothing=float(args.molecular_prior_smoothing),
        )
        energy_model = MolecularFeatureEnergy(
            train_graphs,
            target_mode=args.target_mode,
            degree_match_weight=float(args.degree_match_weight),
            atom_match_weight=float(args.atom_match_weight),
            include_spectral=bool(args.include_spectral_energy),
            spectral_k=int(args.spectral_k),
            include_modularity=bool(args.include_modularity_energy),
            prior=bond_prior,
        )
        print("Running random molecular rewiring control and GraphER energy-guided refinement...")
        for idx, graph in enumerate(digress_graphs):
            target, target_idx = energy_model.target_for(graph)
            target_indices.append(target_idx)
            random_graph, random_trace = random_molecular_rewire_graph(
                graph,
                prior=bond_prior,
                steps=int(args.steps),
                candidate_budget=int(args.candidate_budget),
                k_hop=k_hop,
                ensure_connected=ensure_connected,
                seed=int(args.seed) + 10_000 + idx,
                proposals_per_edge=int(args.molecular_proposals_per_edge),
                proposal_mode=args.molecular_bond_proposal_mode,
                allow_global_backoff=bool(args.allow_global_bond_backoff),
                reject_unseen_endpoint_pairs=bool(args.reject_unseen_endpoint_pairs),
                valence_tolerance=float(args.valence_tolerance),
                require_valid_candidates=bool(args.molecular_require_valid_candidates),
                isomeric_smiles=bool(args.isomeric_smiles),
            )
            refined_graph, refine_trace = refine_molecular_graph_energy(
                graph,
                energy_model=energy_model,
                target=target,
                prior=bond_prior,
                steps=int(args.steps),
                candidate_budget=int(args.candidate_budget),
                k_hop=k_hop,
                ensure_connected=ensure_connected,
                seed=int(args.seed) + 20_000 + idx,
                min_improvement=float(args.min_improvement),
                proposals_per_edge=int(args.molecular_proposals_per_edge),
                proposal_mode=args.molecular_bond_proposal_mode,
                allow_global_backoff=bool(args.allow_global_bond_backoff),
                reject_unseen_endpoint_pairs=bool(args.reject_unseen_endpoint_pairs),
                valence_tolerance=float(args.valence_tolerance),
                require_valid_candidates=bool(args.molecular_require_valid_candidates),
                isomeric_smiles=bool(args.isomeric_smiles),
            )
            random_graphs.append(random_graph)
            refined_graphs.append(refined_graph)
            random_traces.append(random_trace)
            refine_traces.append(refine_trace)
            if (idx + 1) % max(int(args.progress_every), 1) == 0 or (idx + 1) == len(digress_graphs):
                print(f"  refined {idx + 1}/{len(digress_graphs)} molecular graphs")

        print("Evaluating before/after molecular metrics...")
        metrics, molecular_protocol = evaluate_molecular_sets(
            reference_graphs=reference_graphs,
            train_graphs=train_graphs,
            base_graphs=digress_graphs,
            random_graphs=random_graphs,
            refined_graphs=refined_graphs,
            nspdk_backend=args.nspdk_backend,
            nspdk_complexity=int(args.nspdk_complexity),
            skip_nspdk=bool(args.skip_nspdk),
            skip_fcd=bool(args.skip_fcd),
            require_fcd=bool(args.require_fcd),
            fcd_device=args.fcd_device,
            fcd_n_jobs=int(args.fcd_n_jobs),
            fcd_batch_size=int(args.fcd_batch_size),
            keep_explicit_hydrogens=bool(args.keep_explicit_hydrogens),
            isomeric_smiles=bool(args.isomeric_smiles),
        )
    else:
        print("Fitting GraphER optimizer energy on training graphs...")
        energy_model = FeatureEnergy(
            train_graphs,
            target_mode=args.target_mode,
            degree_match_weight=float(args.degree_match_weight),
            include_spectral=bool(args.include_spectral_energy),
            spectral_k=int(args.spectral_k),
            include_modularity=bool(args.include_modularity_energy),
        )
        print("Running random rewiring control and GraphER energy-guided refinement...")
        for idx, graph in enumerate(digress_graphs):
            target, target_idx = energy_model.target_for(graph)
            target_indices.append(target_idx)
            random_graph, random_trace = random_rewire_graph(
                graph,
                steps=int(args.steps),
                candidate_budget=int(args.candidate_budget),
                k_hop=k_hop,
                ensure_connected=ensure_connected,
                seed=int(args.seed) + 10_000 + idx,
            )
            refined_graph, refine_trace = refine_graph_energy(
                graph,
                energy_model=energy_model,
                target=target,
                steps=int(args.steps),
                candidate_budget=int(args.candidate_budget),
                k_hop=k_hop,
                ensure_connected=ensure_connected,
                seed=int(args.seed) + 20_000 + idx,
                min_improvement=float(args.min_improvement),
            )
            random_graphs.append(random_graph)
            refined_graphs.append(refined_graph)
            random_traces.append(random_trace)
            refine_traces.append(refine_trace)
            if (idx + 1) % max(int(args.progress_every), 1) == 0 or (idx + 1) == len(digress_graphs):
                print(f"  refined {idx + 1}/{len(digress_graphs)} graphs")

        print("Evaluating before/after metrics...")
        metrics = evaluate_sets(
            reference_graphs=reference_graphs,
            train_graphs=train_graphs,
            base_graphs=digress_graphs,
            random_graphs=random_graphs,
            refined_graphs=refined_graphs,
            dataset=args.dataset,
            degree_bins=int(args.degree_bins),
            clustering_bins=int(args.clustering_bins),
            spectral_bins=int(args.spectral_bins),
            sigma=float(args.sigma),
        )
        molecular_protocol = None

    accepted = np.asarray([trace.get("accepted_steps", 0) for trace in refine_traces], dtype=np.float64)
    energy_delta = np.asarray([trace.get("energy_delta", 0.0) for trace in refine_traces], dtype=np.float64)
    diagnostics = {
        "refiner_accepted_steps_mean": float(accepted.mean()) if accepted.size else 0.0,
        "refiner_accepted_steps_std": float(accepted.std(ddof=0)) if accepted.size else 0.0,
        "refiner_energy_delta_mean": float(energy_delta.mean()) if energy_delta.size else 0.0,
        "refiner_energy_delta_std": float(energy_delta.std(ddof=0)) if energy_delta.size else 0.0,
        "refiner_degree_preservation_rate": float(np.mean([bool(t.get("degree_preserved", False)) for t in refine_traces])) if refine_traces else 0.0,
        "random_degree_preservation_rate": float(np.mean([bool(t.get("degree_preserved", False)) for t in random_traces])) if random_traces else 0.0,
        "target_mode": args.target_mode,
        "num_unique_matched_targets": int(len({idx for idx in target_indices if idx is not None})),
        "mode": "molecular" if is_molecular else "generic",
    }
    if is_molecular:
        diagnostics["refiner_node_type_preservation_rate"] = float(np.mean([bool(t.get("node_types_preserved", False)) for t in refine_traces])) if refine_traces else 0.0
        diagnostics["random_node_type_preservation_rate"] = float(np.mean([bool(t.get("node_types_preserved", False)) for t in random_traces])) if random_traces else 0.0
        diagnostics["molecular_require_valid_candidates"] = bool(args.molecular_require_valid_candidates)
        diagnostics["refiner_rdkit_valid_final_rate"] = float(np.mean([bool(t.get("rdkit_valid_final", False)) for t in refine_traces])) if refine_traces else 0.0
        diagnostics["random_rdkit_valid_final_rate"] = float(np.mean([bool(t.get("rdkit_valid_final", False)) for t in random_traces])) if random_traces else 0.0
        diagnostics["refiner_rejected_invalid_candidates_mean"] = float(np.mean([int(t.get("rejected_invalid_candidates", 0)) for t in refine_traces])) if refine_traces else 0.0
        diagnostics["random_rejected_invalid_candidates_mean"] = float(np.mean([int(t.get("rejected_invalid_candidates", 0)) for t in random_traces])) if random_traces else 0.0

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    refined_path = output_dir / "digress_grapher_optimizer_refined.pkl"
    random_path = output_dir / "digress_random_rewire.pkl"
    base_path = output_dir / "digress_filtered_base.pkl"
    metrics_path = output_dir / "digress_grapher_optimizer_metrics.json"
    config_path = output_dir / "digress_grapher_optimizer_config.json"

    save_pickle(refined_graphs, refined_path, force=True)
    save_pickle(random_graphs, random_path, force=True)
    save_pickle(digress_graphs, base_path, force=True)

    paths: dict[str, str] = {
        "digress_sample_path": str(sample_path),
        "base_graphs": str(base_path),
        "random_rewire_graphs": str(random_path),
        "refined_graphs": str(refined_path),
        "metrics": str(metrics_path),
        "config": str(config_path),
    }
    if is_molecular:
        # Convenience exports for inspecting molecules without loading pickle.
        base_conv = direct_molecular_conversions(digress_graphs, isomeric_smiles=bool(args.isomeric_smiles))
        rand_conv = direct_molecular_conversions(random_graphs, isomeric_smiles=bool(args.isomeric_smiles))
        ref_conv = direct_molecular_conversions(refined_graphs, isomeric_smiles=bool(args.isomeric_smiles))
        base_jsonl = write_molecular_jsonl(digress_graphs, output_dir / "digress_filtered_base.jsonl", conversions=base_conv)
        random_jsonl = write_molecular_jsonl(random_graphs, output_dir / "digress_random_rewire.jsonl", conversions=rand_conv)
        refined_jsonl = write_molecular_jsonl(refined_graphs, output_dir / "digress_grapher_optimizer_refined.jsonl", conversions=ref_conv)
        heavy_atoms_only = not bool(args.keep_explicit_hydrogens)
        base_smiles = valid_smiles(base_conv, remove_explicit_hydrogens=heavy_atoms_only, isomeric_smiles=bool(args.isomeric_smiles))
        random_smiles = valid_smiles(rand_conv, remove_explicit_hydrogens=heavy_atoms_only, isomeric_smiles=bool(args.isomeric_smiles))
        refined_smiles = valid_smiles(ref_conv, remove_explicit_hydrogens=heavy_atoms_only, isomeric_smiles=bool(args.isomeric_smiles))
        paths.update(
            {
                "base_jsonl": str(base_jsonl),
                "random_jsonl": str(random_jsonl),
                "refined_jsonl": str(refined_jsonl),
                "base_smiles": str(write_smiles_file(base_smiles, output_dir / "digress_filtered_base_valid.smi")),
                "random_smiles": str(write_smiles_file(random_smiles, output_dir / "digress_random_rewire_valid.smi")),
                "refined_smiles": str(write_smiles_file(refined_smiles, output_dir / "digress_grapher_optimizer_refined_valid.smi")),
            }
        )
        if bool(args.write_sdf):
            try:
                paths["base_sdf"] = str(write_molecular_sdf(base_conv, output_dir / "digress_filtered_base_valid.sdf")[0])
                paths["random_sdf"] = str(write_molecular_sdf(rand_conv, output_dir / "digress_random_rewire_valid.sdf")[0])
                paths["refined_sdf"] = str(write_molecular_sdf(ref_conv, output_dir / "digress_grapher_optimizer_refined_valid.sdf")[0])
            except Exception as exc:
                paths["sdf_write_error"] = f"{type(exc).__name__}:{exc}"

    payload = {
        "dataset": args.dataset,
        "experiment": "digress_grapher_optimizer",
        "mode": "molecular" if is_molecular else "generic",
        "runtime_seconds": float(time.perf_counter() - start),
        "paths": paths,
        "filter_info": filter_info,
        "num_graphs_used": int(len(digress_graphs)),
        "num_train_graphs": int(len(train_graphs)),
        "num_reference_graphs": int(len(reference_graphs)),
        "diagnostics": diagnostics,
        "metrics": metrics,
        "molecular_protocol": molecular_protocol,
        "refine_traces": refine_traces if bool(args.save_traces) else None,
        "random_traces": random_traces if bool(args.save_traces) else None,
    }
    save_json(payload, metrics_path, force=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    _print_compact_table(metrics, molecular=is_molecular)
    print(f"\nSaved refined graphs to: {refined_path}")
    print(f"Saved metrics to:        {metrics_path}")
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Small experiment: use DiGress as a coarse graph/molecule generator and GraphER-style "
            "degree/connectivity-preserving rewiring as a topology or molecular optimizer."
        )
    )
    parser.add_argument("--dataset", default="sbm", help="GraphES dataset name, e.g. sbm, planar, grid, qm9.")
    parser.add_argument("--mode", default="auto", choices=["auto", "generic", "molecular"], help="Experiment mode. Auto uses molecular mode for qm9/zinc and generic otherwise.")
    parser.add_argument("--dataset-root", default="outputs/datasets", help="Prepared GraphES dataset root.")
    parser.add_argument(
        "--digress-sample-path",
        default=None,
        help=(
            "Path to DiGress samples. Supports DiGress generated_samples*.txt, generated_adjs.npz, "
            ".pkl/.pt lists of NetworkX graphs, adjacency matrices, or DiGress (X,E) tuples. "
            "If omitted, the script checks baselines/DiGress/generated_adjs.npz and generated_samples1.txt."
        ),
    )
    parser.add_argument("--num-graphs", type=int, default=64, help="Maximum number of DiGress samples to refine.")
    parser.add_argument("--max-train-graphs", type=int, default=512, help="Training graphs used to fit target statistics.")
    parser.add_argument("--max-reference-graphs", type=int, default=512, help="Test/reference graphs used for MMD evaluation.")
    parser.add_argument("--reference-split", default="test", choices=["train", "val", "test"], help="Reference split for metrics.")
    parser.add_argument("--steps", type=int, default=20, help="Maximum valid rewiring steps per graph.")
    parser.add_argument("--candidate-budget", type=int, default=64, help="Valid candidate swaps sampled per refinement step.")
    parser.add_argument("--k-hop", type=int, default=2, help="Locality radius for candidate swaps; use -1 for unrestricted sampled swaps.")
    parser.add_argument("--min-improvement", type=float, default=1e-9, help="Minimum energy decrease required to accept a step.")
    parser.add_argument("--target-mode", default="nearest", choices=["nearest", "mean"], help="Use nearest real graph target or mean training target statistics.")
    parser.add_argument("--degree-match-weight", type=float, default=1.0, help="Weight for degree-histogram nearest-target matching.")
    parser.add_argument("--include-spectral-energy", action="store_true", help="Also optimize a small spectral vector. Slower but stronger.")
    parser.add_argument("--spectral-k", type=int, default=8, help="Number of spectral coordinates if --include-spectral-energy is used.")
    parser.add_argument("--include-modularity-energy", action="store_true", help="Include greedy modularity in the optimizer energy. Slower.")
    parser.add_argument("--allow-disconnected", action="store_true", help="Do not filter disconnected DiGress samples and do not enforce connected rewiring.")
    parser.add_argument("--min-nodes", type=int, default=4, help="Drop DiGress samples with fewer nodes.")
    parser.add_argument("--degree-bins", type=int, default=20)
    parser.add_argument("--clustering-bins", type=int, default=20)
    parser.add_argument("--spectral-bins", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--save-traces", action="store_true", help="Store per-step energy histories in the metrics JSON.")

    # Molecular/QM9 options.
    parser.add_argument("--qm9-with-h", action="store_true", help="Interpret DiGress QM9 node ids as H,C,N,O,F instead of C,N,O,F.")
    parser.add_argument("--digress-atom-decoder", default=None, help="Comma-separated atom decoder for DiGress node ids, e.g. C,N,O,F or H,C,N,O,F or 6,7,8,9.")
    parser.add_argument("--atom-match-weight", type=float, default=1.0, help="Nearest-target matching weight for molecular atom histograms.")
    parser.add_argument("--molecular-prior-smoothing", type=float, default=1e-3, help="Smoothing for endpoint-conditioned empirical bond-type proposals.")
    parser.add_argument("--molecular-proposals-per-edge", type=int, default=2, help="Typed bond-label proposals per new edge.")
    parser.add_argument("--molecular-bond-proposal-mode", default="topk", choices=["sample", "topk", "mode", "top"], help="How to propose bond labels for typed molecular rewiring.")
    parser.add_argument("--allow-global-bond-backoff", action="store_true", help="Allow global bond-type proposal backoff for unseen endpoint atom pairs.")
    parser.add_argument("--reject-unseen-endpoint-pairs", action="store_true", help="Reject typed rewires whose new atom-pair type was unseen in training.")
    parser.add_argument("--valence-tolerance", type=float, default=1e-6, help="Tolerance for valence-filtered molecular rewiring.")
    parser.add_argument("--molecular-require-valid-candidates", action="store_true", help="Conservative QM9 mode: reject every candidate rewiring that does not sanitize in RDKit. This prevents valid DiGress molecules from being refined into chemically invalid molecules, at the cost of slower candidate evaluation and fewer accepted moves.")
    parser.add_argument("--nspdk-backend", default="auto", choices=["auto", "eden", "builtin"], help="NSPDK backend for molecular metrics.")
    parser.add_argument("--nspdk-complexity", type=int, default=4, help="NSPDK neighborhood complexity.")
    parser.add_argument("--skip-nspdk", action="store_true", help="Skip molecular NSPDK MMD.")
    parser.add_argument("--skip-fcd", action="store_true", help="Skip molecular FCD.")
    parser.add_argument("--require-fcd", action="store_true", help="Raise an error if FCD cannot be computed.")
    parser.add_argument("--fcd-device", default="cpu")
    parser.add_argument("--fcd-n-jobs", type=int, default=1)
    parser.add_argument("--fcd-batch-size", type=int, default=512)
    parser.add_argument("--keep-explicit-hydrogens", action="store_true", help="Keep explicit H in NSPDK/FCD representations.")
    parser.add_argument("--isomeric-smiles", action="store_true", help="Use isomeric SMILES for molecular validity/FCD/novelty.")
    parser.add_argument("--write-sdf", action="store_true", help="Also write valid base/random/refined molecules as SDF files.")

    parser.add_argument("--output-dir", default="outputs/experiments/digress_grapher_optimizer")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
