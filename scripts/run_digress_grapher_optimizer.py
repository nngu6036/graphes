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


def _load_digress_txt(path: Path) -> list[nx.Graph]:
    """Parse DiGress generated_samples*.txt files.

    The DiGress discrete model writes blocks of the form:

        N=...
        X:
        ... node types ...
        E:
        ... N rows of edge-type ids ...
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    graphs: list[nx.Graph] = []
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
        # Skip until the E: marker.
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
        if len(rows) == n and all(len(row) == n for row in rows):
            graphs.append(_graph_from_adjacency(np.asarray(rows, dtype=np.int64)))
    if not graphs:
        raise ValueError(f"No DiGress graph blocks could be parsed from {path}.")
    return graphs


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


def _print_compact_table(metrics: dict[str, Any]) -> None:
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
    print("method".ljust(30) + " ".join(k.rjust(22) for k in keys))
    for name in names:
        row = metrics.get(name, {})
        values = []
        for key in keys:
            value = row.get(key)
            if value is None:
                values.append("None".rjust(22))
            elif isinstance(value, (float, int)):
                values.append(f"{float(value):22.6g}")
            else:
                values.append(str(value).rjust(22))
        print(name.ljust(30) + " ".join(values))


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    set_seed(int(args.seed), include_torch=False)

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

    print(f"Loading DiGress samples from {sample_path}...")
    raw_digress_graphs = load_digress_graphs(sample_path)
    require_connected = not bool(args.allow_disconnected)
    digress_graphs, filter_info = _filter_graphs(
        raw_digress_graphs,
        require_connected=require_connected,
        min_nodes=int(args.min_nodes),
    )
    digress_graphs = _subsample(digress_graphs, args.num_graphs, int(args.seed) + 3)
    if not digress_graphs:
        raise RuntimeError(
            "No usable DiGress graphs remained after filtering. Use --allow-disconnected "
            "or check the DiGress sample file."
        )

    print(f"Using {len(digress_graphs)} DiGress graphs; filter_info={filter_info}")
    print("Fitting GraphER optimizer energy on training graphs...")
    energy_model = FeatureEnergy(
        train_graphs,
        target_mode=args.target_mode,
        degree_match_weight=float(args.degree_match_weight),
        include_spectral=bool(args.include_spectral_energy),
        spectral_k=int(args.spectral_k),
        include_modularity=bool(args.include_modularity_energy),
    )

    refined_graphs: list[nx.Graph] = []
    random_graphs: list[nx.Graph] = []
    refine_traces: list[dict[str, Any]] = []
    random_traces: list[dict[str, Any]] = []
    target_indices: list[int | None] = []

    ensure_connected = not bool(args.allow_disconnected)
    k_hop = None if int(args.k_hop) < 0 else int(args.k_hop)
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
    }

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

    payload = {
        "dataset": args.dataset,
        "experiment": "digress_grapher_optimizer",
        "runtime_seconds": float(time.perf_counter() - start),
        "paths": {
            "digress_sample_path": str(sample_path),
            "base_graphs": str(base_path),
            "random_rewire_graphs": str(random_path),
            "refined_graphs": str(refined_path),
            "metrics": str(metrics_path),
            "config": str(config_path),
        },
        "filter_info": filter_info,
        "num_graphs_used": int(len(digress_graphs)),
        "num_train_graphs": int(len(train_graphs)),
        "num_reference_graphs": int(len(reference_graphs)),
        "diagnostics": diagnostics,
        "metrics": metrics,
        "refine_traces": refine_traces if bool(args.save_traces) else None,
        "random_traces": random_traces if bool(args.save_traces) else None,
    }
    save_json(payload, metrics_path, force=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    _print_compact_table(metrics)
    print(f"\nSaved refined graphs to: {refined_path}")
    print(f"Saved metrics to:        {metrics_path}")
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Small experiment: use DiGress as a coarse graph generator and GraphER-style "
            "degree/connectivity-preserving rewiring as a topology optimizer."
        )
    )
    parser.add_argument("--dataset", default="sbm", help="GraphES dataset name, e.g. sbm, planar, grid.")
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
    parser.add_argument("--output-dir", default="outputs/experiments/digress_grapher_optimizer")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
