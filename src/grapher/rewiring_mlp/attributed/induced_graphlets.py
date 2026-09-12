from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from math import comb
from typing import Any

import networkx as nx
import numpy as np

from grapher.utils.motifs import _canonicalize_attributed_tokens
from grapher.rewiring_mlp.attributed.data import GraphletBasis
from grapher.rewiring_mlp.generic.induced_graphlets import InducedGraphletSpec, validate_k
from grapher.rewiring_mlp.attributed.graphlet_diffusion import (
    candidate_attributed_graphlet_counts,
    extract_attributed_graphlet_counts,
)

VERSION = "attributed_induced_histogram_v1"


def _single_k(basis: GraphletBasis) -> str:
    if len(basis.sizes) != 1:
        raise ValueError(
            "Attributed induced graphlet histograms currently require exactly one graphlet order."
        )
    return basis.sizes[0]


def validate_supported_basis(basis: GraphletBasis) -> int:
    if not basis.attributed:
        raise ValueError("Attributed induced graphlet histograms require an attributed GraphletBasis.")
    if basis.topology_filter != "all":
        raise ValueError(
            "Attributed induced graphlet histograms currently support topology_filter='all' only."
        )
    if basis.connected_only:
        raise ValueError(
            "Attributed induced graphlet histograms currently support scope='all' only."
        )
    if not basis.node_attribute or not basis.edge_attribute:
        raise ValueError(
            "Attributed induced graphlet histograms require node_attribute and edge_attribute."
        )
    k = validate_k(int(_single_k(basis)))
    if basis.attributed_backend != "python":
        raise ValueError("Attributed histogram extraction and local deltas require the python canonicalizer.")
    keys = basis.keys_by_k[str(k)]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Attributed vocabulary must have unique nonempty bin identifiers.")
    if basis.overflow_key is None or basis.overflow_key not in keys:
        raise ValueError("Attributed vocabulary requires an explicit unseen-type overflow bin.")
    return k


def metadata(basis: GraphletBasis) -> dict[str, Any]:
    k = validate_supported_basis(basis)
    payload = {
        "version": VERSION,
        "k": k,
        "scope": "all",
        "width": basis.width,
        "attributed": True,
        "node_attribute": str(basis.node_attribute),
        "edge_attribute": str(basis.edge_attribute),
        "overflow_key": basis.overflow_key,
        "normalization": f"choose(n,{k})",
        "vocabulary_policy": "observed_training_graphlets_plus_unseen_overflow",
        "canonical_backend": basis.attributed_backend,
        "bin_ids": list(basis.keys_by_k[str(k)]),
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _placeholder(basis: GraphletBasis) -> np.ndarray:
    out = np.zeros(basis.width, dtype=np.float64)
    if basis.width > 0:
        out[0] = 1.0
    return out


def validate_histogram(values: Any, basis: GraphletBasis) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.shape != (basis.width,):
        raise ValueError("Attributed induced graphlet histogram width mismatch.")
    if not np.isfinite(x).all() or np.any(x < -1e-7) or np.any(x > 1 + 1e-7):
        raise ValueError(
            "Attributed induced graphlet histogram must be finite and nonnegative."
        )
    total = float(x.sum())
    if not np.isclose(total, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(
            "Attributed induced graphlet histogram must be a normalized probability vector."
        )
    x = np.clip(x, 0.0, 1.0)
    return x / max(float(x.sum()), 1.0e-12)


def histogram_from_counts(
    counts_by_size: dict[str, dict[str, int]],
    num_nodes: int,
    basis: GraphletBasis,
) -> np.ndarray:
    k = validate_supported_basis(basis)
    total = comb(int(num_nodes), k) if int(num_nodes) >= k else 0
    if total <= 0:
        return _placeholder(basis)
    size = str(k)
    counts = counts_by_size.get(size, {}) or {}
    block = np.asarray(
        [float(counts.get(key, 0)) for key in basis.keys_by_k[size]],
        dtype=np.float64,
    )
    histogram = block / float(total)
    return validate_histogram(histogram, basis)


def extract_histogram(graph: nx.Graph, basis: GraphletBasis) -> np.ndarray:
    counts = extract_attributed_graphlet_counts(graph, graphlet_basis=basis)
    return histogram_from_counts(counts, graph.number_of_nodes(), basis)


def histogram_distance(left: Any, right: Any, basis: GraphletBasis) -> float:
    a = validate_histogram(left, basis)
    b = validate_histogram(right, basis)
    return float(0.5 * np.abs(a - b).sum())


@dataclass
class AttributedInducedGraphletCounter:
    graph: nx.Graph
    basis: GraphletBasis

    def __post_init__(self) -> None:
        self.k = validate_supported_basis(self.basis)
        self.graph = self.graph.copy()
        self.counts_by_size = extract_attributed_graphlet_counts(
            self.graph,
            graphlet_basis=self.basis,
        )

    def histogram(self) -> np.ndarray:
        return histogram_from_counts(
            self.counts_by_size,
            self.graph.number_of_nodes(),
            self.basis,
        )

    def candidate_counts(
        self,
        candidate: nx.Graph,
        action: Any | None = None,
    ) -> dict[str, dict[str, int]]:
        if action is None:
            return extract_attributed_graphlet_counts(candidate, graphlet_basis=self.basis)
        return candidate_attributed_graphlet_counts(
            self.graph,
            candidate,
            action,
            current_counts=self.counts_by_size,
            graphlet_basis=self.basis,
        )

    def candidate_histogram(
        self,
        candidate: nx.Graph,
        action: Any | None = None,
    ) -> np.ndarray:
        counts = self.candidate_counts(candidate, action)
        return histogram_from_counts(counts, candidate.number_of_nodes(), self.basis)

    def accept(self, candidate: nx.Graph, action: Any | None = None) -> None:
        self.counts_by_size = self.candidate_counts(candidate, action)
        self.graph = candidate.copy()


def prediction_and_loss(logits, target, sizes, basis: GraphletBasis):
    import torch
    import torch.nn.functional as F

    k = validate_supported_basis(basis)
    if logits.ndim != 2 or logits.shape[-1] != basis.width:
        raise ValueError("Wrong attributed induced graphlet output shape.")
    if target.shape != logits.shape:
        raise ValueError(
            "Attributed induced graphlet target/prediction shape mismatch."
        )
    target = target.to(logits)
    if (
        not torch.isfinite(target).all()
        or (target < -1e-6).any()
        or not torch.allclose(
            target.sum(-1),
            torch.ones_like(sizes, dtype=target.dtype),
            atol=1e-5,
        )
    ):
        raise ValueError(
            "Attributed induced graphlet targets must be probability distributions."
        )
    valid = (sizes >= k).to(logits.dtype)
    denominator = valid.sum().clamp_min(1)
    p = logits.softmax(-1)
    delta = p - target
    brier = (delta.square().sum(-1) * valid).sum() / denominator
    ce = (-(target * F.log_softmax(logits, -1)).sum(-1) * valid).sum() / denominator
    totals = torch.as_tensor(
        [comb(int(n), k) if int(n) >= k else 0 for n in sizes.detach().cpu().tolist()],
        dtype=logits.dtype,
        device=logits.device,
    )
    metrics = {
        "induced_graphlet_histogram_loss": brier,
        "induced_graphlet_histogram_ce": ce,
        "induced_graphlet_histogram_tv": (0.5 * delta.abs().sum(-1) * valid).sum() / denominator,
        "induced_graphlet_count_mae": (delta.abs().mean(-1) * totals * valid).sum() / denominator,
        "induced_graphlet_valid_fraction": valid.mean(),
    }
    return brier, ce, metrics


def mask_prediction(probabilities, sizes, basis: GraphletBasis):
    import torch

    k = validate_supported_basis(basis)
    placeholder = torch.zeros_like(probabilities)
    placeholder[:, 0] = 1
    return torch.where((sizes >= k).unsqueeze(-1), probabilities, placeholder)


__all__ = [
    "AttributedInducedGraphletCounter",
    "extract_histogram",
    "histogram_distance",
    "histogram_from_counts",
    "mask_prediction",
    "metadata",
    "prediction_and_loss",
    "validate_histogram",
    "validate_supported_basis",
]


def wants_attributed_histogram(config: dict) -> bool:
    values = config.get("structure_summary_prediction", {})
    flag = values.get("induced_graphlet_attributed", True)
    if not isinstance(flag, bool):
        raise ValueError("induced_graphlet_attributed must be boolean.")
    return bool(values.get("induced_graphlet_histogram", False) and flag)


def validate_model_graphlets(model, config: dict) -> None:
    """Never interpret an old topology head as a labeled graphlet vocabulary."""
    requested = InducedGraphletSpec.from_config(config.get("structure_summary_prediction"))
    basis = model.induced_graphlet_basis
    if basis is not None:
        k = validate_supported_basis(basis)
        if (not wants_attributed_histogram(config) or requested is None
                or requested.k != k or requested.scope != "all"):
            raise ValueError("Attributed induced graphlet config/catalogue differs from checkpoint.")
        cat = config.get("categorical_state", {})
        if (cat.get("node_attribute", "atomic_num") != basis.node_attribute or
                cat.get("edge_attribute", "bond_type") != basis.edge_attribute):
            raise ValueError("Attributed graphlet label attributes differ from checkpoint.")
    else:
        if requested is not None and wants_attributed_histogram(config):
            raise ValueError("This checkpoint does not have an attributed graphlet vocabulary; "
                             "train a new checkpoint. A topology-only head is not compatible.")
        if requested != model.induced_graphlet_spec:
            raise ValueError("Induced graphlet config/catalogue differs from checkpoint.")


def fit_training_basis(config: dict, train_graphs) -> GraphletBasis | None:
    if not wants_attributed_histogram(config):
        return None
    requested = InducedGraphletSpec.from_config(config.get("structure_summary_prediction"))
    if requested.scope != "all":
        raise ValueError("Attributed induced graphlets currently support scope=all only.")
    cat = config["categorical_state"]
    settings = dict(
        graphlet_history=True, graphlet_k_min=requested.k, graphlet_k_max=requested.k,
        graphlet_connected_only=False, graphlet_topology_filter="all", graphlet_num_samples=None,
        attributed=True, node_attribute=cat.get("node_attribute", "atomic_num"),
        edge_attribute=cat.get("edge_attribute", "bond_type"), attributed_backend="python",
    )
    print(f"[AttributedGraphlets] fitting TRAIN-only labeled vocabulary: "
          f"graphs={len(train_graphs)} k={requested.k} scope=all; "
          f"labels={settings['node_attribute']}/{settings['edge_attribute']}", flush=True)
    node_counts = [graph.number_of_nodes() for graph in train_graphs]
    subset_counts = [comb(n, requested.k) if n >= requested.k else 0 for n in node_counts]
    total_subsets = sum(subset_counts)
    pc = config.get("attributed_predictor", {})
    seconds = float(pc.get("progress_interval_seconds", 60))
    graph_interval = int(pc.get("graphlet_progress_interval", 1000))
    print(f"[AttributedGraphlets] CPU preprocessing: induced_subsets={total_subsets:,} "
          f"(including disconnected graphlets); progress_interval_seconds={seconds:g} "
          f"graphlet_progress_interval={graph_interval}", flush=True)
    print(f"[AttributedGraphlets] workload nodes_min={min(node_counts, default=0)} "
          f"nodes_max={max(node_counts, default=0)} "
          f"nodes_mean={sum(node_counts) / max(len(node_counts), 1):.2f} "
          f"graphs_with_no_k_subsets={sum(count == 0 for count in subset_counts)} "
          f"max_subsets_per_graph={max(subset_counts, default=0):,}", flush=True)
    initial_cache = _canonicalize_attributed_tokens.cache_info()
    print(f"[AttributedGraphlets] enumeration starting backend=python "
          f"cache_entries={initial_cache.currsize}/{initial_cache.maxsize}; "
          "bins exclude overflow until finalization", flush=True)
    started = last_progress = time.perf_counter()
    completed_subsets = last_subsets = last_bins = 0
    previous_graph_finished = started

    def progress(k: int, done: int, total: int, bins: int) -> None:
        nonlocal last_progress, completed_subsets, last_subsets, last_bins, previous_graph_finished
        if done:
            completed_subsets += subset_counts[done - 1]
        now = time.perf_counter()
        graph_seconds = now - previous_graph_finished
        previous_graph_finished = now
        by_graph = graph_interval > 0 and done % graph_interval == 0
        by_time = seconds > 0 and now - last_progress >= seconds
        if done not in (0, 1, total) and not by_graph and not by_time:
            return
        elapsed = now - started
        rate = completed_subsets / elapsed if elapsed > 0 else 0.0
        recent_seconds = now - last_progress
        recent_rate = (completed_subsets - last_subsets) / recent_seconds if recent_seconds > 0 else 0.0
        eta = f"{(total_subsets - completed_subsets) / rate:.1f}s" if rate > 0 else "unknown"
        print(f"[AttributedGraphlets] vocabulary progress k={k} graphs={done}/{total} "
              f"subsets={completed_subsets:,}/{total_subsets:,} bins={bins} "
              f"elapsed={elapsed:.1f}s subsets_per_second={rate:.1f} eta={eta} "
              f"new_bins_since_report={bins - last_bins} "
              f"recent_subsets_per_second={recent_rate:.1f}", flush=True)
        if done:
            current = train_graphs[done - 1]
            print(f"[AttributedGraphlets] last_graph_index={done - 1} "
                  f"nodes={node_counts[done - 1]} edges={current.number_of_edges()} "
                  f"subsets={subset_counts[done - 1]:,} wall_seconds={graph_seconds:.4f}", flush=True)
        cache = _canonicalize_attributed_tokens.cache_info()
        hits, misses = cache.hits - initial_cache.hits, cache.misses - initial_cache.misses
        hit_rate = hits / max(hits + misses, 1)
        print(f"[AttributedGraphlets] canonical_cache entries={cache.currsize}/{cache.maxsize} "
              f"fit_hits={hits:,} fit_misses={misses:,} hit_rate={hit_rate:.1%}", flush=True)
        last_progress = now
        last_subsets, last_bins = completed_subsets, bins

    basis = GraphletBasis.fit_from_graphs(train_graphs, settings, attributed=True,
                                         seed=int(config.get("seed", 42)),
                                         progress_callback=progress)
    print("[AttributedGraphlets] enumeration complete; validating and fingerprinting vocabulary", flush=True)
    info = metadata(basis)
    print(f"[AttributedGraphlets] attributed=True bins={info['width']} "
          f"including one overflow bin; fingerprint={info['fingerprint']}", flush=True)
    return basis
