from __future__ import annotations

"""Molecular NSPDK evaluation compatible with the HOG-Diff/EDeN protocol.

HOG-Diff vectorizes labeled molecular graphs with EDeN's neighborhood-pair
features (``complexity=4, discrete=True``) and uses a linear-kernel biased MMD.
For a linear kernel this MMD is exactly the squared Euclidean distance between
mean feature vectors, so we compute that equivalent form without materializing
quadratic Gram matrices.
"""

from pathlib import Path
from typing import Iterable
import hashlib
import os
import pickle

import networkx as nx
import numpy as np
from scipy import sparse

from grapher.rewiring_mlp.evaluation.eden import vectorize


def _atomic_symbol(atomic_num: int) -> str:
    try:
        from rdkit import Chem  # type: ignore

        return str(Chem.GetPeriodicTable().GetElementSymbol(int(atomic_num)))
    except Exception:
        return str(int(atomic_num))


def to_eden_molecular_graph(
    graph: nx.Graph,
    *,
    bond_label_mode: str = "hogdiff",
) -> nx.Graph:
    """Convert a GraphER molecular graph to EDeN label conventions.

    ``hogdiff`` exactly follows HOG-Diff's ``mols_to_nx`` convention:
    ``label=int(bond.GetBondTypeAsDouble())``.  In particular an aromatic
    RDKit bond (1.5) becomes label 1.  ``categorical`` instead preserves
    GraphER's categorical bond id, which is useful as an attributed-kernel
    diagnostic but is not the HOG-Diff benchmark protocol.
    """

    bond_label_mode = str(bond_label_mode).lower()
    if bond_label_mode not in {"hogdiff", "categorical"}:
        raise ValueError("bond_label_mode must be 'hogdiff' or 'categorical'.")

    out = nx.Graph()
    for node, data in graph.nodes(data=True):
        atomic_num = int(data.get("atomic_num", data.get("atom_type", 0)))
        out.add_node(int(node), label=_atomic_symbol(atomic_num))
    for u, v, data in graph.edges(data=True):
        if bond_label_mode == "categorical":
            bond_label = int(data.get("bond_type", data.get("bond_order", 1)))
        else:
            bond_type = int(data.get("bond_type", 1))
            default_order = {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}.get(bond_type, float(bond_type))
            bond_order = float(data.get("bond_order", default_order))
            bond_label = int(bond_order)
        out.add_edge(int(u), int(v), label=bond_label)
    return nx.convert_node_labels_to_integers(out, ordering="sorted")


def _mean_sparse_rows(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    if matrix.shape[0] == 0:
        return sparse.csr_matrix((1, matrix.shape[1]), dtype=np.float64)
    mean = matrix.mean(axis=0)
    return sparse.csr_matrix(mean, dtype=np.float64)


def _reference_cache_key(
    graphs: Iterable[nx.Graph],
    *,
    complexity: int,
    bond_label_mode: str,
) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(
        f"complexity={int(complexity)}|bond_label_mode={bond_label_mode}|"
        f"hashseed={os.getenv('PYTHONHASHSEED','unset')}".encode()
    )
    for graph in graphs:
        g = to_eden_molecular_graph(graph, bond_label_mode=bond_label_mode)
        nodes = sorted((int(n), str(d.get("label", ""))) for n, d in g.nodes(data=True))
        edges = sorted(
            (min(int(u), int(v)), max(int(u), int(v)), str(d.get("label", "")))
            for u, v, d in g.edges(data=True)
        )
        digest.update(repr((nodes, edges)).encode())
    return digest.hexdigest()


def _load_or_compute_reference_mean(
    reference_graphs: list[nx.Graph],
    *,
    complexity: int,
    cache_dir: str | Path | None,
    bond_label_mode: str,
) -> sparse.csr_matrix:
    cache_path: Path | None = None
    # EDeN's feature hashing uses Python's hash().  Cross-process cache reuse is
    # only safe when PYTHONHASHSEED is explicitly fixed, exactly as HOG-Diff's
    # evaluator assumes.  With an unset seed we still compute the exact metric
    # but deliberately skip persistent reference caching.
    if cache_dir is not None and os.getenv("PYTHONHASHSEED") is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        key = _reference_cache_key(reference_graphs, complexity=complexity, bond_label_mode=bond_label_mode)
        cache_path = cache_root / f"nspdk_eden_reference_mean_{key}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if sparse.issparse(cached):
                return sparse.csr_matrix(cached)

    ref_features = vectorize(
        [
            to_eden_molecular_graph(g, bond_label_mode=bond_label_mode)
            for g in reference_graphs
        ],
        complexity=int(complexity),
        discrete=True,
    )
    mean = _mean_sparse_rows(ref_features)
    if cache_path is not None:
        with cache_path.open("wb") as handle:
            pickle.dump(mean, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return mean


def eden_nspdk_mmd(
    reference_graphs: list[nx.Graph],
    generated_graphs: list[nx.Graph],
    *,
    complexity: int = 4,
    cache_dir: str | Path | None = None,
    bond_label_mode: str = "hogdiff",
) -> float | None:
    """Return the HOG-Diff-compatible EDeN NSPDK linear-kernel MMD."""

    reference_graphs = [g for g in reference_graphs if g.number_of_nodes() > 0]
    generated_graphs = [g for g in generated_graphs if g.number_of_nodes() > 0]
    if not reference_graphs or not generated_graphs:
        return None

    ref_mean = _load_or_compute_reference_mean(
        reference_graphs,
        complexity=int(complexity),
        cache_dir=cache_dir,
        bond_label_mode=bond_label_mode,
    )
    gen_features = vectorize(
        [
            to_eden_molecular_graph(g, bond_label_mode=bond_label_mode)
            for g in generated_graphs
        ],
        complexity=int(complexity),
        discrete=True,
    )
    gen_mean = _mean_sparse_rows(gen_features)

    # Feature widths are determined by EDeN's hash dimensionality and should be
    # identical.  Padding is defensive for unusual custom Vectorizer settings.
    width = max(ref_mean.shape[1], gen_mean.shape[1])
    if ref_mean.shape[1] < width:
        ref_mean = sparse.hstack(
            [ref_mean, sparse.csr_matrix((1, width - ref_mean.shape[1]))],
            format="csr",
        )
    if gen_mean.shape[1] < width:
        gen_mean = sparse.hstack(
            [gen_mean, sparse.csr_matrix((1, width - gen_mean.shape[1]))],
            format="csr",
        )
    delta = ref_mean - gen_mean
    return float(delta.multiply(delta).sum())
