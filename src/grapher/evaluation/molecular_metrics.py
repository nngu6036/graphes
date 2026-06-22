from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import networkx as nx
import numpy as np

from grapher.molecules.representation import MolecularConversion, molecular_graph_to_rdkit
from grapher.generation.molecular_rewiring import edge_type_value, node_type_value


def direct_molecular_conversions(
    graphs: Sequence[nx.Graph],
    *,
    isomeric_smiles: bool = False,
) -> list[MolecularConversion]:
    return [
        molecular_graph_to_rdkit(graph, sanitize=True, isomeric_smiles=isomeric_smiles)
        for graph in graphs
    ]


def validity_without_correction(conversions: Sequence[MolecularConversion]) -> float:
    if not conversions:
        return 0.0
    return float(np.mean([bool(item.valid) for item in conversions]))


def valid_smiles(
    conversions: Sequence[MolecularConversion],
    *,
    remove_explicit_hydrogens: bool = True,
    isomeric_smiles: bool = False,
) -> list[str]:
    smiles: list[str] = []
    try:
        from rdkit import Chem
    except Exception:
        return [str(item.smiles) for item in conversions if item.valid and item.smiles]
    for item in conversions:
        if not item.valid or item.mol is None:
            continue
        mol = item.mol
        try:
            if remove_explicit_hydrogens:
                mol = Chem.RemoveHs(mol, sanitize=True)
            smiles.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=bool(isomeric_smiles)))
        except Exception:
            if item.smiles:
                smiles.append(str(item.smiles))
    return smiles


def molecular_uniqueness(smiles: Sequence[str]) -> float:
    values = [s for s in smiles if s]
    if not values:
        return 0.0
    return float(len(set(values)) / len(values))


def molecular_novelty(generated_smiles: Sequence[str], train_smiles: Sequence[str]) -> float:
    values = [s for s in generated_smiles if s]
    if not values:
        return 0.0
    train = set(s for s in train_smiles if s)
    return float(sum(1 for s in values if s not in train) / len(values))


def _hash_token(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % int(dim)


def _heavy_atom_subgraph(graph: nx.Graph) -> nx.Graph:
    keep = []
    for node in graph.nodes():
        try:
            if int(node_type_value(graph, int(node))) != 1:
                keep.append(node)
        except Exception:
            keep.append(node)
    return nx.convert_node_labels_to_integers(graph.subgraph(keep).copy(), ordering="sorted")


def _builtin_molecular_features(
    graph: nx.Graph,
    *,
    complexity: int,
    dim: int,
    heavy_atoms_only: bool,
) -> np.ndarray:
    graph = _heavy_atom_subgraph(graph) if heavy_atoms_only else nx.convert_node_labels_to_integers(nx.Graph(graph), ordering="sorted")
    vec = np.zeros(int(dim), dtype=np.float64)
    if graph.number_of_nodes() == 0:
        return vec
    for node in graph.nodes():
        try:
            atom = node_type_value(graph, int(node))
        except Exception:
            atom = -1
        degree = int(graph.degree(int(node)))
        token = f"atom:{atom}:degree:{degree}"
        vec[_hash_token(token, dim)] += 1.0
    for u, v in graph.edges():
        try:
            a = node_type_value(graph, int(u))
            b = node_type_value(graph, int(v))
            bond = edge_type_value(graph, int(u), int(v))
        except Exception:
            a, b, bond = -1, -1, -1
        left, right = sorted((int(a), int(b)))
        token = f"edge:{left}:{bond}:{right}"
        vec[_hash_token(token, dim)] += 1.0
    # Rooted neighborhood-pair features inspired by NSPDK.  This is not a full
    # replacement for EDeN's exact NSPDK vectorizer, but it is deterministic and
    # records the fallback backend in evaluation metadata.
    max_radius = max(int(complexity), 1)
    labels = {}
    for node in graph.nodes():
        try:
            labels[int(node)] = str(node_type_value(graph, int(node)))
        except Exception:
            labels[int(node)] = "?"
    for radius in range(1, max_radius + 1):
        rooted: dict[int, str] = {}
        for node in graph.nodes():
            frontier = nx.single_source_shortest_path_length(graph, int(node), cutoff=radius)
            atoms = []
            bonds = []
            for other, dist in frontier.items():
                atoms.append((int(dist), labels[int(other)]))
            for a, b in graph.subgraph(frontier.keys()).edges():
                try:
                    bond = edge_type_value(graph, int(a), int(b))
                except Exception:
                    bond = -1
                bonds.append(tuple(sorted((labels[int(a)], str(bond), labels[int(b)]))))
            rooted[int(node)] = f"r{radius}:A{sorted(atoms)}:B{sorted(bonds)}"
            vec[_hash_token("root:" + rooted[int(node)], dim)] += 1.0
        nodes = sorted(rooted)
        for i, u in enumerate(nodes):
            lengths = nx.single_source_shortest_path_length(graph, int(u), cutoff=max_radius)
            for v in nodes[i + 1 :]:
                if int(v) not in lengths:
                    continue
                distance = int(lengths[int(v)])
                token = f"pair:r{radius}:d{distance}:{rooted[int(u)]}|{rooted[int(v)]}"
                vec[_hash_token(token, dim)] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    def pdist2(a, b):
        aa = np.sum(a * a, axis=1, keepdims=True)
        bb = np.sum(b * b, axis=1, keepdims=True).T
        return np.maximum(aa + bb - 2.0 * a @ b.T, 0.0)
    z = np.vstack([x, y])
    d = pdist2(z, z)
    upper = d[np.triu_indices_from(d, k=1)]
    sigma2 = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
    sigma2 = max(sigma2, 1e-12)
    kxx = np.exp(-pdist2(x, x) / (2.0 * sigma2))
    kyy = np.exp(-pdist2(y, y) / (2.0 * sigma2))
    kxy = np.exp(-pdist2(x, y) / (2.0 * sigma2))
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def _eden_nspdk_features(graphs: Sequence[nx.Graph], *, complexity: int, heavy_atoms_only: bool) -> np.ndarray:
    # EDeN's API has varied across versions.  We try the common vectorizer and
    # surface any failure to the caller, which will either fall back or raise.
    from eden.graph import Vectorizer  # type: ignore

    values = [_heavy_atom_subgraph(g) if heavy_atoms_only else nx.Graph(g) for g in graphs]
    vectorizer = Vectorizer(complexity=max(int(complexity), 1))
    matrix = vectorizer.transform(values)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


def compute_nspdk_mmd(
    reference_graphs: Sequence[nx.Graph],
    generated_graphs: Sequence[nx.Graph],
    *,
    complexity: int = 4,
    backend: str = "auto",
    heavy_atoms_only: bool = True,
) -> tuple[float, dict[str, Any]]:
    backend = str(backend).lower()
    if not reference_graphs or not generated_graphs:
        return float("nan"), {"backend": backend, "status": "empty_population"}
    if backend in {"auto", "eden"}:
        try:
            ref = _eden_nspdk_features(reference_graphs, complexity=complexity, heavy_atoms_only=heavy_atoms_only)
            gen = _eden_nspdk_features(generated_graphs, complexity=complexity, heavy_atoms_only=heavy_atoms_only)
            return _mmd_rbf(ref, gen), {"backend": "eden", "complexity": int(complexity), "heavy_atoms_only": bool(heavy_atoms_only)}
        except Exception as exc:
            if backend == "eden":
                raise
            fallback_reason = f"{type(exc).__name__}:{exc}"
    elif backend != "builtin":
        raise ValueError("backend must be one of auto, eden, builtin")
    else:
        fallback_reason = "requested_builtin"
    dim = 2048
    ref = np.asarray([_builtin_molecular_features(g, complexity=complexity, dim=dim, heavy_atoms_only=heavy_atoms_only) for g in reference_graphs], dtype=np.float64)
    gen = np.asarray([_builtin_molecular_features(g, complexity=complexity, dim=dim, heavy_atoms_only=heavy_atoms_only) for g in generated_graphs], dtype=np.float64)
    return _mmd_rbf(ref, gen), {"backend": "builtin", "complexity": int(complexity), "feature_dim": dim, "heavy_atoms_only": bool(heavy_atoms_only), "fallback_reason": fallback_reason}


def compute_fcd(
    reference_smiles: Sequence[str],
    generated_smiles: Sequence[str],
    *,
    device: str = "cpu",
    n_jobs: int = 1,
    batch_size: int = 512,
) -> tuple[float | None, dict[str, Any]]:
    ref = [s for s in reference_smiles if s]
    gen = [s for s in generated_smiles if s]
    if not ref or not gen:
        return None, {"status": "empty_smiles_population", "num_reference": len(ref), "num_generated": len(gen)}
    try:
        import fcd_torch  # type: ignore
    except Exception as exc:
        return None, {"status": "missing_fcd_torch", "error": f"{type(exc).__name__}:{exc}", "num_reference": len(ref), "num_generated": len(gen)}
    attempts = []
    for attr in ("FCD", "FCDMetric"):
        cls = getattr(fcd_torch, attr, None)
        if cls is None:
            continue
        try:
            metric = cls(device=device, n_jobs=int(n_jobs), batch_size=int(batch_size))
            score = metric(ref, gen)
            return float(score), {"status": "ok", "backend": f"fcd_torch.{attr}", "num_reference": len(ref), "num_generated": len(gen)}
        except Exception as exc:
            attempts.append(f"{attr}:{type(exc).__name__}:{exc}")
    # Try common functional names if present.
    for attr in ("get_fcd", "calculate_fcd", "compute_fcd"):
        fn = getattr(fcd_torch, attr, None)
        if fn is None:
            continue
        try:
            score = fn(ref, gen, device=device, n_jobs=int(n_jobs), batch_size=int(batch_size))
            return float(score), {"status": "ok", "backend": f"fcd_torch.{attr}", "num_reference": len(ref), "num_generated": len(gen)}
        except Exception as exc:
            attempts.append(f"{attr}:{type(exc).__name__}:{exc}")
    return None, {"status": "fcd_torch_api_unavailable", "attempts": attempts, "num_reference": len(ref), "num_generated": len(gen)}
