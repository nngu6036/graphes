#!/usr/bin/env python
"""Evaluate generated molecular graphs.

This script is intended for outputs from
`scripts/generate_grapher_molecular_mixture_flow.py`, which writes:

  molecular_graphs.pkl   # all generated graphs, including invalid ones
  generated.smi          # valid SMILES only

Metrics:
  - validity: direct RDKit sanitization success, without post-hoc correction
  - validity_with_correction: deterministic valency-correction diagnostic
  - uniqueness_rate / novelty_rate on the configured valid-molecule source
  - FCD: optional, with reusable reference-statistics caching when supported
  - NSPDK MMD: HOG-Diff-compatible EDeN neighborhood-pair features by default

The previous deterministic hashed neighborhood-pair proxy is retained as an
explicit fallback/diagnostic backend.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from grapher.rewiring_mlp.evaluation.molecular_nspdk import eden_nspdk_mmd
from grapher.rewiring_mlp.molecular.graph_io import (
    graph_to_smiles,
    graphs_from_smiles,
    nx_to_rdkit_mol,
    read_smiles_file,
    require_rdkit,
)
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir, load_pickle, save_json


def _canonicalize_smiles(smiles: str) -> str | None:
    Chem = require_rdkit()
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
    except Exception:
        return None


def _graph_to_canonical_smiles_and_error(
    graph: nx.Graph,
) -> tuple[str | None, str | None]:
    Chem = require_rdkit()
    try:
        mol = nx_to_rdkit_mol(graph, sanitize=True)
        smi = str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
        return smi, None
    except Exception as exc:
        return None, type(exc).__name__


def _reduced_rdkit_bond_type(Chem: Any, bond: Any):
    """Return the next-lower RDKit bond type, or ``None`` to delete the bond.

    This is the deterministic bond-order correction convention used only for
    the *with-correction* validity diagnostic.  The generated graph itself is
    never modified, and ``validity_without_correction`` remains the primary
    pre-repair validity audit.
    """

    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.TRIPLE:
        return Chem.BondType.DOUBLE
    if bond_type == Chem.BondType.DOUBLE:
        return Chem.BondType.SINGLE
    if bond_type == Chem.BondType.AROMATIC:
        return Chem.BondType.SINGLE
    if bond_type == Chem.BondType.SINGLE:
        return None

    order = float(bond.GetBondTypeAsDouble())
    if order > 2.0:
        return Chem.BondType.DOUBLE
    if order > 1.0:
        return Chem.BondType.SINGLE
    return None


def _corrected_canonical_smiles_and_error(
    graph: nx.Graph,
    *,
    max_steps: int = 100,
) -> tuple[str | None, str | None, int]:
    """Return canonical SMILES after deterministic RDKit valency correction.

    The routine follows the correction convention commonly used by molecular
    graph-generation evaluators: repeatedly identify an atom-valence problem
    and lower the highest-order incident bond until the molecule sanitizes.
    Neutral tetravalent nitrogen (and trivalent oxygen) are first represented
    using the corresponding positive formal charge.  A single bond may be
    removed if no lower bond order exists.

    The operation is evaluation-only.  It does not mutate ``graph`` and does
    not affect the raw ``validity_without_correction`` metric.
    """

    Chem = require_rdkit()
    max_steps = max(int(max_steps), 0)
    try:
        rw_mol = Chem.RWMol(nx_to_rdkit_mol(graph, sanitize=False))
    except Exception as exc:
        return None, type(exc).__name__, 0

    last_error = "InvalidMolecule"
    for step in range(max_steps + 1):
        candidate = rw_mol.GetMol()
        try:
            candidate.UpdatePropertyCache(strict=False)
        except Exception:
            pass

        try:
            Chem.SanitizeMol(candidate)
            smiles = str(
                Chem.MolToSmiles(
                    candidate,
                    canonical=True,
                    isomericSmiles=False,
                )
            )
            if not smiles:
                return None, "EmptySMILES", step
            return smiles, None, step
        except Exception as exc:
            last_error = type(exc).__name__

        if step >= max_steps:
            break

        try:
            problems = list(Chem.DetectChemistryProblems(candidate))
        except Exception:
            problems = []

        atom_indices: list[int] = []
        for problem in problems:
            getter = getattr(problem, "GetAtomIdx", None)
            if getter is None:
                continue
            try:
                atom_indices.append(int(getter()))
            except Exception:
                continue
        if not atom_indices:
            return None, last_error, step

        atom_idx = min(atom_indices)
        atom = rw_mol.GetAtomWithIdx(atom_idx)
        incident_bonds = list(atom.GetBonds())
        total_bond_order = float(
            sum(float(bond.GetBondTypeAsDouble()) for bond in incident_bonds)
        )

        # Charged representations are standard for these otherwise-valid
        # local valence patterns and avoid needlessly deleting a bond.
        atomic_num = int(atom.GetAtomicNum())
        formal_charge = int(atom.GetFormalCharge())
        if atomic_num == 7 and formal_charge == 0 and abs(total_bond_order - 4.0) < 1e-8:
            atom.SetFormalCharge(1)
            continue
        if atomic_num == 8 and formal_charge == 0 and abs(total_bond_order - 3.0) < 1e-8:
            atom.SetFormalCharge(1)
            continue

        if not incident_bonds:
            return None, last_error, step

        def _bond_sort_key(bond: Any) -> tuple[float, int, int]:
            begin = int(bond.GetBeginAtomIdx())
            end = int(bond.GetEndAtomIdx())
            return (
                float(bond.GetBondTypeAsDouble()),
                -min(begin, end),
                -max(begin, end),
            )

        selected = max(incident_bonds, key=_bond_sort_key)
        begin = int(selected.GetBeginAtomIdx())
        end = int(selected.GetEndAtomIdx())
        reduced_type = _reduced_rdkit_bond_type(Chem, selected)
        rw_mol.RemoveBond(begin, end)
        if reduced_type is not None:
            rw_mol.AddBond(begin, end, reduced_type)

    return None, last_error, max_steps


def _load_graphs_from_path(path: str | Path) -> list[nx.Graph]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    obj = load_pickle(path)
    if isinstance(obj, dict):
        for key in ["graphs", "molecular_graphs", "generated_graphs"]:
            if key in obj:
                obj = obj[key]
                break
    graphs = list(obj)
    if not all(isinstance(g, nx.Graph) for g in graphs):
        raise TypeError(f"{path} does not contain a list of NetworkX graphs.")
    return graphs


def _load_smiles_as_graphs(path: str | Path) -> tuple[list[nx.Graph], list[str]]:
    smiles = read_smiles_file(path)
    canonical = []
    for smi in smiles:
        c = _canonicalize_smiles(smi)
        if c is not None:
            canonical.append(c)
    graphs, _errors = graphs_from_smiles(canonical, remove_h=True, kekulize=True)
    return graphs, canonical


def _resolve_generated_graphs(
    args: argparse.Namespace,
) -> tuple[list[nx.Graph], str, bool]:
    """Return graphs, source description, whether invalid graphs can be counted."""
    if args.generated_graphs:
        return (
            _load_graphs_from_path(args.generated_graphs),
            str(args.generated_graphs),
            True,
        )

    if args.generated_dir:
        d = Path(args.generated_dir)
        pkl = d / "molecular_graphs.pkl"
        if pkl.exists():
            return _load_graphs_from_path(pkl), str(pkl), True
        refined = d / "generated_graphs.pkl"
        if refined.exists():
            return _load_graphs_from_path(refined), str(refined), True
        smi = d / "generated.smi"
        if smi.exists():
            graphs, _ = _load_smiles_as_graphs(smi)
            return graphs, str(smi), False

    if args.generated_smiles:
        graphs, _ = _load_smiles_as_graphs(args.generated_smiles)
        return graphs, str(args.generated_smiles), False

    raise ValueError(
        "Provide --generated-dir, --generated-graphs, or --generated-smiles."
    )


def _load_reference_graphs_and_smiles(
    *,
    dataset_root: str | Path,
    dataset_name: str,
    split: str,
    smiles_file: str | None = None,
) -> tuple[list[nx.Graph], list[str], str]:
    if smiles_file:
        graphs, smiles = _load_smiles_as_graphs(smiles_file)
        return graphs, smiles, str(smiles_file)

    path = Path(dataset_root) / dataset_name / f"{split}.pkl"
    graphs = _load_graphs_from_path(path)
    smiles = []
    for g in graphs:
        smi = graph_to_smiles(g, canonical=True, sanitize=True)
        if smi is not None:
            c = _canonicalize_smiles(smi)
            if c is not None:
                smiles.append(c)
    return graphs, smiles, str(path)


def _validity_and_smiles(graphs: list[nx.Graph]) -> dict[str, Any]:
    valid_smiles: list[str] = []
    all_smiles: list[str | None] = []
    errors: Counter[str] = Counter()
    invalid_indices: list[int] = []

    for idx, graph in enumerate(graphs):
        smi, err = _graph_to_canonical_smiles_and_error(graph)
        all_smiles.append(smi)
        if smi is None:
            invalid_indices.append(idx)
            errors[str(err or "InvalidMolecule")] += 1
        else:
            valid_smiles.append(smi)

    num_graphs = len(graphs)
    num_valid = len(valid_smiles)
    unique_valid = sorted(set(valid_smiles))

    return {
        "all_smiles": all_smiles,
        "valid_smiles": valid_smiles,
        "unique_valid_smiles": unique_valid,
        "invalid_indices": invalid_indices,
        "conversion_error_counts": dict(errors),
        "num_graphs": num_graphs,
        "num_valid": num_valid,
        "num_invalid": num_graphs - num_valid,
        "validity_without_correction": num_valid / max(num_graphs, 1),
        "uniqueness_rate": len(unique_valid) / max(num_valid, 1),
        "unique_valid_count": len(unique_valid),
    }


def _validity_with_correction(
    graphs: list[nx.Graph],
    *,
    raw_smiles: list[str | None],
    max_steps: int,
) -> dict[str, Any]:
    """Evaluate deterministic valency-corrected validity.

    Raw-valid molecules are reused directly.  Correction is attempted only for
    molecules that fail the no-correction RDKit sanitization test.
    """

    if len(graphs) != len(raw_smiles):
        raise ValueError("graphs and raw_smiles must have identical lengths.")

    corrected_all_smiles: list[str | None] = []
    corrected_valid_smiles: list[str] = []
    corrected_indices: list[int] = []
    correction_steps: dict[int, int] = {}
    correction_errors: Counter[str] = Counter()

    for idx, (graph, raw_smi) in enumerate(zip(graphs, raw_smiles)):
        if raw_smi is not None:
            corrected_all_smiles.append(raw_smi)
            corrected_valid_smiles.append(raw_smi)
            correction_steps[idx] = 0
            continue

        smi, err, steps = _corrected_canonical_smiles_and_error(
            graph,
            max_steps=max_steps,
        )
        corrected_all_smiles.append(smi)
        correction_steps[idx] = int(steps)
        if smi is None:
            correction_errors[str(err or "CorrectionFailed")] += 1
            continue
        corrected_valid_smiles.append(smi)
        corrected_indices.append(idx)

    num_graphs = len(graphs)
    num_valid = len(corrected_valid_smiles)
    unique_valid = sorted(set(corrected_valid_smiles))
    raw_invalid_count = sum(smi is None for smi in raw_smiles)

    return {
        "all_smiles": corrected_all_smiles,
        "valid_smiles": corrected_valid_smiles,
        "unique_valid_smiles": unique_valid,
        "num_graphs": num_graphs,
        "num_valid": num_valid,
        "num_invalid": num_graphs - num_valid,
        "validity": num_valid / max(num_graphs, 1),
        "validity_with_correction": num_valid / max(num_graphs, 1),
        "num_corrected": len(corrected_indices),
        "corrected_indices": corrected_indices,
        "correction_steps": correction_steps,
        "correction_error_counts": dict(correction_errors),
        "correction_success_rate_on_raw_invalid": (
            len(corrected_indices) / raw_invalid_count
            if raw_invalid_count > 0
            else 1.0
        ),
        "unique_valid_count": len(unique_valid),
        "uniqueness_rate": len(unique_valid) / max(num_valid, 1),
    }


def _novelty(
    unique_valid_smiles: Iterable[str], train_smiles: Iterable[str]
) -> tuple[float | None, int]:
    unique = set(unique_valid_smiles)
    if not unique:
        return 0.0, 0
    train = set(train_smiles)
    if not train:
        return None, 0
    novel = [s for s in unique if s not in train]
    return len(novel) / max(len(unique), 1), len(novel)


def _hash_str(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def _node_label(graph: nx.Graph, node: int) -> str:
    data = graph.nodes[node]
    return str(int(data.get("atomic_num", data.get("atom_type", 0))))


def _edge_label(graph: nx.Graph, u: int, v: int) -> str:
    data = graph.edges[u, v]
    return str(int(data.get("bond_type", data.get("bond_order", 1))))


def _rooted_neighborhood_labels(graph: nx.Graph, radius: int) -> list[dict[int, str]]:
    labels: list[dict[int, str]] = []
    labels.append({int(v): _node_label(graph, int(v)) for v in graph.nodes()})

    for r in range(1, radius + 1):
        prev = labels[-1]
        cur: dict[int, str] = {}
        for v in graph.nodes():
            parts = []
            for u in graph.neighbors(v):
                a, b = int(v), int(u)
                parts.append(f"{_edge_label(graph, a, b)}:{prev[int(u)]}")
            parts.sort()
            raw = f"r={r}|self={prev[int(v)]}|nbr={'/'.join(parts)}"
            cur[int(v)] = _hash_str(raw)
        labels.append(cur)
    return labels


def nspdk_feature_counter(
    graph: nx.Graph,
    *,
    radius: int = 2,
    distance: int = 4,
    normalize: bool = True,
) -> Counter[str]:
    """Approximate NSPDK feature counter with rooted neighborhood-pair hashes."""
    g = nx.convert_node_labels_to_integers(nx.Graph(graph), ordering="sorted")
    features: Counter[str] = Counter()
    if g.number_of_nodes() == 0:
        return features

    labels_by_radius = _rooted_neighborhood_labels(g, radius)
    path_lengths = dict(nx.all_pairs_shortest_path_length(g, cutoff=distance))

    for r in range(radius + 1):
        labels = labels_by_radius[r]
        for u, dist_map in path_lengths.items():
            for v, d in dist_map.items():
                if int(u) > int(v):
                    continue
                if d > distance:
                    continue
                lu = labels[int(u)]
                lv = labels[int(v)]
                if lu <= lv:
                    key = f"r{r}|d{int(d)}|{lu}|{lv}"
                else:
                    key = f"r{r}|d{int(d)}|{lv}|{lu}"
                features[key] += 1

    if normalize:
        total = float(sum(features.values()))
        if total > 0:
            for k in list(features.keys()):
                features[k] = features[k] / total  # type: ignore[assignment]

    return features


def _mean_sparse(features: list[Counter[str]]) -> dict[str, float]:
    if not features:
        return {}
    acc: defaultdict[str, float] = defaultdict(float)
    for feat in features:
        for k, v in feat.items():
            acc[k] += float(v)
    n = float(len(features))
    return {k: v / n for k, v in acc.items()}


def _sparse_l2_squared(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return float(sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in keys))


def builtin_nspdk_mmd(
    reference_graphs: list[nx.Graph],
    generated_graphs: list[nx.Graph],
    *,
    radius: int = 2,
    distance: int = 4,
    normalize: bool = True,
) -> float | None:
    if not reference_graphs or not generated_graphs:
        return None
    ref_features = [
        nspdk_feature_counter(g, radius=radius, distance=distance, normalize=normalize)
        for g in reference_graphs
    ]
    gen_features = [
        nspdk_feature_counter(g, radius=radius, distance=distance, normalize=normalize)
        for g in generated_graphs
    ]
    return _sparse_l2_squared(_mean_sparse(ref_features), _mean_sparse(gen_features))


def compute_fcd(
    reference_smiles: list[str],
    generated_smiles: list[str],
    *,
    device: str = "auto",
    skip: bool = False,
    cache_dir: str | Path | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """Compute FCD, reusing reference ChemNet statistics when supported.

    HOG-Diff precomputes the reference FCD representation once and reuses it
    across generated batches.  We follow that protocol when the installed
    ``fcd_torch`` API exposes ``precalc``; older APIs fall back to direct calls.
    """

    if skip:
        return None, {"status": "skipped_by_user"}
    if not reference_smiles or not generated_smiles:
        return None, {
            "status": "not_computed_empty_smiles",
            "num_reference_smiles": len(reference_smiles),
            "num_generated_smiles": len(generated_smiles),
        }

    resolved_device = str(resolve_torch_device(device))

    def _reference_key() -> str:
        digest = hashlib.blake2b(digest_size=16)
        for smi in reference_smiles:
            digest.update(str(smi).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _cached_precalc(fcd: Any) -> tuple[Any | None, str | None]:
        if not hasattr(fcd, "precalc"):
            return None, None
        cache_path: Path | None = None
        if cache_dir is not None:
            import pickle

            cache_root = Path(cache_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_path = cache_root / f"fcd_reference_{_reference_key()}.pkl"
            if cache_path.exists():
                try:
                    with cache_path.open("rb") as handle:
                        return pickle.load(handle), str(cache_path)
                except Exception:
                    pass
        pref = fcd.precalc(reference_smiles)
        if cache_path is not None:
            try:
                import pickle

                with cache_path.open("wb") as handle:
                    pickle.dump(pref, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                pass
        return pref, None if cache_path is None else str(cache_path)

    errors: list[str] = []
    for import_path in ("fcd_torch", "fcd_torch.fcd"):
        try:
            if import_path == "fcd_torch":
                from fcd_torch import FCD  # type: ignore
            else:
                from fcd_torch.fcd import FCD  # type: ignore
            fcd = FCD(device=resolved_device)
            pref, cache_path = _cached_precalc(fcd)
            if pref is not None:
                try:
                    value = fcd(gen=generated_smiles, pref=pref)
                except TypeError:
                    value = fcd(generated_smiles, pref=pref)
                return float(value), {
                    "status": "ok",
                    "backend": f"{import_path}.FCD.precalc",
                    "reference_cache": cache_path,
                }
            value = fcd(reference_smiles, generated_smiles)
            return float(value), {"status": "ok", "backend": f"{import_path}.FCD"}
        except Exception as exc:
            errors.append(f"{import_path}: {type(exc).__name__}")

    try:
        from fcd_torch import get_fcd  # type: ignore

        value = get_fcd(generated_smiles, reference_smiles, device=resolved_device)
        return float(value), {"status": "ok", "backend": "fcd_torch.get_fcd"}
    except Exception as exc:
        errors.append(f"fcd_torch.get_fcd: {type(exc).__name__}")
        return None, {
            "status": "not_available_or_failed",
            "backend_attempts": [
                "fcd_torch.FCD.precalc",
                "fcd_torch.fcd.FCD.precalc",
                "fcd_torch.get_fcd",
            ],
            "errors": errors,
            "message": "Install a compatible fcd_torch package to compute FCD, or pass --skip-fcd.",
        }


def _graphs_from_metric_smiles(smiles: Iterable[str]) -> list[nx.Graph]:
    """Build lightweight attributed graphs for kernel metrics from valid SMILES."""

    Chem = require_rdkit()
    bond_map = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    graphs: list[nx.Graph] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        graph = nx.Graph()
        for atom in mol.GetAtoms():
            atomic_num = int(atom.GetAtomicNum())
            graph.add_node(int(atom.GetIdx()), atomic_num=atomic_num, atom_type=atomic_num)
        supported = True
        for bond in mol.GetBonds():
            bond_type = bond_map.get(bond.GetBondType())
            if bond_type is None:
                supported = False
                break
            graph.add_edge(
                int(bond.GetBeginAtomIdx()),
                int(bond.GetEndAtomIdx()),
                bond_type=int(bond_type),
                bond_order=float(bond.GetBondTypeAsDouble()),
            )
        if supported and graph.number_of_nodes() > 0:
            graphs.append(nx.convert_node_labels_to_integers(graph, ordering="sorted"))
    return graphs


def _select_valid_graphs(
    graphs: list[nx.Graph], smiles: list[str | None]
) -> list[nx.Graph]:
    return [g for g, smi in zip(graphs, smiles) if smi is not None]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    generated_graphs, generated_source, validity_denominator_complete = (
        _resolve_generated_graphs(args)
    )

    reference_graphs, reference_smiles, reference_source = (
        _load_reference_graphs_and_smiles(
            dataset_root=args.dataset_root,
            dataset_name=args.dataset,
            split=args.reference_split,
            smiles_file=args.reference_smiles,
        )
    )
    train_graphs, train_smiles, train_source = _load_reference_graphs_and_smiles(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset,
        split=args.train_split,
        smiles_file=args.train_smiles,
    )

    # Optional limits for speed/debugging.
    if args.max_generated is not None:
        generated_graphs = generated_graphs[: int(args.max_generated)]
    if args.max_reference is not None:
        reference_graphs = reference_graphs[: int(args.max_reference)]
        reference_smiles = reference_smiles[: int(args.max_reference)]
    if args.max_train is not None:
        train_graphs = train_graphs[: int(args.max_train)]
        train_smiles = train_smiles[: int(args.max_train)]

    valid_info = _validity_and_smiles(generated_graphs)
    corrected_valid_info = _validity_with_correction(
        generated_graphs,
        raw_smiles=valid_info["all_smiles"],
        max_steps=args.correction_max_steps,
    )
    hogdiff_compat = bool(getattr(args, "hogdiff_compatible_metrics", False))
    metric_source = (
        "corrected_valid"
        if hogdiff_compat
        else str(getattr(args, "metric_molecule_source", "raw_valid")).lower()
    )
    if metric_source not in {"raw_valid", "corrected_valid"}:
        raise ValueError("--metric-molecule-source must be raw_valid or corrected_valid.")

    raw_valid_graphs = _select_valid_graphs(generated_graphs, valid_info["all_smiles"])
    corrected_metric_graphs = _graphs_from_metric_smiles(
        corrected_valid_info["valid_smiles"]
    )
    metric_smiles = (
        corrected_valid_info["valid_smiles"]
        if metric_source == "corrected_valid"
        else valid_info["valid_smiles"]
    )
    metric_graphs = (
        corrected_metric_graphs if metric_source == "corrected_valid" else raw_valid_graphs
    )
    metric_unique_smiles = sorted(set(metric_smiles))
    novelty_rate, novel_count = _novelty(metric_unique_smiles, train_smiles)
    metric_uniqueness_rate = len(metric_unique_smiles) / max(len(metric_smiles), 1)

    # Retain the legacy proxy for debugging, but use HOG-Diff's EDeN feature
    # protocol as the benchmark NSPDK implementation by default.
    nspdk_proxy_all = builtin_nspdk_mmd(
        reference_graphs,
        generated_graphs,
        radius=int(getattr(args, "nspdk_radius", 2)),
        distance=int(getattr(args, "nspdk_distance", 4)),
        normalize=not bool(getattr(args, "no_nspdk_normalize", False)),
    )
    nspdk_proxy_valid = builtin_nspdk_mmd(
        reference_graphs,
        metric_graphs,
        radius=int(getattr(args, "nspdk_radius", 2)),
        distance=int(getattr(args, "nspdk_distance", 4)),
        normalize=not bool(getattr(args, "no_nspdk_normalize", False)),
    )
    nspdk_backend = (
        "eden"
        if hogdiff_compat
        else str(getattr(args, "nspdk_backend", "eden")).lower()
    )
    nspdk_bond_label_mode = (
        "hogdiff"
        if hogdiff_compat
        else str(getattr(args, "nspdk_bond_label_mode", "hogdiff"))
    )
    metric_cache_dir = getattr(args, "metric_cache_dir", None)
    if metric_cache_dir is None:
        metric_cache_dir = str(Path(args.dataset_root) / args.dataset / "evaluation_cache")
    if nspdk_backend == "eden":
        nspdk_value = eden_nspdk_mmd(
            reference_graphs,
            metric_graphs,
            complexity=int(getattr(args, "nspdk_complexity", 4)),
            cache_dir=metric_cache_dir,
            bond_label_mode=nspdk_bond_label_mode,
        )
    elif nspdk_backend == "proxy":
        nspdk_value = nspdk_proxy_valid
    else:
        raise ValueError("--nspdk-backend must be eden or proxy.")

    # HOG-Diff evaluates distributional metrics after its deterministic validity
    # correction. GraphER defaults to strict raw-valid molecules, but this flag
    # and --metric-molecule-source make cross-codebase reproduction explicit.
    fcd_use_corrected = hogdiff_compat or bool(getattr(args, "fcd_use_corrected", False))
    fcd_generated_smiles = (
        corrected_valid_info["valid_smiles"]
        if fcd_use_corrected
        else metric_smiles
    )
    fcd_value, fcd_info = compute_fcd(
        reference_smiles,
        fcd_generated_smiles,
        device=getattr(args, "fcd_device", "auto"),
        skip=bool(getattr(args, "skip_fcd", False)),
        cache_dir=metric_cache_dir,
    )
    if args.require_fcd and fcd_value is None:
        raise RuntimeError(
            "FCD was requested but could not be computed. "
            f"Backend status: {fcd_info}. Install a compatible fcd_torch package "
            "or omit --require-fcd."
        )

    metrics = {
        # Benchmark headline: direct RDKit validity, matching the no-repair
        # molecular protocol used in the paper and HOG-Diff's reported validity.
        "validity": float(valid_info["validity_without_correction"]),
        "validity_with_correction": float(
            corrected_valid_info["validity_with_correction"]
        ),
        # Primary implementation audit: validity before any repair/correction.
        "validity_without_correction": float(
            valid_info["validity_without_correction"]
        ),
        "fcd": fcd_value,
        "num_generated_graphs": int(len(generated_graphs)),
        "num_valid_generated_molecules": int(valid_info["num_valid"]),
        "num_invalid_generated_molecules": int(valid_info["num_invalid"]),
        "num_valid_generated_molecules_with_correction": int(
            corrected_valid_info["num_valid"]
        ),
        "num_invalid_generated_molecules_after_correction": int(
            corrected_valid_info["num_invalid"]
        ),
        "num_molecules_corrected_to_valid": int(
            corrected_valid_info["num_corrected"]
        ),
        "correction_success_rate_on_raw_invalid": float(
            corrected_valid_info["correction_success_rate_on_raw_invalid"]
        ),
        "metric_molecule_source": metric_source,
        "hogdiff_compatible_distribution_metrics": hogdiff_compat,
        "uniqueness_rate": float(metric_uniqueness_rate),
        "unique_valid_count": int(len(metric_unique_smiles)),
        "novelty_rate": None if novelty_rate is None else float(novelty_rate),
        "novel_unique_valid_count": int(novel_count),
        "nspdk_mmd": None if nspdk_value is None else float(nspdk_value),
        "nspdk_mmd_valid_only": None if nspdk_value is None else float(nspdk_value),
        "nspdk_proxy_mmd_all_generated": (
            None if nspdk_proxy_all is None else float(nspdk_proxy_all)
        ),
        "nspdk_proxy_mmd_metric_source": (
            None if nspdk_proxy_valid is None else float(nspdk_proxy_valid)
        ),
        "fcd_num_reference_molecules": int(len(reference_smiles)),
        "fcd_num_valid_generated_molecules": int(len(fcd_generated_smiles)),
        "fcd_generated_smiles_source": (
            "valid_with_correction"
            if fcd_use_corrected
            else metric_source
        ),
    }

    report = {
        "metrics": metrics,
        "paths": {
            "generated_source": generated_source,
            "reference_source": reference_source,
            "train_source": train_source,
        },
        "protocol": {
            "validity_denominator_complete": bool(validity_denominator_complete),
            "validity_note": (
                "Validity denominator includes all generated graphs."
                if validity_denominator_complete
                else "Generated source was SMILES-only, so invalid generated graphs absent from the file cannot be counted. Prefer molecular_graphs.pkl."
            ),
            "canonical_smiles": True,
            "isomeric_smiles": False,
            "validity": "RDKit Mol construction + Chem.SanitizeMol, with no correction or repair.",
            "validity_with_correction": (
                "Deterministic evaluation-only valency correction diagnostic. "
                "Raw-valid molecules are retained; for raw-invalid molecules the "
                "highest-order bond incident to an atom-valence problem is lowered "
                "until sanitization succeeds or the correction budget is exhausted."
            ),
            "validity_without_correction": "RDKit Mol construction + Chem.SanitizeMol, no valency correction or edge resampling.",
            "correction_max_steps": int(args.correction_max_steps),
            "metric_molecule_source": metric_source,
            "hogdiff_compatible_distribution_metrics": hogdiff_compat,
            "uniqueness_definition": "unique canonical SMILES / molecules in metric_molecule_source",
            "novelty_definition": "unique canonical SMILES not in training set / unique canonical SMILES in metric_molecule_source",
            "nspdk": {
                "backend": (
                    "hogdiff_eden_neighborhood_pair_linear_mmd"
                    if nspdk_backend == "eden"
                    else "builtin_hashed_neighborhood_pair_proxy"
                ),
                "complexity": int(getattr(args, "nspdk_complexity", 4)),
                "bond_label_mode": nspdk_bond_label_mode,
                "proxy_radius": int(getattr(args, "nspdk_radius", 2)),
                "proxy_distance": int(getattr(args, "nspdk_distance", 4)),
                "proxy_normalized_features": bool(
                    not getattr(args, "no_nspdk_normalize", False)
                ),
                "metric_cache_dir": str(metric_cache_dir),
            },
            "fcd": {
                **fcd_info,
                "generated_smiles_source": (
                    "valid_with_correction"
                    if fcd_use_corrected
                    else metric_source
                ),
            },
        },
        "conversion_error_counts": valid_info["conversion_error_counts"],
        "invalid_indices": valid_info["invalid_indices"],
        "correction": {
            "corrected_indices": corrected_valid_info["corrected_indices"],
            "correction_steps_by_index": corrected_valid_info["correction_steps"],
            "correction_error_counts": corrected_valid_info[
                "correction_error_counts"
            ],
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated molecular graphs from GraphER + mixture CatFlow."
    )

    source = parser.add_argument_group("Generated molecules")
    source.add_argument(
        "--generated-dir",
        default=None,
        help="Directory containing molecular_graphs.pkl and/or generated.smi.",
    )
    source.add_argument(
        "--generated-graphs",
        default=None,
        help="Path to molecular_graphs.pkl. Preferred for validity.",
    )
    source.add_argument(
        "--generated-smiles",
        default=None,
        help="Path to generated SMILES. Validity will only cover listed SMILES.",
    )

    ref = parser.add_argument_group("Reference / train molecules")
    ref.add_argument("--dataset-root", default="outputs/datasets")
    ref.add_argument("--dataset", default="qm9_attributed")
    ref.add_argument("--reference-split", default="test")
    ref.add_argument("--train-split", default="train")
    ref.add_argument("--reference-smiles", default=None)
    ref.add_argument("--train-smiles", default=None)

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to generated-dir/evaluation if --generated-dir is used.",
    )
    parser.add_argument("--max-generated", type=int, default=None)
    parser.add_argument("--max-reference", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)

    parser.add_argument(
        "--hogdiff-compatible-metrics",
        action="store_true",
        help=(
            "Use HOG-Diff's corrected-molecule source for uniqueness/novelty/FCD/NSPDK, "
            "the EDeN NSPDK backend, and HOG-Diff bond labels. The headline validity "
            "remains strict raw RDKit validity so GraphER's no-repair protocol is not weakened."
        ),
    )
    parser.add_argument(
        "--metric-molecule-source",
        choices=["raw_valid", "corrected_valid"],
        default="raw_valid",
        help=(
            "Molecule subset used for uniqueness, novelty and NSPDK. "
            "Use corrected_valid to reproduce HOG-Diff's post-correction "
            "distributional-metric convention; raw_valid is the strict GraphER protocol."
        ),
    )
    parser.add_argument(
        "--nspdk-backend",
        choices=["eden", "proxy"],
        default="eden",
        help="EDeN/HOG-Diff-compatible NSPDK (default) or the legacy hashed proxy.",
    )
    parser.add_argument("--nspdk-complexity", type=int, default=4)
    parser.add_argument(
        "--nspdk-bond-label-mode",
        choices=["hogdiff", "categorical"],
        default="hogdiff",
        help=(
            "hogdiff reproduces int(RDKit bond order), including aromatic 1.5 -> 1; "
            "categorical preserves GraphER's bond-type ids."
        ),
    )
    parser.add_argument("--nspdk-radius", type=int, default=2)
    parser.add_argument("--nspdk-distance", type=int, default=4)
    parser.add_argument("--no-nspdk-normalize", action="store_true")
    parser.add_argument(
        "--metric-cache-dir",
        default=None,
        help=(
            "Cache reusable EDeN/FCD reference statistics here. Defaults to "
            "<dataset-root>/<dataset>/evaluation_cache."
        ),
    )

    parser.add_argument("--skip-fcd", action="store_true")
    parser.add_argument("--fcd-device", default="auto")
    parser.add_argument(
        "--fcd-use-corrected",
        action="store_true",
        help=(
            "Compute FCD from molecules valid after deterministic valency "
            "correction. This matches HOG-Diff's distributional-metric source. "
            "Otherwise FCD uses --metric-molecule-source."
        ),
    )
    parser.add_argument(
        "--require-fcd",
        action="store_true",
        help="Fail instead of reporting fcd=None when the FCD backend is unavailable.",
    )
    parser.add_argument(
        "--correction-max-steps",
        type=int,
        default=100,
        help="Maximum deterministic bond-order corrections attempted per molecule.",
    )

    args = parser.parse_args()

    report = evaluate(args)

    if args.output_dir is not None:
        out_dir = ensure_dir(args.output_dir)
    elif args.generated_dir is not None:
        out_dir = ensure_dir(Path(args.generated_dir) / "evaluation")
    else:
        out_dir = ensure_dir("outputs/molecular/evaluation")

    save_json(report, out_dir / "molecular_evaluation_metrics.json")

    valid_smiles = []
    corrected_valid_smiles = []
    # Recompute canonical valid SMILES for saving in graph order.
    generated_graphs, _, _ = _resolve_generated_graphs(args)
    if args.max_generated is not None:
        generated_graphs = generated_graphs[: int(args.max_generated)]
    for g in generated_graphs:
        raw_smi, _ = _graph_to_canonical_smiles_and_error(g)
        if raw_smi is not None:
            valid_smiles.append(raw_smi)
            corrected_valid_smiles.append(raw_smi)
            continue
        corrected_smi, _, _ = _corrected_canonical_smiles_and_error(
            g,
            max_steps=args.correction_max_steps,
        )
        if corrected_smi is not None:
            corrected_valid_smiles.append(corrected_smi)
    with (out_dir / "valid_generated.smi").open("w", encoding="utf-8") as f:
        for smi in valid_smiles:
            f.write(smi + "\n")
    with (out_dir / "corrected_valid_generated.smi").open(
        "w", encoding="utf-8"
    ) as f:
        for smi in corrected_valid_smiles:
            f.write(smi + "\n")

    print("Molecular evaluation")
    for key, value in report["metrics"].items():
        print(f"  {key}: {value}")
    print(f"Saved metrics to: {out_dir / 'molecular_evaluation_metrics.json'}")
    print(f"Saved valid SMILES to: {out_dir / 'valid_generated.smi'}")
    print(
        "Saved validity-corrected SMILES to: "
        f"{out_dir / 'corrected_valid_generated.smi'}"
    )


if __name__ == "__main__":
    main()
