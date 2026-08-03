#!/usr/bin/env python
"""Evaluate generated molecular graphs.

This script is intended for outputs from
`scripts/generate_grapher_molecular_mixture_flow.py`, which writes:

  molecular_graphs.pkl   # all generated graphs, including invalid ones
  generated.smi          # valid SMILES only

Metrics:
  - validity_without_correction: RDKit sanitization success rate, no correction
  - uniqueness_rate: unique valid canonical SMILES / valid generated molecules
  - novelty_rate: unique valid generated SMILES not in training set / unique valid generated
  - FCD: optional, computed with fcd_torch when installed
  - NSPDK MMD: built-in approximate NSPDK-style graph kernel MMD

The built-in NSPDK is a deterministic approximation based on hashed rooted
neighborhood-pair features. If you use it in a paper, report it as the
"builtin NSPDK proxy" unless you replace it with an official NSPDK backend.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from grapher.molecular.graph_io import (
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
) -> tuple[float | None, dict[str, Any]]:
    if skip:
        return None, {"status": "skipped_by_user"}
    if not reference_smiles or not generated_smiles:
        return None, {
            "status": "not_computed_empty_smiles",
            "num_reference_smiles": len(reference_smiles),
            "num_generated_smiles": len(generated_smiles),
        }

    resolved_device = str(resolve_torch_device(device))

    # fcd_torch has had a few APIs. Try the common ones.
    try:
        from fcd_torch import FCD  # type: ignore

        fcd = FCD(device=resolved_device)
        value = fcd(reference_smiles, generated_smiles)
        return float(value), {"status": "ok", "backend": "fcd_torch.FCD"}
    except Exception as exc1:
        try:
            from fcd_torch.fcd import FCD  # type: ignore

            fcd = FCD(device=resolved_device)
            value = fcd(reference_smiles, generated_smiles)
            return float(value), {"status": "ok", "backend": "fcd_torch.fcd.FCD"}
        except Exception as exc2:
            try:
                from fcd_torch import get_fcd  # type: ignore

                value = get_fcd(
                    generated_smiles, reference_smiles, device=resolved_device
                )
                return float(value), {"status": "ok", "backend": "fcd_torch.get_fcd"}
            except Exception as exc3:
                return None, {
                    "status": "not_available_or_failed",
                    "backend_attempts": [
                        "fcd_torch.FCD",
                        "fcd_torch.fcd.FCD",
                        "fcd_torch.get_fcd",
                    ],
                    "errors": [
                        type(exc1).__name__,
                        type(exc2).__name__,
                        type(exc3).__name__,
                    ],
                    "message": "Install a compatible fcd_torch package to compute FCD, or pass --skip-fcd.",
                }


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
    novelty_rate, novel_count = _novelty(
        valid_info["unique_valid_smiles"], train_smiles
    )

    generated_valid_graphs = _select_valid_graphs(
        generated_graphs, valid_info["all_smiles"]
    )

    nspdk_all = builtin_nspdk_mmd(
        reference_graphs,
        generated_graphs,
        radius=args.nspdk_radius,
        distance=args.nspdk_distance,
        normalize=not args.no_nspdk_normalize,
    )
    nspdk_valid = builtin_nspdk_mmd(
        reference_graphs,
        generated_valid_graphs,
        radius=args.nspdk_radius,
        distance=args.nspdk_distance,
        normalize=not args.no_nspdk_normalize,
    )

    fcd_value, fcd_info = compute_fcd(
        reference_smiles,
        valid_info["valid_smiles"],
        device=args.fcd_device,
        skip=args.skip_fcd,
    )

    metrics = {
        "num_generated_graphs": int(len(generated_graphs)),
        "num_valid_generated_molecules": int(valid_info["num_valid"]),
        "num_invalid_generated_molecules": int(valid_info["num_invalid"]),
        "validity_without_correction": float(valid_info["validity_without_correction"]),
        "uniqueness_rate": float(valid_info["uniqueness_rate"]),
        "unique_valid_count": int(valid_info["unique_valid_count"]),
        "novelty_rate": None if novelty_rate is None else float(novelty_rate),
        "novel_unique_valid_count": int(novel_count),
        "nspdk_mmd": None if nspdk_all is None else float(nspdk_all),
        "nspdk_mmd_valid_only": None if nspdk_valid is None else float(nspdk_valid),
        "fcd": fcd_value,
        "fcd_num_reference_molecules": int(len(reference_smiles)),
        "fcd_num_valid_generated_molecules": int(len(valid_info["valid_smiles"])),
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
            "validity_without_correction": "RDKit Mol construction + Chem.SanitizeMol, no valency correction or edge resampling.",
            "uniqueness_definition": "unique valid canonical SMILES / valid generated molecules",
            "novelty_definition": "unique valid canonical SMILES not in training set / unique valid canonical SMILES",
            "nspdk": {
                "backend": "builtin_hashed_neighborhood_pair_proxy",
                "radius": int(args.nspdk_radius),
                "distance": int(args.nspdk_distance),
                "normalized_features": bool(not args.no_nspdk_normalize),
            },
            "fcd": fcd_info,
        },
        "conversion_error_counts": valid_info["conversion_error_counts"],
        "invalid_indices": valid_info["invalid_indices"],
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

    parser.add_argument("--nspdk-radius", type=int, default=2)
    parser.add_argument("--nspdk-distance", type=int, default=4)
    parser.add_argument("--no-nspdk-normalize", action="store_true")

    parser.add_argument("--skip-fcd", action="store_true")
    parser.add_argument("--fcd-device", default="auto")

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
    # Recompute canonical valid smiles for saving in stable order.
    generated_graphs, _, _ = _resolve_generated_graphs(args)
    if args.max_generated is not None:
        generated_graphs = generated_graphs[: int(args.max_generated)]
    for g in generated_graphs:
        smi = graph_to_smiles(g, canonical=True, sanitize=True)
        if smi:
            valid_smiles.append(smi)
    with (out_dir / "valid_generated.smi").open("w", encoding="utf-8") as f:
        for smi in valid_smiles:
            f.write(smi + "\n")

    print("Molecular evaluation")
    for key, value in report["metrics"].items():
        print(f"  {key}: {value}")
    print(f"Saved metrics to: {out_dir / 'molecular_evaluation_metrics.json'}")
    print(f"Saved valid SMILES to: {out_dir / 'valid_generated.smi'}")


if __name__ == "__main__":
    main()
