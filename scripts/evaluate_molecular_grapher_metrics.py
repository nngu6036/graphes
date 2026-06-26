from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grapher.evaluation.data_io import load_dataset_splits
from grapher.evaluation.molecular_metrics import (
    compute_fcd,
    compute_nspdk_mmd,
    direct_molecular_conversions,
    molecular_novelty,
    molecular_uniqueness,
    valid_smiles,
    validity_without_correction,
)
from grapher.evaluation.run_utils import make_model_run_config, metric_path, sample_path
from grapher.generation.molecular_rewiring import edge_type_value, node_type_value
from grapher.generation.rewiring import (
    check_sequence_validity,
    connected_sequence_feasible,
    degree_sequence,
)
from grapher.molecules.representation import molecular_graph_schema
from grapher.utils.io import load_pickle, load_yaml, save_json
from grapher.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_NAME = "grapher_molecular"
METRIC_FILENAME = "molecular_grapher_metrics.json"


def _default_model_config(dataset: str) -> Path:
    if dataset == "qm9":
        return Path("configs/models/grapher_molecular_qm9.yaml")
    if dataset == "zinc":
        return Path("configs/models/grapher_molecular_zinc.yaml")
    raise ValueError("Molecular evaluation supports only qm9 and zinc.")


def _subsample(items: Sequence[Any], max_items: int | None, seed: int) -> list[Any]:
    values = list(items)
    if max_items is None or max_items <= 0 or len(values) <= int(max_items):
        return values
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(values), size=int(max_items), replace=False)
    return [values[int(index)] for index in indices]


def _resolved_sample_candidate(
    cfg: dict[str, Any],
    dataset: str,
    run_id: int | None,
) -> Path:
    configured = cfg.get("samples_path")
    candidates: list[Path] = [sample_path(dataset, MODEL_NAME, run_id=run_id)]
    if run_id is not None and configured:
        base = Path(configured)
        candidates.append(base.parent / MODEL_NAME / f"run_{run_id:03d}.pkl")
    if configured:
        candidates.append(Path(configured))
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find molecular sample bundle. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_molecular_payload(path: Path) -> tuple[list[nx.Graph], dict[str, Any], str]:
    payload = load_pickle(path)
    if isinstance(payload, list) and all(isinstance(item, nx.Graph) for item in payload):
        graphs = [
            nx.convert_node_labels_to_integers(nx.Graph(item), ordering="sorted")
            for item in payload
        ]
        return graphs, {}, "legacy_graph_list"
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Expected a molecular sample bundle dict or list[nx.Graph], got {type(payload)}."
        )
    graphs = payload.get("graphs")
    if not isinstance(graphs, list) or not all(isinstance(item, nx.Graph) for item in graphs):
        raise TypeError("Molecular sample bundle must contain graphs: list[nx.Graph].")
    return (
        [
            nx.convert_node_labels_to_integers(nx.Graph(item), ordering="sorted")
            for item in graphs
        ],
        dict(payload),
        str(payload.get("schema_version", "unknown_bundle")),
    )


def _topological_quality(graphs: Sequence[nx.Graph]) -> dict[str, float | int]:
    if not graphs:
        return {
            "connectedness_rate": 0.0,
            "no_self_loop_rate": 0.0,
            "simple_graph_rate": 0.0,
            "degree_sequence_graphicality_rate": 0.0,
            "degree_sequence_connected_feasible_rate": 0.0,
        }
    connected: list[float] = []
    no_self_loops: list[float] = []
    simple: list[float] = []
    graphical: list[float] = []
    connected_feasible: list[float] = []
    for graph in graphs:
        connected.append(
            float(
                graph.number_of_nodes() > 0
                and (graph.number_of_nodes() == 1 or nx.is_connected(graph))
            )
        )
        no_self_loops.append(float(nx.number_of_selfloops(graph) == 0))
        simple.append(float(not isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph))))
        sequence = degree_sequence(graph)
        graphical.append(float(check_sequence_validity(sequence)[0]))
        connected_feasible.append(float(connected_sequence_feasible(sequence)[0]))
    return {
        "connectedness_rate": float(np.mean(connected)),
        "no_self_loop_rate": float(np.mean(no_self_loops)),
        "simple_graph_rate": float(np.mean(simple)),
        "degree_sequence_graphicality_rate": float(np.mean(graphical)),
        "degree_sequence_connected_feasible_rate": float(np.mean(connected_feasible)),
    }


def _attribute_summary(graphs: Sequence[nx.Graph]) -> dict[str, Any]:
    atom_counts: dict[str, int] = {}
    bond_counts: dict[str, int] = {}
    for graph in graphs:
        for node in graph.nodes():
            try:
                key = str(int(node_type_value(graph, int(node))))
            except Exception:
                key = "missing"
            atom_counts[key] = atom_counts.get(key, 0) + 1
        for u, v in graph.edges():
            try:
                key = str(int(edge_type_value(graph, int(u), int(v))))
            except Exception:
                key = "missing"
            bond_counts[key] = bond_counts.get(key, 0) + 1
    return {
        "atom_type_counts": atom_counts,
        "bond_type_counts": bond_counts,
    }


def evaluate_molecular_grapher(
    *,
    dataset: str,
    model_config: dict[str, Any],
    dataset_root: str,
    reference_split: str,
    max_reference_molecules: int | None,
    max_generated_molecules: int | None,
    seed: int,
    run_id: int | None,
    sample_file: str | None,
    output: str | None,
    nspdk_complexity: int,
    nspdk_backend: str,
    skip_nspdk: bool,
    skip_fcd: bool,
    require_fcd: bool,
    fcd_device: str,
    fcd_n_jobs: int,
    fcd_batch_size: int,
    keep_explicit_hydrogens: bool,
    isomeric_smiles: bool,
) -> dict[str, Any]:
    if dataset not in {"qm9", "zinc"}:
        raise ValueError("Molecular evaluation supports only qm9 and zinc.")
    start = time.perf_counter()
    cfg = make_model_run_config(
        model_config,
        dataset=dataset,
        model=MODEL_NAME,
        run_id=run_id,
        seed=seed,
        use_run_paths=run_id is not None,
    )
    splits = load_dataset_splits(
        dataset,
        output_root=dataset_root,
        build_if_missing=dataset != "zinc",
    )
    if reference_split not in splits:
        raise KeyError(f"Unknown split {reference_split!r}; available={sorted(splits)}")
    reference_graphs = _subsample(
        splits[reference_split],
        max_reference_molecules,
        seed,
    )
    train_graphs = list(splits.get("train", []))

    sample_path_used = (
        Path(sample_file)
        if sample_file
        else _resolved_sample_candidate(cfg, dataset, run_id)
    )
    generated_graphs, sample_payload, payload_type = _load_molecular_payload(sample_path_used)
    generated_graphs = _subsample(
        generated_graphs,
        max_generated_molecules,
        seed + 17,
    )

    # Validity is always measured on the original hard attributed graph.  No
    # graph repair, valence correction, bond resampling, or fragment selection
    # is performed before RDKit sanitization.
    generated_conversions = direct_molecular_conversions(
        generated_graphs,
        isomeric_smiles=isomeric_smiles,
    )
    reference_conversions = direct_molecular_conversions(
        reference_graphs,
        isomeric_smiles=isomeric_smiles,
    )
    train_conversions = direct_molecular_conversions(
        train_graphs,
        isomeric_smiles=isomeric_smiles,
    )

    heavy_atoms_only = not bool(keep_explicit_hydrogens)
    generated_valid_smiles = valid_smiles(
        generated_conversions,
        remove_explicit_hydrogens=heavy_atoms_only,
        isomeric_smiles=isomeric_smiles,
    )
    reference_valid_smiles = valid_smiles(
        reference_conversions,
        remove_explicit_hydrogens=heavy_atoms_only,
        isomeric_smiles=isomeric_smiles,
    )
    train_valid_smiles = valid_smiles(
        train_conversions,
        remove_explicit_hydrogens=heavy_atoms_only,
        isomeric_smiles=isomeric_smiles,
    )
    generated_valid_graphs = [
        graph
        for graph, conversion in zip(generated_graphs, generated_conversions)
        if conversion.valid
    ]
    reference_valid_graphs = [
        graph
        for graph, conversion in zip(reference_graphs, reference_conversions)
        if conversion.valid
    ]

    results: dict[str, float | int | None] = {}
    results["validity_without_correction"] = validity_without_correction(
        generated_conversions
    )
    results["validity_rate"] = results["validity_without_correction"]
    results["reference_validity_without_correction"] = validity_without_correction(
        reference_conversions
    )
    results["num_valid_generated_molecules"] = int(
        sum(item.valid for item in generated_conversions)
    )
    results["num_invalid_generated_molecules"] = int(
        len(generated_graphs) - results["num_valid_generated_molecules"]
    )
    results["uniqueness_rate"] = molecular_uniqueness(generated_valid_smiles)
    results["novelty_rate"] = molecular_novelty(
        generated_valid_smiles,
        train_valid_smiles,
    )
    results.update(_topological_quality(generated_graphs))

    nspdk_protocol: dict[str, Any]
    if skip_nspdk:
        results["nspdk_mmd"] = None
        results["nspdk_mmd_valid_only"] = None
        nspdk_protocol = {"status": "skipped_by_user"}
    else:
        nspdk_all, nspdk_protocol = compute_nspdk_mmd(
            reference_graphs,
            generated_graphs,
            complexity=max(int(nspdk_complexity), 1),
            backend=nspdk_backend,
            heavy_atoms_only=heavy_atoms_only,
        )
        nspdk_valid, nspdk_valid_protocol = compute_nspdk_mmd(
            reference_valid_graphs,
            generated_valid_graphs,
            complexity=max(int(nspdk_complexity), 1),
            backend=nspdk_backend,
            heavy_atoms_only=heavy_atoms_only,
        )
        results["nspdk_mmd"] = (
            float(nspdk_all) if np.isfinite(nspdk_all) else None
        )
        results["nspdk_mmd_valid_only"] = (
            float(nspdk_valid) if np.isfinite(nspdk_valid) else None
        )
        nspdk_protocol["primary_population"] = (
            "all hard attributed generated graphs, including RDKit-invalid outputs"
        )
        nspdk_protocol["valid_only_metric"] = "nspdk_mmd_valid_only"
        nspdk_protocol["valid_only_backend"] = nspdk_valid_protocol

    fcd_status: dict[str, Any]
    if skip_fcd:
        results["fcd"] = None
        fcd_status = {"status": "skipped_by_user"}
    else:
        score, fcd_status = compute_fcd(
            reference_valid_smiles,
            generated_valid_smiles,
            device=fcd_device,
            n_jobs=fcd_n_jobs,
            batch_size=fcd_batch_size,
        )
        results["fcd"] = score
        if require_fcd and score is None:
            raise RuntimeError(
                "FCD was required but could not be computed: " + str(fcd_status)
            )
    results["fcd_validity_without_correction"] = results["validity_without_correction"]
    results["fcd_num_valid_generated_molecules"] = len(generated_valid_smiles)
    results["fcd_num_generated_molecules"] = len(generated_graphs)
    fcd_status.update(
        {
            "population": "valid generated and valid reference molecular representations only",
            "paired_validity_metric": "validity_without_correction",
            "validity_without_correction": results["validity_without_correction"],
            "num_total_generated_molecules": len(generated_graphs),
            "num_valid_generated_molecules": len(generated_valid_smiles),
        }
    )

    elapsed = time.perf_counter() - start
    output_path = (
        Path(output)
        if output
        else metric_path(dataset, MODEL_NAME, METRIC_FILENAME, run_id=run_id)
    )
    conversion_error_counts: dict[str, int] = {}
    for conversion in generated_conversions:
        if conversion.valid:
            continue
        key = conversion.error.split(":", 1)[0] if conversion.error else "unknown"
        conversion_error_counts[key] = conversion_error_counts.get(key, 0) + 1

    payload_out = {
        "dataset": dataset,
        "model": MODEL_NAME,
        "run_id": run_id,
        "metric_family": "molecular_grapher_metrics",
        "num_reference_graphs": len(reference_graphs),
        "num_generated_graphs": len(generated_graphs),
        "num_valid_reference_smiles": len(reference_valid_smiles),
        "num_valid_generated_smiles": len(generated_valid_smiles),
        "runtime_seconds": elapsed,
        "protocol": {
            "seed": int(seed),
            "reference_split": reference_split,
            "max_reference_molecules": max_reference_molecules,
            "max_generated_molecules": max_generated_molecules,
            "validity": (
                "direct RDKit sanitization of the original hard attributed graph; "
                "no graph repair, valence correction, bond resampling, or fragment selection"
            ),
            "metric_representation": (
                "heavy-atom graph/SMILES" if heavy_atoms_only else "explicit-hydrogen graph/SMILES"
            ),
            "nspdk": nspdk_protocol,
            "fcd": fcd_status,
            "isomeric_smiles": bool(isomeric_smiles),
            "paper_metrics": [
                "validity_without_correction",
                "nspdk_mmd",
                "fcd",
            ],
        },
        "representation": sample_payload.get(
            "representation", molecular_graph_schema()
        ),
        "debug": {
            "sample_path": str(sample_path_used),
            "payload_type": payload_type,
            "conversion_error_counts": conversion_error_counts,
            "generated_attribute_summary": _attribute_summary(generated_graphs),
            "reference_attribute_summary": _attribute_summary(reference_graphs),
        },
        "results": results,
    }
    save_json(payload_out, output_path, force=True)
    logger.info("Saved molecular metrics to %s", output_path)
    return payload_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate QM9/ZINC molecular GraphER samples using the paper metrics: "
            "validity without correction, NSPDK MMD, and Fréchet ChemNet Distance."
        )
    )
    parser.add_argument("--dataset", required=True, choices=["qm9", "zinc"])
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--reference-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-reference-molecules", type=int, default=None)
    parser.add_argument("--max-generated-molecules", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--sample-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--nspdk-backend",
        choices=["auto", "eden", "builtin"],
        default="auto",
        help="Prefer EDeN in auto mode and record any fallback in metric metadata.",
    )
    parser.add_argument("--nspdk-complexity", type=int, default=4)
    parser.add_argument("--skip-nspdk", action="store_true")
    parser.add_argument("--skip-fcd", action="store_true")
    parser.add_argument("--require-fcd", action="store_true")
    parser.add_argument(
        "--fcd-device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--fcd-n-jobs", type=int, default=1)
    parser.add_argument("--fcd-batch-size", type=int, default=512)
    parser.add_argument(
        "--keep-explicit-hydrogens",
        action="store_true",
        help=(
            "Keep explicit H atoms in NSPDK/FCD representations. By default the original "
            "hard graph is used for validity, while heavy-atom representations are used for "
            "NSPDK and FCD to align common QM9/ZINC protocols."
        ),
    )
    parser.add_argument("--isomeric-smiles", action="store_true")
    args = parser.parse_args()
    config_path = (
        Path(args.model_config)
        if args.model_config
        else _default_model_config(args.dataset)
    )
    evaluate_molecular_grapher(
        dataset=args.dataset,
        model_config=load_yaml(config_path),
        dataset_root=args.dataset_root,
        reference_split=args.reference_split,
        max_reference_molecules=args.max_reference_molecules,
        max_generated_molecules=args.max_generated_molecules,
        seed=args.seed,
        run_id=args.run_id,
        sample_file=args.sample_path,
        output=args.output,
        nspdk_complexity=args.nspdk_complexity,
        nspdk_backend=args.nspdk_backend,
        skip_nspdk=args.skip_nspdk,
        skip_fcd=args.skip_fcd,
        require_fcd=args.require_fcd,
        fcd_device=args.fcd_device,
        fcd_n_jobs=args.fcd_n_jobs,
        fcd_batch_size=args.fcd_batch_size,
        keep_explicit_hydrogens=args.keep_explicit_hydrogens,
        isomeric_smiles=args.isomeric_smiles,
    )


if __name__ == "__main__":
    main()
