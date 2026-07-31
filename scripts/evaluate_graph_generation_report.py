#!/usr/bin/env python
"""Create the paper-facing MMD table and generated-graph sample figure."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/grapher-matplotlib")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

from grapher.data.io import load_dataset_splits
from grapher.evaluation.metrics import descriptor_matrix, mmd_orbit, mmd_rbf
from grapher.molecular.graph_io import nx_to_rdkit_mol, require_rdkit
from grapher.properties.summary import (
    clustering_histogram,
    configure_orca_executable,
    degree_histogram,
)
from grapher.utils.io import ensure_dir, load_pickle, load_yaml, save_json


REPORT_METRICS = ("degree_mmd", "clustering_mmd", "orbit_mmd")
MOLECULAR_METRICS = (
    "validity_without_correction",
    "uniqueness_rate",
    "novelty_rate",
)
ATOM_COLORS = {
    1: "#F3F4F6",
    6: "#374151",
    7: "#2563EB",
    8: "#DC2626",
    9: "#16A34A",
}
ATOM_LABELS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def _load_graph_list(path: Path) -> list[nx.Graph]:
    value = load_pickle(path)
    if isinstance(value, dict):
        for key in ("graphs", "generated_graphs", "hybrid_refined_graphs"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{path} does not contain a graph list.")
    graphs = list(value)
    if not all(isinstance(graph, nx.Graph) for graph in graphs):
        raise TypeError(f"{path} contains a non-NetworkX graph.")
    if not graphs:
        raise ValueError(f"{path} contains no graphs.")
    return graphs


def _paper_mmd(
    reference: Sequence[nx.Graph],
    candidate: Sequence[nx.Graph],
) -> dict[str, float]:
    """Evaluate the three generic-graph statistics used in the manuscript."""

    max_degree = max(
        (
            max((int(degree) for _, degree in graph.degree()), default=0)
            for graph in [*reference, *candidate]
        ),
        default=0,
    )
    degree_reference = descriptor_matrix(
        reference,
        lambda graph: degree_histogram(graph, max_degree),
    )
    degree_candidate = descriptor_matrix(
        candidate,
        lambda graph: degree_histogram(graph, max_degree),
    )
    clustering_reference = descriptor_matrix(
        reference,
        lambda graph: clustering_histogram(graph, 20),
    )
    clustering_candidate = descriptor_matrix(
        candidate,
        lambda graph: clustering_histogram(graph, 20),
    )
    return {
        "degree_mmd": mmd_rbf(degree_reference, degree_candidate),
        "clustering_mmd": mmd_rbf(
            clustering_reference,
            clustering_candidate,
        ),
        "orbit_mmd": mmd_orbit(reference, candidate),
    }


def is_molecular_evaluation(
    dataset_config: dict[str, Any],
    graphs: Sequence[nx.Graph],
) -> bool:
    """Detect an attributed molecular dataset without relying on its name alone."""

    dataset_name = str(dataset_config.get("name", "")).lower()
    if any(token in dataset_name for token in ("qm9", "zinc", "molecule")):
        return True
    return any(
        "atomic_num" in data or "atom_type" in data
        for graph in graphs
        for _, data in graph.nodes(data=True)
    )


def _canonical_molecular_smiles(
    graph: nx.Graph,
) -> tuple[str | None, str | None]:
    """Return an RDKit-sanitized canonical SMILES and any conversion error."""

    if graph.number_of_nodes() == 0:
        return None, "EmptyMolecule"
    if any(
        "atomic_num" not in data and "atom_type" not in data
        for _, data in graph.nodes(data=True)
    ):
        return None, "MissingAtomType"
    if any(
        "bond_type" not in data and "bond_order" not in data
        for _, _, data in graph.edges(data=True)
    ):
        return None, "MissingBondType"
    Chem = require_rdkit()
    try:
        molecule = nx_to_rdkit_mol(graph, sanitize=True)
        smiles = str(
            Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=False,
            )
        )
        if not smiles:
            return None, "EmptySMILES"
        return smiles, None
    except Exception as exc:
        return None, type(exc).__name__


def molecular_quality_metrics(
    generated_graphs: Sequence[nx.Graph],
    train_graphs: Sequence[nx.Graph],
) -> tuple[dict[str, Any], list[str], list[int], dict[str, int]]:
    """Compute RDKit validity, canonical-SMILES uniqueness, and novelty."""

    # Fail early with the existing installation guidance when RDKit is absent.
    require_rdkit()

    valid_smiles: list[str] = []
    invalid_indices: list[int] = []
    conversion_errors: Counter[str] = Counter()
    for index, graph in enumerate(generated_graphs):
        smiles, error = _canonical_molecular_smiles(graph)
        if smiles is None:
            invalid_indices.append(index)
            conversion_errors[str(error or "InvalidMolecule")] += 1
        else:
            valid_smiles.append(smiles)

    train_smiles = {
        smiles
        for graph in train_graphs
        if (smiles := _canonical_molecular_smiles(graph)[0]) is not None
    }
    unique_valid_smiles = sorted(set(valid_smiles))
    novel_smiles = [
        smiles for smiles in unique_valid_smiles if smiles not in train_smiles
    ]

    num_generated = len(generated_graphs)
    num_valid = len(valid_smiles)
    num_unique = len(unique_valid_smiles)
    novelty_rate: float | None
    if not train_smiles:
        novelty_rate = None
    elif num_unique == 0:
        novelty_rate = 0.0
    else:
        novelty_rate = len(novel_smiles) / num_unique

    metrics = {
        "num_generated_graphs": num_generated,
        "num_valid_generated_molecules": num_valid,
        "num_invalid_generated_molecules": num_generated - num_valid,
        "validity_without_correction": num_valid / max(num_generated, 1),
        "unique_valid_count": num_unique,
        "uniqueness_rate": num_unique / max(num_valid, 1),
        "num_valid_training_molecules": len(train_smiles),
        "novel_unique_valid_count": len(novel_smiles),
        "novelty_rate": novelty_rate,
    }
    return (
        metrics,
        valid_smiles,
        invalid_indices,
        dict(conversion_errors),
    )


def select_sample_indices(
    graphs: Sequence[nx.Graph],
    count: int,
    *,
    selection: str,
    seed: int,
) -> list[int]:
    count = min(max(int(count), 0), len(graphs))
    if count == 0:
        return []
    if selection == "random":
        rng = np.random.default_rng(seed)
        return sorted(
            int(index)
            for index in rng.choice(len(graphs), size=count, replace=False)
        )
    if selection != "stratified":
        raise ValueError(f"Unknown sample selection: {selection!r}.")

    # Cover the generated size range rather than showing only the first graphs.
    ordered = sorted(
        range(len(graphs)),
        key=lambda index: (
            graphs[index].number_of_nodes(),
            graphs[index].number_of_edges(),
            index,
        ),
    )
    positions = np.linspace(0, len(ordered) - 1, num=count)
    return [ordered[int(round(position))] for position in positions]


def _atomic_number(data: dict[str, Any]) -> int | None:
    value = data.get("atomic_num", data.get("atom_type"))
    return int(value) if value is not None else None


def _draw_graph(
    axis: Any,
    graph: nx.Graph,
    *,
    graph_index: int,
    layout_seed: int,
) -> bool:
    molecular = any(
        _atomic_number(data) is not None for _, data in graph.nodes(data=True)
    )
    positions = nx.spring_layout(
        graph,
        seed=int(layout_seed + graph_index),
        iterations=150,
    )
    node_colors = []
    labels: dict[Any, str] = {}
    for node, data in graph.nodes(data=True):
        atomic_number = _atomic_number(data)
        node_colors.append(
            ATOM_COLORS.get(atomic_number, "#60A5FA")
            if molecular
            else "#4C78A8"
        )
        if molecular and atomic_number is not None:
            labels[node] = ATOM_LABELS.get(atomic_number, str(atomic_number))

    edge_widths = []
    edge_colors = []
    for _, _, data in graph.edges(data=True):
        bond_type = int(data.get("bond_type", 1))
        edge_widths.append({1: 1.4, 2: 2.4, 3: 3.4, 4: 2.0}.get(bond_type, 1.4))
        edge_colors.append("#7C3AED" if bond_type == 4 else "#6B7280")

    node_size = max(90, min(360, int(9000 / max(graph.number_of_nodes(), 1))))
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=axis,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.8,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axis,
        node_color=node_colors,
        node_size=node_size,
        edgecolors="#111827",
        linewidths=0.6,
    )
    if labels:
        nx.draw_networkx_labels(
            graph,
            positions,
            labels=labels,
            ax=axis,
            font_size=8,
            font_color="white",
        )
    axis.set_title(
        f"Sample {graph_index + 1}\n"
        f"$|V|={graph.number_of_nodes()}$, $|E|={graph.number_of_edges()}$",
        fontsize=9,
    )
    axis.set_axis_off()
    return molecular


def plot_generated_graphs(
    graphs: Sequence[nx.Graph],
    indices: Sequence[int],
    *,
    output_png: Path,
    output_pdf: Path,
    columns: int,
    layout_seed: int,
    dpi: int,
) -> None:
    if not indices:
        raise ValueError("At least one generated graph is required for plotting.")
    columns = max(1, min(int(columns), len(indices)))
    rows = int(np.ceil(len(indices) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns, 3.0 * rows),
        squeeze=False,
    )
    molecular = False
    for axis, index in zip(axes.flat, indices):
        molecular = (
            _draw_graph(
                axis,
                graphs[index],
                graph_index=index,
                layout_seed=layout_seed,
            )
            or molecular
        )
    for axis in axes.flat[len(indices) :]:
        axis.set_axis_off()
    if molecular:
        atoms_present = sorted(
            {
                atomic_number
                for index in indices
                for _, data in graphs[index].nodes(data=True)
                if (atomic_number := _atomic_number(data)) is not None
            }
        )
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=ATOM_COLORS.get(atomic_number, "#60A5FA"),
                markeredgecolor="#111827",
                label=ATOM_LABELS.get(atomic_number, str(atomic_number)),
                markersize=8,
            )
            for atomic_number in atoms_present
        ]
        figure.legend(
            handles=handles,
            loc="lower center",
            ncol=max(1, len(handles)),
            frameon=False,
        )
        figure.subplots_adjust(bottom=0.09)
    figure.suptitle("Representative generated graphs", fontsize=13)
    figure.tight_layout(rect=(0, 0.04 if molecular else 0, 1, 0.96))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=dpi, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("comparison", *REPORT_METRICS),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_molecular_csv(
    rows: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _print_table(rows: Sequence[dict[str, Any]]) -> None:
    print("Graph-distribution MMD against held-out test graphs (lower is better)")
    print(
        f"{'Comparison':30s}"
        f"{'Degree MMD':>14s}"
        f"{'Clustering MMD':>18s}"
        f"{'Orbit MMD':>14s}"
    )
    for row in rows:
        print(
            f"{str(row['comparison']):30s}"
            f"{float(row['degree_mmd']):14.6f}"
            f"{float(row['clustering_mmd']):18.6f}"
            f"{float(row['orbit_mmd']):14.6f}"
        )


def _print_molecular_metrics(rows: Sequence[dict[str, Any]]) -> None:
    print("\nMolecular quality by stage (RDKit; higher is better)")
    print(
        f"{'Stage':24s}"
        f"{'Validity':>12s}"
        f"{'Uniqueness':>14s}"
        f"{'Novelty':>12s}"
        f"{'Valid count':>16s}"
    )
    for row in rows:
        novelty = row["novelty_rate"]
        novelty_text = (
            "n/a" if novelty is None else f"{float(novelty):.6f}"
        )
        count = (
            f"{row['num_valid_generated_molecules']}/"
            f"{row['num_generated_graphs']}"
        )
        print(
            f"{str(row['stage']):24s}"
            f"{float(row['validity_without_correction']):12.6f}"
            f"{float(row['uniqueness_rate']):14.6f}"
            f"{novelty_text:>12s}"
            f"{count:>16s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved generated graphs with degree, clustering, and "
            "four-node ORCA orbit MMD, then plot representative samples."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--generated-graphs", default=None)
    parser.add_argument("--coarse-graphs", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument(
        "--sample-selection",
        choices=("stratified", "random"),
        default="stratified",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--layout-seed", type=int, default=42)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    config = load_yaml(args.config)
    generated_dir = Path(args.generated_dir)
    generated_path = Path(
        args.generated_graphs
        or generated_dir / "hybrid_refined_graphs.pkl"
    )
    coarse_path = Path(
        args.coarse_graphs or generated_dir / "coarse_graphs.pkl"
    )
    output_dir = ensure_dir(
        args.output_dir or generated_dir / "evaluation_report"
    )

    evaluation_cfg = config.get("evaluation", {}) or {}
    orca_exec = configure_orca_executable(
        evaluation_cfg.get("orca_exec"),
        required=True,
    )
    print(f"ORCA orbit evaluation enabled: {orca_exec}", flush=True)

    dataset_cfg = config.get("dataset", {}) or {}
    splits = load_dataset_splits(
        str(dataset_cfg.get("name", "sbm")),
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", True)),
        config_path=dataset_cfg.get("config_path"),
    )
    train_graphs = list(splits.get("train", []))
    test_graphs = list(splits.get("test", []))
    generated_graphs = _load_graph_list(generated_path)
    coarse_graphs = _load_graph_list(coarse_path) if coarse_path.is_file() else []
    if not test_graphs:
        raise ValueError("The configured dataset has no test graphs.")

    available = min(len(test_graphs), len(generated_graphs))
    if args.max_graphs is not None:
        available = min(available, int(args.max_graphs))
    if available <= 0:
        raise ValueError("No common generated/test graph subset is available.")
    reference = test_graphs[:available]
    generated = generated_graphs[:available]
    molecular = is_molecular_evaluation(dataset_cfg, generated_graphs)

    rows: list[dict[str, Any]] = []
    if train_graphs:
        rows.append(
            {
                "comparison": "train_to_test",
                **_paper_mmd(reference, train_graphs[:available]),
            }
        )
    if coarse_graphs:
        coarse_count = min(available, len(coarse_graphs))
        rows.append(
            {
                "comparison": "hh_source_to_test",
                **_paper_mmd(
                    reference[:coarse_count],
                    coarse_graphs[:coarse_count],
                ),
            }
        )
    rows.append(
        {
            "comparison": "hybrid_final_to_test",
            **_paper_mmd(reference, generated),
        }
    )

    indices = select_sample_indices(
        generated,
        args.num_samples,
        selection=args.sample_selection,
        seed=args.sample_seed,
    )
    png_path = output_dir / "generated_graph_samples.png"
    pdf_path = output_dir / "generated_graph_samples.pdf"
    csv_path = output_dir / "graph_mmd_metrics.csv"
    json_path = output_dir / "graph_evaluation_report.json"
    molecular_csv_path = output_dir / "molecular_quality_metrics.csv"
    valid_smiles_path = output_dir / "valid_generated.smi"
    plot_generated_graphs(
        generated,
        indices,
        output_png=png_path,
        output_pdf=pdf_path,
        columns=args.columns,
        layout_seed=args.layout_seed,
        dpi=args.dpi,
    )
    _write_csv(rows, csv_path)
    molecular_metrics: dict[str, Any] | None = None
    molecular_stage_rows: list[dict[str, Any]] = []
    molecular_stage_errors: dict[str, dict[str, int]] = {}
    invalid_molecule_indices: list[int] = []
    conversion_error_counts: dict[str, int] = {}
    if molecular:
        stage_graphs: list[tuple[str, Sequence[nx.Graph]]] = [
            ("real_test", reference),
        ]
        if coarse_graphs:
            coarse_count = min(available, len(coarse_graphs))
            stage_graphs.append(
                ("hh_source", coarse_graphs[:coarse_count])
            )
        stage_graphs.append(("hybrid_final", generated))
        valid_smiles: list[str] = []
        for stage, graphs in stage_graphs:
            (
                stage_metrics,
                stage_smiles,
                stage_invalid_indices,
                stage_errors,
            ) = molecular_quality_metrics(graphs, train_graphs)
            molecular_stage_rows.append(
                {"stage": stage, **stage_metrics}
            )
            molecular_stage_errors[stage] = stage_errors
            if stage == "hybrid_final":
                molecular_metrics = stage_metrics
                valid_smiles = stage_smiles
                invalid_molecule_indices = stage_invalid_indices
                conversion_error_counts = stage_errors
        _write_molecular_csv(molecular_stage_rows, molecular_csv_path)
        with valid_smiles_path.open("w", encoding="utf-8") as handle:
            for smiles in valid_smiles:
                handle.write(smiles + "\n")

    save_json(
        {
            "format": "graph_generation_evaluation_report_v3",
            "orca_exec": orca_exec,
            "generated_graphs": str(generated_path),
            "num_graphs_evaluated": available,
            "metrics": rows,
            "molecular_evaluation": molecular,
            "molecular_quality": molecular_metrics,
            "molecular_quality_by_stage": {
                str(row["stage"]): {
                    key: value
                    for key, value in row.items()
                    if key != "stage"
                }
                for row in molecular_stage_rows
            },
            "molecular_protocol": (
                {
                    "validity_without_correction": (
                        "RDKit molecule construction followed by "
                        "Chem.SanitizeMol, without valency correction or "
                        "edge resampling."
                    ),
                    "uniqueness": (
                        "unique valid canonical SMILES / valid generated "
                        "molecules"
                    ),
                    "novelty": (
                        "unique valid generated canonical SMILES absent from "
                        "the complete training split / unique valid generated "
                        "canonical SMILES"
                    ),
                    "canonical_smiles": True,
                    "isomeric_smiles": False,
                }
                if molecular
                else None
            ),
            "invalid_molecule_indices_zero_based": invalid_molecule_indices,
            "molecular_conversion_error_counts": conversion_error_counts,
            "molecular_conversion_error_counts_by_stage": (
                molecular_stage_errors if molecular else None
            ),
            "sample_selection": args.sample_selection,
            "sample_indices_zero_based": indices,
            "sample_figure_png": str(png_path),
            "sample_figure_pdf": str(pdf_path),
        },
        json_path,
    )
    _print_table(rows)
    if molecular_metrics is not None:
        _print_molecular_metrics(molecular_stage_rows)
    print(f"Saved metrics: {csv_path}")
    if molecular_metrics is not None:
        print(f"Saved metrics: {molecular_csv_path}")
        print(f"Saved SMILES:  {valid_smiles_path}")
    print(f"Saved report:  {json_path}")
    print(f"Saved figure:  {png_path}")
    print(f"Saved figure:  {pdf_path}")


if __name__ == "__main__":
    main()
