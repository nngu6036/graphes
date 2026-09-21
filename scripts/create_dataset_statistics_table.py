#!/usr/bin/env python
"""Retrieve prepared dataset statistics and write the paper's LaTeX table.

Run from the repository root:

    PYTHONPATH=src python scripts/create_dataset_statistics_table.py

All five benchmarks are included by default. Existing train/val/test pickles
are read one split at a time; no graphs are generated or plotted. Training
budgets come from configs/baselines/common_<dataset>.yaml. Generation counts
are reporting targets, not requests to run the generators. Missing datasets
must be prepared separately; this command does not download or rebuild them.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import yaml

from grapher.data.builders import SPLIT_NAMES
from grapher.data.statistics import PreparedDataset, resolve_prepared_dataset
from grapher.utils.io import load_pickle, load_yaml, save_json


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "community_small": ("Community-small", 1024),
    "ego_small": ("Ego-small", 1024),
    "grid": ("Grid", 1024),
    "qm9": ("QM9", 10000),
    "zinc": ("ZINC250k", 10000),
}


def dataset_extrema(dataset: PreparedDataset) -> dict[str, Any]:
    """Count actual nodes/edges across all splits without attribute analysis."""
    result: dict[str, Any] = {
        "dataset": dataset.requested_name,
        "serialized_dataset": dataset.serialized_name,
        "dataset_directory": str(dataset.directory),
        "split_sizes": {},
        "min_nodes": None,
        "max_nodes": None,
        "min_edges": None,
        "max_edges": None,
    }
    for split in SPLIT_NAMES:
        path = dataset.directory / f"{split}.pkl"
        print(f"Reading {dataset.requested_name}/{split}: {path}", file=sys.stderr)
        graphs = load_pickle(path)
        if not isinstance(graphs, (list, tuple)):
            raise TypeError(f"{path} must contain a list or tuple of NetworkX graphs.")
        result["split_sizes"][split] = len(graphs)
        for index, graph in enumerate(graphs):
            if not isinstance(graph, nx.Graph):
                raise TypeError(f"{path}: item {index} is not a NetworkX graph.")
            if graph.is_directed() or graph.is_multigraph():
                raise ValueError(f"{path}: item {index} must be a simple undirected graph.")
            for feature, count in (
                ("nodes", graph.number_of_nodes()),
                ("edges", graph.number_of_edges()),
            ):
                low, high = f"min_{feature}", f"max_{feature}"
                result[low] = count if result[low] is None else min(result[low], count)
                result[high] = count if result[high] is None else max(result[high], count)
        # Release this split before unpickling the next large molecular split.
        del graphs
    result["num_graphs"] = sum(result["split_sizes"].values())
    if result["num_graphs"] == 0:
        raise ValueError(f"Dataset {dataset.requested_name!r} contains no graphs.")
    return result


def training_epochs(dataset: str, config_dir: Path) -> int:
    """Use the common exposure target, rather than a model's native loop count."""
    path = config_dir / f"common_{dataset}.yaml"
    common = load_yaml(path)["baseline_common"]
    if common.get("dataset") != dataset:
        raise ValueError(f"{path} does not target dataset {dataset!r}.")
    value = (common.get("policy") or {}).get("target_train_passes")
    if value is None:
        value = common["training"]["epochs"]
    epochs = int(value)
    if isinstance(value, bool) or epochs <= 0 or float(value) != epochs:
        raise ValueError(f"{path}: training epochs must be a positive integer.")
    return epochs


def render_latex(rows: Sequence[dict[str, Any]]) -> str:
    def number(value: int) -> str:
        return f"{int(value):,}".replace(",", "{,}")

    lines = [
        r"% Preamble: \usepackage{booktabs}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Dataset statistics, training budgets, and generation sample counts.",
        "Dataset size is the total number of graphs across training, validation, and test splits.",
        "Node and edge extrema are measured over the combined training, validation,",
        "and test splits. Molecular datasets use their prepared heavy-atom",
        "representation, and undirected edges are counted once. Training epochs",
        "denote equivalent passes over the training set, with exposure-matched",
        "budgets for GraphRNN and HOG-Diff. Generated samples are per evaluation run.}",
        r"\label{tab:dataset_statistics}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        "Dataset & Dataset size & Max nodes & Min nodes & Max edges & Min edges",
        r"        & Training epochs & Generated samples \\",
        r"\midrule",
    ]
    for row in rows:
        values = [row["name"]] + [
            number(row[key]) for key in (
                "num_graphs", "max_nodes", "min_nodes", "max_edges", "min_edges",
                "training_epochs", "generated_samples",
            )
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", "--dataset-root", type=Path, default=Path("outputs/datasets"),
        help="Root containing prepared <dataset>/{train,val,test}.pkl files.",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS),
        help="Benchmarks to include, in table order; defaults to all five.",
    )
    parser.add_argument(
        "--baseline-config-dir", type=Path,
        default=REPO_ROOT / "configs/baselines",
        help="Directory containing common_<dataset>.yaml training budgets.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/reports/dataset_statistics.tex"),
        help="LaTeX output file; the table is also printed to standard output.",
    )
    parser.add_argument("--json-out", type=Path, help="Optional machine-readable table data.")
    parser.add_argument("--force", action="store_true", help="Replace existing report files.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    names = list(dict.fromkeys(args.datasets))
    try:
        # Resolve every dataset before scanning any large files or writing outputs.
        datasets = [resolve_prepared_dataset(
            name, root=args.root, config_dir=REPO_ROOT / "configs/datasets",
        ) for name in names]
        epochs = {name: training_epochs(name, args.baseline_config_dir) for name in names}
        outputs = [args.output] + ([args.json_out] if args.json_out else [])
        if len({path.resolve() for path in outputs}) != len(outputs):
            raise ValueError("LaTeX and JSON outputs must be different files.")
        for path in outputs:
            if path.resolve().is_relative_to(args.root.resolve()):
                raise ValueError("Report outputs must be outside the dataset root.")
            if path.exists() and not args.force:
                raise ValueError(f"Report already exists: {path}. Use --force to replace it.")
        rows = []
        for dataset in datasets:
            name = dataset.requested_name
            rows.append({
                **dataset_extrema(dataset),
                "name": DATASETS[name][0],
                "training_epochs": epochs[name],
                "generated_samples": DATASETS[name][1],
                "budget_config": str(args.baseline_config_dir / f"common_{name}.yaml"),
            })
        latex = render_latex(rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(latex, encoding="utf-8")
        if args.json_out:
            save_json({"format": "dataset_statistics_table_v1", "datasets": rows}, args.json_out)
    except (OSError, ValueError, TypeError, KeyError, EOFError, pickle.UnpicklingError,
            yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(latex, end="")
    print(f"Saved LaTeX table: {args.output}", file=sys.stderr)
    if args.json_out:
        print(f"Saved table data: {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
