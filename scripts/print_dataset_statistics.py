from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grapher.datasets.statistics import (  # noqa: E402
    compute_degree_sequence_collection_statistics,
    compute_split_statistics,
    format_degree_sequence_statistics_table,
    format_graph_statistics_table,
)
from grapher.evaluation.data_io import (  # noqa: E402
    build_dataset_splits,
    load_dataset_splits,
    save_dataset_splits,
    split_path,
)
from grapher.datasets.zinc_utils import is_zinc_dataset, zinc_preparation_hint  # noqa: E402
from grapher.registry import available_datasets  # noqa: E402
from grapher.utils.io import load_pickle, load_yaml, save_json  # noqa: E402
from grapher.utils.logging import get_logger  # noqa: E402
from grapher.utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.exists() or p.is_absolute():
        return p
    repo_p = ROOT / p
    return repo_p if repo_p.exists() else p


def _default_config_path(dataset: str) -> Path:
    return ROOT / "configs" / "datasets" / f"{dataset}.yaml"


def _load_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    cfg_path = _resolve_repo_path(args.config) if args.config else _default_config_path(args.dataset)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {cfg_path}")
    cfg = load_yaml(cfg_path)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.download_root is not None:
        cfg["pyg_root"] = args.download_root
    if args.raw_graph_path is not None:
        cfg["raw_graph_path"] = args.raw_graph_path
    if args.max_graphs is not None:
        cfg["max_graphs"] = int(args.max_graphs)
    if getattr(args, "strict_num_graphs", False):
        cfg["strict_num_graphs"] = True
    return cfg, cfg_path


def _persisted_splits_exist(dataset: str, output_root: str) -> bool:
    return all(split_path(dataset, split, output_root).exists() for split in ("train", "val", "test"))




def _cap_mapping(mapping: Mapping[str, Sequence[Any]], max_items: int | None) -> dict[str, list[Any]]:
    if max_items is None:
        return {str(k): list(v) for k, v in mapping.items()}
    limit = max(0, int(max_items))
    return {str(k): list(v)[:limit] for k, v in mapping.items()}


def _load_dataset_from_registry(args: argparse.Namespace) -> tuple[dict[str, list[nx.Graph]], dict[str, Any], Path | None, str]:
    cfg, cfg_path = _load_config(args)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    if _persisted_splits_exist(args.dataset, args.output_root) and not args.rebuild:
        logger.info("Loading persisted splits for %s from %s", args.dataset, args.output_root)
        splits = load_dataset_splits(args.dataset, output_root=args.output_root, build_if_missing=False)
        splits = _cap_mapping(splits, args.max_graphs_per_split)
        source = f"persisted:{Path(args.output_root) / args.dataset}"
        return splits, cfg, cfg_path, source

    if not args.build_if_missing:
        hint = zinc_preparation_hint() if is_zinc_dataset(args.dataset) else "Run scripts/prepare_dataset.py first or omit --no-build-if-missing."
        raise FileNotFoundError(
            f"Persisted split files for dataset {args.dataset!r} were not found under {args.output_root!r}. "
            f"{hint}"
        )

    if is_zinc_dataset(args.dataset):
        raise SystemExit(
            "ERROR: persisted ZINC split files were not found. "
            + zinc_preparation_hint()
        )

    logger.info("Building dataset %s in memory from %s", args.dataset, cfg_path)
    splits = build_dataset_splits(args.dataset, cfg)
    if args.save_built:
        save_dataset_splits(args.dataset, splits, cfg, output_root=args.output_root, force=args.force)
        logger.info("Saved built splits to %s", Path(args.output_root) / args.dataset)
    splits = _cap_mapping(splits, args.max_graphs_per_split)
    source = f"builder:{args.dataset}"
    return splits, cfg, cfg_path, source


def _is_graph_like(obj: Any) -> bool:
    return hasattr(obj, "number_of_nodes") and hasattr(obj, "number_of_edges") and hasattr(obj, "degree")


def _is_degree_sequence(obj: Any) -> bool:
    if not isinstance(obj, (list, tuple)):
        return False
    try:
        return all(isinstance(v, int) or str(v).lstrip("-").isdigit() for v in obj)
    except Exception:
        return False


def _normalize_loaded_payload(payload: Any) -> tuple[str, dict[str, Any]]:
    """Return (kind, split mapping) from a pickle payload.

    kind is either ``graphs`` or ``degree_sequences``. The mapping values are
    lists containing the corresponding objects.
    """
    if isinstance(payload, Mapping):
        items = {str(k): list(v) for k, v in payload.items() if isinstance(v, (list, tuple))}
    elif isinstance(payload, (list, tuple)):
        items = {"input": list(payload)}
    else:
        raise TypeError(f"Unsupported pickle payload type: {type(payload).__name__}")

    non_empty_values = [values for values in items.values() if values]
    first = non_empty_values[0][0] if non_empty_values else None
    if first is None or _is_graph_like(first):
        return "graphs", {k: [nx.Graph(g) for g in values] for k, values in items.items()}
    if _is_degree_sequence(first):
        return "degree_sequences", {k: [[int(d) for d in seq] for seq in values] for k, values in items.items()}
    raise TypeError(
        "The input pickle must contain a list/dict of NetworkX graphs or a list/dict of integer degree sequences."
    )


def _selected_split_names(args: argparse.Namespace) -> list[str]:
    return list(args.splits)


def _rows_for_input_pickle(args: argparse.Namespace) -> tuple[str, list[dict[str, Any]], dict[str, Any], str]:
    payload_path = _resolve_repo_path(args.input_pkl)
    payload = load_pickle(payload_path)
    kind, mapping = _normalize_loaded_payload(payload)
    mapping = _cap_mapping(mapping, args.max_graphs_per_split)
    selected = _selected_split_names(args)
    if selected == ["all", "train", "val", "test"]:
        selected = ["all", *mapping.keys()]

    if kind == "graphs":
        rows = compute_split_statistics(
            mapping,
            selected_splits=selected,
            schema=None,
            include_path_stats=args.include_path_stats,
            skip_planarity=args.skip_planarity,
            include_exact_isomorphism=args.include_exact_isomorphism,
            include_local_structure=args.full,
            include_wl_hashes=args.include_wl_hashes or args.full,
        )
    else:
        rows = []
        for split in selected:
            if split == "all":
                sequences: list[Sequence[int]] = []
                for values in mapping.values():
                    sequences.extend(values)
            else:
                if split not in mapping:
                    continue
                sequences = list(mapping[split])
            rows.append(compute_degree_sequence_collection_statistics(sequences, split=split))

    metadata = {"input_pkl": str(payload_path), "kind": kind, "splits": {k: len(v) for k, v in mapping.items()}}
    return kind, rows, metadata, f"pickle:{payload_path}"


def _rows_for_dataset(args: argparse.Namespace) -> tuple[str, list[dict[str, Any]], dict[str, Any], str]:
    splits, cfg, cfg_path, source = _load_dataset_from_registry(args)
    rows = compute_split_statistics(
        splits,
        selected_splits=args.splits,
        schema=cfg.get("graph_attributes", cfg),
        include_path_stats=args.include_path_stats,
        skip_planarity=args.skip_planarity,
        include_exact_isomorphism=args.include_exact_isomorphism,
        include_local_structure=args.full,
        include_wl_hashes=args.include_wl_hashes or args.full,
    )
    metadata = {
        "dataset": args.dataset,
        "config_path": str(cfg_path) if cfg_path else None,
        "output_root": args.output_root,
        "source": source,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "config": cfg,
    }
    return "graphs", rows, metadata, source


def _flatten_row_for_csv(row: Mapping[str, Any]) -> dict[str, Any]:
    def get_nested(name: str, subname: str) -> Any:
        value = row.get(name, {})
        return value.get(subname) if isinstance(value, Mapping) else None

    if row.get("kind") == "degree_sequences":
        return {
            "split": row.get("split"),
            "num_sequences": row.get("num_sequences"),
            "length_min": get_nested("length", "min"),
            "length_max": get_nested("length", "max"),
            "length_mean": get_nested("length", "mean"),
            "length_std": get_nested("length", "std"),
            "edge_count_mean": get_nested("edge_count_implied", "mean"),
            "max_degree": row.get("max_degree"),
            "graphical_rate": row.get("graphical_rate"),
            "connected_feasible_rate": row.get("connected_feasible_rate"),
        }
    return {
        "split": row.get("split"),
        "num_graphs": row.get("num_graphs"),
        "node_min": get_nested("node_count", "min"),
        "node_max": get_nested("node_count", "max"),
        "node_mean": get_nested("node_count", "mean"),
        "node_std": get_nested("node_count", "std"),
        "edge_min": get_nested("edge_count", "min"),
        "edge_max": get_nested("edge_count", "max"),
        "edge_mean": get_nested("edge_count", "mean"),
        "edge_std": get_nested("edge_count", "std"),
        "max_degree": row.get("max_degree"),
        "avg_degree_mean": get_nested("avg_degree", "mean"),
        "density_mean": get_nested("density", "mean"),
        "connected_rate": row.get("connected_rate"),
        "avg_clustering_mean": get_nested("avg_clustering", "mean"),
        "transitivity_mean": get_nested("transitivity", "mean"),
        "triangle_count_mean": get_nested("triangle_count", "mean"),
        "planarity_rate": row.get("planarity_rate"),
        "wl_unique_rate": row.get("wl_unique_rate"),
        "has_attributes": row.get("has_attributes"),
        "node_label_vocab_size": row.get("node_label_vocab_size"),
        "edge_label_vocab_size": row.get("edge_label_vocab_size"),
    }


def _write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [_flatten_row_for_csv(row) for row in rows]
    if not flat_rows:
        return
    fieldnames: list[str] = []
    for row in flat_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print graph or degree-sequence statistics for a registered dataset or pickle file."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--dataset", choices=available_datasets(), help="Registered dataset name.")
    source_group.add_argument("--input-pkl", help="Pickle containing a graph list, split dict, degree-sequence list, or split dict.")

    parser.add_argument("--config", type=str, default=None, help="Dataset YAML. Defaults to configs/datasets/{dataset}.yaml.")
    parser.add_argument("--output-root", "--dataset-root", dest="output_root", type=str, default="outputs/datasets", help="Prepared dataset split root.")
    parser.add_argument("--download-root", type=str, default=None, help="Override PyG/raw cache root for builders that support it. ZINC is not built here; use prepare_zinc_from_smiles.py first.")
    parser.add_argument("--raw-graph-path", type=str, default=None, help="Optional source graph file, e.g. ind.citeseer.graph.")
    parser.add_argument("--max-graphs", type=int, default=None, help="Optional cap passed to builders that support it.")
    parser.add_argument("--strict-num-graphs", action="store_true", help="For finite-source datasets such as ego_citeseer, fail instead of using min(requested, available).")
    parser.add_argument("--max-graphs-per-split", type=int, default=None, help="Optional reporting cap applied after loading/building each split.")
    parser.add_argument("--seed", type=int, default=None, help="Override dataset seed before building.")
    parser.add_argument("--splits", nargs="+", default=["all", "train", "val", "test"], help="Splits to print. Use 'all' for aggregate statistics.")
    parser.add_argument("--rebuild", action="store_true", help="Build from config even if persisted split files already exist.")
    parser.add_argument("--build-if-missing", action=argparse.BooleanOptionalAction, default=True, help="Build in memory when persisted split files are missing.")
    parser.add_argument("--save-built", action="store_true", help="Persist splits built by this script.")
    parser.add_argument("--force", action="store_true", help="Overwrite saved splits/JSON/CSV outputs where applicable.")
    parser.set_defaults(skip_planarity=True)
    parser.add_argument("--include-planarity", dest="skip_planarity", action="store_false", help="Compute planarity rate. This can be slow on large datasets.")
    parser.add_argument("--skip-planarity", dest="skip_planarity", action="store_true", help="Skip planarity checks.")
    parser.add_argument("--full", action="store_true", help="Compute local-structure diagnostics such as clustering, transitivity, and triangle counts.")
    parser.add_argument("--include-wl-hashes", action="store_true", help="Compute fast Weisfeiler-Lehman hash uniqueness diagnostics.")
    parser.add_argument("--include-path-stats", action="store_true", help="Also compute diameter and average shortest path for connected graphs.")
    parser.add_argument("--include-exact-isomorphism", action="store_true", help="Compute exact isomorphism uniqueness; can be slow on large datasets.")
    parser.add_argument("--json-out", type=str, default=None, help="Optional JSON output path for full nested statistics.")
    parser.add_argument("--csv-out", type=str, default=None, help="Optional CSV output path for flattened table statistics.")
    args = parser.parse_args()

    if args.input_pkl:
        kind, rows, metadata, source = _rows_for_input_pickle(args)
    else:
        kind, rows, metadata, source = _rows_for_dataset(args)

    print(f"Source: {source}")
    print(f"Kind:   {kind}")
    print()
    if kind == "degree_sequences":
        print(format_degree_sequence_statistics_table(rows))
    else:
        print(format_graph_statistics_table(rows))

    payload = {"source": source, "kind": kind, "metadata": metadata, "statistics": rows}
    if args.json_out:
        save_json(payload, args.json_out, force=args.force)
        print(f"\nSaved JSON statistics to {args.json_out}")
    if args.csv_out:
        if Path(args.csv_out).exists() and not args.force:
            raise FileExistsError(f"CSV file exists: {args.csv_out}. Use --force to overwrite.")
        _write_csv(rows, args.csv_out)
        print(f"Saved CSV statistics to {args.csv_out}")


if __name__ == "__main__":
    main()
