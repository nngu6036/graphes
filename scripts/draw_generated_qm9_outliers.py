#!/usr/bin/env python3
"""Rank and draw generated QM9 molecules farthest from the training set.

FCD and NSPDK are set-level distances, not per-molecule properties.  This
script uses two principled individual outlier scores:

* ``fcd_mean_distance`` is the squared distance from a molecule's ChemNet
  activation to the reference activation mean.  It is the molecule-dependent
  mean term of singleton FCD (the reference covariance term is constant).
* ``nspdk_singleton_mmd`` is the exact linear-kernel NSPDK MMD between the
  reference set and a singleton generated set.

Both are distances, so larger values mean farther from the training data.  The
default joint ranking maximizes the lower of the two percentile ranks, finding
molecules that are outliers under both representations.

Example::

    PYTHONPATH=src python scripts/draw_generated_qm9_outliers.py \
      --generated-dir outputs/attributed_grapher/run/seed_42 \
      --dataset qm9_attributed --split train \
      --count 16 --row 4 --col 4 \
      --fcd-device gpu \
      --output outputs/qm9_generated_outliers.png
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.evaluation.molecular_nspdk import (
    eden_nspdk_singleton_mmd,
)
from grapher.rewiring_mlp.molecular.graph_io import graph_to_smiles
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import load_pickle, save_json

if __package__:
    from scripts import draw_dataset as draw
else:
    import draw_dataset as draw


@dataclass(frozen=True)
class ScoredMolecule:
    generated_index: int
    graph: nx.Graph
    smiles: str
    fcd_mean_distance: float | None
    nspdk_singleton_mmd: float
    fcd_percentile: float | None
    nspdk_percentile: float
    ranking_score: float


def _load_generated_graphs(
    *,
    generated_dir: str | Path | None,
    generated_graphs: str | Path | None,
) -> tuple[list[nx.Graph], Path]:
    if generated_graphs is not None:
        path = Path(generated_graphs).expanduser()
    elif generated_dir is not None:
        directory = Path(generated_dir).expanduser()
        candidates = (
            directory / "molecular_graphs.pkl",
            directory / "generated_graphs.pkl",
        )
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    else:
        raise ValueError("Provide --generated-dir or --generated-graphs.")

    if not path.exists():
        raise FileNotFoundError(path)
    payload = load_pickle(path)
    if isinstance(payload, dict):
        for key in ("graphs", "molecular_graphs", "generated_graphs"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, (list, tuple)):
        raise TypeError(f"{path} must contain a list of NetworkX graphs.")
    graphs = list(payload)
    if not all(isinstance(graph, nx.Graph) for graph in graphs):
        raise TypeError(f"{path} contains non-NetworkX values.")
    return graphs, path


def _valid_graphs_and_smiles(
    graphs: Sequence[nx.Graph],
) -> tuple[list[int], list[nx.Graph], list[str]]:
    indices: list[int] = []
    valid_graphs: list[nx.Graph] = []
    smiles: list[str] = []
    for index, graph in enumerate(graphs):
        value = graph_to_smiles(graph, canonical=True, sanitize=True)
        if value is None:
            continue
        indices.append(index)
        valid_graphs.append(graph)
        smiles.append(value)
    return indices, valid_graphs, smiles


def _smiles_digest(smiles: Sequence[str]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(b"chemnet-reference-mean-v1\0")
    for value in smiles:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_fcd_class():
    errors: list[str] = []
    for module_name in ("fcd_torch", "fcd_torch.fcd"):
        try:
            if module_name == "fcd_torch":
                from fcd_torch import FCD  # type: ignore
            else:
                from fcd_torch.fcd import FCD  # type: ignore
            return FCD
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}")
    raise ImportError(
        "fcd_torch is required for ChemNet outlier scoring. Install "
        "requirements-molecular.txt or pass --skip-fcd. Attempts: "
        + ", ".join(errors)
    )


def fcd_mean_distances(
    reference_smiles: Sequence[str],
    generated_smiles: Sequence[str],
    *,
    device: str = "auto",
    n_jobs: int = 1,
    batch_size: int = 512,
    cache_dir: str | Path | None = None,
) -> np.ndarray:
    """Return squared ChemNet distances to the reference activation mean."""

    if not reference_smiles:
        raise ValueError("FCD outlier scoring requires reference SMILES.")
    if not generated_smiles:
        return np.empty(0, dtype=np.float64)

    FCD = _load_fcd_class()
    resolved_device = str(resolve_torch_device(device))
    fcd = FCD(
        device=resolved_device,
        n_jobs=max(int(n_jobs), 1),
        batch_size=max(int(batch_size), 1),
    )

    reference_mean: np.ndarray | None = None
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_root = Path(cache_dir).expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"fcd_reference_mean_{_smiles_digest(reference_smiles)}.npy"
        if cache_path.exists():
            try:
                reference_mean = np.asarray(np.load(cache_path), dtype=np.float64)
            except Exception:
                reference_mean = None

    if reference_mean is None:
        reference_activations = np.asarray(
            fcd.get_predictions(list(reference_smiles)), dtype=np.float64
        )
        if reference_activations.ndim != 2 or reference_activations.shape[0] == 0:
            raise RuntimeError("ChemNet returned no reference activations.")
        reference_mean = reference_activations.mean(axis=0)
        if cache_path is not None:
            np.save(cache_path, reference_mean)

    generated_activations = np.asarray(
        fcd.get_predictions(list(generated_smiles)), dtype=np.float64
    )
    if generated_activations.ndim != 2:
        raise RuntimeError("ChemNet generated activations must be a matrix.")
    if generated_activations.shape[1] != reference_mean.size:
        raise RuntimeError("Reference and generated ChemNet widths do not match.")
    return np.square(generated_activations - reference_mean.reshape(1, -1)).sum(axis=1)


def _percentile_ranks(values: Sequence[float]) -> np.ndarray:
    """Return average tie-aware percentile ranks in [0, 1]."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return np.empty(0, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("Outlier scores must be finite.")
    if array.size == 1:
        return np.ones(1, dtype=np.float64)

    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    cursor = 0
    while cursor < order.size:
        stop = cursor + 1
        while stop < order.size and array[order[stop]] == array[order[cursor]]:
            stop += 1
        ranks[order[cursor:stop]] = (cursor + stop - 1) / 2.0
        cursor = stop
    return ranks / float(array.size - 1)


def rank_outliers(
    generated_indices: Sequence[int],
    generated_graphs: Sequence[nx.Graph],
    generated_smiles: Sequence[str],
    *,
    fcd_scores: Sequence[float] | None,
    nspdk_scores: Sequence[float],
    ranking: str = "joint",
) -> list[ScoredMolecule]:
    size = len(generated_indices)
    if len(generated_graphs) != size or len(generated_smiles) != size:
        raise ValueError("Generated indices, graphs, and SMILES must have equal lengths.")
    nspdk = np.asarray(nspdk_scores, dtype=np.float64).reshape(-1)
    if nspdk.size != size:
        raise ValueError("NSPDK score count does not match generated molecules.")
    nspdk_percentiles = _percentile_ranks(nspdk)

    fcd: np.ndarray | None = None
    fcd_percentiles: np.ndarray | None = None
    if fcd_scores is not None:
        fcd = np.asarray(fcd_scores, dtype=np.float64).reshape(-1)
        if fcd.size != size:
            raise ValueError("FCD score count does not match generated molecules.")
        fcd_percentiles = _percentile_ranks(fcd)

    ranking = str(ranking).lower()
    if ranking == "joint":
        if fcd_percentiles is None:
            ranking_values = nspdk_percentiles
        else:
            ranking_values = np.minimum(fcd_percentiles, nspdk_percentiles)
    elif ranking == "fcd":
        if fcd_percentiles is None:
            raise ValueError("--ranking fcd cannot be used with --skip-fcd.")
        ranking_values = fcd_percentiles
    elif ranking == "nspdk":
        ranking_values = nspdk_percentiles
    else:
        raise ValueError("ranking must be joint, fcd, or nspdk.")

    rows = [
        ScoredMolecule(
            generated_index=int(generated_indices[position]),
            graph=generated_graphs[position],
            smiles=str(generated_smiles[position]),
            fcd_mean_distance=None if fcd is None else float(fcd[position]),
            nspdk_singleton_mmd=float(nspdk[position]),
            fcd_percentile=(
                None if fcd_percentiles is None else float(fcd_percentiles[position])
            ),
            nspdk_percentile=float(nspdk_percentiles[position]),
            ranking_score=float(ranking_values[position]),
        )
        for position in range(size)
    ]
    return sorted(
        rows,
        key=lambda row: (
            row.ranking_score,
            -1.0 if row.fcd_mean_distance is None else row.fcd_mean_distance,
            row.nspdk_singleton_mmd,
            -row.generated_index,
        ),
        reverse=True,
    )


def _default_output(count: int, ranking: str) -> Path:
    return Path(f"outputs/qm9_generated_{ranking}_outliers_n{count}.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw valid generated QM9 molecules farthest from a reference split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--generated-dir", help="Directory containing molecular_graphs.pkl.")
    source.add_argument("--generated-graphs", help="Generated NetworkX graph pickle.")
    parser.add_argument("--dataset", default="qm9_attributed")
    parser.add_argument(
        "--root",
        "--dataset-root",
        dest="root",
        default="outputs/datasets",
        help="Root containing <dataset>/{train,val,test}.pkl.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="Randomly subsample valid generated candidates before scoring.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ranking", choices=("joint", "fcd", "nspdk"), default="joint")
    parser.add_argument("--min-fcd-distance", type=float)
    parser.add_argument("--min-nspdk-distance", type=float)
    parser.add_argument("--nspdk-complexity", type=int, default=4)
    parser.add_argument(
        "--nspdk-bond-label-mode",
        choices=("hogdiff", "categorical"),
        default="hogdiff",
    )
    parser.add_argument("--skip-fcd", action="store_true")
    parser.add_argument("--fcd-device", default="auto")
    parser.add_argument("--fcd-jobs", type=int, default=1)
    parser.add_argument("--fcd-batch-size", type=int, default=512)
    parser.add_argument(
        "--cache-dir",
        default="outputs/cache/molecular_outlier_metrics",
        help="Reference feature cache directory.",
    )
    parser.add_argument("--row", type=int, default=4)
    parser.add_argument("--col", type=int, default=4)
    parser.add_argument("--panel-width", type=int, default=400)
    parser.add_argument("--panel-height", type=int, default=320)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, help="JSON report path.")
    parser.add_argument("--show-hydrogens", action="store_true")
    parser.add_argument("--atom-indices", action="store_true")
    parser.add_argument("--bond-labels", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count <= 0:
        raise ValueError("--count must be positive.")
    if args.row <= 0 or args.col <= 0:
        raise ValueError("--row and --col must be positive.")
    if args.panel_width < 180 or args.panel_height < 180:
        raise ValueError("--panel-width and --panel-height must be at least 180.")
    if args.skip_fcd and args.min_fcd_distance is not None:
        raise ValueError("--min-fcd-distance cannot be used with --skip-fcd.")

    reference_graphs, reference_path, dataset_name = draw._load_prepared_dataset_split(
        args.dataset, args.root, args.split
    )
    generated_graphs, generated_path = _load_generated_graphs(
        generated_dir=args.generated_dir,
        generated_graphs=args.generated_graphs,
    )
    reference_indices, valid_reference, reference_smiles = _valid_graphs_and_smiles(
        reference_graphs
    )
    generated_indices, valid_generated, generated_smiles = _valid_graphs_and_smiles(
        generated_graphs
    )
    if not valid_reference:
        raise ValueError(f"Reference split contains no RDKit-valid molecules: {reference_path}")
    if not valid_generated:
        raise ValueError(f"Generated file contains no RDKit-valid molecules: {generated_path}")
    num_valid_generated = len(valid_generated)

    if args.max_candidates is not None:
        if args.max_candidates <= 0:
            raise ValueError("--max-candidates must be positive.")
        if args.max_candidates < len(valid_generated):
            positions = sorted(
                random.Random(args.seed).sample(
                    range(len(valid_generated)), int(args.max_candidates)
                )
            )
            generated_indices = [generated_indices[position] for position in positions]
            valid_generated = [valid_generated[position] for position in positions]
            generated_smiles = [generated_smiles[position] for position in positions]

    print(
        f"Scoring valid generated molecules: candidates={len(valid_generated)} "
        f"invalid_excluded={len(generated_graphs) - num_valid_generated}",
        flush=True,
    )
    nspdk_scores = eden_nspdk_singleton_mmd(
        valid_reference,
        valid_generated,
        complexity=args.nspdk_complexity,
        cache_dir=args.cache_dir,
        bond_label_mode=args.nspdk_bond_label_mode,
    )
    fcd_scores = None
    if not args.skip_fcd:
        fcd_scores = fcd_mean_distances(
            reference_smiles,
            generated_smiles,
            device=args.fcd_device,
            n_jobs=args.fcd_jobs,
            batch_size=args.fcd_batch_size,
            cache_dir=args.cache_dir,
        )

    ranked = rank_outliers(
        generated_indices,
        valid_generated,
        generated_smiles,
        fcd_scores=fcd_scores,
        nspdk_scores=nspdk_scores,
        ranking=args.ranking,
    )
    if args.min_fcd_distance is not None:
        ranked = [
            row
            for row in ranked
            if row.fcd_mean_distance is not None
            and row.fcd_mean_distance >= args.min_fcd_distance
        ]
    if args.min_nspdk_distance is not None:
        ranked = [
            row
            for row in ranked
            if row.nspdk_singleton_mmd >= args.min_nspdk_distance
        ]
    selected = ranked[: args.count]
    if not selected:
        raise ValueError("No generated molecules satisfy the requested thresholds.")

    output = (
        args.output or _default_output(args.count, args.ranking)
    ).expanduser().resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("--output must be a PNG path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = (
        args.report or output.with_suffix(".json")
    ).expanduser().resolve()

    loaded_items: list[draw.LoadedItem] = []
    report_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        mol, base_info = draw._load_from_prepared_graph(
            row.graph, row.generated_index, "generated", "outliers"
        )
        mol = draw._prepare_molecule(
            mol,
            show_hydrogens=args.show_hydrogens,
            atom_indices=args.atom_indices,
            bond_labels=args.bond_labels,
        )
        fcd_label = (
            "FCD skipped"
            if row.fcd_mean_distance is None
            else f"FCD*={row.fcd_mean_distance:.3g}"
        )
        metric_label = f"{fcd_label} | NSPDK={row.nspdk_singleton_mmd:.3g}"
        info = draw.MoleculeInfo(
            source=str(generated_path),
            name=metric_label,
            smiles=row.smiles,
            dataset_index=row.generated_index,
            source_index=base_info.source_index,
            index_label=f"rank {rank} | generated[{row.generated_index}]",
        )
        loaded_items.append(draw.LoadedItem(info=info, mol=mol, render_mode="molecule"))
        report_rows.append(
            {
                "rank": rank,
                "generated_index": row.generated_index,
                "smiles": row.smiles,
                "ranking_score": row.ranking_score,
                "fcd_mean_distance": row.fcd_mean_distance,
                "fcd_percentile": row.fcd_percentile,
                "nspdk_singleton_mmd": row.nspdk_singleton_mmd,
                "nspdk_percentile": row.nspdk_percentile,
            }
        )

    per_page = args.row * args.col
    total_pages = math.ceil(len(loaded_items) / per_page)
    for page_index in range(total_pages):
        page_items = loaded_items[page_index * per_page : (page_index + 1) * per_page]
        canvas = draw._compose_page(
            page_items,
            row=args.row,
            col=args.col,
            panel_width=args.panel_width,
            panel_height=args.panel_height,
            page_index=page_index,
            total_pages=total_pages,
            count=len(selected),
            seed=args.seed,
            dataset_label=f"{dataset_name}/{args.split}",
            dataset_path=generated_path,
            show_hydrogens=args.show_hydrogens,
            page_title=f"Generated molecular outliers: {args.ranking} ranking",
        )
        page_output = draw._page_output_path(output, page_index, total_pages)
        canvas.save(page_output)
        print(f"Saved: {page_output}")

    save_json(
        {
            "generated_graphs": str(generated_path.resolve()),
            "reference_split": str(reference_path.resolve()),
            "num_generated_graphs": len(generated_graphs),
            "num_valid_generated_graphs": num_valid_generated,
            "num_reference_graphs": len(reference_graphs),
            "num_valid_reference_graphs": len(reference_indices),
            "num_candidates_scored": len(valid_generated),
            "ranking": args.ranking,
            "score_direction": "higher_is_farther_from_reference",
            "fcd_score_definition": (
                None
                if args.skip_fcd
                else "squared ChemNet activation distance to reference mean"
            ),
            "nspdk_score_definition": "linear-kernel NSPDK MMD for singleton generated set",
            "selected": report_rows,
        },
        report_path,
    )
    print(f"Saved metrics: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
