from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grapher.evaluation.data_io import build_dataset_splits, save_dataset_splits
from grapher.datasets.zinc_utils import is_zinc_dataset, zinc_preparation_hint
from grapher.registry import available_datasets
from grapher.utils.io import load_yaml
from grapher.utils.logging import get_logger
from grapher.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist benchmark dataset splits.")
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="outputs/datasets")
    parser.add_argument("--download-root", type=str, default=None, help="Override raw download/cache root for PyG-backed datasets such as ego_citeseer and QM9.")
    parser.add_argument("--raw-graph-path", type=str, default=None, help="Optional local source graph file, e.g. ind.citeseer.graph for ego_citeseer.")
    parser.add_argument("--max-graphs", type=int, default=None, help="Optional cap on the number of raw graphs converted before splitting.")
    parser.add_argument("--strict-num-graphs", action="store_true", help="For finite-source datasets such as ego_citeseer, fail instead of using min(requested, available).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing persisted splits and metadata.")
    args = parser.parse_args()

    if is_zinc_dataset(args.dataset):
        parser.error(
            "ZINC cannot be prepared with scripts/prepare_dataset.py because the PyG ZINC "
            "node labels are categorical atom-type ids, not atomic numbers. "
            + zinc_preparation_hint()
        )

    cfg_path = Path(args.config) if args.config else Path("configs/datasets") / f"{args.dataset}.yaml"
    cfg = load_yaml(cfg_path)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.download_root is not None:
        cfg["pyg_root"] = args.download_root
    if args.raw_graph_path is not None:
        cfg["raw_graph_path"] = args.raw_graph_path
    if args.max_graphs is not None:
        cfg["max_graphs"] = int(args.max_graphs)
        # Synthetic and ego builders use `num_graphs`; molecular builders use `max_graphs`.
        # Keep both in sync so the CLI flag actually affects all registered datasets.
        if args.dataset in {"ego_citeseer", "planar", "sbm"}:
            cfg["num_graphs"] = int(args.max_graphs)
    if args.strict_num_graphs:
        cfg["strict_num_graphs"] = True
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    out_dir = Path(args.output_root) / args.dataset
    logger.info("dataset=%s config=%s seed=%s output=%s", args.dataset, cfg_path, seed, out_dir)
    start = time.perf_counter()
    splits = build_dataset_splits(args.dataset, cfg)
    save_dataset_splits(args.dataset, splits, cfg, output_root=args.output_root, force=args.force)
    elapsed = time.perf_counter() - start
    logger.info("Prepared dataset %s with sizes: %s", args.dataset, {k: len(v) for k, v in splits.items()})
    logger.info("Saved dataset artifacts to %s in %.2fs", out_dir, elapsed)


if __name__ == "__main__":
    main()
