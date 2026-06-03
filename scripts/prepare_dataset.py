from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grapher.evaluation.data_io import build_dataset_splits, save_dataset_splits
from grapher.registry import available_datasets
from grapher.utils.io import load_pickle, load_yaml
from grapher.utils.logging import get_logger
from grapher.utils.seed import set_seed

logger = get_logger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist benchmark dataset splits.")
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="outputs/datasets")
    parser.add_argument("--download-root", type=str, default=None, help="Override raw download/cache root for PyG-backed datasets such as ego_citeseer, QM9, and ZINC.")
    parser.add_argument("--raw-graph-path", type=str, default=None, help="Optional local source graph file, e.g. ind.citeseer.graph for ego_citeseer.")
    parser.add_argument("--max-graphs", type=int, default=None, help="Optional cap on the number of raw graphs converted before splitting.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing persisted splits and metadata.")
    args = parser.parse_args()

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
