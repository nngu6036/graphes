#!/usr/bin/env python
"""List saved joint-model selections; optionally verify each pair's file hashes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from grapher.rewiring_mlp.generic.joint_checkpointing import (
    REGISTRY_NAME, verify_checkpoint_registry,
)
from grapher.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--verify", action="store_true", help="Check all model/export file hashes and best-joint aliases without unpickling.")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    root = Path(args.training_dir)
    if args.verify:
        registry = verify_checkpoint_registry(root)
    else:
        with (root / REGISTRY_NAME).open(encoding="utf-8") as handle:
            registry = json.load(handle)
    print("Joint checkpoint selections (fixed-bridge validation, not generated MMD)")
    print(f"Training complete: {registry['training_complete']}; last completed epoch: {registry['last_completed_epoch']}")
    print(f"{'Selection':18s} {'Epoch':>6s} {'Joint loss':>12s} {'Structure':>12s} {'Hist W1':>12s} {'Orbit logRMSE':>14s}")
    for kind in ("best_joint", "best_histogram", "best_orbit", "best_graphlet", "last"):
        row = registry["selections"].get(kind)
        if row is None:
            if kind in registry.get("unavailable", {}):
                print(f"{kind}: unavailable ({registry['unavailable'][kind]})")
            continue
        metrics = row["validation_metrics"]
        def value(key):
            result = metrics.get(key)
            return "n/a" if result is None else f"{float(result):.6f}"
        print(f"{kind:18s} {row['epoch']:6d} {value('val_joint_loss'):>12s} {value('val_structure_loss'):>12s} "
              f"{value('val_clustering_histogram_w1'):>12s} {value('val_orbit_summary_log_rmse'):>14s}")
        if 'val_induced_graphlet_histogram_tv' in metrics:
            print(f"  induced graphlet TV: {value('val_induced_graphlet_histogram_tv')}")
        print(f"  joint:  {root / row['checkpoint']}")
        print(f"  degree: {root / row['degree_checkpoint']}")
    print("Default checkpoint.pt / degree_checkpoint.pt: best_joint")
    if args.verify:
        print("File integrity and paired selection markers: PASS")
    if args.json_out:
        save_json(registry, args.json_out)


if __name__ == "__main__":
    main()
