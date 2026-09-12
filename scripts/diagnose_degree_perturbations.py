#!/usr/bin/env python
"""Audit perturbation coverage without loading a neural checkpoint or rewiring.

Only training invariants define the kernels. This script does not regenerate data
or evaluate against held-out degrees. Use the usual graph evaluator for MMD.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from grapher.data.io import load_dataset_splits
from grapher.models.dhvae_hh.degree_perturbation import (
    METHODS, PerturbedEmpiricalDegreeSampler,
)
from grapher.utils.io import apply_config_overrides, ensure_dir, load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probability", type=float, default=None,
                        help="Use 1 to test coverage when every sample requests a perturbation.")
    parser.add_argument("--max-degree", type=int, default=None,
                        help="Optional checkpoint degree-support ceiling; no checkpoint is loaded.")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    config = load_yaml(args.config)
    apply_config_overrides(config, args.overrides)
    data = config.get("dataset", {}) or {}
    if config.get("categorical_state") or config.get("molecular_generation"):
        raise ValueError("This diagnostic supports ordinary generic degree priors only.")
    splits = load_dataset_splits(data.get("name", "sbm"), root=data.get("root", "outputs/datasets"),
                                build_if_missing=False, config_path=data.get("config_path"))
    base = deepcopy((config.get("generation", {}) or {}).get("degree_perturbation", {}) or {})
    # Explicit diagnostic policy: report unavailable kernels instead of aborting
    # coverage analysis. Generation itself obeys its configured policy.
    base["failure_policy"] = "keep_original"
    if args.probability is not None:
        base["probability"] = args.probability
    output = ensure_dir(args.output_dir)
    result = {"seed": args.seed, "num_samples": args.num_samples, "training_only": True,
              "checkpoint_loaded": False, "reports": {}, "dataset_split_sha256": {}}
    for split in ("train", "val", "test"):
        path = Path(data.get("root", "outputs/datasets")) / data.get("name", "sbm") / f"{split}.pkl"
        result["dataset_split_sha256"][split] = hashlib.sha256(path.read_bytes()).hexdigest()
    print("Prior-only coverage audit (not generated-graph MMD)")
    print(f"{'Method':22s} {'Requested':>10s} {'Changed':>10s} {'Novel':>10s} {'Fallbacks':>10s}")
    parent_fingerprint = None
    for method in args.methods:
        settings = {**base, "method": method}
        sampler = PerturbedEmpiricalDegreeSampler.fit_from_graphs(
            splits["train"], settings, seed=args.seed, support_max_degree=args.max_degree)
        rng = np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(3,)))
        for _ in range(args.num_samples):
            sampler.sample(rng)
        report = sampler.report()
        if parent_fingerprint is None:
            parent_fingerprint = report["parent_degree_fingerprint"]
        assert parent_fingerprint == report["parent_degree_fingerprint"], "Parent pairing failed."
        report["checkpoint_support_checked"] = args.max_degree is not None
        save_json(report, output / f"{method}.json")
        result["reports"][method] = {k: v for k, v in report.items() if k != "records"}
        print(f"{method:22s} {report['requested_fraction']:10.4f} {report['changed_fraction']:10.4f} "
              f"{report['novel_degree_fraction']:10.4f} {report['num_identity_fallbacks']:10d}")
        if report["failure_reasons"]:
            print(f"  failure reasons: {report['failure_reasons']}")
    result["parents_identical_across_methods"] = True
    save_json(result, output / "report.json")
    print(f"Saved prior diagnostics to: {output}")


if __name__ == "__main__":
    main()
