#!/usr/bin/env python
"""Compare empirical/perturbed priors at a fixed checkpoint and refiner.

Requires matching parent-degree batches within each generation seed, NOT matching
source graphs: changing source degrees is precisely the ablation being measured.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from grapher.models.dhvae_hh.degree_perturbation import METHODS
from grapher.utils.io import ensure_dir, save_json

METRICS = ("degree_mmd", "clustering_mmd", "orbit_mmd")


def summarize(root: str | Path, *, methods=("empirical", *METHODS), seeds=(42,43,44), reference_split="val") -> dict:
    root = Path(root)
    rows, signatures, parent_hashes = [], [], {}
    for seed in seeds:
        for method in methods:
            folder = root / f"generation_seed_{seed}" / method
            with (folder / "report.json").open(encoding="utf-8") as f:
                generation = json.load(f)
            with (folder / f"evaluation_{reference_split}" / "graph_evaluation_report.json").open(encoding="utf-8") as f:
                evaluation = json.load(f)
            with (folder / "degree_prior_report.json").open(encoding="utf-8") as f:
                prior = json.load(f)
            if generation.get("seed") != seed or evaluation.get("reference_split") != reference_split:
                raise ValueError(f"{folder}: seed/reference mismatch.")
            if not generation.get("checkpoint_sha256") or not evaluation.get("reference_split_sha256"):
                raise ValueError(f"{folder}: missing checkpoint/reference provenance.")
            if method != "empirical" and prior.get("method") != method:
                raise ValueError(f"{folder}: prior method does not match the directory.")
            if method == "empirical" and generation.get("degree_source") != "train_empirical":
                raise ValueError(f"{folder}: expected the training-empirical control.")
            if generation.get("degree_rng_mode") != "independent":
                raise ValueError(f"{folder}: use the new independent-parent configs, not legacy empirical runs.")
            cfg = generation.get("config", {})
            signatures.append({
                "checkpoint_sha256": generation["checkpoint_sha256"],
                "refiner": cfg.get("topology_refiner"), "constructor":cfg.get("constructor"),
                "source_enrichment":cfg.get("source_enrichment"),
                "reference":{k:evaluation.get(k) for k in (
                    "reference_split", "reference_split_sha256", "reference_graph_indices_zero_based",
                    "num_reference_graphs", "num_graphs_evaluated", "generic_mmd_protocol",
                    "generic_clustering_bins", "compute_orbit", "orca_exec")},
            })
            if signatures[-1] != signatures[0]:
                raise ValueError("Different checkpoint, reference, sample counts, or constructor/refiner settings.")
            fingerprint = generation.get("parent_degree_fingerprint")
            if not fingerprint:
                raise ValueError(f"{folder}: missing parent fingerprint.")
            if seed in parent_hashes and parent_hashes[seed] != fingerprint:
                raise ValueError(f"Parent degrees differ within seed {seed}; inspect constructor rejection records.")
            parent_hashes[seed] = fingerprint
            def get_row(name):
                found = [r for r in evaluation["metrics"] if r["comparison"] == name]
                if len(found) != 1:
                    raise ValueError(f"{folder}: missing or ambiguous {name}.")
                return found[0]
            final = get_row(f"{evaluation['generated_stage']}_to_{reference_split}")
            source = get_row(f"hh_source_to_{reference_split}")
            for key in METRICS:
                if not math.isfinite(float(final[key])):
                    raise ValueError(f"{folder}: nonfinite {key}; enable the common orbit evaluator.")
            if not math.isclose(final["degree_mmd"], source["degree_mmd"], abs_tol=1e-12, rel_tol=1e-10):
                raise ValueError(f"{folder}: source/final degree MMD mismatch.")
            if generation.get("num_generated") != evaluation.get("num_graphs_evaluated"):
                raise ValueError(f"{folder}: evaluate the whole generated batch for this comparison.")
            returned = prior.get("returned_records", [])
            diag = [r.get("sampling_diagnostics", {}) for r in returned]
            row = {"method":method, "seed":seed, **{k:float(final[k]) for k in METRICS},
                   **{f"hh_{k}":float(source[k]) for k in METRICS},
                   "mean_graph_runtime_seconds":generation["diagnostics"]["mean_graph_runtime_seconds"],
                   "changed_fraction":sum(bool(d.get("changed")) for d in diag)/max(len(diag),1),
                   "novel_degree_fraction":sum(bool(d.get("novel_vs_training")) for d in diag)/max(len(diag),1),
                   "identity_fallbacks":sum(bool(d.get("fallback_used")) for d in diag)}
            rows.append(row)
    if not rows:
        raise ValueError("No methods/seeds selected.")
    aggregates = []
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        stats = {"method":method, "num_sampling_seeds":len(subset)}
        for metric in (*METRICS,"mean_graph_runtime_seconds","changed_fraction","novel_degree_fraction"):
            vals = [float(r[metric]) for r in subset]
            stats[metric+"_mean"] = statistics.mean(vals)
            stats[metric+"_sample_std"] = statistics.stdev(vals) if len(vals)>1 else None
        aggregates.append(stats)
    return {"reference_split":reference_split, "matched_parent_degrees_per_seed":True,
            "parent_degree_fingerprints":parent_hashes, "protocol":signatures[0],
            "aggregation":"run-level arithmetic mean and sample SD across generation seeds; one fixed checkpoint; not pooled MMD",
            "rows":rows, "aggregates":aggregates}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root",required=True)
    parser.add_argument("--methods",nargs="+",choices=("empirical",*METHODS),default=["empirical",*METHODS])
    parser.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    parser.add_argument("--reference-split",choices=("val","test"),default="val")
    parser.add_argument("--output-dir",default=None)
    args=parser.parse_args()
    result=summarize(args.generation_root, methods=args.methods, seeds=args.seeds, reference_split=args.reference_split)
    out=ensure_dir(args.output_dir or str(Path(args.generation_root)/f"summary_{args.reference_split}"))
    save_json(result,out/"comparison.json")
    for name in ("rows","aggregates"):
        with (out/f"{name}.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(result[name][0]))
            writer.writeheader();writer.writerows(result[name])
    print("Matched parent degree batches per seed: True")
    print("Means across sampling seeds (not independent training runs):")
    print(f"{'Method':22s} {'Degree MMD':>12s} {'Clustering':>12s} {'Orbit':>12s} {'Changed':>10s} {'Time(s)':>10s}")
    for r in result['aggregates']:
        print(f"{r['method']:22s} {r['degree_mmd_mean']:12.6f} {r['clustering_mmd_mean']:12.6f} "
              f"{r['orbit_mmd_mean']:12.6f} {r['changed_fraction_mean']:10.4f} {r['mean_graph_runtime_seconds_mean']:10.2f}")
    print(f"Saved run-level metrics and aggregates to {out}")


if __name__ == "__main__":
    main()
