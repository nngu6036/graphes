#!/usr/bin/env python3
"""Run verified generation-only GraphES ablations from a saved parent population.

Budget, temperature and molecular-mask studies reuse exact saved endpoints.
Basis studies rerun reverse diffusion with explicitly matched node counts and
noise streams.  No training, checkpoint modification, retry draw, or repair is
performed.  Pickle/checkpoint inputs must be trusted project artifacts.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

STUDIES = ("rewiring_budget", "gumbel_temperature", "molecular_masks",
           "basis_reservoir", "basis_sensitivity", "donor_replay")


def token(value):
    return format(float(value), ".8g").replace("-", "m").replace(".", "p")


def parse_budget(value):
    pieces = value.split(":")
    if len(pieces) not in (1, 2):
        raise ValueError(f"Invalid budget {value!r}; use B or MIN:MAX")
    lo, hi = (int(pieces[0]), int(pieces[-1]))
    if lo < 0 or hi < lo:
        raise ValueError(f"Invalid nonnegative budget range: {value}")
    return lo, hi


def arms_for(args):
    if args.study == "rewiring_budget":
        return [{"name": f"{decoder}_B{lo}" + (f"to{hi}" if lo != hi else ""),
                 "decoder": decoder, "budget_min": lo, "budget_max": hi}
                for decoder in ("T", "D") for lo, hi in map(parse_budget, args.budgets)]
    if args.study == "gumbel_temperature":
        lo, hi = parse_budget(args.temperature_rewire_budget)
        return [{"name": "D_tau" + token(t), "temperature": t, "budget": 0,
                 "rewire_budget_min": lo, "rewire_budget_max": hi}
                for t in args.temperatures]
    if args.study == "molecular_masks":
        return [{"name": "M1_same_bond", "strict_neighbor_mask": False},
                {"name": "M1_typed_mask", "strict_neighbor_mask": True}]
    if args.study == "donor_replay":
        return [{"name": "B_donor_replay", "control": "replay"}]
    if args.study == "basis_reservoir":
        return [{"name": "B_reservoir_" + str(cap), "control": "reservoir",
                 "cap_per_size": None if cap == "all" else int(cap)}
                for cap in args.reservoir_caps]
    return ([{"name": "B_original", "control": "original"}]
            + [{"name": f"B_donor_{offset}", "control": "donor", "donor_offset": offset}
               for offset in args.donor_offsets]
            + [{"name": "B_sign", "control": "sign"},
               {"name": "B_repeated", "control": "repeated"},
               {"name": "B_random_rotation", "control": "random_rotation"}])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", choices=STUDIES, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--generation-dir", type=Path, required=True)
    p.add_argument("--dataset", choices=("community_small", "ego_small", "qm9", "zinc"), required=True)
    p.add_argument("--serialized-dataset", required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num-samples", type=int, required=True)
    p.add_argument("--output-root", type=Path, required=True,
                   help="One unique root per study/dataset/seed; existing arms are verified and reused")
    p.add_argument("--device", default="cpu")
    p.add_argument("--orca-exec", type=Path, help="ORCA binary; otherwise use ORCA_EXEC or PATH")
    p.add_argument("--budgets", nargs="+", default=["0", "1", "1:2", "1:8"])
    p.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.4, 0.8])
    p.add_argument("--temperature-rewire-budget", default="1:8",
                   help="Common rewiring budget distribution for each paired DR temperature arm")
    p.add_argument("--reservoir-caps", nargs="+", default=["1", "4", "16", "all"],
                   help="Training donors per exact node-count stratum (not a total bank cap)")
    p.add_argument("--donor-offsets", nargs="+", type=int, default=[1, 2])
    p.add_argument("--perturbation-sigma", type=float, default=0.5)
    p.add_argument("--rotations", type=int, default=8)
    p.add_argument("--repeated-atol", type=float, default=1e-6)
    p.add_argument("--repeated-rtol", type=float, default=1e-5)
    p.add_argument("--dry-run", action="store_true", help="Print the plan without loading checkpoints or writing files")
    args = p.parse_args(argv)
    args.kind = "molecular" if args.dataset in {"qm9", "zinc"} else "matched"
    try:
        if args.seed < 0 or args.num_samples < 1:
            raise ValueError("Seed must be nonnegative and sample count positive")
        if args.study in {"rewiring_budget", "gumbel_temperature"} and args.kind != "matched":
            raise ValueError("Exact-degree T/D studies are generic-only; molecular D is not supported")
        if args.study == "molecular_masks" and args.kind != "molecular":
            raise ValueError("Molecular-mask studies require qm9 or zinc")
        if any(not math.isfinite(v) or v < 0 for v in args.temperatures):
            raise ValueError("Temperatures must be finite and nonnegative")
        if any(v != "all" and (not v.isdigit() or int(v) < 1) for v in args.reservoir_caps):
            raise ValueError("Reservoir caps must be positive integers or 'all'")
        if any(v <= 0 for v in args.donor_offsets):
            raise ValueError("Donor offsets must be positive integers")
        if args.rotations < 1 or any(not math.isfinite(v) or v < 0 for v in
                                     (args.perturbation_sigma, args.repeated_atol, args.repeated_rtol)):
            raise ValueError("Perturbation parameters must be finite/nonnegative; rotations positive")
        arms = arms_for(args)
        if len({arm["name"] for arm in arms}) != len(arms):
            raise ValueError("Study contains duplicate arm names")
        if args.dry_run:
            print(json.dumps({"study": args.study, "dataset": args.dataset,
                              "seed": args.seed, "num_samples": args.num_samples,
                              "source_generation": str(args.generation_dir),
                              "output_root": str(args.output_root), "arms": arms,
                              "exported_variants": ([arm["name"] + suffix for arm in arms for suffix in ("_T", "_TR")]
                                                    if args.study.startswith("basis_") else
                                                    [name for arm in arms for name in (arm["name"], arm["name"].replace("D_", "DR_", 1))]
                                                    if args.study == "gumbel_temperature" else [arm["name"] for arm in arms]),
                              "requires_verified_200_epoch_parent": True,
                              "reverse_diffusion_rerun": args.study.startswith("basis_")}, indent=2))
            return 0
        import check_grapher_ablation_artifacts as checker
        if args.orca_exec is not None:
            from grapher.properties.summary import configure_orca_executable
            configure_orca_executable(args.orca_exec, required=True)
        cfg = checker.config_options(args.config)
        checker.require(cfg.get("train", {}).get("epochs") == 200,
                        "Paper experiments require the managed final 200-epoch checkpoint")
        generation, context, _ = checker.verify_generation(args, cfg)
        trained, _ = checker.training(args, cfg)
        checkpoint = checker.artifact(args.run_dir / "train", trained["checkpoint"], "checkpoint")
        args.output_root.parent.mkdir(parents=True, exist_ok=True)
        lock = args.output_root.parent / ("." + args.output_root.name + ".lock")
        with lock.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            from grapher.models.grapher.sampling_ablation import run_study
            result = run_study(args, cfg, generation, context, checkpoint, arms)
        print(json.dumps(result, indent=2))
        return 0 if result["failed_count"] == 0 else 3
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f"Sampling ablation rejected: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
