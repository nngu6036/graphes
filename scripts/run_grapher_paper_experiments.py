#!/usr/bin/env python3
"""Concrete run plan for the missing GraphES paper experiments.

Default: print commands only. Add --execute to run. Run from the repository
root, or use commands/paper_missing.sh from any directory. The source's current
200-epoch protocol is retained. No training or sample count is silently changed.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("community_small", "ego_small", "qm9", "zinc")
SERIALIZED = dict(zip(DATASETS, ("sbm", "ego_small", "qm9_attributed", "zinc")))
GENERIC = DATASETS[:2]
STUDIES = {"budgets": "rewiring_budget", "temperature": "gumbel_temperature", "masks": "molecular_masks", "reservoir": "basis_reservoir", "basis": "basis_sensitivity", "donor_replay": "donor_replay"}
EXPERIMENTS = ("reference", "d0", "learning", "learning_summary", *STUDIES, "diagnostics", "summary_consistency", "audit", "drift", "uncertainty", "coverage", "gallery", "trajectory", "runtime", "training_cost", "cost_summary", "provenance", "baseline", "baseline_summary")
GENERIC_BASELINES = ("graphrnn", "gdss", "digress", "edge", "spectre", "gdsm", "catflow", "defog", "hog_diff")
MOLECULAR_BASELINES = ("gdss", "digress", "catflow", "defog", "hog_diff")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", choices=EXPERIMENTS)
    p.add_argument("--datasets", nargs="+", choices=DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    p.add_argument("--execute", action="store_true", help="Actually run the printed plan; omitted means a read-only preview")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dataset-root", type=Path, default=Path("outputs/datasets"))
    p.add_argument("--output-root", type=Path, default=Path("outputs/baselines"))
    p.add_argument("--report-root", type=Path, default=Path("outputs/ablations/grapher_missing"))
    p.add_argument("--orca-executable", default=os.environ.get("ORCA_EXEC", "/home/quang/orca/orca.out"))
    p.add_argument("--num-samples", type=int, help="Default 1024 generic / 10000 molecular; an override creates a distinct population")
    p.add_argument("--reference-split", choices=("val", "test"), default="test")
    p.add_argument("--evaluation-subdir", help="Read an existing report subdirectory, e.g. evaluation_test/primary")
    p.add_argument("--reference-run-template", default="seed_{seed}_{dataset}_grapher_final_{profile}_g345_v2")
    p.add_argument("--generation-tag", default="paper_ablation_v1")
    p.add_argument("--train-reference", action="store_true", help="For reference/d0 only: train missing compatible 200-epoch reference checkpoints")
    p.add_argument("--phase", choices=("all", "generate", "evaluate"), default="all", help="Sampling studies: generation and evaluation, or either stage alone")
    p.add_argument("--learning-stage", choices=("plan", "train", "generate", "evaluate", "all"), default="all")
    p.add_argument("--variants", nargs="+", help="Learning controls; default all applicable S0-S4,K3,K34,A1,A2")
    p.add_argument("--run-tag", default="paper_learning_v1")
    p.add_argument("--study-tag", default="sampling_v1")
    p.add_argument("--fresh-tag", default="paper_cost_v1", help="Fresh runtime-generation/training namespace; collisions rejected")
    p.add_argument("--cold-preprocessing", action="store_true", help="Training costs: recompute molecular targets with their persistent cache disabled")
    p.add_argument("--profile-batch-size", type=int, help="Runtime only: explicit generation batch size; use 1 under a fresh tag for serial latency measurements")
    p.add_argument("--budgets", nargs="+", default=["0", "1", "1:2", "1:8"])
    p.add_argument("--temperatures", nargs="+", type=float, default=[0, .1, .2, .4, .8])
    p.add_argument("--temperature-rewire-budget", default="1:8")
    p.add_argument("--reservoir-caps", nargs="+", default=["1", "4", "16", "all"])
    p.add_argument("--donor-offsets", nargs="+", type=int, default=[1, 2])
    p.add_argument("--epsilons", nargs="+", type=float, default=[0, .01, .025, .05, .1, .2, .4])
    p.add_argument("--wl-iterations", type=int, default=3)
    p.add_argument("--diagnostic-times", nargs="+", type=float, default=[.1, .3, .5, .7, .9])
    p.add_argument("--noise-repeats", type=int, default=3)
    p.add_argument("--max-heldout-graphs", type=int, help="Optional declared diagnostic prefix; full split by default")
    p.add_argument("--models", nargs="+", choices=GENERIC_BASELINES, help="Baseline evidence commands only; defaults to paper baselines supported on each dataset")
    p.add_argument("--baseline-stage", choices=("train", "generate", "evaluate", "all"), default="evaluate")
    p.add_argument("--baseline-run-template", default="seed_{seed}_{model}_{dataset}_paper_baseline_v1")
    p.add_argument("--baseline-generation-template", default="paper_baseline_v1_seed_{seed}_n_{n}")
    return p


def count(args, dataset):
    return args.num_samples or (1024 if dataset in GENERIC else 10000)


def reference(args, dataset, seed):
    n = count(args, dataset)
    profile = "tau0p4_B1to4" if dataset in GENERIC else "B1to4"
    run = args.reference_run_template.format(seed=seed, dataset=dataset, profile=profile, n=n)
    run_dir = args.output_root / "grapher" / dataset / run
    generation = run_dir / "generations" / f"{args.generation_tag}_seed_{seed}_n_{n}"
    suffix = "matched" if dataset in GENERIC else "paired"
    config = ROOT / f"configs/experiments/grapher_paper_ablation/{dataset}_seed_{seed}_{suffix}.yaml"
    return run, run_dir, generation, config


def command(args, values, seed=None, env=None, accepted=(0,)):
    environment = {"PYTHONPATH": str(ROOT / "src"), "GRAPHER_EIGH_BACKEND": "cpu", "ORCA_EXEC": args.orca_executable}
    if seed is not None:
        environment["PYTHONHASHSEED"] = str(seed)
    environment.update(env or {})
    rendered = ["env", *(f"{key}={value}" for key, value in environment.items()), *map(str, values)]
    print(shlex.join(rendered), flush=True)
    if args.execute:
        process_env = os.environ.copy()
        process_env.update(environment)
        result = subprocess.run(list(map(str, values)), cwd=ROOT, env=process_env)
        if result.returncode not in accepted:
            raise RuntimeError(f"Command exited {result.returncode}; inspect its retained failure report before rerunning")
        if result.returncode:
            args.partial = True


def py(args, script, values, **kwargs):
    return command(args, [sys.executable, "scripts/" + script, *values], **kwargs)


def write_config(args, path, config):
    import yaml
    contents = yaml.safe_dump(config, sort_keys=False)
    print(f"# Configuration: {path}", flush=True)
    if args.execute:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != contents:
            raise RuntimeError(f"Configuration already differs: {path}; choose a new --fresh-tag")
        path.write_text(contents)


def _evaluation(args, dataset, seed, branch, variant):
    destination = branch / f"evaluation_{args.reference_split}"
    graphs = branch / "base_graphs.pkl"
    if dataset in GENERIC:
        values = ["--config", f"configs/evaluation/{dataset}.yaml", "--dataset-root", args.dataset_root,
                  "--reference-split", args.reference_split, "--generic-mmd-protocol", "graphrnn",
                  "--generic-validity-require-connected", "--generic-validity-min-nodes", "1",
                  "--generic-validity-max-nodes", "20" if dataset == "community_small" else "18",
                  "--generated-dir", branch, "--generated-graphs", graphs, "--generated-stage", variant, "--output-dir", destination]
        py(args, "evaluate_graph_generation_report.py", values, seed=seed)
    else:
        values = ["--generated-dir", branch, "--generated-graphs", graphs, "--dataset-root", args.dataset_root,
                  "--dataset", SERIALIZED[dataset], "--reference-split", args.reference_split, "--train-split", "train",
                  "--output-dir", destination, "--strict-raw-metrics", "--metric-molecule-source", "raw_valid",
                  "--nspdk-backend", "eden", "--nspdk-complexity", "4", "--nspdk-bond-label-mode", "hogdiff",
                  "--require-fcd", "--fcd-device", args.device, "--graphlet-mmd", "--graphlet-k-min", "3",
                  "--graphlet-k-max", "5", "--graphlet-topology-filter", "all", "--graphlet-node-attribute", "atomic_num",
                  "--graphlet-edge-attribute", "bond_type", "--graphlet-attributed-backend", "python"]
        if dataset == "zinc":
            values += ["--graphlet-num-samples", "4096"]
        py(args, "evaluate_generated_molecules.py", values, seed=seed)
        py(args, "evaluate_qed.py", ["--generated-graphs", graphs,
            "--reference-graphs", args.dataset_root / SERIALIZED[dataset] / f"{args.reference_split}.pkl",
            "--molecular-report", destination / "molecular_evaluation_metrics.json",
            "--expected-num-generated", count(args, dataset), "--output-dir", destination / "qed"], seed=seed)


def evaluation(args, dataset, seed, branch, variant):
    status_path = branch / f"evaluation_{args.reference_split}" / "evaluation_status.json"
    def status(value, error=None):
        if not args.execute:
            return
        status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = status_path.with_name(status_path.name + ".tmp")
        temporary.write_text(json.dumps({"status": value, "error": error}, indent=2) + "\n")
        temporary.replace(status_path)
    if args.execute and not (branch / "base_graphs.pkl").is_file():
        raise RuntimeError(f"No saved branch graphs to evaluate: {branch}; inspect generation failures")
    status("running")
    try:
        _evaluation(args, dataset, seed, branch, variant)
    except BaseException as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise
    status("complete")


def sampling(args, dataset):
    from types import SimpleNamespace
    from run_grapher_sampling_ablation import arms_for
    study = STUDIES[args.experiment]
    spec = SimpleNamespace(study=study, budgets=args.budgets, temperatures=args.temperatures, reservoir_caps=args.reservoir_caps, donor_offsets=args.donor_offsets, temperature_rewire_budget=args.temperature_rewire_budget)
    arms = [a["name"] for a in arms_for(spec)]
    if study.startswith("basis_"):
        arms = [name + suffix for name in arms for suffix in ("_T", "_TR")]
    elif study == "gumbel_temperature":
        arms = [name for original in arms for name in (original, original.replace("D_tau", "DR_tau", 1))]
    seed_dirs = []
    for seed in args.seeds:
        _, run_dir, generation, config = reference(args, dataset, seed)
        destination = args.report_root / args.study_tag / dataset / study / f"seed_{seed}_n_{count(args,dataset)}"
        seed_dirs += ["--seed-dir", f"{seed}={destination}"]
        if args.phase != "evaluate":
            py(args, "run_grapher_sampling_ablation.py", ["--study", study, "--config", config, "--run-dir", run_dir,
                "--generation-dir", generation, "--dataset", dataset, "--serialized-dataset", SERIALIZED[dataset],
                "--dataset-root", args.dataset_root, "--seed", seed, "--num-samples", count(args,dataset), "--device", args.device,
                "--output-root", destination, "--budgets", *args.budgets, "--temperatures", *args.temperatures,
                "--reservoir-caps", *args.reservoir_caps, "--donor-offsets", *args.donor_offsets,
                "--temperature-rewire-budget", args.temperature_rewire_budget], seed=seed, accepted=(0,3))
        if args.phase != "generate":
            for variant in arms:
                evaluation(args, dataset, seed, destination / variant, variant)
    if args.phase != "generate":
        # Mean/SD only; primary matched decoder contrasts are a separate command.
        py(args, "analyze_grapher_paper_experiments.py", ["summarize", *seed_dirs, "--variants", *arms,
            "--evaluation-subdir", f"evaluation_{args.reference_split}", "--output-dir", args.report_root / args.study_tag / dataset / study / "summary"])


def costs(args, dataset, seed):
    import yaml
    run_id, _, _, cfg_path = reference(args, dataset, seed)
    config = yaml.safe_load(cfg_path.read_text())
    variants = ["training"] if args.experiment == "training_cost" else (["TR", "DR"] if dataset in GENERIC else ["T", "TR"])
    for variant in variants:
        local = copy.deepcopy(config)
        opts = local["grapher"]
        opts["sample"].pop("matched_ablation", None)
        if variant == "training" and args.cold_preprocessing and dataset not in GENERIC:
            opts.setdefault("preprocess", {})["cache_structure_targets"] = False
        if args.experiment == "runtime":
            if args.profile_batch_size is not None:
                opts["generation_batch_size"] = args.profile_batch_size
            if dataset in GENERIC:
                opts["sample"]["degree_constrained_realization"]["enabled"] = variant == "DR"
                opts["graphlet_refinement"]["enabled"] = True
            else:
                opts["attributed_diversification"]["enabled"] = variant == "TR"
        config_path = args.report_root / "costs" / args.fresh_tag / "configs" / f"{dataset}_{seed}_{variant}.yaml"
        write_config(args, config_path, local)
        stage = "train" if variant == "training" else "generate"
        measured_run = f"seed_{seed}_{dataset}_{args.fresh_tag}_training" if stage == "train" else run_id
        generation_id = f"{args.fresh_tag}_{variant}_seed_{seed}_n_{count(args,dataset)}"
        values = ["--output", args.report_root / "costs" / args.fresh_tag / dataset / f"seed_{seed}_{variant}.json",
                  "--label", f"{dataset}_{variant}_seed_{seed}", "--requested-count", count(args,dataset), "--",
                  "--stage", stage, "--dataset", dataset, "--no-common-config", "--wrapper-config", config_path,
                  "--dataset-root", args.dataset_root, "--serialized-dataset", SERIALIZED[dataset], "--output-root", args.output_root,
                  "--seed-id", seed, "--run-id", measured_run, "--device", args.device]
        if stage == "generate":
            values += ["--generation-seed", seed, "--generation-id", generation_id, "--num-samples", count(args,dataset)]
        py(args, "profile_grapher_paper.py", values, seed=seed)


def per_dataset(args, dataset):
    n = count(args, dataset)
    variants = ["T", "D", "TR", "DR"] if dataset in GENERIC else ["T", "TR"]
    evaluation_subdir = args.evaluation_subdir or f"evaluation_{args.reference_split}"
    if args.experiment in ("baseline", "baseline_summary"):
        models = args.models or list(GENERIC_BASELINES if dataset in GENERIC else MOLECULAR_BASELINES)
        unsupported = set(models) - set(GENERIC_BASELINES if dataset in GENERIC else MOLECULAR_BASELINES)
        if unsupported:
            raise ValueError(f"Unsupported molecular baseline combination: {sorted(unsupported)}")
        summary_runs = []
        for seed in args.seeds:
            for model in models:
                run = args.baseline_run_template.format(seed=seed, model=model, dataset=dataset, n=n)
                generation_id = args.baseline_generation_template.format(seed=seed, model=model, dataset=dataset, n=n)
                generation = args.output_root / model / dataset / run / "generations" / generation_id
                summary_runs += ["--run", f"{seed},{model}={generation}"]
                if args.experiment == "baseline_summary":
                    continue
                common = ["--dataset", dataset, "--seed-id", seed, "--run-id", run,
                          "--dataset-root", args.dataset_root, "--serialized-dataset", SERIALIZED[dataset],
                          "--output-root", args.output_root, "--device", args.device,
                          "--wrapper-config", f"configs/baselines/{model}_{dataset}.yaml"]
                if args.baseline_stage in ("train", "all"):
                    py(args, f"run_{model}_baseline.py", [*common, "--stage", "train"], seed=seed)
                if args.baseline_stage in ("generate", "all"):
                    py(args, f"run_{model}_baseline.py", [*common, "--stage", "generate", "--generation-seed", seed,
                        "--generation-id", generation_id, "--num-samples", n], seed=seed)
                if args.baseline_stage in ("evaluate", "all"):
                    evaluation(args, dataset, seed, generation, model)
        if args.experiment == "baseline_summary":
            py(args, "analyze_grapher_paper_experiments.py", ["summarize", *summary_runs, "--variants", *models,
                "--pairing", "seed", "--evaluation-subdir", evaluation_subdir,
                "--output-dir", args.report_root / "baseline" / dataset / f"n_{n}" / "summary"])
            py(args, "analyze_grapher_paper_experiments.py", ["audit", *summary_runs, "--variants", *models,
                "--evaluation-subdir", evaluation_subdir,
                "--output-dir", args.report_root / "baseline" / dataset / f"n_{n}" / "audit"])
        return
    if args.experiment in ("reference", "d0"):
        env = {"SEEDS": " ".join(map(str, args.seeds)), "N": str(n), "DEVICE": args.device,
               "DO_TRAIN": str(int(args.train_reference)), "DO_GENERATE": "1", "DO_EVALUATE": "1", "REUSE": "1",
               "RUN_S0": "0", "REFERENCE_SPLIT": args.reference_split, "GENERATION_TAG": args.generation_tag,
               "RUN_TEMPLATE": args.reference_run_template, "DATASET_ROOT": str(args.dataset_root), "OUTPUT_ROOT": str(args.output_root),
               "REPORT_ROOT": str(args.report_root / "reference"), "PYTHON_BIN": sys.executable,
               "ABLATION_STUDY": "decoder_temperature" if args.experiment == "d0" else "all", "TRAIN_EPOCHS": ""}
        command(args, ["bash", "commands/ablation.sh", dataset], env=env)
        return
    if args.experiment == "learning":
        values = ["--dataset", dataset, "--variants", *(args.variants or ["all"]), "--seeds", *args.seeds,
                  "--stage", args.learning_stage, "--device", args.device, "--dataset-root", args.dataset_root, "--output-root", args.output_root,
                  "--report-root", args.report_root / "learning", "--num-samples", n, "--reference-split", args.reference_split,
                  "--orca-executable", args.orca_executable, "--reference-run-template", args.reference_run_template,
                  "--reference-generation-template", args.generation_tag + "_seed_{seed}_n_{n}", "--run-tag", args.run_tag]
        py(args, "run_grapher_learning_ablation.py", values)
        return
    if args.experiment == "learning_summary":
        sys.path.insert(0, str(ROOT / "src"))
        from grapher.experiments.learning_ablation import applicable_controls
        controls = args.variants or list(applicable_controls(dataset))
        runs = []
        for seed in args.seeds:
            _, _, generation, _ = reference(args, dataset, seed)
            runs += ["--run", f"{seed},T={generation / 'ablations/T'}"]
            for variant in controls:
                run = args.output_root / "grapher" / dataset / f"seed_{seed}_{dataset}_{args.run_tag}_{variant}"
                generated = run / "generations" / f"{args.run_tag}_seed_{seed}_n_{n}"
                runs += ["--run", f"{seed},{variant}={generated}"]
        contrasts = []
        for variant in controls:
            if variant == "A0":
                continue
            reference_variant = "A0" if variant in ("A1", "A2") else "T"
            if reference_variant == "A0" and "A0" not in controls:
                raise ValueError("A1/A2 summaries require A0 among --variants")
            contrasts += ["--contrast", f"{variant}_minus_{reference_variant}={variant},{reference_variant}"]
        py(args, "analyze_grapher_paper_experiments.py", ["summarize", *runs, "--variants", "T", *controls,
            "--pairing", "seed", *contrasts, "--evaluation-subdir", evaluation_subdir,
            "--output-dir", args.report_root / "learning" / dataset / args.run_tag / "summary"])
        return
    if args.experiment in STUDIES:
        sampling(args, dataset)
        return
    if args.experiment in ("audit", "uncertainty"):
        seed_dirs = []
        for seed in args.seeds:
            _, _, generation, _ = reference(args, dataset, seed)
            seed_dirs += ["--seed-dir", f"{seed}={generation / 'ablations'}"]
        mode = "audit" if args.experiment == "audit" else "summarize"
        py(args, "analyze_grapher_paper_experiments.py", [mode, *seed_dirs, "--variants", *variants,
            "--evaluation-subdir", evaluation_subdir, "--output-dir", args.report_root / args.experiment / dataset / f"n_{n}"])
        return
    for seed in args.seeds:
        _, run_dir, generation, config = reference(args, dataset, seed)
        destination = args.report_root / args.experiment / dataset / f"seed_{seed}_n_{n}"
        if args.experiment == "diagnostics":
            values = ["--run-dir", run_dir, "--config", config, "--dataset", dataset, "--serialized-dataset", SERIALIZED[dataset],
                      "--dataset-root", args.dataset_root, "--seed", seed, "--generation-dir", generation, "--num-samples", n,
                      "--split", args.reference_split, "--times", *args.diagnostic_times, "--noise-repeats", args.noise_repeats,
                      "--device", args.device, "--output-dir", destination]
            if args.max_heldout_graphs is not None:
                values += ["--max-graphs", args.max_heldout_graphs]
            py(args, "evaluate_grapher_prediction_diagnostics.py", values, seed=seed)
        elif args.experiment == "summary_consistency":
            # Parent predictions correspond to these pre-rewiring populations.
            graphs = generation / ("pre_realization_graphs.pkl" if dataset in GENERIC else "pre_refinement_graphs.pkl")
            py(args, "evaluate_grapher_structure_summary_predictions.py", ["--generated-dir", generation, "--graphs", graphs,
                "--output", generation / "T_structure_summary_consistency.json"], seed=seed)
            if dataset in GENERIC:
                py(args, "evaluate_grapher_structure_summary_predictions.py", ["--generated-dir", generation, "--graphs", generation / "pre_refinement_graphs.pkl",
                    "--output", generation / "D_structure_summary_consistency.json"], seed=seed)
            else:
                py(args, "evaluate_grapher_typed_graphlet_summary_predictions.py", ["--generated-dir", generation, "--graphs", graphs,
                    "--checkpoint", run_dir / "train/checkpoints/grapher.pt", "--output", generation / "T_typed_summary_consistency.json"], seed=seed)
        elif args.experiment == "coverage":
            populations = [item for variant in variants for item in ("--population", f"{variant}={generation / 'ablations' / variant / 'base_graphs.pkl'}")]
            py(args, "analyze_grapher_paper_experiments.py", ["coverage", "--train-graphs", args.dataset_root / SERIALIZED[dataset] / "train.pkl",
                "--heldout-graphs", args.dataset_root / SERIALIZED[dataset] / f"{args.reference_split}.pkl", "--reference-split", args.reference_split,
                *populations, "--epsilons", *args.epsilons, "--wl-iterations", args.wl_iterations, "--plot", "--output-dir", destination], seed=seed)
        elif args.experiment == "gallery":
            py(args, "analyze_grapher_paper_experiments.py", ["plot", "--ablation-dir", generation / "ablations", "--num-samples", 9, "--layout-seed", 42, "--output-dir", destination], seed=seed)
        elif args.experiment in ("drift", "trajectory"):
            pairs = (("T", "TR"), ("D", "DR")) if dataset in GENERIC else (("T", "TR"),)
            for before, after in pairs:
                values = [args.experiment, "--before-branch", generation / "ablations" / before,
                          "--after-branch", generation / "ablations" / after, "--output-dir", destination / f"{before}_to_{after}"]
                if args.experiment == "drift":
                    values += ["--graphlet-backend", "sampled" if dataset == "zinc" else "exact", "--graphlet-seed", seed, "--plot"]
                    if dataset == "zinc":
                        values += ["--graphlet-samples", 4096]
                else:
                    values += ["--kind", "generic" if dataset in GENERIC else "molecular", "--num-samples", 3, "--layout-seed", 42]
                py(args, "analyze_grapher_rewiring_evidence.py", values, seed=seed)
        elif args.experiment in ("runtime", "training_cost"):
            costs(args, dataset, seed)


def main(argv=None):
    args = parser().parse_args(argv)
    os.chdir(ROOT)
    for name in ("dataset_root", "output_root", "report_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.partial = False
    if args.datasets is None:
        args.datasets = list(GENERIC if args.experiment in ("d0", "budgets", "temperature", "gallery") else (DATASETS[2:] if args.experiment == "masks" else DATASETS))
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(seed not in range(41,46) for seed in args.seeds):
        raise SystemExit("Use distinct configured seeds 41 through 45")
    if args.num_samples is not None and args.num_samples < 1:
        raise SystemExit("--num-samples must be positive")
    if args.profile_batch_size is not None and args.profile_batch_size < 1:
        raise SystemExit("--profile-batch-size must be positive")
    for tag in (args.generation_tag, args.run_tag, args.study_tag, args.fresh_tag):
        if not tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in tag):
            raise SystemExit("Tags must be nonempty path-free identifiers")
    for dataset in args.datasets:
        if args.experiment in ("d0", "budgets", "temperature", "gallery") and dataset not in GENERIC:
            raise SystemExit(f"{args.experiment} is generic-only; molecular D/DR are outside the paper's scope")
        if args.experiment == "masks" and dataset in GENERIC:
            raise SystemExit("masks requires QM9 or ZINC")
    print("# EXECUTING" if args.execute else "# PREVIEW ONLY: add --execute to run", flush=True)
    if args.experiment == "provenance":
        py(args, "inventory_grapher_experiment_artifacts.py", ["--root", args.output_root, "--output", args.report_root / "provenance" / "artifact_inventory.json"])
    elif args.experiment == "cost_summary":
        py(args, "summarize_grapher_paper_costs.py", ["--profile-root", args.report_root / "costs" / args.fresh_tag,
            "--output-dir", args.report_root / "costs" / args.fresh_tag / "summary"])
    else:
        for dataset in args.datasets:
            per_dataset(args, dataset)
    return 3 if args.partial else 0


if __name__ == "__main__":
    raise SystemExit(main())
