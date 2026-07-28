#!/usr/bin/env python
"""Evaluate p_phi(s* | D, z) before building GraphER teacher trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from grapher.data.io import load_dataset_splits
from grapher.evaluation.target_summaries import (
    active_component_names,
    conditional_sample_metrics,
    degree_condition_match_rate,
    evaluate_summary_sets,
    fit_mmd_bandwidths,
    paired_summary_errors,
)
from grapher.generators.summary_vae import (
    ConditionalSummaryVAE,
    load_summary_vae_checkpoint,
)
from grapher.properties.sampler import EmpiricalSummarySampler
from grapher.properties.summary import (
    SummaryConfig,
    extract_summary,
    summary_to_jsonable,
)
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir, load_yaml, save_json


def _subsample(
    values: list[Any],
    limit: int | None,
    rng: np.random.Generator,
) -> list[Any]:
    if limit is None or int(limit) <= 0 or len(values) <= int(limit):
        return list(values)
    indices = rng.choice(len(values), size=int(limit), replace=False)
    return [values[int(index)] for index in indices]


def _apply_condition(
    structural: dict[str, Any],
    condition: dict[str, Any],
) -> dict[str, Any]:
    out = dict(structural)
    sequence = sorted(
        [int(value) for value in condition["degree_sequence"]],
        reverse=True,
    )
    n = int(condition["num_nodes"])
    m = int(sum(sequence) // 2)
    out.update(
        {
            "num_nodes": n,
            "num_edges": m,
            "degree_sequence": sequence,
            "degree_hist": np.asarray(
                condition["degree_hist"], dtype=np.float64
            ),
            "density": float(2.0 * m / (n * (n - 1))) if n > 1 else 0.0,
        }
    )
    return out


def _posterior_reconstructions(
    model: ConditionalSummaryVAE,
    vectorizer: Any,
    targets: list[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    x = vectorizer.to_training_arrays(targets)[0]
    condition = vectorizer.to_condition_array(targets)
    reconstructed: list[dict[str, Any]] = []
    mus: list[np.ndarray] = []
    logvars: list[np.ndarray] = []
    for start in range(0, len(targets), batch_size):
        stop = min(start + batch_size, len(targets))
        batch_x = torch.as_tensor(
            x[start:stop], dtype=torch.float32, device=device
        )
        batch_condition = torch.as_tensor(
            condition[start:stop], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            mu, logvar = model.encode(batch_x, batch_condition)
            outputs = model.decode(mu, batch_condition)
        reconstructed.extend(
            vectorizer.outputs_to_summaries(
                outputs,
                deterministic=True,
                condition_summaries=targets[start:stop],
            )
        )
        mus.append(mu.detach().cpu().numpy())
        logvars.append(logvar.detach().cpu().numpy())
    return reconstructed, np.concatenate(mus), np.concatenate(logvars)


def _conditional_prior_samples(
    model: ConditionalSummaryVAE,
    vectorizer: Any,
    conditions: list[dict[str, Any]],
    *,
    samples_per_condition: int,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> list[list[dict[str, Any]]]:
    repeated_conditions = [
        condition
        for condition in conditions
        for _ in range(samples_per_condition)
    ]
    condition_array = vectorizer.to_condition_array(repeated_conditions)
    flat: list[dict[str, Any]] = []
    for start in range(0, len(repeated_conditions), batch_size):
        stop = min(start + batch_size, len(repeated_conditions))
        batch_condition = torch.as_tensor(
            condition_array[start:stop],
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            outputs = model.sample_outputs(batch_condition, device=device)
        flat.extend(
            vectorizer.outputs_to_summaries(
                outputs,
                rng=rng,
                deterministic=False,
                condition_summaries=repeated_conditions[start:stop],
            )
        )
    return [
        flat[index : index + samples_per_condition]
        for index in range(0, len(flat), samples_per_condition)
    ]


def _aggregate_posterior_samples(
    model: ConditionalSummaryVAE,
    vectorizer: Any,
    train_targets: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    *,
    samples_per_condition: int,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> list[list[dict[str, Any]]]:
    train_x = vectorizer.to_training_arrays(train_targets)[0]
    train_condition = vectorizer.to_condition_array(train_targets)
    with torch.no_grad():
        mu, logvar = model.encode(
            torch.as_tensor(train_x, dtype=torch.float32, device=device),
            torch.as_tensor(train_condition, dtype=torch.float32, device=device),
        )

    count = len(conditions) * samples_per_condition
    indices = rng.choice(len(train_targets), size=count, replace=True)
    selected_mu = mu[torch.as_tensor(indices, dtype=torch.long, device=device)]
    selected_logvar = logvar[
        torch.as_tensor(indices, dtype=torch.long, device=device)
    ]
    repeated_conditions = [
        condition
        for condition in conditions
        for _ in range(samples_per_condition)
    ]
    condition_array = vectorizer.to_condition_array(repeated_conditions)
    flat: list[dict[str, Any]] = []
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        with torch.no_grad():
            z = model.reparameterize(
                selected_mu[start:stop],
                selected_logvar[start:stop],
            )
            batch_condition = torch.as_tensor(
                condition_array[start:stop],
                dtype=torch.float32,
                device=device,
            )
            outputs = model.decode(z, batch_condition)
        flat.extend(
            vectorizer.outputs_to_summaries(
                outputs,
                rng=rng,
                deterministic=False,
                condition_summaries=repeated_conditions[start:stop],
            )
        )
    return [
        flat[index : index + samples_per_condition]
        for index in range(0, len(flat), samples_per_condition)
    ]


def _empirical_conditioned_samples(
    train_targets: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    *,
    samples_per_condition: int,
    rng: np.random.Generator,
) -> list[list[dict[str, Any]]]:
    sampler = EmpiricalSummarySampler(train_targets)
    return [
        [
            _apply_condition(
                sampler.sample_conditioned(condition, rng),
                condition,
            )
            for _ in range(samples_per_condition)
        ]
        for condition in conditions
    ]


def _flatten(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [item for group in groups for item in group]


def _latent_diagnostics(mu: np.ndarray, logvar: np.ndarray) -> dict[str, Any]:
    posterior_variance = np.exp(logvar)
    per_dimension_kl = 0.5 * np.mean(
        mu**2 + posterior_variance - 1.0 - logvar,
        axis=0,
    )
    return {
        "mean_kl": float(np.sum(per_dimension_kl)),
        "active_units_kl_gt_0_01": int(np.sum(per_dimension_kl > 0.01)),
        "per_dimension_kl": per_dimension_kl.tolist(),
        "aggregate_mu_mean_l2": float(np.linalg.norm(mu.mean(axis=0))),
        "aggregate_mu_std_mean": float(np.mean(mu.std(axis=0))),
        "posterior_std_mean": float(np.mean(np.sqrt(posterior_variance))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the conditional target-summary CVAE on held-out graphs "
            "before GraphER teacher construction."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-train-graphs", type=int, default=None)
    parser.add_argument("--max-test-graphs", type=int, default=None)
    parser.add_argument("--samples-per-condition", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    dataset_cfg = config.get("dataset", {}) or {}
    generator_cfg = config.get("summary_generator", {}) or {}
    evaluation_cfg = config.get("target_summary_evaluation", {}) or {}
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    checkpoint_path = Path(
        args.checkpoint
        or generator_cfg.get(
            "checkpoint_path",
            "outputs/target_summary_generators/checkpoint.pt",
        )
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing target-summary checkpoint: {checkpoint_path}"
        )
    device = resolve_torch_device(
        args.device or generator_cfg.get("device", "auto")
    )
    model, vectorizer, checkpoint = load_summary_vae_checkpoint(
        checkpoint_path,
        device=device,
    )
    if not isinstance(model, ConditionalSummaryVAE):
        raise TypeError(
            "Target-summary evaluation requires a conditional_summary_vae "
            "checkpoint."
        )

    splits = load_dataset_splits(
        dataset_cfg.get("name", "sbm"),
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", True)),
        config_path=dataset_cfg.get("config_path"),
    )
    train_graphs = list(splits["train"])
    test_graphs = list(splits["test"])
    if not train_graphs or not test_graphs:
        raise RuntimeError(
            "Target-summary evaluation requires non-empty train and test splits."
        )
    train_limit = (
        args.max_train_graphs
        if args.max_train_graphs is not None
        else evaluation_cfg.get("max_train_graphs", 512)
    )
    test_limit = (
        args.max_test_graphs
        if args.max_test_graphs is not None
        else evaluation_cfg.get("max_test_graphs", 256)
    )
    train_graphs = _subsample(train_graphs, train_limit, rng)
    test_graphs = _subsample(test_graphs, test_limit, rng)
    summary_cfg = SummaryConfig.from_dict(
        config.get("summary", {}) or {},
        train_graphs,
    )
    train_targets = [
        extract_summary(graph, summary_cfg) for graph in train_graphs
    ]
    test_targets = [extract_summary(graph, summary_cfg) for graph in test_graphs]

    samples_per_condition = int(
        args.samples_per_condition
        if args.samples_per_condition is not None
        else evaluation_cfg.get("samples_per_condition", 8)
    )
    batch_size = max(
        int(
            args.batch_size
            if args.batch_size is not None
            else evaluation_cfg.get("batch_size", 256)
        ),
        1,
    )
    reconstruction, mu, logvar = _posterior_reconstructions(
        model,
        vectorizer,
        test_targets,
        batch_size=batch_size,
        device=device,
    )
    prior_groups = _conditional_prior_samples(
        model,
        vectorizer,
        test_targets,
        samples_per_condition=samples_per_condition,
        batch_size=batch_size,
        device=device,
        rng=rng,
    )
    aggregate_groups = _aggregate_posterior_samples(
        model,
        vectorizer,
        train_targets,
        test_targets,
        samples_per_condition=samples_per_condition,
        batch_size=batch_size,
        device=device,
        rng=rng,
    )
    empirical_groups = _empirical_conditioned_samples(
        train_targets,
        test_targets,
        samples_per_condition=samples_per_condition,
        rng=rng,
    )

    loss_weights = dict(generator_cfg.get("loss_weights", {}) or {})
    components = active_component_names(vectorizer, loss_weights)
    bandwidths = fit_mmd_bandwidths(
        test_targets,
        train_targets,
        vectorizer,
        component_names=components,
        loss_weights=loss_weights,
    )
    candidates = {
        "train_to_test": train_targets,
        "posterior_reconstruction_to_test": reconstruction,
        "aggregate_posterior_to_test": _flatten(aggregate_groups),
        "empirical_conditioned_to_test": _flatten(empirical_groups),
        "conditional_prior_to_test": _flatten(prior_groups),
    }
    comparisons = {
        name: evaluate_summary_sets(
            test_targets,
            candidate,
            vectorizer,
            component_names=components,
            bandwidths=bandwidths,
            loss_weights=loss_weights,
        )
        for name, candidate in candidates.items()
    }
    paired = {
        "posterior_reconstruction": paired_summary_errors(
            test_targets,
            reconstruction,
            vectorizer,
            component_names=components,
            loss_weights=loss_weights,
        ),
        "conditional_prior": conditional_sample_metrics(
            test_targets,
            prior_groups,
            vectorizer,
            component_names=components,
            loss_weights=loss_weights,
        ),
        "aggregate_posterior": conditional_sample_metrics(
            test_targets,
            aggregate_groups,
            vectorizer,
            component_names=components,
            loss_weights=loss_weights,
        ),
        "empirical_conditioned": conditional_sample_metrics(
            test_targets,
            empirical_groups,
            vectorizer,
            component_names=components,
            loss_weights=loss_weights,
        ),
    }
    flat_prior = _flatten(prior_groups)
    repeated_targets = [
        target
        for target in test_targets
        for _ in range(samples_per_condition)
    ]
    invariants = {
        "posterior_degree_condition_match_rate": degree_condition_match_rate(
            test_targets,
            reconstruction,
        ),
        "prior_degree_condition_match_rate": degree_condition_match_rate(
            repeated_targets,
            flat_prior,
        ),
    }
    report = {
        "dataset": dataset_cfg.get("name", "sbm"),
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "num_train_summaries": len(train_targets),
        "num_test_summaries": len(test_targets),
        "samples_per_condition": samples_per_condition,
        "active_components": components,
        "mmd_bandwidths_from_train_test": bandwidths,
        "comparisons": comparisons,
        "conditional_metrics": paired,
        "invariants": invariants,
        "latent_diagnostics": _latent_diagnostics(mu, logvar),
        "training_final_metrics": (
            checkpoint.get("metrics", {}).get("final", {})
            if isinstance(checkpoint, dict)
            else {}
        ),
    }

    output_dir = ensure_dir(
        args.output_dir
        or evaluation_cfg.get(
            "output_dir",
            checkpoint_path.parent / "evaluation",
        )
    )
    report_path = output_dir / "target_summary_evaluation.json"
    sample_path = output_dir / "generated_target_summaries.json"
    save_json(report, report_path)
    save_json(
        {
            "test_targets": [
                summary_to_jsonable(item) for item in test_targets
            ],
            "posterior_reconstruction": [
                summary_to_jsonable(item) for item in reconstruction
            ],
            "conditional_prior_samples": [
                [summary_to_jsonable(item) for item in group]
                for group in prior_groups
            ],
        },
        sample_path,
    )

    print("\nTarget-summary distribution matching (lower is better)")
    print(
        f"{'Comparison':38s} {'Structural MMD':>15s} "
        + " ".join(f"{name[:10]:>12s}" for name in components)
    )
    for name, metrics in comparisons.items():
        component_values = " ".join(
            f"{metrics['component_mmd'][component]:12.6f}"
            for component in components
        )
        print(
            f"{name:38s} {metrics['structural_mmd']:15.6f} "
            f"{component_values}"
        )

    print("\nHeld-out conditional diagnostics (lower is better)")
    for name in (
        "conditional_prior",
        "aggregate_posterior",
        "empirical_conditioned",
    ):
        metrics = paired[name]
        print(
            f"{name:24s} "
            f"energy={metrics['conditional_energy_score']:.6f} "
            f"mean_l2={metrics['conditional_mean_l2']:.6f} "
            f"diversity={metrics['within_condition_diversity']:.6f}"
        )
    reconstruction_metrics = paired["posterior_reconstruction"]
    print(
        "posterior_reconstruction "
        f"rmse={reconstruction_metrics['structural_rmse']:.6f} "
        f"mae={reconstruction_metrics['structural_mae']:.6f}"
    )
    print("\nHard degree-condition invariants")
    for key, value in invariants.items():
        print(f"{key}: {value:.6f}")
    print(f"\nSaved report to: {report_path}")
    print(f"Saved generated summaries to: {sample_path}")


if __name__ == "__main__":
    main()
