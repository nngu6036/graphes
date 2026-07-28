#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from grapher.data.io import load_dataset_splits
from grapher.generators.summary_vae import (
    ConditionalSummaryVAE,
    load_summary_vae_checkpoint,
)
from grapher.properties.summary import SummaryConfig, extract_summary
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import load_yaml, save_json


def _structural_error(
    vectorizer,
    predicted: dict,
    target: dict,
) -> float:
    left = vectorizer.to_feature_vector(predicted)
    right = vectorizer.to_feature_vector(target)
    return float(np.linalg.norm(left - right) / np.sqrt(max(left.size, 1)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a degree-conditioned target-summary CVAE."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-graphs", type=int, default=128)
    parser.add_argument("--diversity-samples", type=int, default=16)
    parser.add_argument(
        "--output",
        default="outputs/target_summary_verification.json",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    dataset_cfg = config.get("dataset", {}) or {}
    generator_cfg = config.get("summary_generator", {}) or {}
    device = resolve_torch_device(generator_cfg.get("device", "auto"))
    model, vectorizer, _ = load_summary_vae_checkpoint(
        generator_cfg["checkpoint_path"],
        device=device,
    )
    if not isinstance(model, ConditionalSummaryVAE):
        raise TypeError(
            "The checkpoint is not conditional. Set "
            "summary_generator.conditional_on_degree: true and retrain."
        )

    splits = load_dataset_splits(
        dataset_cfg.get("name", "sbm"),
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", True)),
        config_path=dataset_cfg.get("config_path"),
    )
    graphs = list(splits["val"] or splits["test"])[: int(args.max_graphs)]
    if not graphs:
        raise RuntimeError("Validation and test splits are empty.")
    summary_cfg = SummaryConfig.from_dict(config.get("summary", {}) or {}, graphs)
    targets = [extract_summary(graph, summary_cfg) for graph in graphs]
    x = torch.as_tensor(
        np.stack([vectorizer.to_feature_vector(item) for item in targets]),
        dtype=torch.float32,
        device=device,
    )
    condition = torch.as_tensor(
        vectorizer.to_condition_array(targets),
        dtype=torch.float32,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        outputs, _, _ = model(x, condition)
    reconstructed = vectorizer.outputs_to_summaries(
        outputs,
        deterministic=True,
        condition_summaries=targets,
    )
    errors = [
        _structural_error(vectorizer, predicted, target)
        for predicted, target in zip(reconstructed, targets)
    ]
    invariant_matches = [
        predicted["degree_sequence"] == target["degree_sequence"]
        and predicted["num_nodes"] == target["num_nodes"]
        and predicted["num_edges"] == target["num_edges"]
        for predicted, target in zip(reconstructed, targets)
    ]

    diversity_condition = torch.as_tensor(
        np.repeat(
            vectorizer.to_condition_vector(targets[0])[None, :],
            int(args.diversity_samples),
            axis=0,
        ),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        prior_outputs = model.sample_outputs(diversity_condition, device=device)
    prior_summaries = vectorizer.outputs_to_summaries(
        prior_outputs,
        condition_summaries=[targets[0]] * int(args.diversity_samples),
    )
    prior_vectors = np.stack(
        [vectorizer.to_feature_vector(summary) for summary in prior_summaries]
    )
    diversity = float(np.mean(np.std(prior_vectors, axis=0)))

    report = {
        "num_graphs": len(graphs),
        "mean_reconstruction_error": float(np.mean(errors)),
        "median_reconstruction_error": float(np.median(errors)),
        "degree_condition_match_rate": float(np.mean(invariant_matches)),
        "prior_structural_diversity": diversity,
        "checkpoint": str(generator_cfg["checkpoint_path"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, output)
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
