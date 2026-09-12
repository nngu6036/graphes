#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from grapher.data.io import load_dataset_splits
from grapher.rewiring_mlp.generic.clustering import extract_clustering_histogram
from grapher.rewiring_mlp.generic.orbit import extract_orbit_summary
from grapher.rewiring_mlp.generic.cycle_graphlets import extract_cycle_graphlet_histogram
from grapher.rewiring_mlp.generic.induced_graphlets import extract_histogram as extract_induced_histogram
from math import comb
from grapher.rewiring_mlp.generic.spectral_data import (
    build_spectral_diffusion_examples,
    collate_spectral_examples,
)
from grapher.rewiring_mlp.generic.spectral_model import load_topology_spectral_checkpoint
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import load_yaml, save_json


def _limited(values: list[Any], limit: int | None) -> list[Any]:
    if limit is None or int(limit) <= 0:
        return values
    return values[: int(limit)]


def _masked_rmse(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    delta = (a - b) / scale.unsqueeze(1)
    weight = mask.to(delta.dtype)
    numerator = (delta.square() * weight).sum(dim=1)
    denominator = weight.sum(dim=1).clamp_min(1.0)
    return torch.sqrt(numerator / denominator)


def _masked_mae(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    delta = torch.abs((a - b) / scale.unsqueeze(1))
    weight = mask.to(delta.dtype)
    return (delta * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the GraphER spectral denoiser without any graph rewiring. "
            "It samples the same continuous spectral bridge states used in training "
            "and compares source/noisy/predicted spectra against the clean target."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--samples-per-graph", type=int, default=None)
    parser.add_argument("--paths-per-graph", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--source-endpoint-only", action="store_true", help="Evaluate t=0 on HH spectra only, without clean-target information in the input.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    dataset_cfg = dict(config.get("dataset", {}) or {})
    splits = load_dataset_splits(
        str(dataset_cfg.get("name", "sbm")),
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", False)),
        config_path=dataset_cfg.get("config_path"),
    )
    graphs = _limited(list(splits[args.split]), args.max_graphs)
    if not graphs:
        raise ValueError(f"Dataset split {args.split!r} is empty.")

    predictor_cfg = dict(config.get("topology_predictor", {}) or {})
    checkpoint_path = Path(
        args.checkpoint or predictor_cfg.get("checkpoint_path", "")
    )
    if not str(checkpoint_path):
        raise ValueError("No topology predictor checkpoint was supplied.")
    device = resolve_torch_device(args.device)
    model, _summary_cfg, checkpoint = load_topology_spectral_checkpoint(
        checkpoint_path,
        device=device,
    )

    diffusion_cfg = dict(config.get("summary_diffusion", {}) or {})
    if args.samples_per_graph is not None:
        diffusion_cfg["samples_per_graph"] = int(args.samples_per_graph)
    if args.paths_per_graph is not None:
        diffusion_cfg["paths_per_graph"] = int(args.paths_per_graph)

    constructor_cfg = dict(config.get("constructor", {}) or {})
    source_cfg = {
        "ensure_connected_source": bool(
            diffusion_cfg.get(
                "ensure_connected_source",
                constructor_cfg.get("ensure_connected", True),
            )
        ),
        "random_relabel_source": bool(
            diffusion_cfg.get(
                "random_relabel_source",
                constructor_cfg.get("random_relabel", True),
            )
        ),
        "max_repair_trials": int(
            diffusion_cfg.get(
                "max_repair_trials",
                constructor_cfg.get("max_repair_trials", 10000),
            )
        ),
        "source_randomization_steps": int(
            diffusion_cfg.get("source_randomization_steps", 0)
        ),
    }
    spectral_cfg = dict(config.get("spectral_prediction", {}) or {})
    examples, diffusion_report = build_spectral_diffusion_examples(
        graphs,
        diffusion_config=diffusion_cfg,
        source_config=source_cfg,
        spectral_config=spectral_cfg,
        graphlet_basis=None,
        structure_summary_config={
            "clustering_histogram": bool(getattr(model, "predict_clustering_histogram", False)),
            "clustering_bins": int(getattr(model, "clustering_histogram_bins", 100)),
            "orbit_summary": bool(getattr(model, "predict_orbit_summary", False)),
            "orbit_width": int(getattr(model, "orbit_summary_width", 15)),
            "cycle_graphlet_histogram": bool(getattr(model, "predict_cycle_graphlet_histogram", False)),
            "cycle_graphlet_k": int(getattr(model, "cycle_graphlet_k", 3)),
            "induced_graphlet_histogram": bool(getattr(model, "predict_induced_graphlet_histogram", False)),
            "induced_graphlet_k": getattr(model, "induced_graphlet_k", 5),
            "induced_graphlet_scope": getattr(model, "induced_graphlet_scope", "all"),
        },
        seed=int(args.seed),
    )
    if args.source_endpoint_only:
        for example in examples:
            example.time = 0.0
            example.current_spectrum = example.source_spectrum.copy()
    source_histograms = (
        [extract_clustering_histogram(example.current_graph, model.clustering_histogram_bins) for example in examples]
        if getattr(model, "predict_clustering_histogram", False) else None
    )
    source_orbits = (
        [extract_orbit_summary(example.current_graph) for example in examples]
        if getattr(model, "predict_orbit_summary", False) else None
    )
    source_cycles = (
        [extract_cycle_graphlet_histogram(example.current_graph, k=model.cycle_graphlet_k) for example in examples]
        if getattr(model, "predict_cycle_graphlet_histogram", False) else None
    )
    induced_spec = model.induced_graphlet_spec if getattr(model, "predict_induced_graphlet_histogram", False) else None
    source_induced_histograms = None
    if induced_spec is not None:
        cache = {}
        source_induced_histograms = []
        for example in examples:
            graph = example.current_graph
            key = (len(graph), tuple(sorted(tuple(sorted(e)) for e in graph.edges())))
            if key not in cache:
                cache[key] = extract_induced_histogram(graph, induced_spec)
            source_induced_histograms.append(cache[key])
    example_offset = 0
    loader = DataLoader(
        examples,
        batch_size=max(int(args.batch_size), 1),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_spectral_examples,
    )

    all_rows: list[dict[str, float]] = []
    bins: dict[str, list[dict[str, float]]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            predicted = outputs["clean_spectrum"]
            target = batch.clean_spectrum_target
            current = batch.current_spectrum
            source = batch.source_spectrum
            mask = batch.spectrum_mask.bool()
            # Same normalization used by the predictor/loss.  This is intentionally
            # read from the model so the diagnostic cannot silently use a different
            # scale from training.
            scale = model._spectrum_scale(batch)

            pred_rmse = _masked_rmse(predicted, target, mask, scale)
            noisy_rmse = _masked_rmse(current, target, mask, scale)
            source_rmse = _masked_rmse(source, target, mask, scale)
            pred_mae = _masked_mae(predicted, target, mask, scale)
            noisy_mae = _masked_mae(current, target, mask, scale)

            trace_pred = (predicted * mask.to(predicted.dtype)).sum(dim=1)
            trace_target = (target * mask.to(target.dtype)).sum(dim=1)
            lambda1 = torch.abs(predicted[:, 0])
            predicted_clustering = outputs.get("clean_clustering_coefficient")
            target_clustering = batch.clean_clustering_coefficient_target
            predicted_histogram = outputs.get("clean_clustering_histogram")
            target_histogram = batch.clean_clustering_histogram_target
            predicted_orbit = outputs.get("clean_orbit_summary")
            target_orbit = batch.clean_orbit_summary_target
            hist_w1 = hist_tv = source_hist_w1 = None
            orbit_log_rmse = orbit_raw_nrmse = source_orbit_log_rmse = None
            if predicted_histogram is not None and target_histogram is not None:
                cdf_delta = torch.cumsum(predicted_histogram - target_histogram, dim=-1)[..., :-1]
                hist_w1 = cdf_delta.abs().sum(dim=-1) / model.clustering_histogram_bins
                hist_tv = 0.5 * (predicted_histogram - target_histogram).abs().sum(dim=-1)
                source_histogram = torch.as_tensor(
                    np.stack(source_histograms[example_offset:example_offset + predicted.shape[0]]),
                    dtype=predicted_histogram.dtype, device=predicted_histogram.device,
                )
                source_hist_w1 = torch.cumsum(source_histogram - target_histogram, dim=-1)[..., :-1].abs().sum(dim=-1) / model.clustering_histogram_bins
            if predicted_orbit is not None and target_orbit is not None:
                orbit_log_delta = torch.log1p(predicted_orbit.clamp_min(0.0)) - torch.log1p(target_orbit.clamp_min(0.0))
                orbit_log_rmse = torch.sqrt(torch.mean(orbit_log_delta.square(), dim=-1))
                orbit_scale = torch.sqrt(torch.mean(target_orbit.square(), dim=-1)).clamp_min(1.0)
                orbit_raw_nrmse = torch.sqrt(torch.mean((predicted_orbit - target_orbit).square(), dim=-1)) / orbit_scale
                source_orbit = torch.as_tensor(
                    np.stack(source_orbits[example_offset:example_offset + predicted.shape[0]]),
                    dtype=predicted_orbit.dtype, device=predicted_orbit.device,
                )
                source_orbit_log_rmse = torch.sqrt(
                    torch.mean((torch.log1p(source_orbit) - torch.log1p(target_orbit.clamp_min(0.0))).square(), dim=-1)
                )
            cycle_prediction = outputs.get("clean_cycle_graphlet_histogram")
            cycle_target = batch.clean_cycle_graphlet_histogram_target
            cycle_tv = source_cycle_tv = cycle_count_mae = cycle_orbit_count_gap = None
            if cycle_prediction is not None and cycle_target is not None:
                cycle_tv = (cycle_prediction[:, 0] - cycle_target[:, 0]).abs()
                source_cycle = torch.as_tensor(
                    np.stack(source_cycles[example_offset:example_offset + predicted.shape[0]]),
                    dtype=cycle_prediction.dtype, device=cycle_prediction.device,
                )
                source_cycle_tv = (source_cycle[:, 0] - cycle_target[:, 0]).abs()
                n = batch.graph_size
                subsets = torch.ones_like(n)
                for offset in range(model.cycle_graphlet_k):
                    subsets = subsets * (n - float(offset)).clamp_min(0.0)
                for divisor in range(2, model.cycle_graphlet_k + 1):
                    subsets = subsets / float(divisor)
                cycle_count_mae = cycle_tv * subsets
                if predicted_orbit is not None and model.cycle_graphlet_k == 3:
                    # ORCA orbit 3 is triangle participation per node. This
                    # identity applies only to C3; the 15-D orbit summary does
                    # not cover C5.
                    cycle_orbit_count_gap = (cycle_prediction[:, 0] * subsets - predicted_orbit[:, 3] * n / 3).abs()
            induced_tv = source_induced_tv = induced_count_mae = None
            if induced_spec is not None:
                gp = outputs["clean_induced_graphlet_histogram"]
                gt = batch.clean_induced_graphlet_histogram_target.to(gp)
                gs = torch.as_tensor(np.stack(source_induced_histograms[example_offset:example_offset+len(gp)]), device=gp.device, dtype=gp.dtype)
                induced_tv = 0.5 * (gp-gt).abs().sum(-1)
                source_induced_tv = 0.5 * (gs-gt).abs().sum(-1)
                totals = torch.tensor([comb(int(n), induced_spec.k) if n >= induced_spec.k else 0 for n in batch.graph_size.cpu().tolist()], device=gp.device, dtype=gp.dtype)
                induced_count_mae = (gp-gt).abs().mean(-1) * totals
            example_offset += predicted.shape[0]

            for i in range(predicted.shape[0]):
                t = float(batch.time[i].detach().cpu())
                row = {
                    "time": t,
                    "source_nrmse": float(source_rmse[i].detach().cpu()),
                    "noisy_nrmse": float(noisy_rmse[i].detach().cpu()),
                    "predicted_nrmse": float(pred_rmse[i].detach().cpu()),
                    "noisy_nmae": float(noisy_mae[i].detach().cpu()),
                    "predicted_nmae": float(pred_mae[i].detach().cpu()),
                    "trace_abs_error": float(
                        torch.abs(trace_pred[i] - trace_target[i]).detach().cpu()
                    ),
                    "lambda1_abs": float(lambda1[i].detach().cpu()),
                }
                if predicted_clustering is not None and target_clustering is not None:
                    row["clustering_target"] = float(
                        target_clustering[i].detach().cpu()
                    )
                    row["clustering_prediction"] = float(
                        predicted_clustering[i].detach().cpu()
                    )
                    row["clustering_abs_error"] = abs(
                        row["clustering_prediction"] - row["clustering_target"]
                    )
                if hist_w1 is not None:
                    row["clustering_histogram_w1"] = float(hist_w1[i].detach().cpu())
                    row["clustering_histogram_tv"] = float(hist_tv[i].detach().cpu())
                    row["source_clustering_histogram_w1"] = float(source_hist_w1[i].detach().cpu())
                if orbit_log_rmse is not None:
                    row["orbit_summary_log_rmse"] = float(orbit_log_rmse[i].detach().cpu())
                    row["orbit_summary_raw_nrmse"] = float(orbit_raw_nrmse[i].detach().cpu())
                    row["source_orbit_summary_log_rmse"] = float(source_orbit_log_rmse[i].detach().cpu())
                if cycle_tv is not None and float(batch.graph_size[i]) >= model.cycle_graphlet_k:
                    row["cycle_graphlet_histogram_tv"] = float(cycle_tv[i].cpu())
                    row["source_cycle_graphlet_histogram_tv"] = float(source_cycle_tv[i].cpu())
                    row["cycle_graphlet_count_mae"] = float(cycle_count_mae[i].cpu())
                    if cycle_orbit_count_gap is not None:
                        row["cycle_orbit_triangle_count_gap"] = float(cycle_orbit_count_gap[i].cpu())
                if induced_tv is not None and float(batch.graph_size[i]) >= induced_spec.k:
                    row["induced_graphlet_histogram_tv"] = float(induced_tv[i].cpu())
                    row["source_induced_graphlet_histogram_tv"] = float(source_induced_tv[i].cpu())
                    row["induced_graphlet_count_mae"] = float(induced_count_mae[i].cpu())
                all_rows.append(row)
                if t < 0.25:
                    label = "[0.00,0.25)"
                elif t < 0.50:
                    label = "[0.25,0.50)"
                elif t < 0.75:
                    label = "[0.50,0.75)"
                else:
                    label = "[0.75,1.00]"
                bins[label].append(row)

    if not all_rows:
        raise RuntimeError("No spectral diffusion examples were produced.")

    def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
        result = {"count": len(rows)}
        for key in (
            "source_nrmse",
            "noisy_nrmse",
            "predicted_nrmse",
            "noisy_nmae",
            "predicted_nmae",
            "trace_abs_error",
            "lambda1_abs",
        ):
            result[key] = _mean([row[key] for row in rows])
        clustering_rows = [row for row in rows if "clustering_abs_error" in row]
        if clustering_rows:
            result["clustering_coefficient_mae"] = _mean(
                [row["clustering_abs_error"] for row in clustering_rows]
            )
        for key in (
            "clustering_histogram_w1", "clustering_histogram_tv",
            "source_clustering_histogram_w1", "orbit_summary_log_rmse",
            "orbit_summary_raw_nrmse", "source_orbit_summary_log_rmse",
            "cycle_graphlet_histogram_tv", "source_cycle_graphlet_histogram_tv",
            "cycle_graphlet_count_mae", "cycle_orbit_triangle_count_gap",
            "induced_graphlet_histogram_tv", "source_induced_graphlet_histogram_tv", "induced_graphlet_count_mae",
        ):
            values = [row[key] for row in rows if key in row]
            if values:
                result[key] = _mean(values)
        result["denoising_gain_vs_noisy"] = (
            result["noisy_nrmse"] - result["predicted_nrmse"]
        )
        result["relative_nrmse_reduction_vs_noisy"] = (
            result["denoising_gain_vs_noisy"] / max(result["noisy_nrmse"], 1.0e-12)
        )
        return result

    report = {
        "format": "grapher_spectral_denoiser_diagnostic_v1",
        "split": args.split,
        "num_graphs": len(graphs),
        "num_examples": len(all_rows),
        "checkpoint": str(checkpoint_path),
        "checkpoint_format": checkpoint.get("format"),
        "predictor_type": checkpoint.get("predictor_type"),
        "use_graph_context": bool(getattr(model, "use_graph_context", True)),
        "diffusion": diffusion_report,
        "source_endpoint_only": bool(args.source_endpoint_only),
        "clustering_histogram_bins": model.clustering_histogram_bins if getattr(model, "predict_clustering_histogram", False) else None,
        "orbit_summary_width": model.orbit_summary_width if getattr(model, "predict_orbit_summary", False) else None,
        "cycle_graphlet_k": model.cycle_graphlet_k if getattr(model, "predict_cycle_graphlet_histogram", False) else None,
        "cycle_graphlet_representation": (f"[C{model.cycle_graphlet_k}, other] / choose(n,{model.cycle_graphlet_k})" if source_cycles is not None else None),
        "induced_graphlet_metadata": induced_spec.metadata() if induced_spec else None,
        "overall": summarize(all_rows),
        "by_time": {key: summarize(rows) for key, rows in sorted(bins.items())},
    }

    print("Spectral denoiser diagnostic (no rewiring)")
    print(f"  split={args.split} graphs={len(graphs)} examples={len(all_rows)}")
    print(f"  use_graph_context={report['use_graph_context']}")
    overall = report["overall"]
    print(f"  source -> clean NRMSE:    {overall['source_nrmse']:.6f}")
    print(f"  noisy  -> clean NRMSE:    {overall['noisy_nrmse']:.6f}")
    print(f"  pred   -> clean NRMSE:    {overall['predicted_nrmse']:.6f}")
    print(f"  denoising gain:           {overall['denoising_gain_vs_noisy']:.6f}")
    print(
        "  relative NRMSE reduction: "
        f"{100.0 * overall['relative_nrmse_reduction_vs_noisy']:.2f}%"
    )
    print(f"  trace abs error:          {overall['trace_abs_error']:.6e}")
    print(f"  lambda1 abs:              {overall['lambda1_abs']:.6e}")
    if "clustering_coefficient_mae" in overall:
        print(
            "  clustering coefficient MAE: "
            f"{overall['clustering_coefficient_mae']:.6f}"
        )
    if "clustering_histogram_w1" in overall:
        print(f"  HH -> clean histogram W1:   {overall['source_clustering_histogram_w1']:.6f}")
        print(f"  pred -> clean histogram W1: {overall['clustering_histogram_w1']:.6f}")
        print(f"  pred -> clean histogram TV: {overall['clustering_histogram_tv']:.6f}")
    if "orbit_summary_log_rmse" in overall:
        print(f"  HH -> clean orbit log-RMSE:   {overall['source_orbit_summary_log_rmse']:.6f}")
        print(f"  pred -> clean orbit log-RMSE: {overall['orbit_summary_log_rmse']:.6f}")
        print(f"  pred -> clean orbit raw-NRMSE:{overall['orbit_summary_raw_nrmse']:.6f}")
    if "cycle_graphlet_histogram_tv" in overall:
        print(f"  HH -> clean C{model.cycle_graphlet_k} histogram TV:   {overall['source_cycle_graphlet_histogram_tv']:.6f}")
        print(f"  pred -> clean C{model.cycle_graphlet_k} histogram TV: {overall['cycle_graphlet_histogram_tv']:.6f}")
        print(f"  pred -> clean C{model.cycle_graphlet_k} count MAE:    {overall['cycle_graphlet_count_mae']:.6f}")
        if "cycle_orbit_triangle_count_gap" in overall:
            print(f"  cycle/orbit predicted triangle-count gap: {overall['cycle_orbit_triangle_count_gap']:.6f}")
    if "induced_graphlet_histogram_tv" in overall:
        print(f"  Induced graphlets: k={induced_spec.k} scope={induced_spec.scope} bins={induced_spec.width}")
        print(f"  HH -> clean induced graphlet TV:   {overall['source_induced_graphlet_histogram_tv']:.6f}")
        print(f"  pred -> clean induced graphlet TV: {overall['induced_graphlet_histogram_tv']:.6f}")
        print(f"  pred -> clean count MAE per bin:   {overall['induced_graphlet_count_mae']:.6f}")
    print("  by diffusion time:")
    for label, row in report["by_time"].items():
        print(
            f"    {label}: noisy={row['noisy_nrmse']:.6f} "
            f"pred={row['predicted_nrmse']:.6f} "
            f"gain={row['denoising_gain_vs_noisy']:.6f}"
            + (f" hist_w1={row['clustering_histogram_w1']:.6f}" if "clustering_histogram_w1" in row else "")
            + (f" orbit_log_rmse={row['orbit_summary_log_rmse']:.6f}" if "orbit_summary_log_rmse" in row else "")
            + (f" induced_graphlet_tv={row['induced_graphlet_histogram_tv']:.6f}" if "induced_graphlet_histogram_tv" in row else "")
            + (f" cycle{model.cycle_graphlet_k}_tv={row['cycle_graphlet_histogram_tv']:.6f}" if "cycle_graphlet_histogram_tv" in row else "")
        )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_json(report, out)
        print(f"  saved={out}")


if __name__ == "__main__":
    main()
