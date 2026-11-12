#!/usr/bin/env python3
"""Held-out log-gap denoising and saved pre-rewiring endpoint diagnostics.

The two populations are deliberately separate. Held-out denoising uses the
held-out graph's eigenbasis (reconstruction only), while endpoint diagnostics
use the training-donor unconditional samples already saved by generation.
Neither operation trains a model or silently replaces a failed sample.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset", choices=("community_small", "ego_small", "qm9", "zinc"), required=True)
    p.add_argument("--serialized-dataset", required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("outputs/datasets"))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--generation-dir", type=Path, required=True)
    p.add_argument("--num-samples", type=int, required=True)
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--times", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    p.add_argument("--noise-repeats", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-graphs", type=int, help="Explicit deterministic held-out prefix; omit for the full split.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--endpoints-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def average(rows: list[dict], key: str):
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(values) / len(values) if values else None


def diagnostic_index(rows: list[dict], ids: list[str]) -> dict[str, dict]:
    """Bind a hash-verified diagnostic list to the parent population."""
    result = {}
    for row in rows:
        sample_id = row.get("sample_id")
        graph_index = row.get("graph_index")
        indexed_id = None
        if graph_index is not None:
            if not isinstance(graph_index, int) or isinstance(graph_index, bool) or not 0 <= graph_index < len(ids):
                raise ValueError("Diagnostic graph index is outside the parent population")
            indexed_id = ids[graph_index]
        if sample_id is None:
            sample_id = indexed_id
        if sample_id not in ids or (indexed_id is not None and sample_id != indexed_id):
            raise ValueError("Diagnostic sample ID/index disagrees with the parent population")
        if sample_id in result:
            raise ValueError(f"Duplicate diagnostic sample ID: {sample_id}")
        result[sample_id] = row
    return result


def endpoint_diagnostics(args, manifest: dict, context: dict) -> dict:
    import networkx as nx
    import numpy as np
    import check_grapher_ablation_artifacts as check
    from grapher.utils.networkx_pickle import load_trusted_networkx_pickle

    root = args.generation_dir
    index = json.loads((root / "ablations" / "ablation_manifest.json").read_text())
    ids = index.get("requested_sample_ids") or index.get("sample_ids")
    check.require(isinstance(ids, list), "Missing parent sample IDs")
    check.require(ids == [f"{args.seed}:{i}" for i in range(args.num_samples)], "Parent sample IDs differ")
    # These arrays contain unconditional diffusion endpoints, before rewiring.
    spectrum_record = manifest.get("sampled_spectra") or manifest.get("spectra")
    spectrum_path = check.artifact(root, spectrum_record, "sampled spectra") if spectrum_record else root / "sampled_spectra.pkl"
    check.require(spectrum_path.is_file(), f"Missing saved spectra: {spectrum_path}")
    # A fallback filename still needs an anchor from the verified manifest.
    if not spectrum_record:
        matches = [v for v in manifest.values() if isinstance(v, dict) and v.get("path") == spectrum_path.name]
        check.require(len(matches) == 1, "Spectra have no unique manifest hash anchor")
        check.artifact(root, matches[0], "sampled spectra")
    spectra = load_trusted_networkx_pickle(spectrum_path)
    check.require(len(spectra) == len(ids), "Saved spectrum population differs")
    id_to_index = {str(s): i for i, s in enumerate(ids)}
    branches = ("T", "D") if args.dataset in ("community_small", "ego_small") else ("T",)
    all_rows = []
    summaries = {}
    for branch in branches:
        folder = root / "ablations" / branch
        branch_manifest = json.loads((folder / "manifest.json").read_text())
        graph_path = check.artifact(folder, branch_manifest["base_graphs"], branch + " graphs")
        graphs = load_trusted_networkx_pickle(graph_path)
        branch_ids = branch_manifest.get("sample_ids", [])
        check.require(len(graphs) == len(branch_ids), "Branch graph/ID counts differ")
        diagnostic_record = branch_manifest.get("diagnostics")
        if diagnostic_record:
            diagnostics = json.loads(check.artifact(folder, diagnostic_record, branch + " diagnostics").read_text())
        else:
            check.require(not (folder / "diagnostics.json").is_file(), "Branch diagnostics exist without a manifest hash anchor")
            diagnostics = {}
        diagnostic_rows = diagnostics.get("per_graph", [])
        by_id = diagnostic_index(diagnostic_rows, ids)
        # D projection diagnostics live in the verified parent generation.
        if branch == "D":
            record = manifest.get("degree_constrained_realization_diagnostics") or manifest.get("degree_realization_diagnostics")
            candidates = [(k, v) for k, v in manifest.items() if isinstance(v, dict) and "realization" in k and str(v.get("path", "")).endswith(".json")]
            check.require(record is not None or len(candidates) <= 1, "Ambiguous parent realization diagnostic artifacts")
            if record is None and candidates:
                record = candidates[0][1]
            if record:
                doc = json.loads(check.artifact(root, record, "realization diagnostics").read_text())
                drows = doc.get("per_graph", []) if isinstance(doc, dict) else doc
                by_id.update(diagnostic_index(drows, ids))
        rows = []
        for sample_id, graph in zip(branch_ids, graphs):
            check.require(str(sample_id) in id_to_index, "Unknown endpoint sample ID")
            n = graph.number_of_nodes()
            order = sorted(graph.nodes())
            a = nx.to_numpy_array(graph, nodelist=order, weight=None, dtype=float)
            actual = np.linalg.eigvalsh(np.diag(a.sum(axis=1)) - a)
            target = np.asarray(spectra[id_to_index[str(sample_id)]], dtype=float)[:n]
            d = by_id.get(str(sample_id), {})
            row = {
                "sample_id": str(sample_id), "variant": branch, "num_nodes": n,
                "num_edges": graph.number_of_edges(), "connected": bool(n > 0 and nx.is_connected(graph)),
                "spectrum_rmse_realized_vs_continuous": float(np.sqrt(np.mean((actual - target) ** 2))),
                "edge_count_residual_vs_continuous_trace": float(graph.number_of_edges() - target.sum() / 2),
                "degree_projection_mae": d.get("raw_to_projected_degree_mae") if branch == "D" else None,
                "degree_exact": d.get("degree_exact") if branch == "D" else None,
                "solver": d.get("realization_solver") if branch == "D" else None,
                "edge_count": d.get("edge_count") if branch == "D" else None,
                "fallback_used": d.get("fallback_used") if branch == "D" else None,
            }
            rows.append(row)
        summaries[branch] = {
            "requested_count": args.num_samples, "completed_count": len(rows),
            "decode_success_per_request": len(rows) / args.num_samples,
            "connected_per_completed": average(rows, "connected"),
            "connected_per_request": sum(r["connected"] for r in rows) / args.num_samples,
            "realized_spectrum_rmse_mean": average(rows, "spectrum_rmse_realized_vs_continuous"),
            "degree_projection_mae_mean": average(rows, "degree_projection_mae"),
            "exact_degree_per_completed": average(rows, "degree_exact"),
            "exact_degree_denominator": sum(r["degree_exact"] is not None for r in rows),
            "edge_count_residual_mean": average(rows, "edge_count_residual_vs_continuous_trace"),
            "failures": branch_manifest.get("failures", []),
        }
        all_rows.extend(rows)
    return {"format": "graphes_endpoint_diagnostics_v1", "population": "unconditional_pre_rewiring", "provenance": context,
            "definitions": {"decode_success_per_request": "retained pre-rewiring graph outputs / requested outputs; not a validity or solver-optimality measure",
                            "exact_degree_per_completed": "fraction among completed outputs with explicitly recorded exact-degree diagnostics; see exact_degree_denominator",
                            "spectrum_rmse": "realized unweighted Laplacian spectrum versus saved continuous endpoint spectrum; distinct from held-out denoising error"},
            "summary": summaries, "per_graph": all_rows}


def heldout_diagnostics(args, training: dict, context: dict) -> dict:
    import networkx as nx
    import numpy as np
    import torch
    import check_grapher_ablation_artifacts as check
    from grapher.models.grapher import vanilla_gsdm as v
    from grapher.utils.networkx_pickle import load_trusted_networkx_pickle

    path = check.artifact(args.run_dir / "train", training["checkpoint"], "checkpoint")
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    device = torch.device(args.device)
    mc = state["model_config"]
    options = training["options"]
    molecular = args.dataset in ("qm9", "zinc")
    if molecular:
        from grapher.models.grapher import hybrid_attributed_loggap as h
        from grapher.models.grapher import hierarchical_attributed_loggap as hh
        check.require(state.get("format") == h.CHECKPOINT_FORMAT, "Diagnostics require the final hybrid molecular architecture")
        mx, ml = h._build_models(mc, state["topology_summary"], state["attribute_summary"], device)
        cfg = hh._topology_summary_config(options)
    else:
        meta = state.get("structure_summary") or state.get("graphlet_summary")
        check.require(bool(meta), "Checkpoint has no summary metadata")
        slices = tuple(tuple(pair) for pair in meta["graphlet_slices"])
        mx, ml = v._build_models(mc, device, graphlet_slices=slices, summary_meta=meta)
        cfg = v._resolved_structure_summary_config(options)
    use_ema = bool(options.get("sample", {}).get("use_ema", False))
    mx.load_state_dict(state["ema_x_state" if use_ema else "model_x_state"])
    ml.load_state_dict(state["ema_spectrum_state" if use_ema else "model_spectrum_state"])
    mx.eval(); ml.eval()
    sx, sl = v._make_sdes({"sde": state["sde"]}, device)
    split_path = args.dataset_root / args.serialized_dataset / f"{args.split}.pkl"
    graphs = load_trusted_networkx_pickle(split_path)
    full_count = len(graphs)
    if args.max_graphs is not None:
        graphs = graphs[:args.max_graphs]
    check.require(bool(graphs), "Held-out split is empty")
    graphs = [nx.convert_node_labels_to_integers(nx.Graph(g), ordering="sorted") for g in graphs]
    check.require(all(g.number_of_nodes() > 0 and nx.is_connected(g) for g in graphs), "Held-out log-gap diagnostics require nonempty connected graphs")
    # Do not refit any normalizer or basis on the test split.
    transform = state["spectral_transform"]
    check.require(transform.get("kind") == "laplacian_log_gap", "Held-out log-gap diagnostics require the checkpoint log-gap parameterization")
    tensors = v._padded_dataset(graphs, max_nodes=int(mc["max_nodes"]), max_feat_num=int(mc["max_feat_num"]), spectral_operator="combinatorial_laplacian")
    targets, _ = v._joint_structure_targets(graphs, cfg)
    target_keys = ["graphlet_histogram", "graphlet_mass", "clustering_histogram", "orbit_histogram", "orbit_log_total", "degree_histogram"]
    rows = []
    with torch.no_grad():
        for time_index, time_value in enumerate(args.times):
            for repeat in range(args.noise_repeats):
                # Restart the same stream for every compatible learning condition.
                rng = torch.Generator(device=device).manual_seed(args.seed + 910001 + 1009 * time_index + 104729 * repeat)
                for start in range(0, len(graphs), args.batch_size):
                    end = min(start + args.batch_size, len(graphs))
                    x0, _adj, flags, sizes, u, lam0 = [t[start:end].to(device) for t in tensors]
                    e_mask = v.eigen_mask_from_flags(flags, str(state["sde"].get("eigen_mask", "laplacian_nonzero_prefix")))
                    s0 = v._operator_eigenvalues_to_spectral_state(lam0, flags, spectral_operator="combinatorial_laplacian", spectral_transform=transform)
                    t = torch.full((end-start,), time_value, device=device)
                    zx = v.mask_x(torch.randn(x0.shape, generator=rng, device=device), flags)
                    zs = torch.randn(s0.shape, generator=rng, device=device) * e_mask
                    mean_x, std_x = sx.marginal_x(x0, t)
                    alpha, sigma = sl.marginal_coeff(t)
                    xt = v.mask_x(mean_x + std_x[:, None, None] * zx, flags)
                    st = (alpha[:, None] * s0 + sigma[:, None] * zs) * e_mask
                    lam_t = v._spectral_state_to_operator_eigenvalues(st, flags, spectral_operator="combinatorial_laplacian", spectral_transform=transform)
                    adj_t = v._operator_to_adjacency_state(v.reconstruct_adjacency(u, lam_t), flags, "combinatorial_laplacian")
                    out = ml.forward_topology(xt, adj_t, flags, u, st) if molecular else ml.forward_all(xt, adj_t, flags, u, st)
                    estimate = ((st - sigma[:, None] * out["spectrum"]) / alpha[:, None]) * e_mask
                    estimate_lam = v._spectral_state_to_operator_eigenvalues(estimate, flags, spectral_operator="combinatorial_laplacian", spectral_transform=transform)
                    summaries = ml.topology_structure_means_from_outputs(out) if molecular else ml.structure_means_from_outputs(out)
                    for j in range(end-start):
                        n = int(sizes[j])
                        err = (estimate[j, 1:n] - s0[j, 1:n]).cpu().numpy()
                        spectrum_err = (estimate_lam[j, :n] - lam0[j, :n]).cpu().numpy()
                        row = {"sample_id": f"{args.split}:{start+j}:t{time_index}:r{repeat}", "graph_index": start+j, "noise_repeat": repeat, "time": time_value, "num_nodes": n,
                               "standardized_log_gap_rmse": float(np.sqrt(np.mean(err**2))) if n > 1 else None,
                               "spectrum_rmse": float(np.sqrt(np.mean(spectrum_err**2))),
                               "epsilon_mse": float(((out["spectrum"][j,1:n] - zs[j,1:n])**2).mean().cpu()) if n > 1 else None}
                        for target_key, target_tensor in zip(target_keys, targets):
                            if target_key in summaries:
                                pred = summaries[target_key][j].detach().cpu().reshape(-1)
                                truth = target_tensor[start+j].reshape(-1)
                                check.require(pred.numel() == truth.numel(), f"Summary target width mismatch: {target_key}")
                                row[target_key + "_mae"] = float((pred - truth).abs().mean())
                        rows.append(row)
                print(f"Held-out diagnostics: t={time_value:g}, repeat={repeat+1}/{args.noise_repeats}", flush=True)
    metric_keys = sorted({k for r in rows for k in r if k.endswith(("_rmse", "_mae", "_mse"))})
    return {"format": "graphes_heldout_prediction_diagnostics_v1", "provenance": context,
            "protocol": {"split": args.split, "split_sha256": check.sha256(split_path), "full_split_count": full_count, "evaluated_graph_count": len(graphs), "selection": "full_split" if args.max_graphs is None else "declared_prefix", "times": args.times, "noise_repeats": args.noise_repeats, "batch_size": args.batch_size, "seed": args.seed, "use_ema": use_ema,
                         "estimator": "s0_hat=(s_t-sigma_t*epsilon_hat)/alpha_t; standardized log gaps; active nonzero modes only", "basis": "held_out_graph_eigenbasis_for_reconstruction_only", "normalizer": "training_checkpoint_only", "aggregation": "mean_of_per_graph_errors_over_fixed_times_and_repeats", "unconditional_generation_metric": False},
            "summary": {k: average(rows, k) for k in metric_keys},
            "metric_denominators": {k: sum(row.get(k) is not None for row in rows) for k in metric_keys},
            "by_time": {str(t): {k: average([r for r in rows if r["time"] == t], k) for k in metric_keys} for t in args.times}, "per_graph": rows}


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.noise_repeats < 1 or args.batch_size < 1 or args.num_samples < 1 or (args.max_graphs is not None and args.max_graphs < 1):
        raise SystemExit("Counts must be positive")
    if not args.times or any(not math.isfinite(t) or not 0 < t < 1 for t in args.times) or len(set(args.times)) != len(args.times):
        raise SystemExit("--times must be distinct finite values strictly between 0 and 1")
    if args.dry_run:
        print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
        return 0
    import check_grapher_ablation_artifacts as check
    cfg = check.config_options(args.config)
    args.kind = "matched" if args.dataset in ("community_small", "ego_small") else "molecular"
    training, context = check.training(args, cfg)
    generation, _generation_context, receipt = check.verify_generation(args, cfg)
    branches = ["T", "D"] if args.kind == "matched" else ["T"]
    check.verify_branches(args, cfg, generation, receipt, branches)
    protocol = {"provenance": context, "config": str(args.config.resolve()), "generation_dir": str(args.generation_dir.resolve()), "times": args.times, "noise_repeats": args.noise_repeats, "split": args.split, "max_graphs": args.max_graphs, "batch_size": args.batch_size, "endpoints_only": args.endpoints_only}
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise SystemExit("Output protocol differs; choose a fresh --output-dir")
    write(protocol_path, protocol)
    endpoints = endpoint_diagnostics(args, generation, context)
    write(args.output_dir / "endpoint_diagnostics.json", endpoints)
    if not args.endpoints_only:
        write(args.output_dir / "heldout_prediction_diagnostics.json", heldout_diagnostics(args, training, context))
    print(f"Saved diagnostics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
