"""Standalone mixed categorical-node / continuous-adjacency generation."""
from __future__ import annotations

import pickle
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import yaml

from .config import generation_contract, validate_config
from .data import TypedGraphletsMulti, GraphCategoryVocabulary, decode_graph
from .diffusion import (MarginalNoise, WeightedEdgeCodec, cosine_alpha_bar, draw_categories,
                        pair_mask, reverse_weighted, sampling_grid, symmetric_noise)
from .model import OptionCDenoiser, prediction_targets
from .refinement import refine
from .runtime import (atomic_write, load_checkpoint, resolve_device, seed_everything, sha256,
                      validate_output_dir, versions, write_json, save_torch, source_fingerprint)

GENERATION_FORMAT = "grapher_option_c_generation_v1"


def _save_graphs(path, graphs):
    def save(tmp):
        with tmp.open("wb") as handle:
            pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    atomic_write(path, save)


@torch.no_grad()
def generate(cfg, checkpoint_path, output_dir, *, num_graphs, seed, device="auto", overwrite=False):
    cfg = validate_config(cfg)
    if num_graphs < 1:
        raise ValueError("num_graphs must be positive")
    where = resolve_device(device)
    state = load_checkpoint(checkpoint_path)
    if state["generation_contract"] != generation_contract(cfg):
        raise ValueError("The requested config changes trained Option-C semantics; use the matching config or retrain")
    if cfg["sampling"]["use_ema"] and state["ema_state"] is None:
        raise ValueError("sampling.use_ema=true but this checkpoint has no EMA weights")
    output = Path(output_dir)
    validate_output_dir(output, format_name=GENERATION_FORMAT, overwrite=overwrite)
    if output.resolve() in Path(checkpoint_path).resolve().parents:
        raise ValueError("Generation output must not contain the training checkpoint")
    if overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    schema = state["schema"]
    vocab = GraphCategoryVocabulary.from_dict(schema["category_vocabulary"])
    codec = WeightedEdgeCodec(vocab, cfg["edge_representation"])
    basis = TypedGraphletsMulti.from_schema(schema)
    manifest = {"format": GENERATION_FORMAT, "model_id": "option_c", "status": "generating",
                "seed": int(seed), "num_requested": int(num_graphs), "dataset": state["dataset"],
                "checkpoint": {"path": str(Path(checkpoint_path).resolve()), "sha256": sha256(checkpoint_path),
                               "train_seed": state["seed"], "epoch": state["epoch"],
                               "weights": "ema" if cfg["sampling"]["use_ema"] else "raw"},
                "created_at": datetime.now(timezone.utc).isoformat(), "config": cfg,
                "versions": versions(), "device": str(where),
                "implementation": source_fingerprint(), "trained_implementation": state.get("implementation"),
                "contract": {"empirical_node_marginal_role": "terminal_noise_only",
                             "node_reverse": "learned_clean_category_exact_posterior_mixture",
                             "node_final_decoder": "categorical_sample_not_argmax",
                             "adjacency_diffusion": "symmetric_scaled_weights_zero_diagonal",
                             "weight_sampler": cfg["sampling"]["sampler"],
                             "clip_clean": cfg["sampling"]["clip_clean"],
                             "quantization": "final_only_midpoint_threshold_higher_weight_at_tie",
                             "thresholds_physical": codec.thresholds.tolist(),
                             "refinement": "final_only" if cfg["refinement"]["enabled"] else "disabled",
                             "refinement_feedback_into_diffusion": False,
                             "categorical_edge_head": False, "spectral_stochastic_state": False,
                             "soft_degree_consistency_trained": bool(cfg["schema_version"] >= 2),
                             "normalized_laplacian_consistency_trained": bool(cfg["schema_version"] >= 2),
                             "degree_vae": False, "eigenbasis": False,
                             "repair": "none", "reject_invalid_graphs": False,
                             "full_trajectory_degree_preservation": False}}
    write_json(output / "manifest.json", manifest)
    atomic_write(output / "resolved_config.yaml", lambda p: p.write_text(yaml.safe_dump({"option_c": cfg}, sort_keys=False)))
    started = time.monotonic()
    try:
        seed_everything(seed)
        model = OptionCDenoiser(**state["model_config"]).to(where)
        model.load_state_dict(state["ema_state"] if cfg["sampling"]["use_ema"] else state["model_state"])
        model.eval()
        total = cfg["diffusion"]["steps"]
        abar = cosine_alpha_bar(total, device=where)
        node_noise = MarginalNoise(schema["node_marginal"], abar)
        grid = sampling_grid(total, cfg["sampling"]["steps"])
        noise_rng = torch.Generator(device=where).manual_seed(seed+1009)
        size_rng = np.random.default_rng(seed+1013)
        # Isolate refinement RNG completely: --no-refine gives exactly the same
        # pre-rewiring samples at the same checkpoint/seed/batch/step settings.
        refine_rng = np.random.default_rng(seed+1019)
        size_values = np.array(schema["graph_sizes"], dtype=np.int64)
        size_probs = np.array(schema["graph_size_counts"], dtype=np.float64)
        size_probs /= size_probs.sum()
        sizes = size_rng.choice(size_values, size=num_graphs, p=size_probs)
        pre_graphs, graphs, diagnostics = [], [], []
        continuous, unbounded_predictions, node_probabilities = [], [], []
        model_calls, denoise_seconds, refine_seconds = 0, 0., 0.
        for start in range(0, num_graphs, cfg["sampling"]["batch_size"]):
            ns = sizes[start:start+cfg["sampling"]["batch_size"]]
            b, width = len(ns), int(ns.max())
            mask = torch.arange(width, device=where)[None] < torch.as_tensor(ns, device=where)[:, None]
            xt = draw_categories(node_noise.marginal.expand(b, width, -1), noise_rng).masked_fill(~mask, 0)
            wt = symmetric_noise(mask, noise_rng)
            tick = time.monotonic()
            for t_value, u_value in zip(grid, grid[1:]):
                t = torch.full((b,), t_value, dtype=torch.long, device=where)
                u = torch.full((b,), u_value, dtype=torch.long, device=where)
                pred = model(xt, wt, t, mask, total)
                clean = pred["clean_adjacency"]
                if not torch.isfinite(clean).all() or not torch.isfinite(pred["node_logits"]).all():
                    raise FloatingPointError(f"Nonfinite denoiser output at timestep {t_value}")
                if cfg["sampling"]["clip_clean"]:
                    clean = clean.clamp(0., codec.max_scaled) * pair_mask(mask)
                px = pred["node_logits"].softmax(-1)
                xt = draw_categories(node_noise.reverse_probs(px, xt, t, u), noise_rng).masked_fill(~mask, 0)
                wt = reverse_weighted(wt, clean, abar, t, u, mask, noise_rng, sampler=cfg["sampling"]["sampler"])
                model_calls += 1
            if where.type == "cuda":
                torch.cuda.synchronize(where)
            denoise_seconds += time.monotonic()-tick
            x_final, w_final = xt.cpu().numpy(), wt.cpu().numpy()
            raw_pred = pred["clean_adjacency"].cpu().numpy()
            px_final = pred["node_logits"].softmax(-1).cpu()
            for j, n in enumerate(ns.tolist()):
                x = x_final[j, :n]
                w = w_final[j, :n, :n]
                e = codec.decode(w)
                raw = decode_graph(x, e, vocab)
                pre_graphs.append(raw)
                tick = time.monotonic()
                if cfg["refinement"]["enabled"]:
                    target = prediction_targets(pred, j, n, basis, scaled_clean=w)
                    final_e, diag = refine(
                        x, e, target, basis, codec, cfg["graphlets"]["clustering_bins"],
                        cfg["refinement"], refine_rng,
                        spectral_mode=cfg["spectral"]["normalization"],
                        consistency=cfg.get("consistency"), edge_scale=codec.scale)
                else:
                    final_e = e.copy()
                    diag = {"accepted_steps": 0, "tested_candidates": 0, "changed": False,
                            "degree_preserved": True, "typed_degrees_preserved": True}
                refine_seconds += time.monotonic()-tick
                final = decode_graph(x, final_e, vocab)
                graphs.append(final)
                diag.update(index=start+j, n=n, raw_edges=raw.number_of_edges(), final_edges=final.number_of_edges(),
                            connected_pre=nx.is_connected(raw), connected_final=nx.is_connected(final))
                diagnostics.append(diag)
                if cfg["sampling"]["save_continuous"]:
                    # These matrices generated the PRE-rewiring graph, not the refined graph.
                    continuous.append(torch.from_numpy(w.copy())*codec.scale)
                    unbounded_predictions.append(torch.from_numpy(raw_pred[j, :n, :n].copy())*codec.scale)
                    node_probabilities.append(px_final[j, :n].clone())
            print(f"[option-c] generated {len(graphs)}/{num_graphs}; "
                  f"denoise_s={denoise_seconds:.1f} refine_s={refine_seconds:.1f}", flush=True)
        _save_graphs(output / "pre_rewire_graphs.pkl", pre_graphs)
        _save_graphs(output / "base_graphs.pkl", graphs)
        # The established evaluator understands base_graphs.pkl; molecular alias
        # avoids any dependency on the old baseline run directory convention.
        if vocab.node_attribute is not None:
            _save_graphs(output / "molecular_graphs.pkl", graphs)
        if cfg["sampling"]["save_continuous"]:
            save_torch(output / "continuous_adjacencies.pt", {
                "format": "option_c_pre_threshold_physical_weights_v1",
                "pre_threshold_weighted_adjacencies": continuous,
                "unclipped_clean_predictions": unbounded_predictions,
                "node_clean_probabilities": node_probabilities,
                "node_category_values": list(vocab.node_values),
                "corresponds_to": "pre_rewire_graphs.pkl"})
        summary = {"num_requested": num_graphs, "num_generated": len(graphs), "batched_model_calls": model_calls,
                   "reverse_steps_per_graph": len(grid)-1,
                   "connected_pre_rate": float(np.mean([d["connected_pre"] for d in diagnostics])),
                   "connected_final_rate": float(np.mean([d["connected_final"] for d in diagnostics])),
                   "refinement_changed_rate": float(np.mean([d["changed"] for d in diagnostics])),
                   "accepted_swaps_mean": float(np.mean([d["accepted_steps"] for d in diagnostics])),
                   "event_local_degree_preservation_rate": float(np.mean([d["degree_preserved"] for d in diagnostics])),
                   "event_local_typed_degree_preservation_rate": float(np.mean([d["typed_degrees_preserved"] for d in diagnostics])),
                   "denoise_seconds": denoise_seconds, "refinement_seconds": refine_seconds,
                   "total_seconds": time.monotonic()-started,
                   "molecular_validity": "not_computed_use_existing_molecular_evaluator"}
        write_json(output / "diagnostics.json", {"summary": summary, "graphs": diagnostics})
        manifest.update(status="completed", num_generated=len(graphs), diagnostics=summary,
                        base_graphs={"path": "base_graphs.pkl", "sha256": sha256(output / "base_graphs.pkl")},
                        pre_rewire_graphs={"path": "pre_rewire_graphs.pkl", "sha256": sha256(output / "pre_rewire_graphs.pkl")})
        write_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise
