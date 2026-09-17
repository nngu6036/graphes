"""Option A: spectral denoising is the graph generator; swaps are local corrections.

Every reverse model evaluation decodes a fresh graph from its clean spectral
estimate. Only selected events run structural refinement. The next spectral
step may change all degrees. No HH construction, degree projection, fixed-edge
count threshold, connectivity repair, or component filtering occurs here.
"""
from __future__ import annotations

import json
import pickle
import shutil
import tempfile
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from grapher.models.artifacts import ArtifactLayout
from grapher.models.base import GenerationArtifacts
from grapher.models.gdsm_simple.degree_initialization import align_basis_rows
from grapher.models.gdsm_simple.model import make_schedule
from grapher.models.gdsm_simple.refiner import normalized_adjacency_eigenvalues
from grapher.models.gdsm_simple.spectral_decode import decode_spectral_batch, project_rewiring_feedback
from grapher.models.gdsm_simple.structure3 import graph_summary3, reconcile_summary3, refine_structure3, structure_distances
from grapher.models.gdsm_simple.structured_model import centered_x0, centered_ddim_step, summary_probabilities
from grapher.models.gdsm_simple.structured_pipeline import _load_degree_sampler, make_anchor, make_condition, make_model
from grapher.models.gdsm_simple.wrapper import _jsonable, _resolve_device, _seed_everything, _sha256, _write_json


def _indexed_degrees(graph):
    return [int(graph.degree(v)) for v in range(len(graph))]


def _sample_batch(b, state, init, degree_sampler, basis_rng, degree_rng, anchor_cache):
    max_nodes = int(state["max_nodes"])
    bank_n = np.asarray(state["basis_num_nodes"], dtype=int)
    bank_d, bank_u = state["basis_degree_sequences"], state["basis_eigenvectors"]
    result = {k: [] for k in ("degrees", "bases", "anchors", "conditions", "basis_indices",
                             "initialization", "degree_sampling")}
    for _ in range(b):
        if degree_sampler is None:
            d = np.sort(np.asarray(bank_d[int(degree_rng.integers(len(bank_d)))], dtype=np.int64))[::-1]
            dd = {"source": "empirical_training_split"}
        else:
            sample = degree_sampler.sample(rng=degree_rng)
            d = np.sort(np.asarray(sample["degree_sequence"], dtype=np.int64))[::-1]
            dd = _jsonable(sample.get("sampling_diagnostics", {}))
        n = len(d)
        choices = np.flatnonzero(bank_n == n)
        if n < 2 or n > max_nodes or not len(choices):
            raise ValueError(f"Sampled degree sequence has unsupported size n={n}; training basis support is {sorted(set(bank_n.tolist()))}.")
        j = int(basis_rng.choice(choices))
        basis = align_basis_rows(bank_u[j], np.asarray(bank_d[j]))
        key = (j, tuple(d))
        if key not in anchor_cache:
            anchor_cache[key] = make_anchor(basis, d, init)
        anchor, ad = anchor_cache[key]
        for key, value in (("degrees", d), ("bases", basis), ("anchors", anchor),
                           ("conditions", make_condition(basis, d, anchor, max_nodes, init)),
                           ("basis_indices", j), ("initialization", ad), ("degree_sampling", dd)):
            result[key].append(value)
    return result


@torch.no_grad()
def generate_spectral_decode(wrapper, request, state, training_manifest, options):
    """Called only after the shared Structure3 checkpoint contract is checked."""
    layout = request.run.layout
    generation_id = request.resolved_generation_id
    target = layout.generation_dir(generation_id)
    ArtifactLayout.require_available(target, overwrite=request.overwrite)
    ext = options["extensions"]
    init = ext.get("initialization", {})
    guide = ext.get("structure_guidance", {})
    decode_options = ext.get("spectral_decode", {})
    device = _resolve_device(options.get("runtime", {}))
    _seed_everything(request.generation_seed)
    model = make_model(state["model_config"], state["summary_config"], device)
    model.load_state_dict(state["model_state"]); model.eval()
    dc = state["diffusion"]
    schedule = make_schedule(steps=int(dc["steps"]), beta_start=float(dc["beta_start"]),
                             beta_end=float(dc["beta_end"]), device=device)
    total = len(schedule.alpha_bar)
    sample_cfg = options.get("sample", state["sample"])
    sample_steps = max(2, min(int(sample_cfg["steps"]), total))
    times = torch.linspace(total-1, 0, sample_steps).round().long().unique_consecutive().tolist()
    max_nodes = int(state["max_nodes"])
    batch_size = int(options.get("generation_batch_size", 128))
    if batch_size < 1:
        raise ValueError("generation_batch_size must be positive.")
    threshold = float(sample_cfg.get("threshold", .5))
    enabled = bool(ext.get("degree_preserving_rewiring", True))
    feedback = float(guide.get("spectrum_feedback", .05)) if enabled else 0.
    start_fraction, every = float(guide.get("start_fraction", .35)), int(guide.get("every", 100))
    bins = int(state["summary_config"].get("clustering_bins", 100))
    save_graphs = bool(guide.get("save_intermediate_graphs", False))
    save_degrees = bool(decode_options.get("save_degree_trajectory", False))
    basis_rng = np.random.default_rng(request.generation_seed)
    degree_rng = np.random.default_rng(request.generation_seed+1009)
    rewire_rng = np.random.default_rng(request.generation_seed+1000003)
    generator = torch.Generator(device=device).manual_seed(request.generation_seed)
    sampler, degree_provenance = _load_degree_sampler(init, state, training_manifest, device, request.generation_seed)
    degree_provenance["role"] = "initial_spectral_prior_not_a_hard_invariant"
    degree_provenance["explicit_denoiser_conditioning"] = init.get("conditioning", "degree_basis")
    names = ("base_graphs.pkl", "initial_graphs.pkl", "first_decoded_graphs.pkl",
             "threshold_graphs.pkl", "final_pre_rewire_graphs.pkl", "target_adjacency_eigenvalues.pkl",
             "final_graph_adjacency_eigenvalues.pkl", "sampled_basis_indices.pkl",
             "sampled_degree_sequences.pkl", "final_degree_sequences.pkl",
             "initial_eigenvalue_anchors.pkl", "predicted_structure_summaries.pkl")
    objects = {name: [] for name in names}
    rows, snapshots_all, degree_trajectories = [], [], []
    anchor_cache = {}
    started = time.monotonic()
    while len(rows) < request.num_graphs:
        b = min(batch_size, request.num_graphs-len(rows))
        batch = _sample_batch(b, state, init, sampler, basis_rng, degree_rng, anchor_cache)
        sizes = torch.tensor([len(d) for d in batch["degrees"]], dtype=torch.long, device=device)
        mask = torch.arange(max_nodes, device=device).unsqueeze(0) < sizes.unsqueeze(1)
        anchor = torch.tensor(np.stack([np.pad(a, (0, max_nodes-len(a))) for a in batch["anchors"]]), dtype=torch.float32, device=device)
        condition = torch.tensor(np.stack(batch["conditions"]), dtype=torch.float32, device=device)
        padded_bases = np.zeros((b, max_nodes, max_nodes), dtype=np.float32)
        for i, basis in enumerate(batch["bases"]):
            padded_bases[i, :len(basis), :len(basis)] = basis
        basis_tensor = torch.tensor(padded_bases, device=device)
        x = (anchor + torch.randn((b, max_nodes), generator=generator, device=device))*mask
        # Graph state is overwritten by each new spectral decode, not by HH.
        current_binary = None
        degree_change_steps = torch.zeros(b, dtype=torch.long, device=device)
        edge_change_steps = torch.zeros(b, dtype=torch.long, device=device)
        degree_l1 = torch.zeros(b, dtype=torch.long, device=device)
        first_graphs, sources, finals, final_pre = [None]*b, [None]*b, [None]*b, [None]*b
        final_raw, final_targets, final_values = [None]*b, [None]*b, [None]*b
        events, snapshots, degree_history = [[] for _ in range(b)], [[] for _ in range(b)], []
        for position, scalar_t in enumerate(times):
            t = torch.full((b,), scalar_t, dtype=torch.long, device=device)
            eps, prediction = model(x, t, mask, sizes, total, condition)
            x0 = centered_x0(x, eps, t, anchor, schedule)*mask
            binary, values, order = decode_spectral_batch(x0, basis_tensor, mask, sizes, threshold)
            decoded_degrees = binary.sum(-1)
            if current_binary is not None:
                change = (decoded_degrees-current_binary.sum(-1)).abs()
                degree_change_steps += change.any(-1).long()
                degree_l1 += change.sum(-1)
                edge_change_steps += (binary != current_binary).any(-1).any(-1).long()
            current_binary = binary.clone()
            if save_degrees:
                # Swaps do not alter these degrees, so one vector per decode suffices.
                degree_history.append(decoded_degrees.cpu().numpy().astype(np.int32))
            if position == 0:
                first = binary.cpu().numpy()
                for row, d in enumerate(batch["degrees"]):
                    first_graphs[row] = nx.from_numpy_array(first[row, :len(d), :len(d)])
            last = position == len(times)-1
            due = last or (scalar_t <= start_fraction*(total-1) and position % every == 0)
            if due:
                predictions = {k: v.cpu().numpy() for k, v in summary_probabilities(prediction).items()}
                binary_np, values_np = binary.cpu().numpy(), values.cpu().numpy()
                for row, d in enumerate(batch["degrees"]):
                    n = len(d)
                    before = nx.from_numpy_array(binary_np[row, :n, :n])
                    local_degrees = np.asarray(_indexed_degrees(before), dtype=np.int64)
                    if sources[row] is None:
                        sources[row] = before.copy()
                    raw = {k: v[row].astype(np.float64) for k, v in predictions.items()}
                    # Only the current decoded degrees constrain the local swap.
                    # Using the originally sampled d here would silently reintroduce
                    # the old global constraint into the target-count reconciliation.
                    summary = reconcile_summary3(raw, local_degrees)
                    spectral_target = values_np[row, :n].astype(np.float64)
                    if enabled:
                        after, diag = refine_structure3(before, spectral_target, summary,
                            source=before, config=guide, rng=rewire_rng, visited=set())
                    else:
                        after = before.copy()
                        diag = {"enabled": False, "accepted_steps": 0, "degree_preserved": True,
                                "all_accepted_steps_improve_energy": True,
                                "all_accepted_steps_improve_structure": True}
                    if _indexed_degrees(after) != local_degrees.tolist():
                        raise AssertionError("A local structural event changed the decoded degrees.")
                    diag.update({"timestep": scalar_t, "reverse_step_index": position,
                                 "degrees_before_rewiring": local_degrees.tolist(),
                                 "degrees_after_rewiring": _indexed_degrees(after),
                                 "decoded_degrees_equal_sampled_prior": bool(np.array_equal(local_degrees, d)),
                                 "decoded_degree_l1_from_prior": int(np.abs(local_degrees-d).sum()),
                                 "decoded_connected": nx.is_connected(before),
                                 "reconciled_against": "current_decoded_degrees_not_sampled_prior",
                                 "source_penalty_reference": "this_events_decoded_graph",
                                 "raw_prediction": _jsonable(raw), "reconciled_target": _jsonable(summary)})
                    apply_feedback = bool(feedback > 0 and not last and diag["accepted_steps"] > 0)
                    if apply_feedback:
                        corrected, projection = project_rewiring_feedback(spectral_target, batch["bases"][row], before, after, feedback)
                        # Restore model-coordinate order; decoding used a sorted copy.
                        x0[row, order[row, :n]] = torch.tensor(corrected, dtype=x0.dtype, device=device)
                        diag["feedback_projection"] = projection
                        apply_feedback = bool(projection["representable_feedback_nonzero"])
                    diag["spectrum_feedback_applied"] = apply_feedback
                    events[row].append(diag)
                    after_array = nx.to_numpy_array(after, nodelist=range(n), dtype=bool)
                    current_binary[row, :n, :n] = torch.tensor(after_array, dtype=torch.bool, device=device)
                    if save_graphs:
                        snapshots[row].append({"timestep": scalar_t, "decoded_graph": before.copy(),
                                               "graph": after.copy(), "degree_sequence": local_degrees.tolist()})
                    finals[row], final_pre[row] = after, before
                    final_raw[row], final_targets[row], final_values[row] = raw, summary, spectral_target
            if not last:
                previous = torch.full((b,), times[position+1], dtype=torch.long, device=device)
                # The graph correction changes x0, retaining this step's eps. This
                # is a guided DDIM heuristic, not an exact conditioned reverse law.
                x = centered_ddim_step(x0, eps, previous, anchor, schedule)*mask
        change_counts = degree_change_steps.cpu().tolist()
        edge_counts = edge_change_steps.cpu().tolist()
        l1_counts = degree_l1.cpu().tolist()
        if save_degrees:
            degree_trajectories.append(np.stack(degree_history, axis=1))  # [B,steps,N]
        for row, d in enumerate(batch["degrees"]):
            g, source, before = finals[row], sources[row], final_pre[row]
            final_d, initial_d = _indexed_degrees(g), _indexed_degrees(source)
            bins_actual = len(final_targets[row]["clustering_histogram"])
            payloads = {"base_graphs.pkl": g, "initial_graphs.pkl": source,
                        "first_decoded_graphs.pkl": first_graphs[row],
                        "threshold_graphs.pkl": before, "final_pre_rewire_graphs.pkl": before,
                        "target_adjacency_eigenvalues.pkl": final_values[row],
                        "final_graph_adjacency_eigenvalues.pkl": normalized_adjacency_eigenvalues(g),
                        "sampled_basis_indices.pkl": batch["basis_indices"][row],
                        "sampled_degree_sequences.pkl": d.tolist(), "final_degree_sequences.pkl": final_d,
                        "initial_eigenvalue_anchors.pkl": batch["anchors"][row],
                        "predicted_structure_summaries.pkl": {"raw": final_raw[row], "reconciled": final_targets[row],
                                                              "reconciled_degrees": _indexed_degrees(before)}}
            for name, value in payloads.items():
                objects[name].append(value)
            row_diag = {
                "basis_index": batch["basis_indices"][row], "initialization": batch["initialization"][row],
                "degree_sampling": batch["degree_sampling"][row], "num_nodes": len(g), "num_edges": g.number_of_edges(),
                "sampled_prior_degrees": d.tolist(), "initial_degrees": initial_d, "final_degrees": final_d,
                "initial_to_final_indexed_degree_preserved": initial_d == final_d,
                "initial_to_final_degree_multiset_preserved": sorted(initial_d) == sorted(final_d),
                "prior_to_final_indexed_degree_preserved": d.tolist() == final_d,
                "prior_to_final_degree_multiset_preserved": sorted(d.tolist()) == sorted(final_d),
                "final_degree_l1_from_prior": int(np.abs(np.asarray(final_d)-d).sum()),
                "initial_connected": nx.is_connected(source), "final_connected": nx.is_connected(g),
                "final_num_components": nx.number_connected_components(g),
                "final_num_isolates": nx.number_of_isolates(g),
                "num_spectral_decodes": len(times), "spectral_degree_change_steps": change_counts[row],
                "spectral_edge_change_steps": edge_counts[row], "spectral_degree_change_l1_sum": l1_counts[row],
                "accepted_steps": sum(e["accepted_steps"] for e in events[row]),
                "num_guidance_events": len(events[row]), "events": events[row],
                "local_rewiring_degree_preserved": all(e["degree_preserved"] for e in events[row]),
                "initial_to_final_structure_target": structure_distances(graph_summary3(source, bins_actual), final_targets[row]),
                "final_pre_rewire_to_final_structure_target": structure_distances(graph_summary3(before, bins_actual), final_targets[row]),
                "final_to_final_structure_target": structure_distances(graph_summary3(g, bins_actual), final_targets[row]),
            }
            rows.append(row_diag); snapshots_all.append(snapshots[row])
        print(f"GSDM-Simple Structure3 Option A generated {len(rows)}/{request.num_graphs}", flush=True)
    mean = lambda name: float(np.mean([r[name] for r in rows]))
    aggregate = {
        "enabled": enabled, "mode": "spectral_decode_option_a", "num_graphs": len(rows),
        "global_degree_constraint": False, "decoder": "threshold_clean_estimate_every_reverse_evaluation",
        "local_rewiring_degree_preservation_rate": mean("local_rewiring_degree_preserved"),
        "initial_to_final_indexed_degree_preservation_rate": mean("initial_to_final_indexed_degree_preserved"),
        "initial_to_final_degree_multiset_preservation_rate": mean("initial_to_final_degree_multiset_preserved"),
        "prior_to_final_indexed_degree_preservation_rate": mean("prior_to_final_indexed_degree_preserved"),
        "prior_to_final_degree_multiset_preservation_rate": mean("prior_to_final_degree_multiset_preserved"),
        "mean_final_degree_l1_from_prior": mean("final_degree_l1_from_prior"),
        "mean_spectral_degree_change_steps": mean("spectral_degree_change_steps"),
        "mean_spectral_edge_change_steps": mean("spectral_edge_change_steps"),
        "initial_connected_rate": mean("initial_connected"), "final_connected_rate": mean("final_connected"),
        "mean_final_isolates": mean("final_num_isolates"),
        "mean_accepted_steps": mean("accepted_steps"), "mean_guidance_events": mean("num_guidance_events"),
        "spectrum_feedback": feedback, "feedback_mode": "fixed_basis_delta",
        "mean_feedback_events": float(np.mean([sum(e["spectrum_feedback_applied"] for e in r["events"]) for r in rows])),
        "all_accepted_steps_improve_event_energy": all(e["all_accepted_steps_improve_energy"] for r in rows for e in r["events"]),
        "all_accepted_steps_improve_event_structure": all(e["all_accepted_steps_improve_structure"] for r in rows for e in r["events"]),
        "note": "Local swaps preserve current decoded degrees only. Later spectral decoding may change degrees, edges and connectivity. Energy improvement is event-local.",
    }
    if save_graphs:
        objects["intermediate_graphs.pkl"] = snapshots_all
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gdsm_option_a_generation_", dir=target.parent))
    try:
        hashes = {}
        for name, value in objects.items():
            with (staging/name).open("wb") as handle:
                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            hashes[name] = _sha256(staging/name)
        if save_degrees:
            path = staging/"degree_trajectories.npz"
            np.savez_compressed(path, degrees=np.concatenate(degree_trajectories, axis=0),
                                timesteps=np.asarray(times), num_nodes=np.asarray([r["num_nodes"] for r in rows]))
            hashes[path.name] = _sha256(path)
        _write_json(staging/"rewiring_diagnostics.json", {"aggregate": aggregate, "per_graph": rows})
        _write_json(staging/"manifest.json", {
            "format": "grapher_gdsm_simple_generation_v3", "model_id": wrapper.model_id,
            "variant": "GSDM-Simple-Structure3-OptionA", "run_id": request.run.run_id,
            "generation_id": generation_id, "generation_seed": request.generation_seed,
            "num_requested": request.num_graphs, "num_generated": len(rows),
            "duration_seconds": time.monotonic()-started,
            "base_graphs": {"path": "base_graphs.pkl", "sha256": hashes["base_graphs.pkl"], "role": "final_decode_after_final_local_rewiring"},
            "initial_graphs": {"path": "initial_graphs.pkl", "role": "spectral_decode_at_first_guidance_event_not_degree_exact"},
            "first_decoded_graphs": {"path": "first_decoded_graphs.pkl", "role": "decode_at_first_reverse_model_evaluation"},
            "threshold_graphs": {"path": "threshold_graphs.pkl", "role": "final_pre_rewire_decode_of_same_guided_chain_not_independent_baseline"},
            "final_pre_rewire_graphs": {"path": "final_pre_rewire_graphs.pkl", "role": "paired_final_event_input_same_degrees_as_final_output"},
            "artifact_sha256": hashes,
            "checkpoint": {"path": str(request.checkpoint_path.resolve()), "sha256": _sha256(request.checkpoint_path)},
            "dataset": training_manifest["dataset"], "degree_prior": degree_provenance,
            "empirical_prior": {"eigenvectors": "independent_same_size_training_basis", "test_conditioning": False},
            "initialization": _jsonable(init), "structure_guidance": _jsonable(guide),
            "structural_rewiring": aggregate, "generation_mode": "spectral_decode",
            "decode": {"every_reverse_evaluation": True, "state": "predicted_clean_normalized_adjacency_spectrum",
                       "threshold": threshold, "basis": "fixed_initial_sample", "degree_projection": False,
                       "connectivity": "unconstrained", "feedback_mode": "fixed_basis_delta"},
            "terminal_alpha_bar": float(schedule.alpha_bar[-1]),
            "prior_note": "A source-centred VP prior uses s+noise at small terminal alpha_bar. s stays a coordinate offset, never an imposed degree sequence.",
            "conditioning_note": ("Explicit degree and anchor conditioning blocks are zero; only basis descriptors remain."
                                  if init.get("conditioning", "degree_basis") == "initialization_only" else
                                  "Legacy checkpoint: the sampled degree and anchor remain soft neural conditions, not hard constraints."),
            "posthoc_repair": False, "largest_component_filter": False,
            "construction_note": "No HH realization, fixed-degree projection, edge-count projection or connectivity repair.",
            "structure_schema": {"graphlet_size": 3, "graphlet_order": ["induced_P3", "triangle"],
                                 "orbit_order": ["edge_endpoint", "P3_endpoint", "P3_centre", "triangle_vertex"],
                                 "orbit_statistic": "log1p(mean_node_orbit_count)",
                                 "count_reconciliation": "current_decoded_degrees_not_initial_prior"},
        })
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return GenerationArtifacts(run_dir=layout.run_dir, generation_dir=target,
                               graphs_path=target/"base_graphs.pkl", manifest_path=target/"manifest.json",
                               num_requested=request.num_graphs, num_generated=len(rows),
                               graphs_sha256=hashes["base_graphs.pkl"])
