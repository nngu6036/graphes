"""Stage 3: graphlet-summary-guided degree-preserving refinement of vanilla GSDM.

The adjacency-spectrum GSDM sampler is left unchanged.  After its final
continuous adjacency is thresholded, the resulting binary graph becomes the
refinement source and its *indexed* ordinary degree sequence is frozen as a hard
constraint.  A separate topology predictor estimates clean connected induced
3/4/5-node graphlet summaries from the current graph.  Those predictions score
valid double-edge swaps; every accepted move preserves the frozen degree vector
exactly and, for connected sources, connectivity.

The predictor is trained only from the training split.  Training examples are
constructed by random degree-preserving corruptions of each clean training
(or validation) graph; the clean graphlet summary is the fixed target.  The
predictor is independent of the GSDM score networks and is trained after the
vanilla GSDM optimization has finished, so it cannot perturb the vanilla GSDM
training trajectory for a fixed seed/configuration.
"""
from __future__ import annotations

import copy
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader

from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.data import (
    TopologyGraphletExample,
    collate_topology_examples,
    normalize_topology_graph,
)
from grapher.rewiring_mlp.generic.graphlets import extract_topology_graphlet_target
from grapher.rewiring_mlp.generic.model import TopologyGraphletPredictor
from grapher.rewiring_mlp.generic.refiner import (
    TopologyPrediction,
    TopologyRefinerConfig,
    predict_topology_target,
    refine_graph_with_topology_predictions,
)
from grapher.rewiring_mlp.generic.rewiring import (
    propose_valid_topology_swaps,
    topology_state_key,
)
from grapher.rewiring_mlp.properties.summary import SummaryConfig


STAGE3_FORMAT = "gdsm_simple_vanilla_graphlet_refine_v1"


def default_graphlet_refinement_options() -> dict[str, Any]:
    return {
        "enabled": True,
        "graphlet_k_min": 3,
        "graphlet_k_max": 5,
        "graphlet_connected_only": True,
        "graphlet_topology_filter": "all",
        "predictor": {
            "epochs": 500,
            "batch_size": 64,
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0e-5,
            "grad_norm": 1.0,
            "hidden_dim": 128,
            "edge_dim": 64,
            "graph_dim": 128,
            "num_layers": 4,
            "dropout": 0.0,
            "min_concentration": 0.05,
            "max_concentration": 50.0,
            "target_epsilon": 1.0e-5,
            "time_conditioning": "constant_zero",
            "validation_every": 10,
            "early_stopping_patience": 100,
            "loss_weights": {
                "graphlet_mean": 1.0,
                "graphlet_distribution": 0.1,
                "graphlet_mass": 0.25,
                "clustering_mean": 0.0,
                "clustering_distribution": 0.0,
                "orbit": 0.0,
            },
            "corruption": {
                "trajectories_per_graph": 4,
                "states_per_trajectory": 8,
                "max_swaps": 24,
                "proposal_budget": 64,
                "valid_candidate_budget": 32,
                "preserve_connectivity_if_source_connected": True,
            },
        },
        "refiner": {
            "steps": 8,
            "proposal_budget": 256,
            "valid_candidate_budget": 128,
            "preserve_connectivity": True,
            "selection": "greedy",
            "temperature": 0.1,
            "graphlet_weight": 1.0,
            "graphlet_mass_weight": 0.10,
            "clustering_weight": 0.0,
            "orbit_weight": 0.0,
            "accept_only_improving": True,
            "min_improvement": 1.0e-8,
            "min_relative_improvement": 0.0,
            "relative_improvement_epsilon": 1.0e-12,
            "sample_graphlet": False,
            "refresh_prediction_every": 1,
            "refresh_on_plateau": False,
            "reject_revisited_states": True,
        },
        # Vanilla GSDM can in principle emit a disconnected graph.  Degree is
        # the only hard invariant requested in this stage.  We preserve a
        # connected source's connectivity, and leave disconnected sources
        # untouched rather than silently repairing/changing their distribution.
        "disconnected_source_policy": "skip",
        "save_prediction_trace": True,
    }


def default_fixed_target_graphlet_refinement_options() -> dict[str, Any]:
    """Post-generation rewiring driven by one joint auxiliary-head prediction.

    This option block is used by the Laplacian log-gap + joint graphlet model.
    No separate topology predictor is trained: the fixed target comes from the
    graphlet auxiliary head evaluated on the final diffusion state.
    """

    return {
        "enabled": True,
        "target_source": "joint_auxiliary_head",
        "graphlet_k_min": 3,
        "graphlet_k_max": 5,
        "graphlet_connected_only": True,
        "graphlet_topology_filter": "all",
        "refiner": {
            "steps": 8,
            "proposal_budget": 256,
            "valid_candidate_budget": 128,
            "preserve_connectivity": True,
            "selection": "greedy",
            "temperature": 0.1,
            "graphlet_weight": 1.0,
            "graphlet_mass_weight": 0.10,
            "clustering_weight": 0.0,
            "orbit_weight": 0.0,
            "accept_only_improving": True,
            "min_improvement": 1.0e-8,
            "min_relative_improvement": 0.0,
            "relative_improvement_epsilon": 1.0e-12,
            "sample_graphlet": False,
            # The prediction is fixed for the whole refinement trajectory.
            "refresh_prediction_every": 8,
            "refresh_on_plateau": False,
            "reject_revisited_states": True,
        },
        "disconnected_source_policy": "skip",
        "save_prediction_trace": True,
    }


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_update(dict(out[key]), value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_graphlet_refinement_options(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = _deep_update(default_graphlet_refinement_options(), dict(raw or {}))
    validate_graphlet_refinement_options(cfg)
    return cfg


def validate_graphlet_refinement_options(cfg: Mapping[str, Any]) -> None:
    if not bool(cfg.get("enabled", False)):
        raise ValueError("Stage-3 graphlet refinement requires graphlet_refinement.enabled=true")
    kmin = int(cfg.get("graphlet_k_min", 3))
    kmax = int(cfg.get("graphlet_k_max", 5))
    if (kmin, kmax) != (3, 5):
        raise ValueError("Stage 3 is intentionally fixed to graphlet orders 3,4,5")
    if not bool(cfg.get("graphlet_connected_only", True)):
        raise ValueError("Stage 3 uses connected induced topology graphlets only")
    if str(cfg.get("graphlet_topology_filter", "all")).lower() != "all":
        raise ValueError("Stage 3 uses the complete connected graphlet basis")
    predictor = cfg.get("predictor", {}) or {}
    for key in ("epochs", "batch_size"):
        if int(predictor.get(key, 0)) <= 0:
            raise ValueError(f"graphlet_refinement.predictor.{key} must be positive")
    if float(predictor.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("graphlet predictor learning_rate must be positive")
    if str(predictor.get("time_conditioning", "constant_zero")).lower() != "constant_zero":
        raise ValueError("Stage 3 predictor time_conditioning is fixed to constant_zero")
    corruption = predictor.get("corruption", {}) or {}
    for key in ("trajectories_per_graph", "states_per_trajectory", "max_swaps", "proposal_budget", "valid_candidate_budget"):
        if int(corruption.get(key, 0)) <= 0:
            raise ValueError(f"graphlet_refinement.predictor.corruption.{key} must be positive")
    policy = str(cfg.get("disconnected_source_policy", "skip")).lower()
    if policy != "skip":
        raise ValueError("Stage 3 currently requires disconnected_source_policy=skip")
    # Reuse the mature generic GraphER refiner validation.  It requires
    # connectivity-preserving moves; disconnected vanilla sources are skipped.
    TopologyRefinerConfig.from_dict(dict(cfg.get("refiner", {}) or {}))


def validate_fixed_target_graphlet_refinement_options(cfg: Mapping[str, Any]) -> None:
    if not bool(cfg.get("enabled", False)):
        raise ValueError("Fixed-target graphlet refinement requires enabled=true")
    target_source = str(cfg.get("target_source", "joint_auxiliary_head")).lower()
    if target_source not in {"joint_auxiliary_head", "joint_structure_summary_head"}:
        raise ValueError(
            "Fixed-target refinement requires target_source=joint_auxiliary_head "
            "or joint_structure_summary_head"
        )
    if (int(cfg.get("graphlet_k_min", 3)), int(cfg.get("graphlet_k_max", 5))) != (3, 5):
        raise ValueError("Fixed-target refinement is intentionally fixed to graphlet orders 3,4,5")
    if not bool(cfg.get("graphlet_connected_only", True)):
        raise ValueError("Fixed-target refinement uses connected induced graphlets only")
    if str(cfg.get("graphlet_topology_filter", "all")).lower() != "all":
        raise ValueError("Fixed-target refinement uses the complete connected graphlet basis")
    if str(cfg.get("disconnected_source_policy", "skip")).lower() != "skip":
        raise ValueError("Fixed-target refinement currently requires disconnected_source_policy=skip")
    TopologyRefinerConfig.from_dict(dict(cfg.get("refiner", {}) or {}))


def _summary_config(cfg: Mapping[str, Any]) -> SummaryConfig:
    return SummaryConfig.from_dict(
        {
            "clustering_summary": False,
            "spectral_summary": False,
            "motif_proxy": False,
            "orbit_count": False,
            "graphlet_history": True,
            "graphlet_k_min": int(cfg.get("graphlet_k_min", 3)),
            "graphlet_k_max": int(cfg.get("graphlet_k_max", 5)),
            "graphlet_connected_only": bool(cfg.get("graphlet_connected_only", True)),
            "graphlet_topology_filter": str(cfg.get("graphlet_topology_filter", "all")),
            "graphlet_backend": "exact",
            "graphlet_num_samples": None,
        }
    )


def _target_for_graph(
    graph: nx.Graph,
    *,
    basis: TopologyGraphletBasis,
    summary_cfg: SummaryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    target, mass = extract_topology_graphlet_target(
        normalize_topology_graph(graph),
        graphlet_basis=basis,
        summary_config=summary_cfg,
    )
    return target.astype(np.float32), mass.astype(np.float32)


def _corruption_examples_for_graph(
    graph: nx.Graph,
    *,
    basis: TopologyGraphletBasis,
    summary_cfg: SummaryConfig,
    corruption: Mapping[str, Any],
    rng: np.random.Generator,
) -> list[TopologyGraphletExample]:
    clean = normalize_topology_graph(graph)
    target, mass = _target_for_graph(clean, basis=basis, summary_cfg=summary_cfg)
    trajectories = int(corruption.get("trajectories_per_graph", 4))
    states_per = int(corruption.get("states_per_trajectory", 8))
    max_swaps = int(corruption.get("max_swaps", 24))
    proposal_budget = int(corruption.get("proposal_budget", 64))
    valid_budget = int(corruption.get("valid_candidate_budget", 32))
    preserve_if_connected = bool(
        corruption.get("preserve_connectivity_if_source_connected", True)
    )
    preserve = preserve_if_connected and (
        clean.number_of_nodes() <= 1 or nx.is_connected(clean)
    )

    # Always include the clean terminal state once.  Stage 3 is a post-GSDM
    # refiner rather than a diffusion process, so predictor time is deliberately
    # disabled (constant zero) and the current graph carries all state information.
    examples = [
        TopologyGraphletExample(
            current_graph=clean.copy(),
            time=0.0,
            graphlet_target=target,
            graphlet_mass_target=mass,
        )
    ]
    # Exclude zero because the clean example is already present.  Evenly spread
    # recording depths make the predictor see the whole degree fibre trajectory.
    record_steps = sorted(
        set(
            int(v)
            for v in np.rint(np.linspace(1, max_swaps, states_per)).astype(int)
        )
    )
    for _trajectory in range(trajectories):
        current = clean.copy()
        visited = {topology_state_key(current)}
        accepted = 0
        record_index = 0
        while accepted < max_swaps and record_index < len(record_steps):
            actions, candidates, _ = propose_valid_topology_swaps(
                current,
                proposal_budget=proposal_budget,
                valid_candidate_budget=valid_budget,
                preserve_connectivity=preserve,
                rng=rng,
                excluded_states=visited,
            )
            if not actions:
                break
            action = actions[int(rng.integers(len(actions)))]
            current = candidates[action]
            visited.add(topology_state_key(current))
            accepted += 1
            while record_index < len(record_steps) and accepted >= record_steps[record_index]:
                depth = record_steps[record_index]
                examples.append(
                    TopologyGraphletExample(
                        current_graph=current.copy(),
                        time=0.0,
                        graphlet_target=target,
                        graphlet_mass_target=mass,
                    )
                )
                record_index += 1
    return examples


def _make_examples(
    graphs: Sequence[nx.Graph],
    *,
    basis: TopologyGraphletBasis,
    summary_cfg: SummaryConfig,
    corruption: Mapping[str, Any],
    seed: int,
) -> list[TopologyGraphletExample]:
    rng = np.random.default_rng(int(seed))
    rows: list[TopologyGraphletExample] = []
    for graph in graphs:
        rows.extend(
            _corruption_examples_for_graph(
                graph,
                basis=basis,
                summary_cfg=summary_cfg,
                corruption=corruption,
                rng=rng,
            )
        )
    if not rows:
        raise ValueError("Graphlet predictor training produced no examples")
    return rows


def _aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    return {
        key: float(np.mean([row[key] for row in rows if key in row]))
        for key in keys
    }


def train_graphlet_predictor(
    train_graphs: Sequence[nx.Graph],
    val_graphs: Sequence[nx.Graph],
    *,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the Stage-3 predictor without touching GSDM model parameters."""

    cfg = resolve_graphlet_refinement_options(config)
    predictor_cfg = dict(cfg["predictor"])
    corruption = dict(predictor_cfg.get("corruption", {}) or {})
    summary_cfg = _summary_config(cfg)
    basis = TopologyGraphletBasis.from_config(summary_cfg)

    train_examples = _make_examples(
        train_graphs,
        basis=basis,
        summary_cfg=summary_cfg,
        corruption=corruption,
        seed=int(seed) + 310001,
    )
    val_examples = _make_examples(
        val_graphs,
        basis=basis,
        summary_cfg=summary_cfg,
        corruption=corruption,
        seed=int(seed) + 320001,
    )

    model = TopologyGraphletPredictor(
        graphlet_slices=basis.slices,
        clustering_width=0,
        orbit_width=0,
        hidden_dim=int(predictor_cfg.get("hidden_dim", 128)),
        edge_dim=int(predictor_cfg.get("edge_dim", 64)),
        graph_dim=int(predictor_cfg.get("graph_dim", 128)),
        num_layers=int(predictor_cfg.get("num_layers", 4)),
        dropout=float(predictor_cfg.get("dropout", 0.0)),
        min_concentration=float(predictor_cfg.get("min_concentration", 0.05)),
        max_concentration=float(predictor_cfg.get("max_concentration", 50.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(predictor_cfg.get("learning_rate", 3.0e-4)),
        weight_decay=float(predictor_cfg.get("weight_decay", 1.0e-5)),
    )
    batch_size = int(predictor_cfg.get("batch_size", 64))
    loader_generator = torch.Generator().manual_seed(int(seed) + 330001)
    train_loader = DataLoader(
        train_examples,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        collate_fn=collate_topology_examples,
    )
    val_loader = DataLoader(
        val_examples,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_topology_examples,
    )
    loss_weights = {
        str(k): float(v)
        for k, v in (predictor_cfg.get("loss_weights", {}) or {}).items()
    }
    target_epsilon = float(predictor_cfg.get("target_epsilon", 1.0e-5))
    epochs = int(predictor_cfg.get("epochs", 500))
    val_every = max(1, int(predictor_cfg.get("validation_every", 10)))
    patience = max(0, int(predictor_cfg.get("early_stopping_patience", 100)))
    grad_norm = float(predictor_cfg.get("grad_norm", 1.0))

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    started = time.monotonic()

    for epoch in range(1, epochs + 1):
        model.train()
        train_rows: list[dict[str, float]] = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.loss(
                batch,
                loss_weights=loss_weights,
                target_epsilon=target_epsilon,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Stage-3 graphlet predictor loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            optimizer.step()
            train_rows.append(metrics)
        record: dict[str, Any] = {"epoch": epoch}
        record.update({f"train_{k}": v for k, v in _aggregate_metrics(train_rows).items()})

        if epoch == 1 or epoch % val_every == 0 or epoch == epochs:
            model.eval()
            val_rows: list[dict[str, float]] = []
            with torch.no_grad():
                for batch in val_loader:
                    _loss, metrics = model.loss(
                        batch.to(device),
                        loss_weights=loss_weights,
                        target_epsilon=target_epsilon,
                    )
                    val_rows.append(metrics)
            val_metrics = _aggregate_metrics(val_rows)
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
            val_loss = float(val_metrics.get("loss", float("inf")))
            if val_loss < best_val - 1.0e-12:
                best_val = val_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += val_every
        history.append(record)
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            val_text = f" val={record['val_loss']:.6f}" if "val_loss" in record else ""
            print(
                f"Stage3 graphlet predictor epoch {epoch}/{epochs} "
                f"train={record.get('train_loss', float('nan')):.6f}{val_text}",
                flush=True,
            )
        if patience > 0 and best_state is not None and stale >= patience:
            break

    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_epoch = len(history)
        best_val = float(history[-1].get("val_loss", history[-1].get("train_loss", float("nan"))))

    payload = {
        "enabled": True,
        "format": STAGE3_FORMAT,
        "model_state_dict": best_state,
        "model_config": model.model_config(),
        "graphlet_basis": basis.to_dict(),
        "summary_config": asdict(summary_cfg),
        "training_config": copy.deepcopy(cfg),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "num_train_examples": len(train_examples),
        "num_val_examples": len(val_examples),
        "history": history,
    }
    report = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "epochs_completed": len(history),
        "num_train_examples": len(train_examples),
        "num_val_examples": len(val_examples),
        "graphlet_orders": [int(k) for k in basis.sizes],
        "graphlet_width": int(basis.width),
        "duration_seconds": float(time.monotonic() - started),
    }
    return payload, report


def load_graphlet_predictor(
    payload: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[TopologyGraphletPredictor, TopologyGraphletBasis, SummaryConfig]:
    if not bool(payload.get("enabled", False)) or payload.get("format") != STAGE3_FORMAT:
        raise RuntimeError("Checkpoint does not contain a compatible Stage-3 graphlet predictor")
    basis = TopologyGraphletBasis.from_dict(dict(payload["graphlet_basis"]))
    summary_cfg = SummaryConfig.from_dict(dict(payload["summary_config"]))
    model = TopologyGraphletPredictor(**dict(payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, basis, summary_cfg


def refine_generated_graphs(
    graphs: Sequence[nx.Graph],
    *,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[list[nx.Graph], list[list[int]], list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Refine vanilla GSDM outputs while freezing each output's degree vector."""

    cfg = resolve_graphlet_refinement_options(config)
    model, basis, summary_cfg = load_graphlet_predictor(payload, device=device)
    refiner_cfg = TopologyRefinerConfig.from_dict(dict(cfg.get("refiner", {}) or {}))
    rng = np.random.default_rng(int(seed) + 410001)

    refined: list[nx.Graph] = []
    frozen_degrees: list[list[int]] = []
    diagnostics: list[dict[str, Any]] = []
    prediction_traces: list[list[dict[str, Any]]] = []

    for index, raw in enumerate(graphs):
        source = normalize_topology_graph(raw)
        degree_vector = [int(source.degree(node)) for node in sorted(source.nodes())]
        frozen_degrees.append(degree_vector)
        source_connected = bool(source.number_of_nodes() <= 1 or nx.is_connected(source))

        if not source_connected:
            # Do not repair or otherwise alter a disconnected vanilla sample in
            # this controlled stage.  Degree remains exact and the baseline
            # distribution is not silently conditioned on connectedness.
            refined.append(source.copy())
            diagnostics.append(
                {
                    "graph_index": index,
                    "source_connected": False,
                    "skipped": True,
                    "skip_reason": "disconnected_vanilla_source",
                    "accepted_steps": 0,
                    "prediction_calls": 0,
                    "changed": False,
                    "degree_preserved": True,
                }
            )
            prediction_traces.append([])
            continue

        predictions: list[dict[str, Any]] = []

        def recording_predict(*args, **kwargs):
            # This stage is not itself a diffusion trajectory.  Keep the
            # summary predictor state-conditioned but time-independent.
            kwargs["time"] = 0.0
            prediction = predict_topology_target(*args, **kwargs)
            predictions.append(
                {
                    "time": 0.0,
                    "graphlet_target": prediction.graphlet_target.tolist(),
                    "graphlet_mass_target": prediction.graphlet_mass_target.tolist(),
                }
            )
            return prediction

        result, trace = refine_graph_with_topology_predictions(
            source,
            model=model,
            graphlet_basis=basis,
            summary_config=summary_cfg,
            refiner_config=refiner_cfg,
            device=device,
            rng=rng,
            return_trace=True,
            prediction_fn=recording_predict,
        )
        final_degrees = [int(result.degree(node)) for node in sorted(result.nodes())]
        if final_degrees != degree_vector:
            raise AssertionError("Stage-3 graphlet refinement changed the frozen indexed degree vector")
        if source_connected and result.number_of_nodes() > 1 and not nx.is_connected(result):
            raise AssertionError("Stage-3 graphlet refinement disconnected a connected vanilla source")
        accepted = sum(bool(row.get("accepted", False)) for row in trace)
        changed = topology_state_key(source) != topology_state_key(result)
        refined.append(result)
        diagnostics.append(
            {
                "graph_index": index,
                "source_connected": True,
                "skipped": False,
                "accepted_steps": int(accepted),
                "prediction_calls": len(predictions),
                "changed": bool(changed),
                "degree_preserved": True,
                "final_connected": bool(result.number_of_nodes() <= 1 or nx.is_connected(result)),
                "trace": trace,
            }
        )
        prediction_traces.append(predictions)
        if (index + 1) % max(1, min(128, len(graphs))) == 0 or index + 1 == len(graphs):
            print(f"Stage3 graphlet refinement {index + 1}/{len(graphs)}", flush=True)

    return refined, frozen_degrees, diagnostics, prediction_traces


def refine_generated_graphs_with_fixed_predictions(
    graphs: Sequence[nx.Graph],
    predictions: Sequence[Mapping[str, Any]],
    *,
    graphlet_summary_payload: Mapping[str, Any],
    config: Mapping[str, Any],
    seed: int,
) -> tuple[list[nx.Graph], list[list[int]], list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Refine generated graphs toward fixed joint auxiliary-head predictions.

    The source graph's indexed degree vector is frozen.  Each prediction is
    computed by the joint diffusion/graphlet model *before* rewiring and is held
    fixed for the complete double-edge-swap trajectory.  No additional neural
    network is trained or evaluated during refinement.
    """

    cfg = copy.deepcopy(dict(config or {}))
    validate_fixed_target_graphlet_refinement_options(cfg)
    if len(graphs) != len(predictions):
        raise ValueError(
            f"Expected one fixed graphlet prediction per graph, got {len(predictions)} for {len(graphs)} graphs"
        )
    if not bool(graphlet_summary_payload.get("enabled", False)):
        raise RuntimeError("Checkpoint is missing the joint graphlet-summary head metadata")

    basis = TopologyGraphletBasis.from_dict(dict(graphlet_summary_payload["graphlet_basis"]))
    summary_cfg = SummaryConfig.from_dict(dict(graphlet_summary_payload["summary_config"]))
    if tuple(int(k) for k in basis.sizes) != (3, 4, 5):
        raise RuntimeError(f"Expected graphlet orders 3,4,5, found {basis.sizes}")
    refiner_cfg = TopologyRefinerConfig.from_dict(dict(cfg.get("refiner", {}) or {}))
    rng = np.random.default_rng(int(seed) + 510001)

    refined: list[nx.Graph] = []
    frozen_degrees: list[list[int]] = []
    diagnostics: list[dict[str, Any]] = []
    prediction_traces: list[list[dict[str, Any]]] = []

    for index, (raw, pred_raw) in enumerate(zip(graphs, predictions)):
        source = normalize_topology_graph(raw)
        degree_vector = [int(source.degree(node)) for node in sorted(source.nodes())]
        frozen_degrees.append(degree_vector)
        source_connected = bool(source.number_of_nodes() <= 1 or nx.is_connected(source))

        hist = np.asarray(pred_raw["graphlet_histogram"], dtype=np.float64).reshape(-1)
        mass = np.asarray(pred_raw["graphlet_mass"], dtype=np.float64).reshape(-1)
        clustering = np.asarray(
            pred_raw.get("clustering_histogram", []), dtype=np.float64
        ).reshape(-1)
        orbit = np.asarray(
            pred_raw.get("orbit_mean_counts", []), dtype=np.float64
        ).reshape(-1)
        if hist.size != basis.width:
            raise ValueError(
                f"Prediction {index} has graphlet width {hist.size}, expected {basis.width}"
            )
        if mass.size != len(basis.sizes):
            raise ValueError(
                f"Prediction {index} has mass width {mass.size}, expected {len(basis.sizes)}"
            )
        expected_clustering = int(graphlet_summary_payload.get("clustering_bins", 0))
        expected_orbit = int(graphlet_summary_payload.get("orbit_width", 0))
        if refiner_cfg.clustering_weight > 0.0:
            if expected_clustering <= 0 or clustering.size != expected_clustering:
                raise ValueError(
                    f"Prediction {index} has clustering width {clustering.size}, "
                    f"expected {expected_clustering}"
                )
        if refiner_cfg.orbit_weight > 0.0:
            if expected_orbit <= 0 or orbit.size != expected_orbit:
                raise ValueError(
                    f"Prediction {index} has orbit width {orbit.size}, expected {expected_orbit}"
                )
        fixed_prediction = TopologyPrediction(
            graphlet_target=hist,
            graphlet_mass_target=mass,
            graphlet_history=basis.unflatten_history(hist),
            graphlet_connected_mass={
                str(k): float(v) for k, v in zip(basis.sizes, mass)
            },
            clustering_target=clustering,
            orbit_target=orbit,
        )
        prediction_record = {
            "time": 0.0,
            "source": "joint_structure_summary_head_final_diffusion_state",
            "graphlet_target": hist.tolist(),
            "graphlet_mass_target": mass.tolist(),
            "clustering_target": clustering.tolist(),
            "orbit_target": orbit.tolist(),
        }

        if not source_connected:
            refined.append(source.copy())
            diagnostics.append(
                {
                    "graph_index": index,
                    "source_connected": False,
                    "skipped": True,
                    "skip_reason": "disconnected_generated_source",
                    "accepted_steps": 0,
                    "prediction_calls": 0,
                    "changed": False,
                    "degree_preserved": True,
                    "target_source": str(cfg.get("target_source", "joint_auxiliary_head")),
                }
            )
            prediction_traces.append([prediction_record])
            continue

        calls = 0

        def fixed_predict(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            return fixed_prediction

        # ``model`` is unused because ``prediction_fn`` supplies the complete
        # frozen prediction.  The generic refiner still provides exact local
        # graphlet-delta scoring and degree/connectivity-preserving swaps.
        result, trace = refine_graph_with_topology_predictions(
            source,
            model=None,  # type: ignore[arg-type]
            graphlet_basis=basis,
            summary_config=summary_cfg,
            refiner_config=refiner_cfg,
            device="cpu",
            rng=rng,
            return_trace=True,
            prediction_fn=fixed_predict,
        )
        final_degrees = [int(result.degree(node)) for node in sorted(result.nodes())]
        if final_degrees != degree_vector:
            raise AssertionError("Fixed-target graphlet refinement changed the frozen indexed degree vector")
        if source_connected and result.number_of_nodes() > 1 and not nx.is_connected(result):
            raise AssertionError("Fixed-target graphlet refinement disconnected a connected source")
        accepted = sum(bool(row.get("accepted", False)) for row in trace)
        changed = topology_state_key(source) != topology_state_key(result)
        refined.append(result)
        diagnostics.append(
            {
                "graph_index": index,
                "source_connected": True,
                "skipped": False,
                "accepted_steps": int(accepted),
                "prediction_calls": int(calls),
                "changed": bool(changed),
                "degree_preserved": True,
                "final_connected": bool(result.number_of_nodes() <= 1 or nx.is_connected(result)),
                "target_source": str(cfg.get("target_source", "joint_auxiliary_head")),
                "trace": trace,
            }
        )
        prediction_traces.append([prediction_record])
        if (index + 1) % max(1, min(128, len(graphs))) == 0 or index + 1 == len(graphs):
            print(
                f"Log-gap joint-head graphlet refinement {index + 1}/{len(graphs)}",
                flush=True,
            )

    return refined, frozen_degrees, diagnostics, prediction_traces
