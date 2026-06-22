from __future__ import annotations

import argparse
from collections import Counter
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grapher.evaluation.data_io import load_dataset_splits
from grapher.evaluation.run_utils import (
    make_model_run_config,
    sample_config_path,
    sample_metadata_path,
    sample_path,
)
from grapher.generation.molecular_rewiring import (
    EmpiricalBondTypePrior,
    apply_molecular_rewire,
    enumerate_molecular_rewire_actions,
    merge_molecular_action_sets,
    molecular_graph_to_data,
    molecular_local_feature_matrix,
)
from grapher.generation.rewiring import (
    check_sequence_validity,
    connected_sequence_feasible,
    degree_sequence,
)
from grapher.models.checkpoint import (
    load_dhvae_checkpoint,
    load_molecular_grapher_checkpoint,
)
from grapher.molecules.representation import (
    canonicalize_molecular_graph,
    molecular_graph_schema,
    molecular_graph_to_rdkit,
    write_molecular_jsonl,
    write_molecular_sdf,
    write_smiles_file,
)
from grapher.molecules.sampling import (
    EmpiricalAtomTypePrior,
    initialize_generated_molecular_source,
)
from grapher.utils.compute import PeakMemoryMonitor, compute_report
from grapher.utils.io import load_yaml, save_json, save_pickle, save_yaml, stable_hash
from grapher.utils.logging import get_logger
from grapher.utils.numerics import assert_model_tensors_finite
from grapher.utils.seed import set_seed

logger = get_logger(__name__)

MODEL_NAME = "grapher_molecular"


def _default_model_config(dataset: str) -> Path:
    if dataset == "qm9":
        return Path("configs/models/grapher_molecular_qm9.yaml")
    if dataset == "zinc":
        return Path("configs/models/grapher_molecular_zinc.yaml")
    raise ValueError("Molecular generation supports only qm9 and zinc.")


def _resolved_sample_output(
    cfg: dict[str, Any],
    dataset: str,
    run_id: int | None,
) -> Path:
    configured = cfg.get("samples_path")
    if run_id is not None:
        if isinstance(configured, str) and "run_id" in configured:
            return Path(configured)
        if isinstance(configured, str) and configured:
            base = Path(configured)
            return base.parent / MODEL_NAME / f"run_{run_id:03d}.pkl"
        return sample_path(dataset, MODEL_NAME, run_id=run_id)
    return Path(configured) if configured else sample_path(dataset, MODEL_NAME, run_id=None)


def _checkpoint_fallback(
    *,
    dataset: str,
    model: str,
    run_id: int | None,
    filename: str,
) -> Path:
    if run_id is None:
        return Path("outputs/checkpoints") / dataset / model / filename
    run_candidate = (
        Path("outputs/checkpoints")
        / dataset
        / model
        / f"run_{run_id:03d}"
        / filename
    )
    if run_candidate.exists():
        return run_candidate
    return Path("outputs/checkpoints") / dataset / model / filename


def _parse_optional_int(value: Any) -> int | None:
    if value in (None, "", "none", "None"):
        return None
    return int(value)


def _split_budget(total: int, global_fraction: float) -> tuple[int, int]:
    total = max(int(total), 1)
    fraction = min(max(float(global_fraction), 0.0), 1.0)
    global_budget = min(int(round(total * fraction)), total)
    return max(total - global_budget, 0), global_budget


def _generation_actions(
    graph: nx.Graph,
    prior: EmpiricalBondTypePrior,
    *,
    rng: random.Random,
    candidate_budget: int,
    k_hop: int | None,
    ensure_connected: bool,
    global_candidate_fraction: float,
    proposals_per_edge: int,
    proposal_mode: str,
    allow_global_backoff: bool,
    reject_unseen_endpoint_pairs: bool,
    valence_tolerance: float,
) -> list:
    local_budget, global_budget = _split_budget(candidate_budget, global_candidate_fraction)
    local = (
        enumerate_molecular_rewire_actions(
            graph,
            prior,
            rng=rng,
            ensure_connected=ensure_connected,
            k_hop=k_hop,
            max_candidates=local_budget,
            proposals_per_edge=proposals_per_edge,
            proposal_mode=proposal_mode,
            allow_global_backoff=allow_global_backoff,
            reject_unseen_endpoint_pairs=reject_unseen_endpoint_pairs,
            valence_tolerance=valence_tolerance,
            shuffle=True,
        )
        if local_budget > 0
        else []
    )
    global_actions = (
        enumerate_molecular_rewire_actions(
            graph,
            prior,
            rng=rng,
            ensure_connected=ensure_connected,
            k_hop=None,
            max_candidates=global_budget,
            proposals_per_edge=proposals_per_edge,
            proposal_mode=proposal_mode,
            allow_global_backoff=allow_global_backoff,
            reject_unseen_endpoint_pairs=reject_unseen_endpoint_pairs,
            valence_tolerance=valence_tolerance,
            shuffle=True,
        )
        if global_budget > 0
        else []
    )
    actions = merge_molecular_action_sets(
        local,
        global_actions,
        max_candidates=max(int(candidate_budget), 1),
    )
    if len(actions) < max(int(candidate_budget), 1):
        fallback = enumerate_molecular_rewire_actions(
            graph,
            prior,
            rng=rng,
            ensure_connected=ensure_connected,
            k_hop=None if global_candidate_fraction > 0.0 else k_hop,
            max_candidates=max(int(candidate_budget), 1) - len(actions),
            proposals_per_edge=proposals_per_edge,
            proposal_mode=proposal_mode,
            allow_global_backoff=allow_global_backoff,
            reject_unseen_endpoint_pairs=reject_unseen_endpoint_pairs,
            valence_tolerance=valence_tolerance,
            shuffle=True,
        )
        actions = merge_molecular_action_sets(
            actions,
            fallback,
            max_candidates=max(int(candidate_budget), 1),
        )
    rng.shuffle(actions)
    return actions


def _sample_action_index(
    logits: torch.Tensor,
    *,
    temperature: float,
    sample_actions: bool,
) -> int:
    if logits.numel() == 0:
        raise ValueError("Cannot select an action from an empty logit tensor.")
    if not sample_actions:
        return int(torch.argmax(logits).item())
    scaled = logits / max(float(temperature), 1e-6)
    probabilities = F.softmax(scaled, dim=0)
    return int(torch.multinomial(probabilities, num_samples=1).item())


def _source_and_flow(
    sequence: Sequence[int],
    *,
    model,
    atom_prior: EmpiricalAtomTypePrior,
    bond_prior: EmpiricalBondTypePrior,
    cfg: dict[str, Any],
    rng: random.Random,
    device: torch.device,
) -> tuple[nx.Graph, dict[str, Any]]:
    source = initialize_generated_molecular_source(
        sequence,
        atom_prior,
        bond_prior,
        rng=rng,
        source_edge_type_strategy=str(cfg.get("source_edge_type_strategy", "sample")),
        allow_global_bond_backoff=bool(cfg.get("allow_global_bond_backoff", True)),
        valence_tolerance=float(cfg.get("valence_tolerance", 1e-6)),
        node_assignment_attempts=int(cfg.get("node_assignment_attempts", 16)),
    )
    graph = source
    accepted_steps = 0
    candidate_sizes: list[int] = []
    num_steps = max(int(cfg.get("num_steps", model.T)), 0)
    candidate_budget = max(int(cfg.get("candidate_budget", 48)), 1)
    k_hop = _parse_optional_int(cfg.get("k_hop", 2))
    ensure_connected = bool(cfg.get("ensure_connected", True))
    proposals_per_edge = max(int(cfg.get("bond_type_proposals_per_edge", 2)), 1)
    proposal_mode = str(cfg.get("bond_type_proposal_mode", "sample"))
    allow_global_backoff = bool(cfg.get("allow_global_bond_backoff", True))
    reject_unseen = bool(cfg.get("reject_unseen_endpoint_pairs", False))
    valence_tolerance = float(cfg.get("valence_tolerance", 1e-6))
    global_fraction = float(cfg.get("global_candidate_fraction", 0.1))
    action_temperature = float(cfg.get("action_temperature", cfg.get("temperature", 1.0)))
    sample_actions = bool(cfg.get("sample_actions", True))

    with torch.no_grad():
        for step in range(num_steps):
            actions = _generation_actions(
                graph,
                bond_prior,
                rng=rng,
                candidate_budget=candidate_budget,
                k_hop=k_hop,
                ensure_connected=ensure_connected,
                global_candidate_fraction=global_fraction,
                proposals_per_edge=proposals_per_edge,
                proposal_mode=proposal_mode,
                allow_global_backoff=allow_global_backoff,
                reject_unseen_endpoint_pairs=reject_unseen,
                valence_tolerance=valence_tolerance,
            )
            if not actions:
                break
            data = molecular_graph_to_data(
                graph,
                node_type_to_index=model.node_type_to_index,
                edge_type_to_index=model.edge_type_to_index,
                k_eigen=model.k_eigen,
            ).to(device)
            local = molecular_local_feature_matrix(
                graph,
                actions,
                bond_prior,
                local_feature_dim=model.local_feature_dim,
            ).to(device)
            logits = model.score_actions(
                node_types=data.node_types,
                degree_features=data.degree_features,
                pe=data.pe,
                edge_index=data.edge_index,
                edge_types=data.edge_types,
                actions=actions,
                t=float(step) / float(max(num_steps, 1)),
                degree_sequence=[int(value) for value in sequence],
                action_local_features=local,
            )
            index = _sample_action_index(
                logits,
                temperature=action_temperature,
                sample_actions=sample_actions,
            )
            candidate = apply_molecular_rewire(
                graph,
                actions[index],
                bond_prior,
                ensure_connected=ensure_connected,
                valence_tolerance=valence_tolerance,
            )
            if candidate is None:
                break
            graph = nx.convert_node_labels_to_integers(candidate, ordering="sorted")
            accepted_steps += 1
            candidate_sizes.append(len(actions))

    return canonicalize_molecular_graph(graph), {
        "accepted_rewiring_steps": int(accepted_steps),
        "requested_rewiring_steps": int(num_steps),
        "stopped_early": bool(accepted_steps < num_steps),
        "avg_candidate_size": float(np.mean(candidate_sizes)) if candidate_sizes else 0.0,
        "initial_degree_sequence": [int(value) for value in sequence],
    }


def generate_molecular_grapher_samples(
    *,
    dataset: str,
    model_config: dict[str, Any],
    dataset_root: str,
    num_samples: int,
    seed: int,
    run_id: int | None,
    device: str,
    force: bool,
    max_rounds: int,
    write_sdf: bool,
    isomeric_smiles: bool,
) -> dict[str, Any]:
    if dataset not in {"qm9", "zinc"}:
        raise ValueError("Molecular generation supports only qm9 and zinc.")
    set_seed(seed, include_torch=True)
    rng = random.Random(int(seed))
    torch_device = torch.device(device)
    cfg = make_model_run_config(
        model_config,
        dataset=dataset,
        model=MODEL_NAME,
        run_id=run_id,
        seed=seed,
        use_run_paths=run_id is not None,
    )

    molecular_checkpoint = Path(
        cfg.get("checkpoint_path")
        or _checkpoint_fallback(
            dataset=dataset,
            model=MODEL_NAME,
            run_id=run_id,
            filename="grapher_molecular.pt",
        )
    )
    dhvae_checkpoint = Path(
        cfg.get("dhvae_checkpoint_path")
        or _checkpoint_fallback(
            dataset=dataset,
            model="dhvae",
            run_id=run_id,
            filename="dhvae.pt",
        )
    )
    if not molecular_checkpoint.exists():
        raise FileNotFoundError(
            f"Molecular GraphER checkpoint not found: {molecular_checkpoint}. "
            "Run scripts/train_molecular_grapher_model.py first."
        )
    if not dhvae_checkpoint.exists():
        raise FileNotFoundError(
            f"DH-VAE checkpoint not found: {dhvae_checkpoint}. "
            "Run scripts/train_dhvae_model.py first."
        )

    model, molecular_payload = load_molecular_grapher_checkpoint(
        molecular_checkpoint,
        device=str(torch_device),
    )
    dhvae, dhvae_payload = load_dhvae_checkpoint(
        dhvae_checkpoint,
        device=str(torch_device),
    )
    assert_model_tensors_finite(model, context=f"molecular_generation/{dataset}")

    splits = load_dataset_splits(
        dataset,
        output_root=dataset_root,
        build_if_missing=dataset != "zinc",
    )
    train_graphs = list(splits.get("train", []))
    if not train_graphs:
        raise ValueError(f"No training graphs are available for fitting the atom prior for {dataset}.")
    bond_prior_payload = molecular_payload.get("empirical_bond_prior")
    if not isinstance(bond_prior_payload, dict):
        raise KeyError("Molecular GraphER checkpoint is missing empirical_bond_prior.")
    bond_prior = EmpiricalBondTypePrior.from_dict(bond_prior_payload)
    atom_prior = EmpiricalAtomTypePrior.fit(
        train_graphs,
        allowed_node_types=model.node_type_values,
        smoothing=float(cfg.get("empirical_atom_smoothing", 0.1)),
    )

    output_path = _resolved_sample_output(cfg, dataset, run_id)
    metadata_path = sample_metadata_path(dataset, MODEL_NAME, run_id=run_id)
    resolved_config_path = sample_config_path(dataset, MODEL_NAME, run_id=run_id)
    if output_path.exists() and not force:
        raise FileExistsError(f"Sample file already exists: {output_path}. Use --force to overwrite.")

    generated: list[nx.Graph] = []
    records: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()
    degree_attempts = 0
    start = time.perf_counter()
    degree_temperature = float(
        cfg.get("degree_sample_temperature", cfg.get("dhvae_temperature", 1.0))
    )
    attempt_factor = max(int(cfg.get("degree_attempt_factor", 8)), 1)

    with PeakMemoryMonitor() as memory_monitor:
        for round_index in range(max(int(max_rounds), 1)):
            if len(generated) >= int(num_samples):
                break
            remaining = int(num_samples) - len(generated)
            request = max(remaining * attempt_factor, remaining)
            sequences = dhvae.generate(request, temperature=degree_temperature)
            for sequence in sequences:
                if len(generated) >= int(num_samples):
                    break
                degree_attempts += 1
                sequence = [int(value) for value in sequence]
                graphical, code = check_sequence_validity(sequence)
                if not graphical:
                    reject_counts[f"degree_not_graphical:{code}"] += 1
                    continue
                feasible, reason = connected_sequence_feasible(sequence)
                if not feasible:
                    reject_counts[f"degree_not_connected_feasible:{reason}"] += 1
                    continue
                if len(sequence) > model.max_nodes:
                    reject_counts["degree_sequence_exceeds_model_max_nodes"] += 1
                    continue
                if sequence and max(sequence) >= model.degree_histogram_dim:
                    reject_counts["degree_exceeds_model_histogram_dim"] += 1
                    continue
                try:
                    graph, record = _source_and_flow(
                        sequence,
                        model=model,
                        atom_prior=atom_prior,
                        bond_prior=bond_prior,
                        cfg=cfg,
                        rng=rng,
                        device=torch_device,
                    )
                except Exception as exc:
                    reject_counts[f"source_or_flow_failure:{type(exc).__name__}"] += 1
                    logger.debug("Rejected molecular source: %s", exc)
                    continue
                graph.graph["source_dataset"] = dataset
                graph.graph["generator"] = "GraphER_molecular"
                graph.graph["sample_index"] = len(generated)
                generated.append(graph)
                records.append(record)
            logger.info(
                "Molecular generation round=%d/%d saved=%d/%d degree_attempts=%d",
                round_index + 1,
                max(int(max_rounds), 1),
                len(generated),
                int(num_samples),
                degree_attempts,
            )

        conversions = [
            molecular_graph_to_rdkit(
                graph,
                sanitize=True,
                isomeric_smiles=isomeric_smiles,
            )
            for graph in generated
        ]
    elapsed = time.perf_counter() - start

    smiles = [item.smiles for item in conversions]
    valid_mask = [bool(item.valid) for item in conversions]
    conversion_errors = Counter(
        item.error.split(":", 1)[0] if item.error else "none"
        for item in conversions
        if not item.valid
    )
    sample_bundle = {
        "schema_version": "grapher.molecular_samples.v1",
        "dataset": dataset,
        "model": MODEL_NAME,
        "run_id": run_id,
        "seed": int(seed),
        "representation": molecular_graph_schema(),
        "graphs": generated,
        "canonical_smiles": smiles,
        "valid_without_correction": valid_mask,
        "generation_records": records,
        "degree_sequences": [degree_sequence(graph) for graph in generated],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(sample_bundle, output_path, force=force)
    smiles_path = output_path.with_suffix(".smi")
    jsonl_path = output_path.with_suffix(".jsonl")
    write_smiles_file(smiles, smiles_path)
    write_molecular_jsonl(
        generated,
        jsonl_path,
        conversions=conversions,
        records=records,
    )
    sdf_path: Path | None = None
    sdf_count = 0
    if write_sdf:
        sdf_path, sdf_count = write_molecular_sdf(
            conversions,
            output_path.with_suffix(".sdf"),
        )
    save_yaml(cfg, resolved_config_path, force=True)

    compute = compute_report(
        operation="molecular_sampling",
        runtime_seconds=elapsed,
        num_graphs=len(generated),
        memory=memory_monitor.to_dict(),
    )
    validity_rate = float(np.mean(valid_mask)) if valid_mask else 0.0
    metadata = {
        "dataset": dataset,
        "model": MODEL_NAME,
        "run_id": run_id,
        "seed": int(seed),
        "num_samples_requested": int(num_samples),
        "num_samples_saved": len(generated),
        "num_valid_without_correction": int(sum(valid_mask)),
        "validity_without_correction": validity_rate,
        "runtime_seconds": elapsed,
        "seconds_per_graph": elapsed / max(len(generated), 1),
        "sample_path": str(output_path),
        "smiles_path": str(smiles_path),
        "jsonl_path": str(jsonl_path),
        "sdf_path": None if sdf_path is None else str(sdf_path),
        "sdf_valid_molecule_count": int(sdf_count),
        "checkpoint_path": str(molecular_checkpoint),
        "dhvae_checkpoint_path": str(dhvae_checkpoint),
        "model_config_hash": stable_hash(cfg),
        "compute": compute,
        "representation": molecular_graph_schema(),
        "degree_sampling": {
            "temperature": degree_temperature,
            "num_attempts": int(degree_attempts),
            "rejection_counts": dict(reject_counts),
        },
        "atom_type_prior": atom_prior.to_dict(),
        "bond_type_prior": {
            "conditioning": "p(edge_type | unordered endpoint atomic numbers)",
            "node_types": bond_prior.node_types,
            "edge_types": bond_prior.edge_types,
        },
        "rewiring_policy": {
            "action_type": "typed_double_edge_swap_(e1,e2,r,c1,c2)",
            "node_types_fixed": True,
            "target_free_generation": True,
            "num_steps": int(cfg.get("num_steps", model.T)),
            "candidate_budget": int(cfg.get("candidate_budget", 48)),
            "k_hop": cfg.get("k_hop", 2),
            "global_candidate_fraction": float(cfg.get("global_candidate_fraction", 0.1)),
            "action_temperature": float(cfg.get("action_temperature", cfg.get("temperature", 1.0))),
            "sample_actions": bool(cfg.get("sample_actions", True)),
            "valence_filter": True,
        },
        "conversion": {
            "protocol": "direct attributed-graph to RDKit conversion; no graph repair, valence correction, bond resampling, or fragment selection",
            "isomeric_smiles": bool(isomeric_smiles),
            "error_counts": dict(conversion_errors),
        },
        "training_stats": {
            "molecular_grapher": molecular_payload.get("training_graph_stats", {}),
            "dhvae": dhvae_payload.get("degree_sequence_stats", {}),
        },
    }
    save_json(metadata, metadata_path, force=True)
    logger.info(
        "Saved %d molecular GraphER samples (%d directly valid) to %s",
        len(generated),
        int(sum(valid_mask)),
        output_path,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate QM9/ZINC attributed graphs from DH-VAE + MolecularGraphER. "
            "Node types are sampled from an empirical degree-conditioned prior and remain fixed; "
            "new bond types are proposed from p(edge_type|endpoint atom types), and valence-invalid actions are rejected."
        )
    )
    parser.add_argument("--dataset", required=True, choices=["qm9", "zinc"])
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--write-sdf", action="store_true")
    parser.add_argument("--isomeric-smiles", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.model_config) if args.model_config else _default_model_config(args.dataset)
    generate_molecular_grapher_samples(
        dataset=args.dataset,
        model_config=load_yaml(config_path),
        dataset_root=args.dataset_root,
        num_samples=args.num_samples,
        seed=args.seed,
        run_id=args.run_id,
        device=args.device,
        force=args.force,
        max_rounds=args.max_rounds,
        write_sdf=args.write_sdf,
        isomeric_smiles=args.isomeric_smiles,
    )


if __name__ == "__main__":
    main()
