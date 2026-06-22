from __future__ import annotations

import argparse
from collections import Counter
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grapher.evaluation.data_io import load_dataset_splits
from grapher.evaluation.run_utils import make_model_run_config, sample_metadata_path, sample_path
from grapher.generation.rewiring import check_sequence_validity, connected_sequence_feasible, degree_sequence
from grapher.models.checkpoint import load_dhvae_checkpoint
from grapher.registry import available_datasets
from grapher.utils.compute import PeakMemoryMonitor, compute_report
from grapher.utils.io import load_yaml, save_json, save_pickle, save_yaml
from grapher.utils.logging import get_logger
from grapher.utils.seed import set_seed

logger = get_logger(__name__)

MOLECULAR_DATASETS = {"qm9", "zinc"}


def _default_model_config(dataset: str) -> Path:
    """Return the dataset-aware default DH-VAE config.

    Generic datasets use configs/models/dhvae.yaml.  Molecular datasets have
    separate defaults because QM9 and ZINC usually need different model capacity
    and training limits, but the DH-VAE still models only topological degree
    sequences.  Atom and bond types are handled later by MolecularGraphER.
    """

    dataset_key = str(dataset).lower()
    if dataset_key == "qm9":
        return Path("configs/models/dhvae_qm9.yaml")
    if dataset_key == "zinc":
        return Path("configs/models/dhvae_zinc.yaml")
    return Path("configs/models/dhvae.yaml")


def _resolved_sample_output(cfg: dict, dataset: str, model: str, run_id: int | None) -> Path:
    configured = cfg.get("samples_path")
    if run_id is not None:
        if isinstance(configured, str) and "run_id" in configured:
            return Path(configured)
        if isinstance(configured, str) and configured:
            base = Path(configured)
            return base.parent / model / f"run_{run_id:03d}.pkl"
        return sample_path(dataset, model, run_id=run_id)
    return Path(configured) if configured else sample_path(dataset, model, run_id=None)


def _increment(counter: dict[str, int], key: str) -> None:
    counter[str(key)] = int(counter.get(str(key), 0)) + 1


def _summary(values: Sequence[int | float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
    }


def _sequence_diagnostics(sequences: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Summarize degree-sequence validity and GraphER compatibility."""

    num = int(len(sequences))
    graphical = 0
    connected_feasible = 0
    invalid_reasons: dict[str, int] = {}
    sizes: list[int] = []
    max_degrees: list[int] = []
    edge_counts: list[float] = []
    zero_degree_multi_node = 0
    degree_value_counts: Counter[int] = Counter()

    for raw_seq in sequences:
        seq = [int(d) for d in raw_seq]
        sizes.append(len(seq))
        max_degrees.append(max(seq) if seq else 0)
        edge_counts.append(float(sum(seq)) / 2.0)
        degree_value_counts.update(seq)
        if len(seq) > 1 and any(d == 0 for d in seq):
            zero_degree_multi_node += 1
        ok, code = check_sequence_validity(seq)
        if ok:
            graphical += 1
            feasible, reason = connected_sequence_feasible(seq)
        else:
            feasible, reason = False, f"not_graphical:{code}"
        if feasible:
            connected_feasible += 1
        else:
            _increment(invalid_reasons, reason)

    return {
        "num_sequences": num,
        "graphicality_rate": float(graphical) / max(float(num), 1.0),
        "connected_feasible_rate": float(connected_feasible) / max(float(num), 1.0),
        "zero_degree_multi_node_count": int(zero_degree_multi_node),
        "zero_degree_multi_node_rate": float(zero_degree_multi_node) / max(float(num), 1.0),
        "invalid_reasons": dict(invalid_reasons),
        "size": _summary(sizes),
        "edge_count": _summary(edge_counts),
        "max_degree": _summary(max_degrees),
        "degree_value_counts": {str(k): int(v) for k, v in sorted(degree_value_counts.items())},
    }


def _reference_degree_diagnostics(
    *,
    dataset: str,
    dataset_root: str,
    reference_split: str,
    max_reference_graphs: int | None,
    seed: int,
) -> dict[str, Any] | None:
    """Load a prepared split and summarize its degree sequences.

    This is best-effort.  Generation only needs a DH-VAE checkpoint, so we avoid
    rebuilding/downloading datasets here.  ZINC, in particular, must have been
    prepared from SMILES beforehand.
    """

    try:
        splits = load_dataset_splits(
            dataset,
            output_root=dataset_root,
            build_if_missing=False,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path.
        logger.warning("Skipping reference degree diagnostics for %s: %s", dataset, exc)
        return None
    if reference_split not in splits:
        logger.warning("Skipping reference degree diagnostics: split %r not found in %s", reference_split, sorted(splits))
        return None
    graphs = list(splits[reference_split])
    if max_reference_graphs is not None and int(max_reference_graphs) > 0 and len(graphs) > int(max_reference_graphs):
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(len(graphs), size=int(max_reference_graphs), replace=False)
        graphs = [graphs[int(i)] for i in indices]
    sequences = [degree_sequence(nx.Graph(g)) for g in graphs]
    diagnostics = _sequence_diagnostics(sequences)
    diagnostics["split"] = reference_split
    diagnostics["num_graphs"] = len(graphs)
    diagnostics["connectedness_rate"] = float(np.mean([nx.is_connected(nx.Graph(g)) if g.number_of_nodes() > 0 else False for g in graphs])) if graphs else 0.0
    return diagnostics


def _checkpoint_degree_stats(payload: dict[str, Any]) -> dict[str, Any]:
    stats = payload.get("degree_sequence_stats", {})
    if isinstance(stats, dict):
        return dict(stats)
    return {}


def _compatibility_report(
    *,
    dataset: str,
    generated_sequences: Sequence[Sequence[int]],
    checkpoint_payload: dict[str, Any],
    require_connected_feasible: bool,
    reference_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    dataset_key = str(dataset).lower()
    kind = "molecular" if dataset_key in MOLECULAR_DATASETS else "generic"
    generated = _sequence_diagnostics(generated_sequences)
    connected_rate = float(generated.get("connected_feasible_rate", 0.0))
    graphicality_rate = float(generated.get("graphicality_rate", 0.0))
    zero_degree_multi = int(generated.get("zero_degree_multi_node_count", 0))
    graph_er_ready = bool(graphicality_rate == 1.0 and (not require_connected_feasible or connected_rate == 1.0))
    molecular_ready = bool(graph_er_ready and zero_degree_multi == 0) if kind == "molecular" else None
    checkpoint_stats = _checkpoint_degree_stats(checkpoint_payload)
    return {
        "dataset_kind": kind,
        "degree_sequences_only": True,
        "generated_degree_sequences": generated,
        "reference_degree_sequences": reference_diagnostics,
        "checkpoint_degree_sequence_stats": checkpoint_stats,
        "graph_er_degree_prior_ready": graph_er_ready,
        "molecular_degree_prior_ready": molecular_ready,
        "requires_connected_feasible_sequences": bool(require_connected_feasible),
        "notes": (
            "DH-VAE outputs topological degree sequences only.  For QM9/ZINC, "
            "MolecularGraphER samples atom types and bond types from empirical priors and keeps node types fixed during rewiring."
            if kind == "molecular"
            else "DH-VAE outputs topological degree sequences for generic GraphER."
        ),
    }


def generate_dhvae_samples(
    *,
    dataset: str,
    model_config: dict,
    num_samples: int,
    seed: int,
    run_id: int | None,
    device: str,
    force: bool,
    require_connected_feasible: bool = True,
    max_attempts: int | None = None,
    temperature: float | None = None,
    dataset_root: str = "outputs/datasets",
    reference_split: str = "train",
    max_reference_graphs: int | None = 2048,
    reference_diagnostics: bool = True,
) -> dict:
    """Generate degree sequences from the size-conditioned DH-VAE prior.

    The generated payload is intentionally a list of degree sequences for every
    dataset.  For QM9 and ZINC, these sequences are the topological degree prior
    used by MolecularGraphER; atom and bond labels are generated downstream by
    empirical molecular proposal priors plus the typed rewiring scorer.
    """

    set_seed(seed, include_torch=True)
    dataset_key = str(dataset).lower()
    dataset_kind = "molecular" if dataset_key in MOLECULAR_DATASETS else "generic"
    cfg = make_model_run_config(model_config, dataset=dataset, model="dhvae", run_id=run_id, seed=seed, use_run_paths=run_id is not None)
    checkpoint = Path(cfg.get("checkpoint_path") or f"outputs/checkpoints/{dataset}/dhvae/dhvae.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(f"DH-VAE checkpoint not found: {checkpoint}. Run scripts/train_dhvae_model.py first.")
    model, checkpoint_payload = load_dhvae_checkpoint(checkpoint, device=device)
    out = _resolved_sample_output(cfg, dataset, "dhvae", run_id)
    metadata_out = sample_metadata_path(dataset, "dhvae", run_id=run_id)
    if out.exists() and not force:
        raise FileExistsError(f"Sample file already exists: {out}. Use --force to overwrite.")

    sample_temperature = float(temperature if temperature is not None else cfg.get("sample_temperature", 1.0))
    max_attempts = int(max_attempts or max(10 * int(num_samples), int(num_samples)))
    accepted: list[list[int]] = []
    rejected = {"not_graphical": 0, "not_connected_feasible": 0, "empty_sequence": 0}
    rejected_reasons: dict[str, int] = {}
    attempts = 0
    start = time.perf_counter()
    with PeakMemoryMonitor() as memory_monitor:
        while len(accepted) < int(num_samples) and attempts < max_attempts:
            batch_size = min(max(32, int(num_samples) - len(accepted)), max_attempts - attempts)
            candidates = model.generate(batch_size, temperature=sample_temperature)
            attempts += batch_size
            for seq in candidates:
                seq = [int(d) for d in seq]
                if not seq:
                    rejected["empty_sequence"] += 1
                    _increment(rejected_reasons, "empty_sequence")
                    continue
                ok, code = check_sequence_validity(seq)
                if not ok:
                    rejected["not_graphical"] += 1
                    _increment(rejected_reasons, f"not_graphical:{code}")
                    continue
                if require_connected_feasible:
                    feasible, reason = connected_sequence_feasible(seq)
                    if not feasible:
                        rejected["not_connected_feasible"] += 1
                        _increment(rejected_reasons, reason)
                        continue
                accepted.append(seq)
                if len(accepted) >= int(num_samples):
                    break
    elapsed = time.perf_counter() - start
    if len(accepted) < int(num_samples):
        logger.warning(
            "Generated only %d/%d accepted DH-VAE sequences after %d attempts. Increase --max-attempts or lower constraints for diagnostics.",
            len(accepted),
            int(num_samples),
            attempts,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(accepted, out, force=force)
    save_yaml(cfg, out.with_suffix(".resolved_model_config.yaml"), force=True)
    acceptance_rate = len(accepted) / max(attempts, 1)
    reference = None
    if reference_diagnostics:
        reference = _reference_degree_diagnostics(
            dataset=dataset,
            dataset_root=dataset_root,
            reference_split=reference_split,
            max_reference_graphs=max_reference_graphs,
            seed=seed,
        )
    compatibility = _compatibility_report(
        dataset=dataset,
        generated_sequences=accepted,
        checkpoint_payload=checkpoint_payload,
        require_connected_feasible=require_connected_feasible,
        reference_diagnostics=reference,
    )
    compute = compute_report(operation="sampling", runtime_seconds=elapsed, num_graphs=len(accepted), memory=memory_monitor.to_dict())
    metadata = {
        "dataset": dataset,
        "dataset_kind": dataset_kind,
        "model": "dhvae",
        "seed": seed,
        "run_id": run_id,
        "checkpoint_path": str(checkpoint),
        "sample_path": str(out),
        "sample_format": "list[list[int]] degree sequences",
        "num_samples_requested": int(num_samples),
        "num_samples_saved": len(accepted),
        "attempts": attempts,
        "acceptance_rate": acceptance_rate,
        "rejected": rejected,
        "rejected_reasons": rejected_reasons,
        "sample_temperature": sample_temperature,
        "require_connected_feasible": bool(require_connected_feasible),
        "runtime_seconds": elapsed,
        "compute": compute,
        "compatibility": compatibility,
        "checkpoint_training_stats": checkpoint_payload.get("degree_sequence_stats", {}),
    }
    save_json(metadata, metadata_out, force=True)
    logger.info(
        "Saved %d DH-VAE degree sequences to %s dataset_kind=%s acceptance_rate=%.4f graph_er_ready=%s",
        len(accepted),
        out,
        dataset_kind,
        acceptance_rate,
        compatibility.get("graph_er_degree_prior_ready"),
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate degree sequences from a trained size-conditioned DH-VAE.")
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="DH-VAE config. Defaults to configs/models/dhvae_qm9.yaml for QM9, dhvae_zinc.yaml for ZINC, and dhvae.yaml otherwise.",
    )
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--reference-split", type=str, default="train")
    parser.add_argument("--max-reference-graphs", type=int, default=2048)
    parser.add_argument("--skip-reference-diagnostics", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-disconnected-degree-sequences", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None, help="Override sample_temperature from the config.")
    args = parser.parse_args()
    config_path = Path(args.model_config) if args.model_config else _default_model_config(args.dataset)
    generate_dhvae_samples(
        dataset=args.dataset,
        model_config=load_yaml(config_path),
        num_samples=args.num_samples,
        seed=args.seed,
        run_id=args.run_id,
        device=args.device,
        force=args.force,
        require_connected_feasible=not args.allow_disconnected_degree_sequences,
        max_attempts=args.max_attempts,
        temperature=args.temperature,
        dataset_root=args.dataset_root,
        reference_split=args.reference_split,
        max_reference_graphs=args.max_reference_graphs,
        reference_diagnostics=not args.skip_reference_diagnostics,
    )


if __name__ == "__main__":
    main()
