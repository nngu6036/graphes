from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import networkx as nx
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grapher.evaluation.data_io import load_dataset_splits
from grapher.evaluation.run_utils import make_model_run_config, sample_config_path, sample_metadata_path, sample_path
from grapher.generation.validity import quality_metrics
from grapher.models.checkpoint import load_grapher_checkpoint, load_msvae_checkpoint
from grapher.registry import available_datasets
from grapher.utils.compute import PeakMemoryMonitor, compute_report
from grapher.utils.io import load_yaml, save_json, save_pickle, save_yaml, stable_hash
from grapher.utils.logging import get_logger
from grapher.utils.numerics import assert_finite_graphs
from grapher.utils.seed import set_seed

logger = get_logger(__name__)


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


def generate_grapher_samples(
    *,
    dataset: str,
    model_config: dict,
    dataset_root: str,
    num_samples: int,
    seed: int,
    run_id: int | None,
    device: str,
    force: bool,
    max_rounds: int = 20,
) -> dict:
    set_seed(seed, include_torch=True)
    cfg = make_model_run_config(model_config, dataset=dataset, model="grapher", run_id=run_id, seed=seed, use_run_paths=run_id is not None)
    grapher_ckpt = Path(cfg.get("checkpoint_path") or f"outputs/checkpoints/{dataset}/grapher/grapher.pt")
    msvae_ckpt = Path(cfg.get("msvae_checkpoint_path") or f"outputs/checkpoints/{dataset}/msvae/msvae.pt")
    if not grapher_ckpt.exists():
        raise FileNotFoundError(f"GraphER checkpoint not found: {grapher_ckpt}. Run scripts/train_grapher_model.py first.")
    if not msvae_ckpt.exists():
        raise FileNotFoundError(f"MS-VAE checkpoint not found: {msvae_ckpt}. Run scripts/train_msvae_model.py first.")
    grapher, grapher_payload = load_grapher_checkpoint(grapher_ckpt, device=device)
    msvae, msvae_payload = load_msvae_checkpoint(msvae_ckpt, device=device)

    out = _resolved_sample_output(cfg, dataset, "grapher", run_id)
    metadata_out = sample_metadata_path(dataset, "grapher", run_id=run_id)
    resolved_cfg_out = sample_config_path(dataset, "grapher", run_id=run_id)
    if out.exists() and not force:
        raise FileExistsError(f"Sample file already exists: {out}. Use --force to overwrite.")

    start = time.perf_counter()
    generated: list[nx.Graph] = []
    degree_sequences: list[list[int]] = []
    with PeakMemoryMonitor() as memory_monitor:
        rounds = 0
        while len(generated) < int(num_samples) and rounds < int(max_rounds):
            rounds += 1
            remaining = int(num_samples) - len(generated)
            graphs, seqs = grapher.generate(
                num_samples=remaining,
                num_steps=int(cfg.get("num_steps", grapher_payload.get("model_params", {}).get("T", 32))),
                msvae_model=msvae,
                k_eigen=int(cfg.get("k_eigen", grapher_payload.get("model_params", {}).get("k_eigen", 4))),
                method=str(cfg.get("init_method", "havel_hakimi")),
                ensure_connected=bool(cfg.get("ensure_connected", True)),
                k_hop=int(cfg.get("k_hop", 2)),
                max_candidates=int(cfg.get("candidate_budget", 64)),
                degree_temperature=float(cfg.get("degree_sample_temperature", cfg.get("msvae_temperature", 1.0))),
                action_temperature=float(cfg.get("action_temperature", cfg.get("temperature", 1.0))),
                sample_actions=bool(cfg.get("sample_actions", True)),
            )
            if not graphs:
                break
            generated.extend(graphs[:remaining])
            degree_sequences.extend(seqs[:remaining])
        assert_finite_graphs(generated, context=f"GraphER sampling {dataset}")
    elapsed = time.perf_counter() - start

    try:
        train_graphs = list(load_dataset_splits(dataset, output_root=dataset_root, build_if_missing=False).get("train", []))
    except Exception:
        train_graphs = []
    quality = quality_metrics(generated, reference_graphs=train_graphs, dataset=dataset)

    out.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(generated, out, force=force)
    save_yaml(cfg, resolved_cfg_out, force=True)
    compute = compute_report(operation="sampling", runtime_seconds=elapsed, num_graphs=len(generated), memory=memory_monitor.to_dict())
    metadata = {
        "dataset": dataset,
        "model": "grapher",
        "seed": seed,
        "run_id": run_id,
        "num_samples_requested": int(num_samples),
        "num_samples_saved": len(generated),
        "runtime_seconds": elapsed,
        "seconds_per_graph": elapsed / max(len(generated), 1),
        "sample_path": str(out),
        "checkpoint_path": str(grapher_ckpt),
        "msvae_checkpoint_path": str(msvae_ckpt),
        "model_config_hash": stable_hash(cfg),
        "compute": compute,
        "quality": quality,
        "degree_sequences": {
            "num_recorded": len(degree_sequences),
            "source": "sampled from size-conditioned DH-VAE and accepted by GraphER canonical-source construction",
            "degree_sample_temperature": float(cfg.get("degree_sample_temperature", cfg.get("msvae_temperature", 1.0))),
        },
        "rewiring_policy": {
            "action_type": "complete_double_edge_swap_(e1,e2,r)",
            "candidate_set": "target-free local/random valid connected actions",
            "action_temperature": float(cfg.get("action_temperature", cfg.get("temperature", 1.0))),
            "sample_actions": bool(cfg.get("sample_actions", True)),
        },
        "training_stats": {
            "grapher": grapher_payload.get("training_graph_stats", {}),
            "msvae": msvae_payload.get("degree_sequence_stats", {}),
        },
    }
    save_json(metadata, metadata_out, force=True)
    logger.info("Saved %d GraphER samples to %s", len(generated), out)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate graph samples from trained DH-VAE + GraphER checkpoints.")
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--model-config", type=str, default="configs/models/grapher_generic.yaml")
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=20)
    args = parser.parse_args()
    generate_grapher_samples(
        dataset=args.dataset,
        model_config=load_yaml(args.model_config),
        dataset_root=args.dataset_root,
        num_samples=args.num_samples,
        seed=args.seed,
        run_id=args.run_id,
        device=args.device,
        force=args.force,
        max_rounds=args.max_rounds,
    )


if __name__ == "__main__":
    main()
