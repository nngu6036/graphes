from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grapher.evaluation.run_utils import make_model_run_config, sample_metadata_path, sample_path
from grapher.generation.rewiring import check_sequence_validity, connected_sequence_feasible
from grapher.models.checkpoint import load_msvae_checkpoint
from grapher.registry import available_datasets
from grapher.utils.compute import PeakMemoryMonitor, compute_report
from grapher.utils.io import load_yaml, save_json, save_pickle, save_yaml
from grapher.utils.logging import get_logger
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


def generate_msvae_samples(
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
) -> dict:
    """Generate degree sequences from the size-conditioned DH-VAE prior."""

    set_seed(seed, include_torch=True)
    cfg = make_model_run_config(model_config, dataset=dataset, model="msvae", run_id=run_id, seed=seed, use_run_paths=run_id is not None)
    checkpoint = Path(cfg.get("checkpoint_path") or f"outputs/checkpoints/{dataset}/msvae/msvae.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(f"DH-VAE checkpoint not found: {checkpoint}. Run scripts/train_msvae_model.py first.")
    model, checkpoint_payload = load_msvae_checkpoint(checkpoint, device=device)
    out = _resolved_sample_output(cfg, dataset, "msvae", run_id)
    metadata_out = sample_metadata_path(dataset, "msvae", run_id=run_id)
    if out.exists() and not force:
        raise FileExistsError(f"Sample file already exists: {out}. Use --force to overwrite.")

    sample_temperature = float(temperature if temperature is not None else cfg.get("sample_temperature", 1.0))
    max_attempts = int(max_attempts or max(10 * int(num_samples), int(num_samples)))
    accepted: list[list[int]] = []
    rejected = {"not_graphical": 0, "not_connected_feasible": 0, "empty_sequence": 0}
    attempts = 0
    start = time.perf_counter()
    with PeakMemoryMonitor() as memory_monitor:
        while len(accepted) < int(num_samples) and attempts < max_attempts:
            batch_size = min(max(32, int(num_samples) - len(accepted)), max_attempts - attempts)
            candidates = model.generate(batch_size, temperature=sample_temperature)
            attempts += batch_size
            for seq in candidates:
                if not seq:
                    rejected["empty_sequence"] += 1
                    continue
                ok, _ = check_sequence_validity(seq)
                if not ok:
                    rejected["not_graphical"] += 1
                    continue
                if require_connected_feasible:
                    feasible, _ = connected_sequence_feasible(seq)
                    if not feasible:
                        rejected["not_connected_feasible"] += 1
                        continue
                accepted.append([int(d) for d in seq])
                if len(accepted) >= int(num_samples):
                    break
    elapsed = time.perf_counter() - start
    out.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(accepted, out, force=force)
    save_yaml(cfg, out.with_suffix(".resolved_model_config.yaml"), force=True)
    validity_rate = len(accepted) / max(attempts, 1)
    compute = compute_report(operation="sampling", runtime_seconds=elapsed, num_graphs=len(accepted), memory=memory_monitor.to_dict())
    metadata = {
        "dataset": dataset,
        "model": "dhvae",
        "model_alias": "msvae",
        "seed": seed,
        "run_id": run_id,
        "checkpoint_path": str(checkpoint),
        "sample_path": str(out),
        "num_samples_requested": int(num_samples),
        "num_samples_saved": len(accepted),
        "attempts": attempts,
        "acceptance_rate": validity_rate,
        "rejected": rejected,
        "sample_temperature": sample_temperature,
        "runtime_seconds": elapsed,
        "compute": compute,
        "checkpoint_training_stats": checkpoint_payload.get("degree_sequence_stats", {}),
    }
    save_json(metadata, metadata_out, force=True)
    logger.info("Saved %d DH-VAE degree sequences to %s", len(accepted), out)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate degree sequences from a trained size-conditioned DH-VAE.")
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--model-config", type=str, default="configs/models/msvae.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-disconnected-degree-sequences", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None, help="Override sample_temperature from the config.")
    args = parser.parse_args()
    generate_msvae_samples(
        dataset=args.dataset,
        model_config=load_yaml(args.model_config),
        num_samples=args.num_samples,
        seed=args.seed,
        run_id=args.run_id,
        device=args.device,
        force=args.force,
        require_connected_feasible=not args.allow_disconnected_degree_sequences,
        max_attempts=args.max_attempts,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
