from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from grapher.models.checkpoint import load_grapher_checkpoint, load_msvae_checkpoint
from grapher.utils.numerics import assert_finite_graphs
from grapher.utils.progress import progress_bar, update_progress
from grapher.utils.seed import set_seed


def sample_graphs(
    model_name: str,
    model_cfg: dict[str, Any],
    num_graphs: int,
    seed: int = 0,
    *,
    show_progress: bool = False,
    progress_desc: str | None = None,
):
    """Sample from the revised local models using resolved checkpoint config."""

    set_seed(seed, include_torch=True)
    model_key = model_name.lower()
    device = model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if model_key in {"msvae", "dhvae"}:
        checkpoint = Path(model_cfg.get("checkpoint_path") or "outputs/checkpoints/msvae.pt")
        model, _ = load_msvae_checkpoint(checkpoint, device=device)
        sequences = model.generate(int(num_graphs), temperature=float(model_cfg.get("sample_temperature", 1.0)))
        return sequences
    if model_key in {"grapher", "grapher_generic", "grapher_attributed"}:
        grapher_checkpoint = Path(model_cfg.get("checkpoint_path") or "outputs/checkpoints/grapher.pt")
        msvae_checkpoint = Path(model_cfg.get("msvae_checkpoint_path") or "outputs/checkpoints/msvae.pt")
        grapher, payload = load_grapher_checkpoint(grapher_checkpoint, device=device)
        msvae, _ = load_msvae_checkpoint(msvae_checkpoint, device=device)
        desc = progress_desc or f"Sampling {model_name}"
        graphs = []
        with progress_bar(total=int(num_graphs), desc=desc, unit="graph", enabled=show_progress) as update:
            rounds = 0
            while len(graphs) < int(num_graphs) and rounds < 20:
                rounds += 1
                need = int(num_graphs) - len(graphs)
                batch, _ = grapher.generate(
                    num_samples=need,
                    num_steps=int(model_cfg.get("num_steps", payload.get("model_params", {}).get("T", 32))),
                    msvae_model=msvae,
                    k_eigen=int(model_cfg.get("k_eigen", payload.get("model_params", {}).get("k_eigen", 4))),
                    method=str(model_cfg.get("init_method", "havel_hakimi")),
                    ensure_connected=bool(model_cfg.get("ensure_connected", True)),
                    k_hop=int(model_cfg.get("k_hop", 2)),
                    max_candidates=int(model_cfg.get("candidate_budget", 64)),
                    degree_temperature=float(model_cfg.get("degree_sample_temperature", model_cfg.get("msvae_temperature", 1.0))),
                    action_temperature=float(model_cfg.get("action_temperature", model_cfg.get("temperature", 1.0))),
                    sample_actions=bool(model_cfg.get("sample_actions", True)),
                )
                if not batch:
                    break
                graphs.extend(batch[:need])
                update_progress(update, min(len(batch), need))
        assert_finite_graphs(graphs, context=f"{model_name}.sample output")
        return graphs
    raise KeyError(f"Unknown model_name={model_name!r}; expected 'msvae'/'dhvae' or 'grapher'.")


def model_capabilities(model_name: str) -> dict[str, bool]:
    model_key = model_name.lower()
    if model_key in {"msvae", "dhvae"}:
        return {
            "supports_training": True,
            "supports_sampling": True,
            "supports_degree_sequences": True,
            "supports_graph_sampling": False,
            "supports_node_features": False,
            "supports_edge_features": False,
            "supports_constraints": True,
            "supports_variable_size": True,
        }
    return {
        "supports_training": True,
        "supports_sampling": True,
        "supports_degree_sequences": False,
        "supports_graph_sampling": True,
        "supports_node_features": False,
        "supports_edge_features": False,
        "supports_node_labels": False,
        "supports_edge_labels": False,
        "supports_graph_labels": False,
        "supports_constraints": True,
        "supports_variable_size": True,
        "supports_featureless_graphs": True,
    }
