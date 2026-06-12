from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from grapher.models.model_grapher import GraphER
from grapher.models.model_dhvae import DHVAE


def load_dhvae_checkpoint(path: str | Path, device: str = "cpu") -> tuple[DHVAE, dict[str, Any]]:
    """Load a size-conditioned DH-VAE degree-prior checkpoint."""

    payload = torch.load(path, map_location=device, weights_only=False)
    params = dict(payload.get("model_params", {}))
    architecture = str(params.get("architecture") or payload.get("model_name") or "").lower()
    state_dict = payload.get("model_state_dict", payload)

    looks_legacy = any(str(k).startswith("decoder.logits_layer") for k in state_dict.keys())
    if looks_legacy and "dhvae" not in architecture and "size_conditioned" not in architecture:
        raise RuntimeError(
            "The checkpoint appears to use an old independent-count architecture. "
            "Please retrain with scripts/train_dhvae_model.py before generating samples."
        )

    max_nodes = params.get("max_nodes")
    if max_nodes is None:
        # Backward-compatible inference for revised checkpoints written during
        # transition periods.
        if params.get("max_frequency") is not None:
            max_nodes = int(params["max_frequency"]) - 1
        elif params.get("num_nodes") is not None:
            max_nodes = int(params["num_nodes"])
    if max_nodes is None:
        raise KeyError("DH-VAE checkpoint is missing model_params['max_nodes'].")

    model = DHVAE(
        max_nodes=int(max_nodes),
        histogram_dim=int(params.get("histogram_dim", params.get("max_input_dim", max_nodes))),
        hidden_dim=int(params.get("hidden_dim", 128)),
        latent_dim=int(params.get("latent_dim", 32)),
        size_embedding_dim=int(params.get("size_embedding_dim", 32)),
    )
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not load the degree-prior checkpoint into the size-conditioned DH-VAE. "
            "This usually means the checkpoint was trained before the DH-VAE revision; retrain "
            "scripts/train_dhvae_model.py."
        ) from exc
    model.to(device)
    model.eval()
    return model, payload


def load_grapher_checkpoint(path: str | Path, device: str = "cpu") -> tuple[GraphER, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    params = dict(payload.get("model_params", {}))
    state_dict = payload.get("model_state_dict", payload)
    required = {"node_in_dim", "hidden_dim", "num_layer", "T"}
    missing = sorted(required - set(params))
    if missing:
        raise KeyError(f"GraphER checkpoint is missing model_params keys: {missing}")

    architecture = str(params.get("architecture") or payload.get("model_architecture") or "").lower()
    looks_legacy_second_edge = any(str(k).startswith("edge_predictor") for k in state_dict.keys())
    if looks_legacy_second_edge and "complete_action" not in architecture:
        raise RuntimeError(
            "The GraphER checkpoint appears to use the old second-edge scorer. "
            "The code now uses the paper-aligned complete-action scorer over (e1,e2,r); "
            "please retrain with scripts/train_grapher_model.py before generating samples."
        )

    max_nodes = params.get("max_nodes") or params.get("degree_histogram_dim") or 64
    degree_histogram_dim = params.get("degree_histogram_dim") or max_nodes
    model = GraphER(
        node_in_dim=int(params["node_in_dim"]),
        hidden_dim=int(params["hidden_dim"]),
        num_layer=int(params["num_layer"]),
        T=int(params["T"]),
        max_nodes=int(max_nodes),
        degree_histogram_dim=int(degree_histogram_dim),
        time_embedding_dim=int(params.get("time_embedding_dim", params["hidden_dim"])),
        local_feature_dim=int(params.get("local_feature_dim", 8)),
        dropout=float(params.get("dropout", 0.0)),
    )
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not load the GraphER checkpoint into the complete-action scorer. "
            "This usually means the checkpoint was trained before the generic GraphER revision; "
            "retrain scripts/train_grapher_model.py."
        ) from exc
    model.to(device)
    model.eval()
    return model, payload
