from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from grapher.models.model_grapher import GraphER
from grapher.models.model_dhvae import DHVAE


def _torch_load_compat(*args: Any, **kwargs: Any) -> Any:
    """torch.load wrapper compatible with old and new PyTorch versions.

    Newer PyTorch supports weights_only=False.
    Older PyTorch, such as torch 1.12, raises TypeError for weights_only.
    """
    try:
        return torch.load(*args, **kwargs)
    except TypeError as exc:
        msg = str(exc).lower()

        is_weights_only_error = (
            "weights_only" in msg
            or "invalid keyword" in msg
            or "unexpected keyword" in msg
        )

        if not is_weights_only_error:
            raise

        kwargs.pop("weights_only", None)
        return torch.load(*args, **kwargs)


def _prepare_grapher_state_dict_for_pyg_compat(
    model: torch.nn.Module,
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Handle harmless GINConv eps differences across PyG versions.

    Some PyG versions store GINConv.eps in the state_dict, while others do not.
    This function:
      1. Adds missing gin_layers.*.eps keys from the initialized model.
      2. Removes unexpected gin_layers.*.eps keys if the current model does not use them.
    It does not ignore any other parameter mismatch.
    """
    model_state = model.state_dict()
    prepared = dict(state_dict)

    # Drop checkpoint eps keys if the current PyG model does not have them.
    for key in list(prepared.keys()):
        if key.startswith("gin_layers.") and key.endswith(".eps") and key not in model_state:
            prepared.pop(key)

    # Add missing eps keys if the current PyG model expects them.
    for key, value in model_state.items():
        if key.startswith("gin_layers.") and key.endswith(".eps") and key not in prepared:
            prepared[key] = value.detach().clone()

    return prepared


def load_dhvae_checkpoint(path: str | Path, device: str = "cpu") -> tuple[DHVAE, dict[str, Any]]:
    """Load a size-conditioned DH-VAE degree-prior checkpoint."""

    payload = _torch_load_compat(path, map_location=device, weights_only=False)
    params = dict(payload.get("model_params", {}))
    state_dict = payload.get("model_state_dict", payload)

    max_nodes = params.get("max_nodes")
    if max_nodes is None:
        raise KeyError("DH-VAE checkpoint is missing model_params['max_nodes'].")

    model = DHVAE(
        max_nodes=int(max_nodes),
        histogram_dim=int(params.get("histogram_dim", max_nodes)),
        hidden_dim=int(params.get("hidden_dim", 128)),
        latent_dim=int(params.get("latent_dim", 32)),
        size_embedding_dim=int(params.get("size_embedding_dim", 32)),
    )

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not load the degree-prior checkpoint into the size-conditioned DH-VAE."
        ) from exc

    model.to(device)
    model.eval()
    return model, payload


def load_grapher_checkpoint(path: str | Path, device: str = "cpu") -> tuple[GraphER, dict[str, Any]]:
    """Load a complete-action GraphER checkpoint."""

    payload = _torch_load_compat(path, map_location=device, weights_only=False)
    params = dict(payload.get("model_params", {}))
    state_dict = payload.get("model_state_dict", payload)

    required = {"node_in_dim", "hidden_dim", "num_layer", "T"}
    missing = sorted(required - set(params))
    if missing:
        raise KeyError(f"GraphER checkpoint is missing model_params keys: {missing}")

    max_nodes = params.get("max_nodes")
    degree_histogram_dim = params.get("degree_histogram_dim")

    if max_nodes is None:
        raise KeyError("GraphER checkpoint is missing model_params['max_nodes'].")
    if degree_histogram_dim is None:
        raise KeyError("GraphER checkpoint is missing model_params['degree_histogram_dim'].")

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

    state_dict = _prepare_grapher_state_dict_for_pyg_compat(model, state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not load the GraphER checkpoint into the complete-action scorer."
        ) from exc

    model.to(device)
    model.eval()
    return model, payload