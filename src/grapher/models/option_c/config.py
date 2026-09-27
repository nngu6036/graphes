"""Explicit, checked configuration for the independent Option-C experiment."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

# Every provided YAML declares these fields; no baseline/common-config merge.
FIELDS = {
    "schema_version": None,
    "dataset": {"benchmark", "name", "root", "config_path"},
    "categories": {"node_attribute", "node_categories", "edge_attribute", "edge_categories"},
    "edge_representation": {"weights", "scale"},
    "node_noise": {"type", "schedule", "pseudocount"},
    "diffusion": {"steps", "prediction"},
    "model": {"max_nodes", "hidden_dim", "num_layers", "num_heads", "ff_dim", "dropout"},
    "spectral": {"normalization", "solver_device", "dtype"},
    "graphlets": {"sizes", "size_weights", "connected_only", "counting", "clustering_bins",
                  "max_vocab_per_size", "min_train_count", "max_connected_subsets"},
    "loss_weights": {"node", "adjacency", "spectral", "graphlet", "mass", "clustering", "orbit"},
    "training": {"epochs", "batch_size", "lr", "weight_decay", "grad_norm", "validation_every",
                 "validation_seed", "log_every", "log_every_batches", "checkpoint_every",
                 "permutation_augmentation", "ema_decay", "cache_dir"},
    "sampling": {"steps", "batch_size", "sampler", "clip_clean", "use_ema", "save_continuous"},
    "refinement": {"enabled", "timing", "max_steps", "proposal_budget", "valid_candidate_budget",
                   "preserve_connectivity_if_connected", "require_structure_improvement",
                   "min_improvement", "weights"},
    "protocol": {"seeds", "generated_graphs_per_seed", "full_training_split"},
}


def _positive(value: Any, name: str, *, integer: bool = False, allow_zero: bool = False) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean")
    if integer and (not isinstance(value, int)):
        raise ValueError(f"{name} must be an integer")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


def validate_config(config: dict) -> dict:
    """Reject misspelled/unimplemented options rather than silently ignoring them."""
    if not isinstance(config, dict) or set(config) != set(FIELDS):
        missing = set(FIELDS) - set(config or {})
        extra = set(config or {}) - set(FIELDS)
        raise ValueError(f"Option-C config fields: missing={sorted(missing)}, unknown={sorted(extra)}")
    cfg = copy.deepcopy(config)
    if cfg["schema_version"] != 1:
        raise ValueError("Only option_c.schema_version=1 is supported")
    for section, keys in FIELDS.items():
        if keys is not None and (not isinstance(cfg[section], dict) or set(cfg[section]) != keys):
            raise ValueError(f"option_c.{section} must declare exactly {sorted(keys)}")
    if cfg["diffusion"]["prediction"] != "clean_adjacency":
        raise ValueError("Option C predicts clean weighted adjacency, not epsilon or an independent spectrum")
    if cfg["node_noise"]["type"] != "marginal" or cfg["node_noise"]["schedule"] != "cosine_exact_terminal":
        raise ValueError("Node diffusion uses the existing marginal, exact-terminal cosine process")
    if cfg["spectral"]["normalization"] != "sqrt_n" or cfg["spectral"]["dtype"] != "float64":
        raise ValueError("Spectral loss uses eigenvalues of scaled W divided by sqrt(n), in float64")
    if cfg["spectral"]["solver_device"] not in ("cpu", "model"):
        raise ValueError("spectral.solver_device must be cpu or model")
    if cfg["sampling"]["sampler"] not in ("ddpm", "ddim"):
        raise ValueError("sampling.sampler must be ddpm or ddim")
    if cfg["refinement"]["timing"] != "final_only":
        raise ValueError("Only final-only refinement is implemented; no noisy-state rounding or feedback")
    for section, keys in {
        "protocol": ("full_training_split",), "graphlets": ("connected_only",),
        "training": ("permutation_augmentation",),
        "sampling": ("clip_clean", "use_ema", "save_continuous"),
        "refinement": ("enabled", "preserve_connectivity_if_connected", "require_structure_improvement"),
    }.items():
        for key in keys:
            if not isinstance(cfg[section][key], bool):
                raise ValueError(f"{section}.{key} must be a YAML boolean")
    if not cfg["protocol"]["full_training_split"]:
        raise ValueError("Option C trains on the entire prepared train split; no implicit subsampling")
    for key in ("benchmark", "name"):
        value = cfg["dataset"][key]
        if not isinstance(value, str) or not value or Path(value).name != value or value in (".", ".."):
            raise ValueError(f"dataset.{key} must be a simple identifier")
    for key in ("root", "config_path"):
        if not isinstance(cfg["dataset"][key], str) or not cfg["dataset"][key]:
            raise ValueError(f"dataset.{key} must be a nonempty path")
    for key in ("node_categories", "edge_categories"):
        values = cfg["categories"][key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"categories.{key} must explicitly list distinct categories")
    for section, keys in {
        "model": ("max_nodes", "hidden_dim", "num_layers", "num_heads", "ff_dim"),
        "training": ("epochs", "batch_size", "validation_every", "log_every", "checkpoint_every"),
        "sampling": ("steps", "batch_size"), "diffusion": ("steps",),
        "graphlets": ("clustering_bins", "min_train_count", "max_connected_subsets"),
        "refinement": ("proposal_budget", "valid_candidate_budget"),
        "protocol": ("generated_graphs_per_seed",),
    }.items():
        for key in keys:
            _positive(cfg[section][key], f"{section}.{key}", integer=True)
    for section, key in (("training", "log_every_batches"), ("refinement", "max_steps"),
                         ("training", "validation_seed")):
        _positive(cfg[section][key], f"{section}.{key}", integer=True, allow_zero=True)
    for section, key in (("training", "lr"), ("training", "grad_norm"),
                         ("node_noise", "pseudocount"), ("edge_representation", "scale")):
        _positive(cfg[section][key], f"{section}.{key}")
    _positive(cfg["training"]["weight_decay"], "training.weight_decay", allow_zero=True)
    _positive(cfg["refinement"]["min_improvement"], "refinement.min_improvement", allow_zero=True)
    m = cfg["model"]
    if m["hidden_dim"] < 4 or m["hidden_dim"] % m["num_heads"]:
        raise ValueError("hidden_dim must be >=4 and divisible by num_heads")
    if not 0 <= float(m["dropout"]) < 1 or not 0 <= float(cfg["training"]["ema_decay"]) < 1:
        raise ValueError("dropout and ema_decay must lie in [0,1)")
    if cfg["diffusion"]["steps"] < 2 or cfg["sampling"]["steps"] > cfg["diffusion"]["steps"]:
        raise ValueError("Require 2 <= diffusion.steps and sampling.steps <= diffusion.steps")
    gc = cfg["graphlets"]
    if gc["sizes"] != sorted(set(gc["sizes"])) or not gc["sizes"] or any(k not in (3, 4, 5) for k in gc["sizes"]):
        raise ValueError("graphlets.sizes must be a sorted subset of [3,4,5]")
    if not gc["connected_only"] or gc["counting"] != "exact_connected":
        raise ValueError("Graphlet targets use exact connected induced counts")
    if len(gc["size_weights"]) != len(gc["sizes"]):
        raise ValueError("One graphlet weight is required per order")
    for weight in gc["size_weights"]:
        _positive(weight, "graphlets.size_weights")
    if gc["max_vocab_per_size"] is not None:
        _positive(gc["max_vocab_per_size"], "graphlets.max_vocab_per_size", integer=True)
    required = {"graphlet", "mass", "clustering", "orbit", "spectral", "adjacency"}
    if set(cfg["refinement"]["weights"]) != required:
        raise ValueError(f"refinement.weights must contain {sorted(required)}; there is no edge NLL")
    for section in (cfg["loss_weights"], cfg["refinement"]["weights"]):
        for name, value in section.items():
            _positive(value, f"loss weight {name}", allow_zero=True)
    if cfg["loss_weights"]["adjacency"] <= 0 or cfg["loss_weights"]["node"] <= 0:
        raise ValueError("Both adjacency and node denoising losses must be positive")
    if cfg["refinement"]["enabled"]:
        for key in ("graphlet", "mass", "clustering", "orbit", "spectral"):
            if cfg["refinement"]["weights"][key] > 0 and cfg["loss_weights"][key] == 0:
                raise ValueError(f"Cannot guide with an untrained {key} target")
    if not isinstance(cfg["protocol"]["seeds"], list):
        raise ValueError("protocol.seeds must be a list")
    for value in cfg["protocol"]["seeds"]:
        _positive(value, "protocol.seeds", integer=True, allow_zero=True)
    if not cfg["protocol"]["seeds"]:
        raise ValueError("protocol.seeds cannot be empty")
    # Map by actual attribute VALUE, not by a sorted vocabulary's tensor index.
    mapping = {str(k): float(v) for k, v in cfg["edge_representation"]["weights"].items()}
    categories = cfg["categories"]["edge_categories"]
    if set(mapping) != {str(v) for v in categories}:
        raise ValueError("Declare one positive scalar weight for each present-edge category")
    weights = [mapping[str(v)] for v in categories]
    if any(not math.isfinite(w) or w <= 0 for w in weights) or len(set(weights)) != len(weights):
        raise ValueError("Present-edge weights must be finite, positive and distinct")
    if max(weights) > cfg["edge_representation"]["scale"]:
        raise ValueError("edge_representation.scale must be at least the largest physical edge weight")
    cfg["edge_representation"]["weights"] = mapping
    return cfg


def load_config(path: str | Path) -> dict:
    with Path(path).open() as handle:
        root = yaml.safe_load(handle)
    if not isinstance(root, dict) or set(root) != {"option_c"}:
        raise ValueError("Expected a standalone YAML with only the option_c root (not gdsm_simple)")
    return validate_config(root["option_c"])


def generation_contract(cfg: dict) -> dict:
    """Training semantics that cannot be changed at sampling time."""
    result = {key: copy.deepcopy(cfg[key]) for key in (
        "schema_version", "categories", "edge_representation", "node_noise", "diffusion",
        "model", "spectral", "graphlets", "loss_weights")}
    result["dataset"] = {k: cfg["dataset"][k] for k in ("benchmark", "name")}
    return result


def resume_contract(cfg: dict) -> dict:
    result = generation_contract(cfg)
    result["training"] = {k: v for k, v in cfg["training"].items()
                          if k not in ("epochs", "log_every", "log_every_batches", "checkpoint_every", "cache_dir")}
    return result
