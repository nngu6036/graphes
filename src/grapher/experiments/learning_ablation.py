"""Learning controls for the GraphES paper, with explicit causal interventions.

The planning/configuration helpers intentionally import neither torch nor the
model package, so plans are reviewable on machines without training dependencies.
Neural checkpoint loading is confined to the attribute-only training hook.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

DATASETS = ("community_small", "ego_small", "qm9", "zinc")
MOLECULAR = ("qm9", "zinc")
CONTROLS = ("S0", "S1", "S2", "S3", "S4", "K3", "K34", "A0", "A1", "A2")
SERIALIZED = {"community_small": "sbm", "ego_small": "ego_small", "qm9": "qm9_attributed", "zinc": "zinc"}
DEFINITIONS = {
    "S0": "Zero total topology-summary loss; retain all heads and backbone capacity.",
    "S1": "Zero topology graphlet histogram and mass losses; retain their heads.",
    "S2": "Zero topology orbit and clustering losses; clustering is already disabled, so this is orbit removal.",
    "S3": "Zero ordinary topology degree histogram and CDF losses; inapplicable when those losses are already off.",
    "S4": "Detach shared features only for topology-summary heads; keep head learning and spectral-denoising gradients.",
    "K3": "Use topology graphlet order {3}; molecular typed graphlet orders remain unchanged.",
    "K34": "Use topology graphlet orders {3,4}; molecular typed graphlet orders remain unchanged.",
    "A0": "Matched attribute-only reference: retrain fresh full attribute branch with reference topology frozen; compare A1/A2 against this row, not joint-training T.",
    "A1": "Retrain only fresh attribute parameters without typed-summary losses, with reference topology parameters frozen.",
    "A2": "Retrain only fresh attribute parameters with global log-gap input to typed-summary readouts zeroed. Local degree/RW/SP/topology context remains; categorical logits have no direct spectral input in this implementation.",
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def applicable_controls(dataset: str) -> tuple[str, ...]:
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}")
    return tuple(v for v in CONTROLS if not (v.startswith("A") and dataset not in MOLECULAR) and not (v == "S3" and dataset == "qm9"))


def _active_task(summary: Mapping[str, Any], task: str, keys: tuple[str, ...]) -> bool:
    cfg = dict(summary.get(task, {}) or {})
    return bool(summary.get("enabled", False)) and float(summary.get("loss_weight", 0)) > 0 and bool(cfg.get("enabled", False)) and any(float(cfg.get(key, 0)) > 0 for key in keys)


def build_control_config(base: Mapping[str, Any], dataset: str, variant: str, seed: int) -> dict[str, Any]:
    """Construct a T/B=0 control, rejecting interventions with no active target.

    Removal controls set loss weights to zero rather than removing modules. This
    preserves network capacity and parameter initialization under the same seed.
    """
    if variant not in applicable_controls(dataset):
        reason = "QM9 ordinary degree supervision is already disabled; S3 is N/A" if dataset == "qm9" and variant == "S3" else f"{variant} is not defined for {dataset}"
        raise ValueError(reason)
    result = copy.deepcopy(dict(base))
    options = result["grapher"]
    if int(options["train"]["epochs"]) != 200:
        raise ValueError("Current paper-ablation protocol requires exactly 200 epochs; do not relabel 10,000-epoch checkpoints")
    summary_key = "topology_summary" if dataset in MOLECULAR else "structure_summary"
    summary = options[summary_key]
    if not bool(summary.get("enabled", False)) or float(summary.get("loss_weight", 0)) <= 0:
        raise ValueError("Reference has no active topology-summary objective")
    if variant == "S0":
        summary["loss_weight"] = 0.0
    elif variant == "S1":
        if not _active_task(summary, "graphlet", ("histogram_weight", "mass_weight")):
            raise ValueError("S1 would remove no active graphlet/mass loss")
        summary["graphlet"].update(histogram_weight=0.0, mass_weight=0.0)
    elif variant == "S2":
        active = False
        for task, weights in (("orbit", ("histogram_weight", "log_total_weight")), ("clustering", ("histogram_weight", "cdf_weight"))):
            active |= _active_task(summary, task, weights)
            if task in summary:
                summary[task].update({key: 0.0 for key in weights})
        if not active:
            raise ValueError("S2 would remove no active orbit/clustering loss")
    elif variant == "S3":
        if not _active_task(summary, "degree", ("histogram_weight", "cdf_weight")):
            raise ValueError("S3 would remove no active ordinary degree loss")
        summary["degree"].update(histogram_weight=0.0, cdf_weight=0.0)
    elif variant == "S4":
        if summary.get("detach_encoder", False):
            raise ValueError("S4 reference already has detached summary features")
        summary["detach_encoder"] = True
    elif variant in {"K3", "K34"}:
        orders = [3] if variant == "K3" else [3, 4]
        if list(summary["graphlet"]["orders"]) == orders:
            raise ValueError(f"{variant} reference already uses the requested orders")
        summary["graphlet"]["orders"] = orders
    elif variant == "A1":
        if float(options["attribute_summary"].get("loss_weight", 0)) <= 0:
            raise ValueError("A1 reference already has no typed-summary objective")
        options["attribute_summary"]["loss_weight"] = 0.0
    elif variant == "A2":
        if options["attributed"].get("spectral_context", "summary_log_gap") == "none":
            raise ValueError("A2 reference already has no summary-side spectral context")
        if float(options["attribute_summary"].get("loss_weight", 0)) <= 0:
            raise ValueError("A2 would be a no-op because only the inactive typed-summary heads consume spectral context")
        options["attributed"]["spectral_context"] = "none"
    # Every learning control is evaluated through threshold decoding with B=0.
    sample = options.setdefault("sample", {})
    sample.setdefault("degree_constrained_realization", {})["enabled"] = False
    sample.setdefault("matched_ablation", {})["enabled"] = False
    if "eigenbasis_perturbation" in sample:
        sample["eigenbasis_perturbation"]["enabled"] = False
    if dataset in MOLECULAR:
        options["attributed_diversification"]["enabled"] = False
        sample["separate_attribute_rng"] = True
        sample["use_ema"] = False
    elif "graphlet_refinement" in options:
        options["graphlet_refinement"]["enabled"] = False
    result["experiment"] = f"{dataset}_learning_v1_{variant}_seed_{seed}"
    result["seed"] = int(seed)
    result["description"] = DEFINITIONS[variant] + " Threshold decoding; B=0; configured 200 epochs."
    return result


def reference_paths(output_root: Path, dataset: str, seed: int, n: int, run_template: str | None = None, generation_template: str | None = None) -> tuple[Path, Path]:
    profile = "B1to4" if dataset in MOLECULAR else "tau0p4_B1to4"
    run = (run_template or "seed_{seed}_{dataset}_grapher_final_{profile}_g345_v2").format(seed=seed, dataset=dataset, profile=profile, n=n)
    generation = (generation_template or "paper_ablation_v1_seed_{seed}_n_{n}").format(seed=seed, dataset=dataset, profile=profile, n=n)
    for value in (run, generation):
        if not value or Path(value).name != value or value in {".", ".."}:
            raise ValueError("Reference templates must resolve to path-free run/generation identifiers")
    root = output_root / "grapher" / dataset / run
    return root, root / "generations" / generation


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def verify_manifest_files(directory: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    """Verify every hashed generated artifact, including optional evidence."""
    files = dict(manifest.get("files", {}) or {})
    # Generation format uses top-level artifact records, while a few exporters
    # use a files mapping. Accept both only when their file hashes are present.
    files.update({key: value for key, value in manifest.items() if isinstance(value, dict) and value.get("path") and value.get("sha256")})
    hashes: dict[str, str] = {}
    for name, record in files.items():
        if name == "checkpoint":
            continue
        if not isinstance(record, Mapping) or not record.get("path") or not record.get("sha256"):
            continue
        path = (directory / str(record["path"])).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError(f"Artifact escapes generation directory: {name}")
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"Missing or modified generated artifact: {path}")
        hashes[str(name)] = str(record["sha256"])
    return hashes


def require_attribute_reference(run_dir: Path, generation_dir: Path, *, dataset: str, seed: int, n: int) -> dict[str, Any]:
    """Fail before expensive work unless A controls have an immutable parent."""
    checkpoint = run_dir / "train" / "checkpoints" / "grapher.pt"
    train_manifest_path = run_dir / "train" / "manifest.json"
    generation_manifest_path = generation_dir / "manifest.json"
    for path in (checkpoint, train_manifest_path, generation_manifest_path, generation_dir / "topology_pairing.json", generation_dir / "sampled_basis_indices.pkl", generation_dir / "topology_graphs.pkl"):
        if not path.is_file():
            raise FileNotFoundError(f"A1/A2 require a compatible reference checkpoint AND parent generation; missing {path}")
    train_manifest = _read_json(train_manifest_path)
    gen_manifest = _read_json(generation_manifest_path)
    actual_sha = sha256(checkpoint)
    if train_manifest.get("checkpoint", {}).get("sha256") != actual_sha:
        raise ValueError("Reference checkpoint hash does not match training manifest")
    if train_manifest.get("dataset", {}).get("benchmark_id") != dataset or int(train_manifest.get("train_seed", -1)) != seed:
        raise ValueError("Reference training dataset/seed mismatch")
    if int(train_manifest.get("checkpoint_selection", {}).get("epoch", -1)) != 200 or int(train_manifest.get("options", {}).get("train", {}).get("epochs", -1)) != 200:
        raise ValueError("Attribute controls require a verified final 200-epoch reference")
    if int(gen_manifest.get("generation_seed", -1)) != seed or int(gen_manifest.get("num_generated", -1)) != n:
        raise ValueError("Reference generation seed or requested population size differs")
    hashes = verify_manifest_files(generation_dir, gen_manifest)
    if not {"topology_graphs", "topology_pairing", "sampled_basis_indices"}.issubset(hashes):
        raise ValueError("Reference generation lacks hashed topology, pairing, or donor evidence")
    pairing = _read_json(generation_dir / "topology_pairing.json")
    if pairing.get("checkpoint_sha256") != actual_sha or len(pairing.get("per_graph_sha256", [])) != n:
        raise ValueError("Reference topology pairing does not identify the reference checkpoint/population")
    sample = dict(gen_manifest.get("effective_sample_config", {}) or {})
    if not sample or not bool(sample.get("separate_attribute_rng", True)):
        raise ValueError("A controls require a recorded separate topology/attribute RNG policy")
    if bool((sample.get("eigenbasis_perturbation", {}) or {}).get("enabled", False)) or sample.get("basis_ablation"):
        raise ValueError("A controls require the unperturbed reference donor policy")
    if bool(sample.get("use_ema", False)):
        raise ValueError("Attribute controls currently require sample.use_ema=false on the reference")
    return {
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": actual_sha,
        "training_manifest": str(train_manifest_path.resolve()),
        "training_manifest_sha256": sha256(train_manifest_path),
        "generation_dir": str(generation_dir.resolve()),
        "generation_manifest_sha256": sha256(generation_manifest_path),
        "topology_pairing_sha256": sha256(generation_dir / "topology_pairing.json"),
        "sampled_basis_indices_sha256": sha256(generation_dir / "sampled_basis_indices.pkl"),
        "batch_size": int(pairing["batch_size"]),
        "effective_sample_config": sample,
        "dataset_fingerprint": train_manifest["dataset"]["fingerprint"],
        "generation_seed": seed, "num_requested": n,
    }


def _topology_parameter_names(model: Any) -> tuple[str, ...]:
    ids = {id(value) for value in model.topology_parameters()}
    names = tuple(name for name, value in model.named_parameters() if id(value) in ids)
    if not names or len(names) != len(ids):
        raise ValueError("Topology parameter partition is incomplete")
    attribute_ids = {id(value) for value in model.attribute_parameters()}
    if ids & attribute_ids:
        raise ValueError("Attribute and topology parameters overlap")
    return names


def _tensor_digest(named_tensors: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initialize_frozen_topology(model_x: Any, model_lam: Any, *, options: Mapping[str, Any], dataset_fingerprint: str, train_seed: int, model_config: Mapping[str, Any], topology_meta: Mapping[str, Any], spectral_transform: Mapping[str, Any], basis_adjacencies: Any, basis_num_nodes: Any) -> dict[str, Any]:
    """Load topology weights only; fresh attribute weights retain their seed initialization."""
    import numpy as np
    import torch

    train_cfg = dict(options.get("train", {}) or {})
    reference = dict(train_cfg.get("frozen_topology_reference", {}) or {})
    required = ("checkpoint", "checkpoint_sha256", "training_manifest", "training_manifest_sha256", "generation_dir", "generation_manifest_sha256", "topology_pairing_sha256", "sampled_basis_indices_sha256")
    if any(not reference.get(key) for key in required):
        raise ValueError("attribute_only requires the runner's complete frozen_topology_reference record")
    for path_key, hash_key in (("checkpoint", "checkpoint_sha256"), ("training_manifest", "training_manifest_sha256")):
        if sha256(reference[path_key]) != reference[hash_key]:
            raise ValueError(f"Reference {path_key} changed after preflight")
    parent_generation = Path(reference["generation_dir"])
    for name, expected_key in (("manifest.json", "generation_manifest_sha256"), ("topology_pairing.json", "topology_pairing_sha256"), ("sampled_basis_indices.pkl", "sampled_basis_indices_sha256")):
        if sha256(parent_generation / name) != reference[expected_key]:
            raise ValueError(f"Reference generation changed after preflight: {name}")
    manifest = _read_json(Path(reference["training_manifest"]))
    if manifest.get("dataset", {}).get("fingerprint") != dataset_fingerprint or int(manifest.get("train_seed", -1)) != train_seed:
        raise ValueError("Frozen reference dataset fingerprint or seed differs from attribute training")
    if int(manifest.get("checkpoint_selection", {}).get("epoch", -1)) != int(train_cfg["epochs"]) or int(train_cfg["epochs"]) != 200:
        raise ValueError("Frozen-reference and attribute training must both follow the 200-epoch protocol")
    state = torch.load(reference["checkpoint"], map_location="cpu", weights_only=False)
    if state.get("format") != "grapher_laplacian_loggap_hybrid_attributed_ppgn_v1":
        raise ValueError("A controls require the hybrid attributed checkpoint format")
    if state.get("attribute_diffusion_training_contract") != "fixed_topology_marginal_attribute_diffusion_v1":
        raise ValueError("Reference does not satisfy the hybrid categorical training contract")
    reference_cfg = dict(state["model_config"])
    reference_cfg.setdefault("detach_topology_summary_encoder", False)
    reference_cfg.setdefault("attribute_spectral_context", "summary_log_gap")
    actual_cfg = dict(model_config)
    # A2 changes only the attribute readout's spectral side-channel.
    reference_cfg.pop("attribute_spectral_context", None)
    actual_cfg.pop("attribute_spectral_context", None)
    if actual_cfg != reference_cfg:
        raise ValueError("Attribute control changed the topology architecture")
    for key in ("graphlet_slices", "degree_bins", "clustering_bins", "orbit_width", "degree_enabled", "clustering_enabled", "orbit_enabled", "graphlet_basis"):
        if topology_meta.get(key) != state["topology_summary"].get(key):
            raise ValueError(f"Attribute control changed topology metadata: {key}")
    def plain(value):
        if isinstance(value, Mapping):
            return {str(k): plain(v) for k, v in value.items()}
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        return value
    if plain(spectral_transform) != plain(state["spectral_transform"]):
        raise ValueError("Attribute control recomputed a different topology spectral transform")
    if not np.array_equal(basis_adjacencies, state["basis_adjacencies"]) or not np.array_equal(basis_num_nodes, state["basis_num_nodes"]):
        raise ValueError("Attribute control's training donor pool differs from the reference")
    if plain(options["sde"]) != plain(state["sde"]):
        raise ValueError("Attribute control changed the topology diffusion process")
    names = _topology_parameter_names(model_lam)
    fresh = model_lam.state_dict()
    for name in names:
        if name not in state["model_spectrum_state"] or tuple(fresh[name].shape) != tuple(state["model_spectrum_state"][name].shape):
            raise ValueError(f"Incompatible reference topology parameter {name}")
        fresh[name] = state["model_spectrum_state"][name]
    model_x.load_state_dict(state["model_x_state"], strict=True)
    model_lam.load_state_dict(fresh, strict=True)
    for parameter in model_x.parameters():
        parameter.requires_grad_(False)
    for parameter in model_lam.topology_parameters():
        parameter.requires_grad_(False)
    if not all(parameter.requires_grad for parameter in model_lam.attribute_parameters()):
        raise AssertionError("Attribute branch must remain trainable")
    topology_state = {"node." + k: v for k, v in model_x.state_dict().items()}
    topology_state.update({"topology." + k: model_lam.state_dict()[k] for k in names})
    return {
        "reference": reference, "topology_parameter_names": names,
        "frozen_topology_sha256": _tensor_digest(topology_state),
        "spectral_transform": copy.deepcopy(state["spectral_transform"]),
        "basis_adjacencies": state["basis_adjacencies"], "basis_num_nodes": state["basis_num_nodes"],
        "ema_x_state": state["ema_x_state"],
        "ema_topology_state": {name: state["ema_spectrum_state"][name] for name in names},
    }


def assert_frozen_topology(model_x: Any, model_lam: Any, frozen: Mapping[str, Any]) -> dict[str, Any]:
    topology_state = {"node." + k: v for k, v in model_x.state_dict().items()}
    topology_state.update({"topology." + k: model_lam.state_dict()[k] for k in frozen["topology_parameter_names"]})
    digest = _tensor_digest(topology_state)
    if digest != frozen["frozen_topology_sha256"]:
        raise AssertionError("Attribute-only training changed frozen topology parameters")
    return {
        "format": "grapher_attribute_only_learning_v1", "training_scope": "fresh_attribute_parameters_only",
        "topology_parameters_frozen": True, "reference": copy.deepcopy(frozen["reference"]),
        "topology_parameter_sha256_before": digest, "topology_parameter_sha256_after": digest,
    }


def verify_attribute_population(generation_dir: Path, reference: Mapping[str, Any]) -> dict[str, Any]:
    """Prove A1/A2 used the same ordered topology states, donors and sample IDs."""
    import pickle

    parent = Path(reference["generation_dir"])
    if sha256(parent / "manifest.json") != reference["generation_manifest_sha256"]:
        raise ValueError("Parent generation changed after training")
    child_manifest = _read_json(generation_dir / "manifest.json")
    verify_manifest_files(generation_dir, child_manifest)
    parent_pair = _read_json(parent / "topology_pairing.json")
    child_pair = _read_json(generation_dir / "topology_pairing.json")
    for key in ("hash_inputs", "per_graph_sha256", "batch_size", "generation_seed"):
        if child_pair.get(key) != parent_pair.get(key):
            raise ValueError(f"Attribute control did not preserve reference topology evidence: {key}")
    if sha256(generation_dir / "sampled_basis_indices.pkl") != reference["sampled_basis_indices_sha256"]:
        raise ValueError("Attribute control did not preserve the ordered donor IDs")
    n = int(reference["num_requested"])
    def topology_signature(path: Path):
        with path.open("rb") as handle:
            graphs = pickle.load(handle)
        if len(graphs) != n:
            raise ValueError("Attribute population size changed")
        signatures = []
        for index, graph in enumerate(graphs):
            sample_id = f"{reference['generation_seed']}:{index}"
            if graph.graph.get("sample_id") != sample_id:
                raise ValueError("Missing, reordered or duplicated requested sample ID")
            signatures.append((sample_id, sorted(graph.nodes()), sorted(tuple(sorted(edge)) for edge in graph.edges())))
        return signatures
    parent_sig = topology_signature(parent / "topology_graphs.pkl")
    child_sig = topology_signature(generation_dir / "topology_graphs.pkl")
    final_sig = topology_signature(generation_dir / "base_graphs.pkl")
    if child_sig != parent_sig or final_sig != parent_sig:
        raise ValueError("Attribute control changed at least one fixed topology")
    return {
        "fixed_topology_verified": True, "ordered_sample_ids_verified": True,
        "donor_ids_verified": True, "continuous_topology_state_hashes_verified": True,
        "num_matched": n, "reference_generation": str(parent),
        "reference_checkpoint_sha256": reference["checkpoint_sha256"],
    }
