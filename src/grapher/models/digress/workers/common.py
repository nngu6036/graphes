"""Shared runtime helpers for isolated DiGress workers.

This module is imported only by worker scripts executed inside the external
DiGress environment. It deliberately has no imports from GraphER's Python
package, so the upstream dependency stack remains isolated.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

SUPPORTED_GENERIC_DATASETS = frozenset({"comm20", "planar", "sbm"})
SUPPORTED_MOLECULAR_DATASETS = frozenset({"qm9", "zinc"})
SUPPORTED_DATASETS = SUPPORTED_GENERIC_DATASETS | SUPPORTED_MOLECULAR_DATASETS
MOLECULAR_PROFILES: dict[str, dict[str, Any]] = {
    "qm9": {
        "atom_decoder": ("C", "N", "O", "F"),
        "valencies": (4, 3, 2, 1),
        "atom_weights": (12.0, 14.0, 16.0, 19.0),
        "edge_orders": (0.0, 1.0, 2.0, 3.0, 1.5),
        "max_weight": 150.0,
    },
    "zinc": {
        "atom_decoder": ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I"),
        "valencies": (4, 3, 2, 1, 5, 6, 1, 1, 1),
        "atom_weights": (12.0, 14.0, 16.0, 19.0, 30.0, 32.0, 35.5, 78.0, 127.0),
        # GraphER's ZINC benchmark follows the kekulized HOG-Diff/GDSS state.
        "edge_orders": (0.0, 1.0, 2.0, 3.0),
        "max_weight": 500.0,
    },
}


def status(message: str) -> None:
    print(f"[GraphER/DiGress] {message}", file=sys.stderr, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except Exception:
        pass
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        import pytorch_lightning as pl

        pl.seed_everything(int(seed), workers=True)
    except TypeError:
        import pytorch_lightning as pl

        pl.seed_everything(int(seed))


def install_upstream_runtime_patches() -> None:
    """Install compatibility fixes without modifying the DiGress checkout.

    DiGress imports its utility module as ``src.utils``. Importing the same
    file as top-level ``utils`` creates a second module object, so patching that
    alias does not affect ``diffusion_model_discrete``. Patch the canonical
    ``src.utils`` module and any already-loaded legacy alias.

    The attached ``encode_no_edge`` and ``sample_discrete_features``
    implementations apply boolean masks to rank-4 CUDA tensors through
    advanced indexing. PyTorch 2.0.x can fail there with an internal size
    assertion. The replacements create masks on the tensor device and use
    broadcasted ``masked_fill_`` operations.
    """
    import torch
    from src import utils as digress_utils
    from src.diffusion import diffusion_utils as digress_diffusion_utils

    def encode_no_edge(E: Any) -> Any:
        if E.dim() != 4:
            raise ValueError(
                "DiGress edge tensor must have shape [B, N, N, C], "
                f"received {tuple(E.shape)}."
            )
        if E.shape[-1] == 0:
            return E

        no_edge = torch.sum(E, dim=-1) == 0
        first_channel = E[..., 0]
        first_channel[no_edge] = 1
        E[..., 0] = first_channel

        diagonal = torch.eye(
            E.shape[1],
            dtype=torch.bool,
            device=E.device,
        ).unsqueeze(0).expand(E.shape[0], -1, -1)
        E.masked_fill_(diagonal.unsqueeze(-1), 0)
        return E

    digress_utils.encode_no_edge = encode_no_edge

    legacy_utils = sys.modules.get("utils")
    if legacy_utils is not None and legacy_utils is not digress_utils:
        legacy_utils.encode_no_edge = encode_no_edge

    if digress_utils.encode_no_edge is not encode_no_edge:
        raise RuntimeError(
            "Failed to install the DiGress encode_no_edge compatibility patch."
        )

    def sample_discrete_features(
        probX: Any,
        probE: Any,
        node_mask: Any,
    ) -> Any:
        """Sample categorical graph state without CUDA advanced indexing."""

        if probX.dim() != 3:
            raise ValueError(
                "DiGress node probabilities must have shape [B, N, C], "
                f"received {tuple(probX.shape)}."
            )
        if probE.dim() != 4:
            raise ValueError(
                "DiGress edge probabilities must have shape [B, N, N, C], "
                f"received {tuple(probE.shape)}."
            )
        batch_size, num_nodes, node_classes = probX.shape
        expected_edge_prefix = (batch_size, num_nodes, num_nodes)
        if tuple(probE.shape[:3]) != expected_edge_prefix:
            raise ValueError(
                "DiGress node/edge probability shapes are incompatible: "
                f"nodes={tuple(probX.shape)}, edges={tuple(probE.shape)}."
            )
        if tuple(node_mask.shape) != (batch_size, num_nodes):
            raise ValueError(
                "DiGress node mask must have shape [B, N], "
                f"received {tuple(node_mask.shape)}."
            )
        edge_classes = int(probE.shape[-1])
        if node_classes == 0 or edge_classes == 0:
            raise ValueError("DiGress categorical probability tensors cannot be empty.")
        if probX.device != probE.device:
            raise ValueError(
                "DiGress node and edge probabilities must use the same device."
            )

        valid_nodes = node_mask.to(device=probX.device, dtype=torch.bool)
        probX.masked_fill_(
            (~valid_nodes).unsqueeze(-1),
            1.0 / int(node_classes),
        )
        sampled_nodes = probX.reshape(batch_size * num_nodes, -1).multinomial(1)
        sampled_nodes = sampled_nodes.reshape(batch_size, num_nodes)

        valid_edges = valid_nodes.unsqueeze(1) & valid_nodes.unsqueeze(2)
        diagonal = torch.eye(
            num_nodes,
            dtype=torch.bool,
            device=probE.device,
        ).unsqueeze(0)
        invalid_edges = (~valid_edges) | diagonal
        probE.masked_fill_(
            invalid_edges.unsqueeze(-1),
            1.0 / edge_classes,
        )
        sampled_edges = probE.reshape(
            batch_size * num_nodes * num_nodes,
            -1,
        ).multinomial(1)
        sampled_edges = sampled_edges.reshape(batch_size, num_nodes, num_nodes)
        sampled_edges = torch.triu(sampled_edges, diagonal=1)
        sampled_edges = sampled_edges + sampled_edges.transpose(1, 2)

        empty_global = torch.zeros(
            (batch_size, 0),
            dtype=sampled_nodes.dtype,
            device=sampled_nodes.device,
        )
        return digress_diffusion_utils.PlaceHolder(
            X=sampled_nodes,
            E=sampled_edges,
            y=empty_global,
        )

    digress_diffusion_utils.sample_discrete_features = sample_discrete_features
    legacy_diffusion_utils = sys.modules.get("diffusion.diffusion_utils")
    if (
        legacy_diffusion_utils is not None
        and legacy_diffusion_utils is not digress_diffusion_utils
    ):
        legacy_diffusion_utils.sample_discrete_features = sample_discrete_features

    if digress_diffusion_utils.sample_discrete_features is not sample_discrete_features:
        raise RuntimeError(
            "Failed to install the DiGress categorical-sampling compatibility patch."
        )

    def mask_distributions(
        true_X: Any,
        true_E: Any,
        pred_X: Any,
        pred_E: Any,
        node_mask: Any,
    ) -> Tuple[Any, Any, Any, Any]:
        """Mask categorical distributions without vector-valued CUDA indexing."""

        node_tensors = (true_X, pred_X)
        edge_tensors = (true_E, pred_E)
        if any(tensor.dim() != 3 for tensor in node_tensors):
            raise ValueError("DiGress node distributions must have shape [B, N, C].")
        if any(tensor.dim() != 4 for tensor in edge_tensors):
            raise ValueError(
                "DiGress edge distributions must have shape [B, N, N, C]."
            )

        batch_size, num_nodes = true_X.shape[:2]
        expected_nodes = (batch_size, num_nodes)
        expected_edges = (batch_size, num_nodes, num_nodes)
        if any(tuple(tensor.shape[:2]) != expected_nodes for tensor in node_tensors):
            raise ValueError("DiGress node distribution shapes are incompatible.")
        if any(tuple(tensor.shape[:3]) != expected_edges for tensor in edge_tensors):
            raise ValueError("DiGress edge distribution shapes are incompatible.")
        if tuple(node_mask.shape) != expected_nodes:
            raise ValueError(
                "DiGress node mask must have shape [B, N], "
                f"received {tuple(node_mask.shape)}."
            )
        if any(int(tensor.shape[-1]) == 0 for tensor in node_tensors + edge_tensors):
            raise ValueError("DiGress categorical distributions cannot be empty.")
        devices = {tensor.device for tensor in node_tensors + edge_tensors}
        if len(devices) != 1:
            raise ValueError("DiGress distributions must use one common device.")

        device = true_X.device
        valid_nodes = node_mask.to(device=device, dtype=torch.bool)
        invalid_nodes = ~valid_nodes
        valid_edges = valid_nodes.unsqueeze(1) & valid_nodes.unsqueeze(2)
        off_diagonal = ~torch.eye(
            num_nodes,
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0)
        invalid_edges = ~(valid_edges & off_diagonal)

        def mask_and_normalize(tensor: Any, invalid: Any) -> Any:
            tensor.masked_fill_(invalid.unsqueeze(-1), 0)
            tensor[..., 0].masked_fill_(invalid, 1)
            smoothed = tensor + 1e-7
            return smoothed / smoothed.sum(dim=-1, keepdim=True)

        true_X_out = mask_and_normalize(true_X, invalid_nodes)
        pred_X_out = mask_and_normalize(pred_X, invalid_nodes)
        true_E_out = mask_and_normalize(true_E, invalid_edges)
        pred_E_out = mask_and_normalize(pred_E, invalid_edges)
        return true_X_out, true_E_out, pred_X_out, pred_E_out

    digress_diffusion_utils.mask_distributions = mask_distributions
    if (
        legacy_diffusion_utils is not None
        and legacy_diffusion_utils is not digress_diffusion_utils
    ):
        legacy_diffusion_utils.mask_distributions = mask_distributions

    if digress_diffusion_utils.mask_distributions is not mask_distributions:
        raise RuntimeError(
            "Failed to install the DiGress distribution-masking compatibility patch."
        )

    status(
        "Installed DiGress CUDA indexing compatibility patches on "
        f"{digress_utils.__name__} and {digress_diffusion_utils.__name__}."
    )


def _mask_reconstruction_distributions(
    node_probabilities: Any,
    edge_probabilities: Any,
    node_mask: Any,
) -> Tuple[Any, Any]:
    """Apply DiGress reconstruction masks without CUDA advanced indexing."""

    import torch

    if node_probabilities.dim() != 3:
        raise ValueError(
            "DiGress reconstruction node probabilities must have shape [B, N, C]."
        )
    if edge_probabilities.dim() != 4:
        raise ValueError(
            "DiGress reconstruction edge probabilities must have shape "
            "[B, N, N, C]."
        )
    batch_size, num_nodes = node_probabilities.shape[:2]
    if tuple(edge_probabilities.shape[:3]) != (
        batch_size,
        num_nodes,
        num_nodes,
    ):
        raise ValueError("DiGress reconstruction probability shapes are incompatible.")
    if tuple(node_mask.shape) != (batch_size, num_nodes):
        raise ValueError(
            "DiGress reconstruction node mask must have shape [B, N], "
            f"received {tuple(node_mask.shape)}."
        )
    if node_probabilities.device != edge_probabilities.device:
        raise ValueError(
            "DiGress reconstruction probabilities must use one common device."
        )

    device = node_probabilities.device
    valid_nodes = node_mask.to(device=device, dtype=torch.bool)
    valid_edges = valid_nodes.unsqueeze(1) & valid_nodes.unsqueeze(2)
    diagonal = torch.eye(
        num_nodes,
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)
    invalid_edges = (~valid_edges) | diagonal
    node_probabilities.masked_fill_((~valid_nodes).unsqueeze(-1), 1)
    edge_probabilities.masked_fill_(invalid_edges.unsqueeze(-1), 1)
    return node_probabilities, edge_probabilities


def install_discrete_model_runtime_patches(model_class: Any) -> None:
    """Patch CUDA-sensitive masking in the upstream discrete Lightning model."""

    if getattr(model_class, "_grapher_cuda_indexing_patch", False):
        return

    def reconstruction_logp(
        self: Any,
        t: Any,
        X: Any,
        E: Any,
        node_mask: Any,
    ) -> Any:
        import torch
        from torch.nn import functional as F

        from src import utils
        from src.diffusion import diffusion_utils

        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)
        probX0 = X @ Q0.X
        probE0 = E @ Q0.E.unsqueeze(1)

        sampled0 = diffusion_utils.sample_discrete_features(
            probX=probX0,
            probE=probE0,
            node_mask=node_mask,
        )
        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        if X.shape != X0.shape or E.shape != E0.shape:
            raise RuntimeError(
                "DiGress reconstruction sampling returned incompatible shapes."
            )

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)
        noisy_data = {
            "X_t": sampled_0.X,
            "E_t": sampled_0.E,
            "y_t": sampled_0.y,
            "node_mask": node_mask,
            "t": torch.zeros(X0.shape[0], 1).type_as(y0),
        }
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)
        probX0, probE0 = _mask_reconstruction_distributions(
            probX0,
            probE0,
            node_mask,
        )
        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    model_class.reconstruction_logp = reconstruction_logp
    model_class._grapher_cuda_indexing_patch = True
    if model_class.reconstruction_logp is not reconstruction_logp:
        raise RuntimeError(
            "Failed to install the DiGress reconstruction compatibility patch."
        )
    status(
        "Installed DiGress reconstruction CUDA indexing compatibility patch on "
        f"{model_class.__module__}.{model_class.__name__}."
    )


def _safe_override(value: str) -> str:
    text = str(value)
    if not text or any(character in text for character in "\x00\r\n"):
        raise ValueError(f"Invalid Hydra override: {value!r}")
    return text


def upstream_config_templates(dataset: str, experiment: str) -> tuple[str, str]:
    """Map a logical GraphER dataset to configs present in stock DiGress."""

    dataset_name = str(dataset).lower()
    experiment_name = str(experiment).lower()
    if dataset_name == "zinc":
        return (
            "qm9",
            "qm9_no_h" if experiment_name == "zinc_no_h" else experiment_name,
        )
    return dataset_name, experiment_name


def compose_config(
    *,
    digress_root: Path,
    dataset: str,
    experiment: str,
    dataset_datadir: Path,
    run_name: str,
    seed: int,
    gpus: int,
    n_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    check_val_every_n_epochs: Optional[int] = None,
    extra_overrides: Sequence[str] = (),
) -> Any:
    """Compose the attached DiGress Hydra configuration without chdir."""

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    dataset_name = str(dataset).lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported DiGress dataset {dataset!r}; supported datasets are "
            f"{sorted(SUPPORTED_DATASETS)}."
        )
    config_dir = (Path(digress_root) / "configs").resolve()
    if not (config_dir / "config.yaml").is_file():
        raise FileNotFoundError(f"Missing DiGress config root: {config_dir}")
    template_dataset, template_experiment = upstream_config_templates(
        dataset_name, experiment
    )
    required_configs = (
        config_dir / "dataset" / f"{template_dataset}.yaml",
        config_dir / "experiment" / f"{template_experiment}.yaml",
    )
    missing_configs = [str(path) for path in required_configs if not path.is_file()]
    if missing_configs:
        raise FileNotFoundError(
            f"Missing DiGress config templates for {dataset_name}: "
            f"{missing_configs}."
        )

    overrides = [
        f"+experiment={_safe_override(template_experiment)}",
        f"dataset={template_dataset}",
        f"dataset.datadir={str(Path(dataset_datadir).resolve())}",
        f"general.name={_safe_override(run_name)}",
        "general.wandb=disabled",
        f"general.gpus={int(gpus)}",
        f"train.seed={int(seed)}",
        "train.save_model=true",
        # Validation generation is disabled by the GraphER training shim. The
        # likelihood validation loop remains available at the configured rate.
        "general.sample_every_val=1000000000",
        "general.samples_to_generate=0",
        "general.samples_to_save=0",
        "general.chains_to_save=0",
        "general.final_model_samples_to_generate=0",
        "general.final_model_samples_to_save=0",
        "general.final_model_chains_to_save=0",
    ]
    if n_epochs is not None:
        overrides.append(f"train.n_epochs={int(n_epochs)}")
    if batch_size is not None:
        overrides.append(f"train.batch_size={int(batch_size)}")
    if num_workers is not None:
        overrides.append(f"train.num_workers={int(num_workers)}")
        # Molecular experiments also place num_workers under the dataset.
        if dataset_name in SUPPORTED_MOLECULAR_DATASETS:
            overrides.append(f"dataset.num_workers={int(num_workers)}")
    if check_val_every_n_epochs is not None:
        overrides.append(
            f"general.check_val_every_n_epochs={int(check_val_every_n_epochs)}"
        )
    overrides.extend(_safe_override(value) for value in extra_overrides)

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=overrides)
    # Restore the logical dataset after composing the stock compatibility
    # template so component construction cannot accidentally use QM9 semantics.
    cfg.dataset.name = dataset_name
    OmegaConf.resolve(cfg)
    return cfg


class NullSamplingMetrics:
    """No-op replacement for expensive validation/test sampling metrics."""

    def reset(self) -> None:
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        del args, kwargs
        return {}

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        del args, kwargs
        return {}


class GraphERMolecularDatasetInfos:
    """Dataset information for a GraphER-owned molecular categorical state.

    Stock DiGress has no ZINC ``DatasetInfos`` class.  This dependency-light
    implementation mirrors the small interface used by the discrete model and
    stays importable as ``common.GraphERMolecularDatasetInfos`` when Lightning
    serializes checkpoint hyperparameters.
    """

    def __init__(self, dataset: str) -> None:
        dataset_name = str(dataset).lower()
        profile = MOLECULAR_PROFILES[dataset_name]
        atom_decoder = list(profile["atom_decoder"])
        self.name = dataset_name
        self.remove_h = True
        self.aromatic = len(profile["edge_orders"]) == 5
        self.need_to_strip = False
        self.compute_fcd = False
        self.atom_decoder = atom_decoder
        self.atom_encoder = {
            atom: index for index, atom in enumerate(atom_decoder)
        }
        self.num_atom_types = len(atom_decoder)
        self.valencies = list(profile["valencies"])
        self.atom_weights = {
            index: float(weight)
            for index, weight in enumerate(profile["atom_weights"])
        }
        self.max_n_nodes = 38 if dataset_name == "zinc" else 9
        self.max_weight = float(profile["max_weight"])
        self.input_dims = None
        self.output_dims = None

    def complete_infos(self, n_nodes: Any, node_types: Any) -> None:
        from src.diffusion.distributions import DistributionNodes

        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(
        self,
        datamodule: Any,
        extra_features: Any,
        domain_features: Any,
    ) -> None:
        from src import utils

        example_batch = next(iter(datamodule.train_dataloader()))
        dense, node_mask = utils.to_dense(
            example_batch.x,
            example_batch.edge_index,
            example_batch.edge_attr,
            example_batch.batch,
        )
        noisy_data = {
            "X_t": dense.X,
            "E_t": dense.E,
            "y_t": example_batch["y"],
            "node_mask": node_mask,
        }
        extra = extra_features(noisy_data)
        domain = domain_features(noisy_data)
        self.input_dims = {
            "X": example_batch["x"].size(1)
            + extra.X.size(-1)
            + domain.X.size(-1),
            "E": example_batch["edge_attr"].size(1)
            + extra.E.size(-1)
            + domain.E.size(-1),
            "y": example_batch["y"].size(1)
            + 1
            + extra.y.size(-1)
            + domain.y.size(-1),
        }
        self.output_dims = {
            "X": example_batch["x"].size(1),
            "E": example_batch["edge_attr"].size(1),
            "y": 0,
        }


class GraphERMolecularFeatures:
    """Molecular features with a dataset-specific bond vocabulary."""

    def __init__(self, dataset_infos: Any, *, dataset: str) -> None:
        profile = MOLECULAR_PROFILES[str(dataset).lower()]
        self.edge_orders = tuple(float(value) for value in profile["edge_orders"])
        self.valencies = tuple(float(value) for value in dataset_infos.valencies)
        self.atom_weights = tuple(
            float(dataset_infos.atom_weights[index])
            for index in range(len(dataset_infos.atom_weights))
        )
        self.max_weight = float(dataset_infos.max_weight)

    def __call__(self, noisy_data: Mapping[str, Any]) -> Any:
        import torch
        from src import utils

        X = noisy_data["X_t"]
        E = noisy_data["E_t"]
        if X.shape[-1] != len(self.valencies):
            raise ValueError(
                "Molecular node-state width does not match its valency profile."
            )
        if E.shape[-1] != len(self.edge_orders):
            raise ValueError(
                "Molecular edge-state width does not match its bond profile."
            )
        edge_orders = E.new_tensor(self.edge_orders).reshape(1, 1, 1, -1)
        current_valencies = (E * edge_orders).sum(dim=-1).sum(dim=-1)
        normal_valencies = (
            X * X.new_tensor(self.valencies).reshape(1, 1, -1)
        ).sum(dim=-1)
        charge = (normal_valencies - current_valencies).unsqueeze(-1)
        valency = current_valencies.unsqueeze(-1)
        atom_classes = torch.argmax(X, dim=-1)
        atom_weights = X.new_tensor(self.atom_weights)
        node_mask = noisy_data["node_mask"].type_as(X)
        weight = (
            atom_weights[atom_classes] * node_mask
        ).sum(dim=-1, keepdim=True)
        weight = weight / self.max_weight
        empty_edges = E.new_zeros((*E.shape[:-1], 0))
        return utils.PlaceHolder(
            X=torch.cat((charge, valency), dim=-1),
            E=empty_edges,
            y=weight,
        )


def _tensor_list(value: Any) -> list[float]:
    return [float(item) for item in value.detach().cpu().tolist()]


def _empirical_molecular_distributions(
    datamodule: Any, *, dataset: str
) -> tuple[Any, Any, Any, Any]:
    """Collect categorical priors from train and safe node support from val."""

    import torch

    profile = MOLECULAR_PROFILES[dataset]
    atom_classes = len(profile["atom_decoder"])
    edge_orders = tuple(float(value) for value in profile["edge_orders"])
    edge_classes = len(edge_orders)
    node_counts = torch.zeros(300, dtype=torch.float64)
    node_types = torch.zeros(atom_classes, dtype=torch.float64)
    edge_types = torch.zeros(edge_classes, dtype=torch.float64)
    valencies = torch.zeros(3 * 300 - 2, dtype=torch.float64)
    multipliers = torch.tensor(edge_orders, dtype=torch.float64)
    loader = datamodule.train_dataloader()
    total_batches = len(loader)
    report_every = max(total_batches // 20, 1)
    started = time.monotonic()
    for batch_index, data in enumerate(loader, start=1):
        if data.x.shape[-1] != atom_classes:
            raise ValueError(
                f"{dataset.upper()} node-state width is {data.x.shape[-1]}; "
                f"expected {atom_classes}."
            )
        if data.edge_attr.shape[-1] != edge_classes:
            raise ValueError(
                f"{dataset.upper()} edge-state width is {data.edge_attr.shape[-1]}; "
                f"expected {edge_classes}."
            )
        _, counts = torch.unique(data.batch.detach().cpu(), return_counts=True)
        for count in counts.tolist():
            if int(count) >= len(node_counts):
                raise ValueError(f"Molecular graph has {count} nodes; maximum is 299.")
            node_counts[int(count)] += 1
        local_x = data.x.detach().cpu().to(dtype=torch.float64)
        local_edges = data.edge_attr.detach().cpu().to(dtype=torch.float64)
        node_types += local_x.sum(dim=0)
        all_pairs = sum(int(count) * (int(count) - 1) for count in counts.tolist())
        present_edges = int(data.edge_index.shape[1])
        non_edges = all_pairs - present_edges
        if non_edges < 0:
            raise ValueError("Molecular batch contains more edges than node pairs.")
        edge_types[0] += non_edges
        edge_types[1:] += local_edges.sum(dim=0)[1:]

        directed_orders = (local_edges * multipliers).sum(dim=-1)
        atom_valencies = torch.zeros(local_x.shape[0], dtype=torch.float64)
        atom_valencies.scatter_add_(
            0,
            data.edge_index[0].detach().cpu().long(),
            directed_orders,
        )
        rounded_valencies = torch.round(atom_valencies)
        if not bool(torch.allclose(atom_valencies, rounded_valencies)):
            raise ValueError("Molecular training data contains non-integral valency.")
        integer_valencies = rounded_valencies.long()
        if bool((integer_valencies < 0).any()) or bool(
            (integer_valencies >= len(valencies)).any()
        ):
            raise ValueError("Molecular training data contains an invalid valency.")
        valencies += torch.bincount(
            integer_valencies, minlength=len(valencies)
        ).to(dtype=torch.float64)
        if (
            batch_index == 1
            or batch_index == total_batches
            or batch_index % report_every == 0
        ):
            status(
                "Molecular-prior scan: "
                f"dataset={dataset}, batch={batch_index}/{total_batches}, "
                f"graphs={int(node_counts.sum().item())}, "
                f"elapsed_seconds={time.monotonic() - started:.1f}."
            )

    # DiGress evaluates log p(N) during validation. Match its native contract
    # by including validation graph sizes in the node-count support, otherwise
    # a validation-only larger molecule would index beyond DistributionNodes.
    validation_loader_factory = getattr(datamodule, "val_dataloader", None)
    if callable(validation_loader_factory):
        validation_loader = validation_loader_factory()
        for data in validation_loader:
            if data.x.shape[-1] != atom_classes:
                raise ValueError(
                    f"{dataset.upper()} validation node-state width is "
                    f"{data.x.shape[-1]}; expected {atom_classes}."
                )
            _, counts = torch.unique(
                data.batch.detach().cpu(), return_counts=True
            )
            for count in counts.tolist():
                if int(count) >= len(node_counts):
                    raise ValueError(
                        f"Molecular validation graph has {count} nodes; "
                        "maximum is 299."
                    )
                node_counts[int(count)] += 1

    nonzero_node_counts = torch.nonzero(node_counts, as_tuple=False)
    if nonzero_node_counts.numel() == 0:
        raise ValueError("Cannot compute molecular priors from an empty split.")
    max_n_nodes = int(nonzero_node_counts.max().item())
    node_counts = node_counts[: max_n_nodes + 1]
    valencies = valencies[: 3 * max_n_nodes - 2]
    distributions = (node_counts, node_types, edge_types, valencies)
    if any(values.sum() <= 0 for values in distributions):
        raise ValueError("A molecular empirical distribution has zero mass.")
    return tuple(
        (values / values.sum()).to(dtype=torch.float) for values in distributions
    )  # type: ignore[return-value]


def compute_molecular_statistics(
    datamodule: Any, *, dataset: str
) -> dict[str, Any]:
    """Compute categorical priors and safe node support from GraphER splits."""

    dataset_name = str(dataset).lower()
    profile = MOLECULAR_PROFILES[dataset_name]
    n_nodes, node_types, edge_types, valencies = (
        _empirical_molecular_distributions(datamodule, dataset=dataset_name)
    )
    max_n_nodes = int(len(n_nodes) - 1)
    if dataset_name == "zinc" and max_n_nodes > 38:
        raise ValueError(
            f"GraphER ZINC contains {max_n_nodes} nodes; the protocol maximum is 38."
        )
    return {
        "format": "grapher_digress_molecular_statistics_v1",
        "dataset": dataset_name,
        "atom_decoder": list(profile["atom_decoder"]),
        "edge_orders": list(profile["edge_orders"]),
        "node_count_source": "train_and_validation",
        "categorical_source": "train",
        "n_nodes": _tensor_list(n_nodes),
        "node_types": _tensor_list(node_types),
        "edge_types": _tensor_list(edge_types),
        "valency_distribution": _tensor_list(valencies),
    }


def apply_molecular_statistics(
    dataset_infos: Any,
    statistics: Mapping[str, Any],
    *,
    dataset: str,
) -> None:
    import torch

    dataset_name = str(dataset).lower()
    profile = MOLECULAR_PROFILES[dataset_name]
    recorded_format = statistics.get("format")
    if recorded_format is not None and (
        recorded_format != "grapher_digress_molecular_statistics_v1"
    ):
        raise ValueError(
            f"Unsupported molecular statistics format {recorded_format!r}."
        )
    recorded_dataset = statistics.get("dataset")
    if recorded_dataset is not None and str(recorded_dataset).lower() != dataset_name:
        raise ValueError(
            "Molecular statistics dataset mismatch: "
            f"expected {dataset_name}, got {recorded_dataset}."
        )
    recorded_atoms = statistics.get("atom_decoder")
    if recorded_atoms is not None and tuple(recorded_atoms) != tuple(
        profile["atom_decoder"]
    ):
        raise ValueError(f"{dataset_name.upper()} atom vocabulary mismatch.")
    recorded_edges = statistics.get("edge_orders")
    if recorded_edges is not None and tuple(
        float(value) for value in recorded_edges
    ) != tuple(float(value) for value in profile["edge_orders"]):
        raise ValueError(f"{dataset_name.upper()} bond vocabulary mismatch.")
    required = (
        "n_nodes",
        "node_types",
        "edge_types",
        "valency_distribution",
    )
    missing = [name for name in required if name not in statistics]
    if missing:
        raise ValueError(
            f"{dataset_name.upper()} statistics are missing fields: {missing}"
        )
    n_nodes = torch.tensor(statistics["n_nodes"], dtype=torch.float)
    node_types = torch.tensor(statistics["node_types"], dtype=torch.float)
    edge_types = torch.tensor(statistics["edge_types"], dtype=torch.float)
    valencies = torch.tensor(
        statistics["valency_distribution"], dtype=torch.float
    )
    distributions = {
        "node-count": n_nodes,
        "atom-type": node_types,
        "edge-type": edge_types,
        "valency": valencies,
    }
    for label, values in distributions.items():
        if (
            values.ndim != 1
            or not bool(torch.isfinite(values).all())
            or bool((values < 0).any())
            or values.sum() <= 0
        ):
            raise ValueError(
                f"Invalid {dataset_name.upper()} {label} distribution."
            )
    expected_atoms = len(profile["atom_decoder"])
    expected_edges = len(profile["edge_orders"])
    if node_types.shape != (expected_atoms,):
        raise ValueError(
            f"Invalid {dataset_name.upper()} atom-type distribution; expected "
            f"{expected_atoms} classes."
        )
    if edge_types.shape != (expected_edges,):
        raise ValueError(
            f"Invalid {dataset_name.upper()} edge-type distribution; expected "
            f"{expected_edges} classes."
        )
    if dataset_name == "zinc" and len(n_nodes) - 1 > 38:
        raise ValueError("GraphER ZINC node-count support exceeds 38 nodes.")

    n_nodes = n_nodes / n_nodes.sum()
    node_types = node_types / node_types.sum()
    edge_types = edge_types / edge_types.sum()
    if valencies.sum() > 0:
        valencies = valencies / valencies.sum()

    dataset_infos.n_nodes = n_nodes
    dataset_infos.node_types = node_types
    dataset_infos.edge_types = edge_types
    dataset_infos.valency_distribution = valencies
    dataset_infos.complete_infos(n_nodes=n_nodes, node_types=node_types)


def compute_qm9_statistics(datamodule: Any) -> dict[str, Any]:
    """Backward-compatible QM9 statistics entry point."""

    return compute_molecular_statistics(datamodule, dataset="qm9")


def apply_qm9_statistics(dataset_infos: Any, statistics: Mapping[str, Any]) -> None:
    """Backward-compatible QM9 statistics entry point."""

    apply_molecular_statistics(dataset_infos, statistics, dataset="qm9")


def build_components(
    cfg: Any,
    *,
    molecular_statistics: Optional[Mapping[str, Any]] = None,
) -> Tuple[Any, dict, Optional[dict]]:
    """Build the upstream data module and model kwargs without main.py.

    Avoiding ``src/main.py`` is intentional: the attached entrypoint imports
    graph-tool unconditionally and forces DDP even for a one-device run.
    """

    dataset = str(cfg.dataset.name).lower()
    if dataset in SUPPORTED_GENERIC_DATASETS:
        from datasets.spectre_dataset import (
            SpectreDatasetInfos,
            SpectreGraphDataModule,
        )
        from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete

        datamodule = SpectreGraphDataModule(cfg)
        dataset_infos = SpectreDatasetInfos(datamodule, cfg.dataset)
        if cfg.model.type == "discrete" and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(
                cfg.model.extra_features, dataset_info=dataset_infos
            )
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()
        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )
        model_kwargs = {
            "dataset_infos": dataset_infos,
            "train_metrics": TrainAbstractMetricsDiscrete(),
            "sampling_metrics": NullSamplingMetrics(),
            "visualization_tools": None,
            "extra_features": extra_features,
            "domain_features": domain_features,
        }
        return datamodule, model_kwargs, None

    if dataset in SUPPORTED_MOLECULAR_DATASETS:
        from datasets import qm9_dataset
        from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete

        datamodule = qm9_dataset.QM9DataModule(cfg)
        if dataset == "qm9":
            from diffusion.extra_features_molecular import ExtraMolecularFeatures
            from metrics.molecular_metrics_discrete import (
                TrainMolecularMetricsDiscrete,
            )

            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
        else:
            dataset_infos = GraphERMolecularDatasetInfos(dataset)
        statistics = (
            dict(molecular_statistics)
            if molecular_statistics is not None
            else compute_molecular_statistics(datamodule, dataset=dataset)
        )
        apply_molecular_statistics(dataset_infos, statistics, dataset=dataset)
        if cfg.model.type == "discrete" and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(
                cfg.model.extra_features, dataset_info=dataset_infos
            )
            domain_features = (
                ExtraMolecularFeatures(dataset_infos=dataset_infos)
                if dataset == "qm9"
                else GraphERMolecularFeatures(dataset_infos, dataset=dataset)
            )
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()
        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )
        model_kwargs = {
            "dataset_infos": dataset_infos,
            "train_metrics": (
                TrainMolecularMetricsDiscrete(dataset_infos)
                if dataset == "qm9"
                else TrainAbstractMetricsDiscrete()
            ),
            "sampling_metrics": NullSamplingMetrics(),
            "visualization_tools": None,
            "extra_features": extra_features,
            "domain_features": domain_features,
        }
        return datamodule, model_kwargs, statistics

    raise ValueError(f"Unsupported DiGress dataset: {dataset}")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
