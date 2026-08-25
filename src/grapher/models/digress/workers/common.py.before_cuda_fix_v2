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
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

SUPPORTED_GENERIC_DATASETS = frozenset({"comm20", "planar", "sbm"})
SUPPORTED_MOLECULAR_DATASETS = frozenset({"qm9"})
SUPPORTED_DATASETS = SUPPORTED_GENERIC_DATASETS | SUPPORTED_MOLECULAR_DATASETS


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


def _safe_override(value: str) -> str:
    text = str(value)
    if not text or any(character in text for character in "\x00\r\n"):
        raise ValueError(f"Invalid Hydra override: {value!r}")
    return text


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

    overrides = [
        f"+experiment={_safe_override(experiment)}",
        f"dataset={dataset_name}",
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
        # QM9's experiment also places num_workers under dataset.
        if dataset_name == "qm9":
            overrides.append(f"dataset.num_workers={int(num_workers)}")
    if check_val_every_n_epochs is not None:
        overrides.append(
            f"general.check_val_every_n_epochs={int(check_val_every_n_epochs)}"
        )
    overrides.extend(_safe_override(value) for value in extra_overrides)

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=overrides)
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


def _tensor_list(value: Any) -> list[float]:
    return [float(item) for item in value.detach().cpu().tolist()]


def compute_qm9_statistics(datamodule: Any) -> dict[str, list[float]]:
    """Compute priors from the exact GraphER-prepared QM9 splits."""

    n_nodes = datamodule.node_counts()
    node_types = datamodule.node_types()
    edge_types = datamodule.edge_counts()
    max_n_nodes = int(len(n_nodes) - 1)
    valencies = datamodule.valency_count(max_n_nodes)
    return {
        "n_nodes": _tensor_list(n_nodes),
        "node_types": _tensor_list(node_types),
        "edge_types": _tensor_list(edge_types),
        "valency_distribution": _tensor_list(valencies),
    }


def apply_qm9_statistics(dataset_infos: Any, statistics: Mapping[str, Any]) -> None:
    import torch

    required = (
        "n_nodes",
        "node_types",
        "edge_types",
        "valency_distribution",
    )
    missing = [name for name in required if name not in statistics]
    if missing:
        raise ValueError(f"QM9 statistics are missing fields: {missing}")
    n_nodes = torch.tensor(statistics["n_nodes"], dtype=torch.float)
    node_types = torch.tensor(statistics["node_types"], dtype=torch.float)
    edge_types = torch.tensor(statistics["edge_types"], dtype=torch.float)
    valencies = torch.tensor(
        statistics["valency_distribution"], dtype=torch.float
    )
    if n_nodes.ndim != 1 or not torch.isfinite(n_nodes).all() or n_nodes.sum() <= 0:
        raise ValueError("Invalid QM9 node-count distribution.")
    if node_types.shape != (4,) or node_types.sum() <= 0:
        raise ValueError("Invalid QM9 atom-type distribution.")
    if edge_types.shape != (5,) or edge_types.sum() <= 0:
        raise ValueError("Invalid QM9 edge-type distribution.")

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

    if dataset == "qm9":
        from datasets import qm9_dataset
        from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete

        datamodule = qm9_dataset.QM9DataModule(cfg)
        dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
        statistics = (
            dict(molecular_statistics)
            if molecular_statistics is not None
            else compute_qm9_statistics(datamodule)
        )
        apply_qm9_statistics(dataset_infos, statistics)
        if cfg.model.type == "discrete" and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(
                cfg.model.extra_features, dataset_info=dataset_infos
            )
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
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
            "train_metrics": TrainMolecularMetricsDiscrete(dataset_infos),
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
