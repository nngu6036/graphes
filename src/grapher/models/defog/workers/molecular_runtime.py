#!/usr/bin/env python
"""Runtime helpers adapting DeFoG molecular priors to GraphER splits.

The attached DeFoG dataset-info classes contain statistics from their original
full QM9/ZINC benchmarks. Wrapper training may use different, explicitly
prepared splits (notably the fixed GraphER ZINC subset), so both training and
sampling must recompute the same node-count, atom, bond, and valency marginals.
This module runs only inside the isolated DeFoG environment.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


STATISTICS_PATH_ENV = "GRAPHER_DEFOG_STATISTICS_PATH"


def _float_values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _write_statistics_record(record: dict[str, Any]) -> None:
    raw_path = os.environ.get(STATISTICS_PATH_ENV)
    if not raw_path:
        return
    path = Path(raw_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def apply_empirical_statistics(
    dataset_infos: Any,
    datamodule: Any,
    *,
    dataset: str,
) -> dict[str, Any]:
    dataset = str(dataset).lower()
    if dataset not in {"qm9", "zinc"}:
        raise ValueError(f"Empirical molecular statistics do not support {dataset!r}.")

    n_nodes = datamodule.node_counts()
    node_types = datamodule.node_types()
    edge_types = datamodule.edge_counts()
    dataset_infos.n_nodes = n_nodes
    dataset_infos.node_types = node_types
    dataset_infos.edge_types = edge_types
    dataset_infos.complete_infos(n_nodes=n_nodes, node_types=node_types)
    dataset_infos.valency_distribution = datamodule.valency_count(
        dataset_infos.max_n_nodes,
        zinc=dataset == "zinc",
    )
    distributions = {
        "n_nodes": _float_values(n_nodes),
        "node_types": _float_values(node_types),
        "edge_types": _float_values(edge_types),
        "valency": _float_values(dataset_infos.valency_distribution),
    }
    distribution_payload = json.dumps(
        distributions,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    record = {
        "format": "grapher_defog_molecular_statistics_v1",
        "dataset": dataset,
        "source": "converted_grapher_train_and_validation_splits",
        "node_count_classes": int(len(n_nodes)),
        "atom_classes": int(len(node_types)),
        "edge_classes": int(len(edge_types)),
        "max_nodes": int(dataset_infos.max_n_nodes),
        "distribution_sha256": hashlib.sha256(distribution_payload).hexdigest(),
        "distributions": distributions,
    }
    dataset_infos.grapher_empirical_statistics = record
    _write_statistics_record(record)
    return record


def apply_cached_statistics(
    dataset_infos: Any,
    path: str | Path,
    *,
    dataset: str,
) -> dict[str, Any]:
    """Restore the audited training priors without rescanning every split."""

    import torch

    source = Path(path).expanduser().resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or (
        record.get("format") != "grapher_defog_molecular_statistics_v1"
        or str(record.get("dataset", "")).lower() != str(dataset).lower()
    ):
        raise RuntimeError(f"Invalid cached DeFoG molecular statistics: {source}.")
    distributions = record.get("distributions")
    if not isinstance(distributions, dict):
        raise RuntimeError("Cached DeFoG statistics contain no distributions.")
    payload = json.dumps(
        distributions,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != str(
        record.get("distribution_sha256", "")
    ):
        raise RuntimeError("Cached DeFoG molecular-statistics digest is invalid.")
    try:
        n_nodes = torch.tensor(distributions["n_nodes"], dtype=torch.float)
        node_types = torch.tensor(distributions["node_types"], dtype=torch.float)
        edge_types = torch.tensor(distributions["edge_types"], dtype=torch.float)
        valency = torch.tensor(distributions["valency"], dtype=torch.float)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Cached DeFoG molecular distributions are malformed.") from exc
    if min(len(n_nodes), len(node_types), len(edge_types), len(valency)) <= 0:
        raise RuntimeError("Cached DeFoG molecular distributions cannot be empty.")
    for name, values in (
        ("n_nodes", n_nodes),
        ("node_types", node_types),
        ("edge_types", edge_types),
        ("valency", valency),
    ):
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise RuntimeError(
                f"Cached DeFoG {name} distribution has invalid values."
            )
        if not bool(torch.isclose(values.sum(), torch.tensor(1.0), atol=1e-5)):
            raise RuntimeError(
                f"Cached DeFoG {name} distribution is not normalized."
            )
    dataset_infos.n_nodes = n_nodes
    dataset_infos.node_types = node_types
    dataset_infos.edge_types = edge_types
    dataset_infos.complete_infos(n_nodes=n_nodes, node_types=node_types)
    dataset_infos.valency_distribution = valency
    dataset_infos.grapher_empirical_statistics = record
    return record


def install_dataset_info_patch(dataset: str) -> None:
    """Patch the class imported by upstream ``main.py`` before Hydra starts.

    The patch modifies the existing top-level class in place.  A locally
    defined subclass is not safe here because DeFoG passes ``dataset_infos`` to
    Lightning's hyperparameter serializer; such an instance would reference a
    non-importable ``<locals>`` class and could make checkpoint saving/loading
    fail.
    """

    dataset = str(dataset).lower()
    if dataset == "qm9":
        from datasets import qm9_dataset as module

        info_class = module.QM9infos
        if getattr(info_class, "_grapher_empirical_patch", False):
            return
        original_init = info_class.__init__

        def empirical_init(self, datamodule, cfg, recompute_statistics=False):
            original_init(
                self,
                datamodule=datamodule,
                cfg=cfg,
                recompute_statistics=False,
            )
            self.grapher_empirical_statistics = apply_empirical_statistics(
                self, datamodule, dataset="qm9"
            )

        info_class.__init__ = empirical_init
        info_class._grapher_empirical_patch = True
        return

    if dataset == "zinc":
        from datasets import zinc_dataset as module

        info_class = module.ZINCinfos
        if getattr(info_class, "_grapher_empirical_patch", False):
            return
        original_init = info_class.__init__

        def empirical_init(self, datamodule, cfg, recompute_statistics=False):
            original_init(
                self,
                datamodule=datamodule,
                cfg=cfg,
                recompute_statistics=False,
            )
            self.grapher_empirical_statistics = apply_empirical_statistics(
                self, datamodule, dataset="zinc"
            )

        info_class.__init__ = empirical_init
        info_class._grapher_empirical_patch = True
        return

    raise ValueError(f"Cannot patch unsupported molecular dataset {dataset!r}.")


__all__ = [
    "apply_cached_statistics",
    "apply_empirical_statistics",
    "install_dataset_info_patch",
]
