from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class RegistrySpec:
    module: str
    class_name: str
    kind: str
    optional_dependency_hint: str | None = None


class LazyRegistry(Mapping[str, type]):
    """Dictionary-like registry that imports objects on first use.

    Dataset preparation should not import model-only dependencies, and metric
    scripts should remain usable without molecular/PyG extras.  The registry
    therefore stores import paths rather than importing everything at package
    import time.
    """

    def __init__(self, specs: dict[str, RegistrySpec]) -> None:
        self._specs = dict(specs)
        self._cache: dict[str, type] = {}

    def __getitem__(self, key: str) -> type:
        key = key.lower()
        if key not in self._specs:
            raise KeyError(f"Unknown registry key {key!r}. Available: {sorted(self._specs)}")
        if key not in self._cache:
            spec = self._specs[key]
            try:
                module = importlib.import_module(spec.module)
            except ModuleNotFoundError as exc:
                hint = f" Hint: {spec.optional_dependency_hint}" if spec.optional_dependency_hint else ""
                raise ModuleNotFoundError(
                    f"Could not import {spec.kind} '{key}' from {spec.module}.{spec.class_name}." + hint
                ) from exc
            try:
                cls = getattr(module, spec.class_name)
            except AttributeError as exc:
                raise AttributeError(
                    f"Registry entry '{key}' points to missing class {spec.class_name!r} in {spec.module!r}."
                ) from exc
            self._cache[key] = cls
        return self._cache[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def specs(self) -> dict[str, RegistrySpec]:
        return dict(self._specs)

    def describe(self) -> dict[str, dict[str, Any]]:
        return {key: spec.__dict__.copy() for key, spec in self._specs.items()}


DATASET_REGISTRY = LazyRegistry(
    {
        "sbm": RegistrySpec("grapher.datasets.sbm", "SBMDatasetBuilder", "dataset"),
        "planar": RegistrySpec("grapher.datasets.planar", "PlanarDatasetBuilder", "dataset"),
        "ego_citeseer": RegistrySpec(
            "grapher.datasets.ego_citeseer",
            "EgoCiteseerDatasetBuilder",
            "dataset",
            "Install torch-geometric to download Planetoid CiteSeer, or provide raw_graph_path for ind.citeseer.graph.",
        ),
        "qm9": RegistrySpec(
            "grapher.datasets.molecular",
            "QM9DatasetBuilder",
            "dataset",
            "Install torch-geometric to download and process QM9.",
        ),
        "zinc": RegistrySpec(
            "grapher.datasets.molecular",
            "ZINCDatasetBuilder",
            "dataset",
            "Prepare ZINC with scripts/prepare_zinc_from_smiles.py and RDKit; PyG ZINC is not used because its atom-type ids are not atomic numbers.",
        ),
    }
)

MODEL_REGISTRY = LazyRegistry(
    {
        "msvae": RegistrySpec("grapher.models.model_msvae", "MSVAE", "model"),
        "dhvae": RegistrySpec("grapher.models.model_msvae", "DHVAE", "model"),
        "grapher": RegistrySpec(
            "grapher.models.model_grapher",
            "GraphER",
            "model",
            "GraphER uses torch; torch-geometric is optional in this revised code because a small fallback layer is provided.",
        ),
        "grapher_generic": RegistrySpec("grapher.models.model_grapher", "GraphER", "model"),
        "grapher_attributed": RegistrySpec("grapher.models.model_grapher", "GraphER", "model"),
    }
)


def get_dataset_builder(name: str) -> type:
    return DATASET_REGISTRY[name]


def get_model_class(name: str) -> type:
    return MODEL_REGISTRY[name]


def available_datasets() -> list[str]:
    return sorted(DATASET_REGISTRY.keys())


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())
