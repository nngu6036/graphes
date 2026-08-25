"""Lazy registry of GraphER base-generator wrappers."""

from __future__ import annotations

import importlib
from typing import Final

from grapher.models.base import BaseGeneratorWrapper

_WRAPPER_PATHS: Final[dict[str, str]] = {
    "dhvae_hh": "grapher.models.dhvae_hh.wrapper:DHVAEHHWrapper",
    "digress": "grapher.models.digress.wrapper:DiGressWrapper",
    "catflow": "grapher.models.catflow:CatFlowWrapper",
    "defog": "grapher.models.defog.wrapper:DeFoGWrapper",
    "hog_diff": "grapher.models.hog_diff:HOGDiffWrapper",
    "flagg": "grapher.models.flagg:FLAGGWrapper",
}

_ALIASES: Final[dict[str, str]] = {
    "dh-vae+hh": "dhvae_hh",
    "dh_vae_hh": "dhvae_hh",
    "dhvae+hh": "dhvae_hh",
    "di-gress": "digress",
    "cat-flow": "catflow",
    "de-fog": "defog",
    "hog-diff": "hog_diff",
}


def available_baselines() -> tuple[str, ...]:
    """Return canonical identifiers without importing wrapper modules."""

    return tuple(_WRAPPER_PATHS)


def normalize_baseline_id(model_id: str) -> str:
    key = str(model_id).strip().lower()
    canonical = _ALIASES.get(key, key)
    if canonical not in _WRAPPER_PATHS:
        raise KeyError(
            f"Unknown baseline {model_id!r}; available baselines are "
            f"{list(available_baselines())}."
        )
    return canonical


def get_wrapper_class(model_id: str) -> type[BaseGeneratorWrapper]:
    """Resolve a wrapper class lazily, without importing upstream projects."""

    canonical = normalize_baseline_id(model_id)
    module_name, class_name = _WRAPPER_PATHS[canonical].split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    wrapper_class = getattr(module, class_name)
    if not isinstance(wrapper_class, type) or not issubclass(
        wrapper_class, BaseGeneratorWrapper
    ):
        raise TypeError(f"Registered object {_WRAPPER_PATHS[canonical]} is invalid.")
    if wrapper_class.model_id != canonical:
        raise ValueError(
            f"Wrapper {wrapper_class.__name__} declares model_id "
            f"{wrapper_class.model_id!r}, expected {canonical!r}."
        )
    return wrapper_class


def create_baseline(model_id: str) -> BaseGeneratorWrapper:
    return get_wrapper_class(model_id)()
