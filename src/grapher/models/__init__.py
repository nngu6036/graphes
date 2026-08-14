"""Uniform training/generation wrappers for GraphER baseline models.

Wrapper modules are loaded lazily. Importing this package never imports an
external baseline repository or its optional dependencies.
"""

from grapher.models.artifacts import ArtifactLayout
from grapher.models.base import (
    BaselineCapabilities,
    BaseGeneratorWrapper,
    DatasetReference,
    GenerateRequest,
    GenerationArtifacts,
    RunSpec,
    TrainRequest,
    TrainingArtifacts,
)
from grapher.models.errors import (
    ArtifactCollisionError,
    BaselineNotImplementedError,
    BaselineWrapperError,
)
from grapher.models.registry import (
    available_baselines,
    create_baseline,
    get_wrapper_class,
    normalize_baseline_id,
)

__all__ = [
    "ArtifactCollisionError",
    "ArtifactLayout",
    "BaselineCapabilities",
    "BaselineNotImplementedError",
    "BaselineWrapperError",
    "BaseGeneratorWrapper",
    "DatasetReference",
    "GenerateRequest",
    "GenerationArtifacts",
    "RunSpec",
    "TrainRequest",
    "TrainingArtifacts",
    "available_baselines",
    "create_baseline",
    "get_wrapper_class",
    "normalize_baseline_id",
]
