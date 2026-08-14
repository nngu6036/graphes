"""Shared contracts for all GraphER base-generator wrappers."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from grapher.models.artifacts import (
    DEFAULT_BASELINE_OUTPUT_ROOT,
    ArtifactLayout,
    validate_identifier,
)
from grapher.models.errors import BaselineNotImplementedError

GraphDomain = Literal["generic", "attributed"]
IsolationMode = Literal["in_process", "subprocess"]
ImplementationStatus = Literal["placeholder", "partial", "ready"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BaselineCapabilities:
    """Declared scope of one GraphER-facing base wrapper."""

    domains: frozenset[GraphDomain]
    isolation: IsolationMode
    status: ImplementationStatus = "placeholder"


@dataclass(frozen=True)
class DatasetReference:
    """Reference to one dataset prepared by the GraphER project.

    ``benchmark_id`` is used in reports and output paths. ``serialized_id`` is
    the directory name under ``root``. ``native_id`` records an optional name
    expected by an upstream baseline (for example, ``comm20`` for DeFoG).
    """

    benchmark_id: str
    root: Path = Path("outputs/datasets")
    serialized_id: str | None = None
    native_id: str | None = None
    config_path: Path | None = None

    def __post_init__(self) -> None:
        benchmark_id = validate_identifier(
            self.benchmark_id, field="dataset.benchmark_id"
        )
        serialized_id = validate_identifier(
            self.serialized_id or benchmark_id,
            field="dataset.serialized_id",
        )
        native_id = (
            validate_identifier(self.native_id, field="dataset.native_id")
            if self.native_id is not None
            else None
        )
        object.__setattr__(self, "benchmark_id", benchmark_id)
        object.__setattr__(self, "serialized_id", serialized_id)
        object.__setattr__(self, "native_id", native_id)
        object.__setattr__(self, "root", Path(self.root))
        if self.config_path is not None:
            object.__setattr__(self, "config_path", Path(self.config_path))

    @property
    def dataset_dir(self) -> Path:
        return self.root / str(self.serialized_id)

    @property
    def split_paths(self) -> dict[str, Path]:
        return {
            split: self.dataset_dir / f"{split}.pkl"
            for split in ("train", "val", "test")
        }

    def require_prepared(self) -> DatasetReference:
        missing = [path for path in self.split_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "The baseline wrapper requires prepared GraphER dataset splits; "
                f"missing: {[str(path) for path in missing]}."
            )
        return self

    def fingerprint(self) -> str:
        """Hash the three serialized splits without loading their pickle data."""

        self.require_prepared()
        record = {
            split: {"path": path.name, "sha256": _sha256(path)}
            for split, path in sorted(self.split_paths.items())
        }
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunSpec:
    """Identity and location of one trained baseline run."""

    model_id: str
    dataset_id: str
    run_id: str
    train_seed: int
    output_root: Path = DEFAULT_BASELINE_OUTPUT_ROOT

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "model_id", validate_identifier(self.model_id, field="model_id")
        )
        object.__setattr__(
            self,
            "dataset_id",
            validate_identifier(self.dataset_id, field="dataset_id"),
        )
        object.__setattr__(
            self, "run_id", validate_identifier(self.run_id, field="run_id")
        )
        object.__setattr__(self, "train_seed", int(self.train_seed))
        object.__setattr__(self, "output_root", Path(self.output_root))

    @classmethod
    def for_seed(
        cls,
        *,
        model_id: str,
        dataset_id: str,
        seed: int,
        output_root: str | Path = DEFAULT_BASELINE_OUTPUT_ROOT,
        run_id: str | None = None,
    ) -> RunSpec:
        return cls(
            model_id=model_id,
            dataset_id=dataset_id,
            run_id=run_id or f"seed_{int(seed)}",
            train_seed=int(seed),
            output_root=Path(output_root),
        )

    @property
    def layout(self) -> ArtifactLayout:
        return ArtifactLayout(
            model_id=self.model_id,
            dataset_id=self.dataset_id,
            run_id=self.run_id,
            output_root=self.output_root,
        )


@dataclass(frozen=True)
class TrainRequest:
    """Model-independent input to :meth:`BaseGeneratorWrapper.train`."""

    run: RunSpec
    dataset: DatasetReference
    config_path: Path | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    resume_from: Path | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.run.dataset_id != self.dataset.benchmark_id:
            raise ValueError(
                "run.dataset_id must match dataset.benchmark_id: "
                f"{self.run.dataset_id!r} != {self.dataset.benchmark_id!r}."
            )
        if self.config_path is not None:
            object.__setattr__(self, "config_path", Path(self.config_path))
        if self.resume_from is not None:
            object.__setattr__(self, "resume_from", Path(self.resume_from))


@dataclass(frozen=True)
class TrainingArtifacts:
    run_dir: Path
    checkpoint_path: Path
    manifest_path: Path
    log_path: Path | None = None
    artifacts: tuple[Path, ...] = ()


@dataclass(frozen=True)
class GenerateRequest:
    """Model-independent input to :meth:`BaseGeneratorWrapper.generate`."""

    run: RunSpec
    checkpoint_path: Path
    num_graphs: int
    generation_seed: int
    generation_id: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        object.__setattr__(self, "num_graphs", int(self.num_graphs))
        object.__setattr__(self, "generation_seed", int(self.generation_seed))
        if self.num_graphs <= 0:
            raise ValueError("num_graphs must be positive.")
        if self.generation_id is not None:
            object.__setattr__(
                self,
                "generation_id",
                validate_identifier(self.generation_id, field="generation_id"),
            )

    @property
    def resolved_generation_id(self) -> str:
        return self.generation_id or ArtifactLayout.default_generation_id(
            seed=self.generation_seed,
            num_graphs=self.num_graphs,
        )


@dataclass(frozen=True)
class GenerationArtifacts:
    run_dir: Path
    generation_dir: Path
    graphs_path: Path
    manifest_path: Path
    num_requested: int
    num_generated: int
    graphs_sha256: str
    log_path: Path | None = None
    native_artifacts: tuple[Path, ...] = ()


class BaseGeneratorWrapper(ABC):
    """Uniform GraphER-facing interface for a frozen upstream generator."""

    model_id: ClassVar[str]
    display_name: ClassVar[str]
    capabilities: ClassVar[BaselineCapabilities]
    implementation_note: ClassVar[str | None] = None

    def validate_train_request(self, request: TrainRequest) -> None:
        if request.run.model_id != self.model_id:
            raise ValueError(
                f"Run model {request.run.model_id!r} does not match wrapper "
                f"{self.model_id!r}."
            )
        request.dataset.require_prepared()

    def validate_generate_request(self, request: GenerateRequest) -> None:
        if request.run.model_id != self.model_id:
            raise ValueError(
                f"Run model {request.run.model_id!r} does not match wrapper "
                f"{self.model_id!r}."
            )
        if not request.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing baseline checkpoint: {request.checkpoint_path}"
            )

    @abstractmethod
    def train(self, request: TrainRequest) -> TrainingArtifacts:
        """Train from GraphER-prepared splits and publish model artifacts."""

    @abstractmethod
    def generate(self, request: GenerateRequest) -> GenerationArtifacts:
        """Publish an ordered, immutable batch of completed raw graphs."""


class PlaceholderBaseGeneratorWrapper(BaseGeneratorWrapper):
    """Registered contract used until an upstream integration is implemented."""

    def train(self, request: TrainRequest) -> TrainingArtifacts:
        # Raise before validating paths or creating any partial run directory.
        raise BaselineNotImplementedError(
            self.model_id,
            "Training",
            detail=self.implementation_note,
        )

    def generate(self, request: GenerateRequest) -> GenerationArtifacts:
        # Raise before validating paths or creating any partial generation.
        raise BaselineNotImplementedError(
            self.model_id,
            "Generation",
            detail=self.implementation_note,
        )

