"""Canonical, collision-resistant paths for baseline-model artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from grapher.models.errors import ArtifactCollisionError

DEFAULT_BASELINE_OUTPUT_ROOT = Path("outputs/baselines")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def validate_identifier(value: str, *, field: str) -> str:
    """Validate one path component used by the artifact layout."""

    normalized = str(value).strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or _SAFE_IDENTIFIER.fullmatch(normalized) is None
    ):
        raise ValueError(
            f"{field} must be a non-empty identifier containing only letters, "
            "digits, '.', '_', or '-'."
        )
    return normalized


def _contained_path(root: Path, *parts: str) -> Path:
    """Join validated components and reject an existing symlink escape."""

    resolved_root = root.expanduser().resolve()
    candidate = resolved_root.joinpath(*parts)
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Artifact path {resolved_candidate} escapes output root {resolved_root}."
        ) from exc
    return candidate


@dataclass(frozen=True)
class ArtifactLayout:
    """Paths for one trained base-model run and all of its raw generations.

    The run path is identified by model, benchmark dataset, and run identifier:

    ``outputs/baselines/<model>/<dataset>/<run>/``.
    """

    model_id: str
    dataset_id: str
    run_id: str
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
        object.__setattr__(self, "output_root", Path(self.output_root))

    @property
    def run_dir(self) -> Path:
        return _contained_path(
            self.output_root, self.model_id, self.dataset_id, self.run_id
        )

    @property
    def run_manifest_path(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def train_dir(self) -> Path:
        return self.run_dir / "train"

    @property
    def checkpoints_dir(self) -> Path:
        return self.train_dir / "checkpoints"

    @property
    def training_manifest_path(self) -> Path:
        return self.train_dir / "manifest.json"

    @property
    def resolved_training_config_path(self) -> Path:
        return self.train_dir / "resolved_config.yaml"

    @property
    def training_log_path(self) -> Path:
        return self.train_dir / "train.log"

    @property
    def generations_dir(self) -> Path:
        return self.run_dir / "generations"

    @staticmethod
    def default_generation_id(*, seed: int, num_graphs: int) -> str:
        if int(num_graphs) <= 0:
            raise ValueError("num_graphs must be positive.")
        return f"seed_{int(seed)}_n_{int(num_graphs)}"

    def generation_dir(self, generation_id: str) -> Path:
        safe_id = validate_identifier(generation_id, field="generation_id")
        return self.generations_dir / safe_id

    def generated_graphs_path(self, generation_id: str) -> Path:
        return self.generation_dir(generation_id) / "base_graphs.pkl"

    def generation_manifest_path(self, generation_id: str) -> Path:
        return self.generation_dir(generation_id) / "manifest.json"

    def generation_log_path(self, generation_id: str) -> Path:
        return self.generation_dir(generation_id) / "generate.log"

    def native_generation_dir(self, generation_id: str) -> Path:
        return self.generation_dir(generation_id) / "native"

    @staticmethod
    def require_available(path: str | Path, *, overwrite: bool = False) -> Path:
        """Refuse accidental reuse before an implementation publishes artifacts.

        This helper never deletes data. An implemented wrapper may explicitly
        reuse a directory when ``overwrite`` is true, but must replace files
        atomically and record that decision in its manifest.
        """

        candidate = Path(path)
        if candidate.exists() and not overwrite:
            raise ArtifactCollisionError(
                f"Artifact path already exists: {candidate}. Choose a new run or "
                "generation identifier, or explicitly enable overwrite."
            )
        return candidate

