"""Exceptions raised by the baseline-model wrapper layer."""

from __future__ import annotations


class BaselineWrapperError(RuntimeError):
    """Base class for wrapper and artifact-contract failures."""


class BaselineNotImplementedError(NotImplementedError, BaselineWrapperError):
    """Raised when a registered wrapper operation is only a placeholder."""

    def __init__(self, model_id: str, operation: str, *, detail: str | None = None):
        message = (
            f"{operation} is not implemented for baseline {model_id!r}. "
            "The wrapper contract is registered, but its upstream integration "
            "must be added before this operation can run."
        )
        if detail:
            message = f"{message} {detail}"
        super().__init__(message)
        self.model_id = model_id
        self.operation = operation
        self.detail = detail


class ArtifactCollisionError(FileExistsError, BaselineWrapperError):
    """Raised when a run would overwrite an existing baseline artifact."""

