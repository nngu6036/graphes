"""GraphER structural correction implementation.

The package groups the Rewiring MLP and its correction-time support code:

``generic``
    Structural-summary prediction and rewiring for unlabelled graphs.
``attributed``
    Attribute-aware prediction, candidate selection, and rewiring.
``core``
    Shared degree-preserving double-edge-swap primitives.
``molecular``
    Molecular schemas, constraints, conversion, and typed invariants.
``evaluation``
    Metrics and studies used to assess raw and corrected graph batches.

The package root deliberately avoids eager imports so that lightweight tools
can inspect the project without importing optional numerical dependencies.
"""

__all__: list[str] = []
