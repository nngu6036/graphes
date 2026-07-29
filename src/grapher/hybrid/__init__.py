"""CatFlow-inspired endpoint prediction with graphlet-guided rewiring.

The hybrid package is intentionally separate from the legacy summary-only
generator.  Its model predicts a complete categorical endpoint law together
with a higher-order graphlet law from an intermediate, degree-preserving graph.
"""

from grapher.hybrid.data import (
    GraphCategoryVocabulary,
    GraphletBasis,
    HybridEndpointBatch,
    HybridEndpointExample,
    aligned_havel_hakimi_source,
    build_endpoint_examples,
    collate_endpoint_examples,
)
from grapher.hybrid.model import (
    HybridEndpointPredictor,
    load_hybrid_endpoint_checkpoint,
    save_hybrid_endpoint_checkpoint,
)
from grapher.hybrid.refiner import (
    HybridPrediction,
    HybridRefinerConfig,
    predict_hybrid_target,
    refine_graph_with_hybrid_predictions,
)

__all__ = [
    "GraphCategoryVocabulary",
    "GraphletBasis",
    "HybridEndpointBatch",
    "HybridEndpointExample",
    "HybridEndpointPredictor",
    "HybridPrediction",
    "HybridRefinerConfig",
    "aligned_havel_hakimi_source",
    "build_endpoint_examples",
    "collate_endpoint_examples",
    "load_hybrid_endpoint_checkpoint",
    "predict_hybrid_target",
    "refine_graph_with_hybrid_predictions",
    "save_hybrid_endpoint_checkpoint",
]
