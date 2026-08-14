"""GraphER wrapper placeholder for CatFlow."""

from grapher.models.base import BaselineCapabilities, PlaceholderBaseGeneratorWrapper


class CatFlowWrapper(PlaceholderBaseGeneratorWrapper):
    model_id = "catflow"
    display_name = "CatFlow"
    capabilities = BaselineCapabilities(
        domains=frozenset({"generic", "attributed"}),
        isolation="subprocess",
        status="placeholder",
    )

