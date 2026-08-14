"""GraphER wrapper placeholder for DiGress."""

from grapher.models.base import BaselineCapabilities, PlaceholderBaseGeneratorWrapper


class DiGressWrapper(PlaceholderBaseGeneratorWrapper):
    model_id = "digress"
    display_name = "DiGress"
    capabilities = BaselineCapabilities(
        domains=frozenset({"generic", "attributed"}),
        isolation="subprocess",
        status="placeholder",
    )

