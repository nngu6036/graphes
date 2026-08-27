"""GraphER wrapper placeholder for HOG-Diff."""

from grapher.models.base import BaselineCapabilities, PlaceholderBaseGeneratorWrapper


class HOGDiffWrapper(PlaceholderBaseGeneratorWrapper):
    model_id = "hog_diff"
    display_name = "HOG-Diff"
    capabilities = BaselineCapabilities(
        domains=frozenset({"generic", "attributed"}),
        isolation="subprocess",
        status="placeholder",
    )

