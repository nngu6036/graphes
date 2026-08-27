"""GraphER wrapper placeholder for FLAGG."""

from grapher.models.base import BaselineCapabilities, PlaceholderBaseGeneratorWrapper


class FLAGGWrapper(PlaceholderBaseGeneratorWrapper):
    model_id = "flagg"
    display_name = "FLAGG"
    capabilities = BaselineCapabilities(
        domains=frozenset({"generic", "attributed"}),
        isolation="subprocess",
        status="placeholder",
    )
    implementation_note = (
        "The future manifest must record the FLAGG variant, insertion policy, "
        "and one-shot filler configuration/checkpoint."
    )

