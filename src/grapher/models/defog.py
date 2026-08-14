"""GraphER wrapper placeholder for DeFoG.

The existing generation backend remains in :mod:`grapher.models.defog_backend`.
The unified wrapper will delegate to it rather than duplicate that validated
subprocess/export implementation.
"""

from grapher.models.base import BaselineCapabilities, PlaceholderBaseGeneratorWrapper


class DeFoGWrapper(PlaceholderBaseGeneratorWrapper):
    model_id = "defog"
    display_name = "DeFoG"
    capabilities = BaselineCapabilities(
        domains=frozenset({"generic", "attributed"}),
        isolation="subprocess",
        status="placeholder",
    )
    implementation_note = (
        "A validated generic generation backend already exists at "
        "grapher.models.defog_backend; the common train/generate artifact wrapper "
        "has not yet been connected to it."
    )

