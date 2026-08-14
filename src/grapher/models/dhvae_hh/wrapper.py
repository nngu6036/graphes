"""GraphER wrapper contract for the isolated DH-VAE + HH baseline."""

from grapher.models.base import BaselineCapabilities, PlaceholderBaseGeneratorWrapper


class DHVAEHHWrapper(PlaceholderBaseGeneratorWrapper):
    model_id = "dhvae_hh"
    display_name = "DH-VAE + randomized Havel--Hakimi"
    capabilities = BaselineCapabilities(
        domains=frozenset({"generic", "attributed"}),
        isolation="in_process",
        status="partial",
    )
    implementation_note = (
        "The DH-VAE, samplers, and Havel--Hakimi constructors are isolated in "
        "grapher.models.dhvae_hh. The common train/generate wrapper orchestration "
        "is intentionally not implemented yet. Havel--Hakimi has no trainable "
        "state."
    )
