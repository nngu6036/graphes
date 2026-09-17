"""Opt-in attributed spectral/categorical diffusion; legacy samplers are untouched."""


def enabled(extensions):
    value = extensions.get("attributed_categorical", {})
    return isinstance(value, dict) and bool(value.get("enabled", False))
