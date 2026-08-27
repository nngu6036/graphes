"""DeFoG baseline integration for GraphER.

The wrapper is re-exported here so callers can continue to import
``grapher.models.defog.DeFoGWrapper`` while the implementation modules remain
grouped under this package.
"""

from grapher.models.defog.wrapper import DeFoGWrapper

__all__ = ["DeFoGWrapper"]
