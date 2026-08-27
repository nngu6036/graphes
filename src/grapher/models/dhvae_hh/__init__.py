"""Isolated DH-VAE + randomized Havel--Hakimi baseline package.

The package root intentionally exports only the lightweight wrapper. Import
the implementation modules explicitly when training or generating with this
baseline; registry discovery must not eagerly import Torch or NetworkX.
"""

from grapher.models.dhvae_hh.wrapper import DHVAEHHWrapper

__all__ = ["DHVAEHHWrapper"]
