"""GSDM spectral diffusion, retaining the user's requested GDSMWrapper spelling."""
from grapher.models.base import BaselineCapabilities
from grapher.models.external_wrapper import SourceBackedWrapper


class GDSMWrapper(SourceBackedWrapper):
    model_id = "gdsm"
    display_name = "GSDM"
    capabilities = BaselineCapabilities(frozenset({"generic"}), "subprocess", "ready")
    supported_datasets = frozenset({"community_small", "ego_small", "grid"})
    source_aliases = ("GSDM",)
    source_markers = ("models/ScoreNetwork_A_eigen.py", "models/ScoreNetwork_X.py", "solver.py", "losses.py")
    default_options = {
        "upstream_config": "community_small.yaml",
        "train": {"epochs": 200, "batch_size": 128, "log_every": 10},
        "sample": {"use_ema": False}, "generation_batch_size": 128, "runtime": {"device": "auto"},
    }
    implementation_note = "Native GSDM spectral loss/sampler; empirical eigenvector bases come exclusively from the training split. Generic graphs only."


GSDMWrapper = GDSMWrapper
