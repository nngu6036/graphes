"""EDGE degree-guided diffusion on GraphER's frozen generic splits."""
from grapher.models.base import BaselineCapabilities
from grapher.models.external_wrapper import SourceBackedWrapper


class EDGEWrapper(SourceBackedWrapper):
    model_id = "edge"
    display_name = "EDGE"
    capabilities = BaselineCapabilities(frozenset({"generic"}), "subprocess", "ready")
    supported_datasets = frozenset({"community_small", "ego_small", "grid"})
    source_markers = ("model.py", "diffusion/diffusion_binomial_active.py", "layers/layers.py")
    default_options = {
        "train": {"epochs": 50000, "batch_size": 8, "lr": 1e-4, "clip_value": 1., "log_every": 100},
        "model": {"arch": "TGNN_degree_guided", "parametrization": "xt_prescribed_st",
                  "loss_type": "vb_ce_xt_prescribred_st", "diffusion_steps": 128, "diffusion_dim": 64,
                  "noise_schedule": "linear", "dp_rate": .1, "num_heads": [8, 8, 8, 8, 1]},
        "generation_batch_size": 32, "runtime": {"device": "auto"},
    }
    implementation_note = "Native active-node diffusion/TGNN, GraphER PyG adapter, and empirical training-degree prior. No learned degree sampler or attributed generation."
