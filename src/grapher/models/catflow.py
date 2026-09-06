"""Source-backed CatFlow wrapper (replaces the previous placeholder)."""
from grapher.models.base import BaselineCapabilities
from grapher.models.external_wrapper import SourceBackedWrapper


class CatFlowWrapper(SourceBackedWrapper):
    model_id = "catflow"
    display_name = "CatFlow"
    capabilities = BaselineCapabilities(frozenset({"generic", "attributed"}), "subprocess", "ready")
    supported_datasets = frozenset({"community_small", "ego_small", "grid", "qm9", "zinc"})
    source_markers = ("utils.py", "flow_matching.py", "models/transformer.py")
    default_options = {
        "train": {"epochs": 1000, "batch_size": 128, "lr": 0.0002, "ema": 0.999, "log_every": 10,
                  "distribution": "normal", "loss_function": "kld"},
        "model": {"num_layers": 6, "small_model": 0},
        "sample": {"method": "dopri5", "t_end": 0.95, "steps": 100, "atol": 1e-5, "rtol": 1e-5, "use_ema": True},
        "generation_batch_size": 128, "runtime": {"device": "auto"},
    }
    implementation_note = "Native CatFlow transformer and default normal/kld path; frozen GraphER splits and exact-count raw exports."
