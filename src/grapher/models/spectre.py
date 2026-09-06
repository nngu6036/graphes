"""SPECTRE GAN wrapper with unconditional all_fake sampling."""
from grapher.models.base import BaselineCapabilities
from grapher.models.external_wrapper import SourceBackedWrapper


class SPECTREWrapper(SourceBackedWrapper):
    model_id = "spectre"
    display_name = "SPECTRE"
    capabilities = BaselineCapabilities(frozenset({"generic", "attributed"}), "subprocess", "ready")
    supported_datasets = frozenset({"community_small", "ego_small", "grid", "qm9"})
    source_markers = ("full_gan.py", "model/lambda_gan.py", "model/SON_gan.py", "model/ppgn_gan.py")
    default_options = {
        "train": {"epochs": 12000, "batch_size": 10, "log_every": 100},
        "model": {"k_eigval": 2, "gen_gelu": True, "disc_gelu": True,
                  "SON_D_full_readout": True, "SON_normalize_left": True, "SON_small": True,
                  "lambda_gating": True, "lambda_last_gating": True, "lambda_upsample": True,
                  "noisy_gen": True, "noisy_disc": True, "derived_eigval_noise": True,
                  "normalize_noise": True, "spectral_norm": True, "eigvec_right_noise": True,
                  "gp_shared_alpha": True, "no_restart": True, "gp_do_backwards": True,
                  "eigvec_sign_flip": True, "ignore_first_eigv": True, "gp_include_unpermuted": True,
                  "clip_grad_norm": 1., "eigvec_temp_decay": True, "eigval_temp_decay": True,
                  "decay_eigvec_temp_over": 2000, "decay_eigval_temp_over": 2000,
                  "n_eigval_warmup_epochs": 2000, "n_eigvec_warmup_epochs": 2000,
                  "min_eigval_temp": .8, "min_eigvec_temp": .8,
                  "SON_gumbel_temperature_decay": True, "decay_SON_gumbel_temp_over": 10000,
                  "SON_gumbel_temperature_warmup_epochs": 0},
        "sample": {"use_ema": True}, "generation_batch_size": 32, "runtime": {"device": "auto"},
    }
    implementation_note = "Native spectral/adjacency GAN losses in a managed loop; all_fake sampling. Fixed four-atom/three-bond QM9 head, no ZINC support."
