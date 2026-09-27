"""Option C: categorical nodes + continuous weighted-adjacency diffusion.

Standalone training/sampling live in training.py and sampling.py. No baseline
wrapper, categorical-edge denoiser, degree prior, or diffused eigenbasis is used.
"""

CHECKPOINT_FORMAT = "grapher_option_c_checkpoint_v1"
