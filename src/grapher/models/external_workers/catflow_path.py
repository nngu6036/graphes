"""Explicit CatFlow probability paths; native source files remain read-only.

The linear path is Eq. (3)/(20) of arXiv:2406.04843v2. The uploaded
VFM_CatFlow source adds time-independent 0.5 noise in its normal branch;
that branch is retained only as an explicitly labelled legacy experiment.
"""
import hashlib
from pathlib import Path

import torch


def linear_interpolant(target, t, edge=False):
    """Sample (1-t) z + t target with the source's symmetric edge prior."""
    noise = torch.randn_like(target)
    if edge:
        noise = (noise + noise.transpose(1, 2)) * 0.5
    return (1.0 - t) * noise + t * target


def training_interpolant(target, t, path, upstream_flow, edge=False):
    if path == "linear":
        return linear_interpolant(target, t, edge=edge)
    if path == "upstream":
        return upstream_flow.conditional_velocity("normal", target, t, None, 8, 0, edge=edge)[0]
    raise ValueError("train.path must be 'linear' or explicitly 'upstream'.")


def probe_upstream_path(flow):
    """Measure residual noise at t=1 without changing the training RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1729)
        x = torch.zeros(4096, 4, 2)
        x[..., 0] = 1
        xt, _ = flow.conditional_velocity("normal", x, torch.ones(4096, 1, 1), None, 8, 0)
        e = torch.zeros(4096, 4, 4, 2)
        e[..., 0] = 1
        et, _ = flow.conditional_velocity("normal", e, torch.ones(4096, 1, 1, 1), None, 8, 0, edge=True)
        upper = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
        return {"node_residual_std_at_t1": float((xt - x).std().item()),
                "edge_residual_std_at_t1": float((et - e)[:, upper].std().item()),
                "seed": 1729, "num_probe_graphs": 4096}


def path_record(path, source_root, probe):
    if path not in {"linear", "upstream"}:
        raise ValueError("Unknown CatFlow probability path: " + str(path))
    return {"name": path, "version": 2,
            "formula": "x_t=(1-t)*z+t*x_1" if path == "linear" else "upstream conditional_velocity(normal)",
            "added_training_noise_std": 0.0 if path == "linear" else None,
            "edge_noise": "(z+z.transpose(1,2))/2",
            "sampling_velocity": "(softmax(logits)-state)/(1-t)",
            "upstream_flow_matching_sha256": hashlib.sha256((Path(source_root) / "flow_matching.py").read_bytes()).hexdigest(),
            "upstream_probe": probe}
