"""Joint DH-VAE / spectral-summary predictor for simple, untyped graphs.

The exact degree histogram is re-encoded on EVERY structural forward pass.
Structural conditioning uses the posterior mean and its decoder features in both
training and inference. Stochastic VAE reconstruction/prior learning is a separate
loss. Integer sampling, HH construction, and swaps remain outside autograd.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn

from grapher.models.dhvae_hh.degree_sampler import DegreeVAESampler
from grapher.models.dhvae_hh.degree_vae import DegreeHistogramVAE, DegreeVectorizer, degree_vae_loss
from grapher.rewiring_mlp.generic.spectral_data import TopologySpectralBatch
from grapher.rewiring_mlp.generic.spectral_model import TopologySpectralTransformerPredictor

JOINT_DEGREE_VERSION = 1


def exact_degree_inputs(
    batch: TopologySpectralBatch, vectorizer: DegreeVectorizer,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Read ONLY the invariant from adjacency (never edge-pattern features).

    batch.degrees is normalized by the collator; use adjacency sums to recover
    exact integer degrees. Padding nodes must never contribute degree-zero mass.
    Fail rather than silently clipping an unsupported graph size or degree.
    Returned totals are [mean_degree, mean_choose(d,2), mean_choose(d,3)].
    """
    mask = batch.node_mask.bool()
    adjacency = batch.adjacency.to(dtype=torch.float32)
    if torch.any((adjacency != 0) & (adjacency != 1)):
        raise ValueError("Joint predictor requires a simple binary adjacency matrix.")
    degrees = adjacency.sum(-1)
    n = mask.sum(-1).long()
    if torch.any(n < 1):
        raise ValueError("Joint degree conditioning requires nonempty graphs.")
    if torch.any(n < vectorizer.min_nodes) or torch.any(n > vectorizer.max_nodes):
        raise ValueError(
            f"Graph size outside joint DH-VAE support [{vectorizer.min_nodes}, {vectorizer.max_nodes}]. "
            "Rebuild/retrain a degree model with suitable support; sizes are never clipped."
        )
    if torch.any(degrees[mask] > vectorizer.max_degree):
        raise ValueError(
            f"Degree exceeds joint DH-VAE support max_degree={vectorizer.max_degree}; "
            "do not silently truncate its histogram."
        )
    if torch.any(degrees[mask] != degrees[mask].round()):
        raise ValueError("Joint predictor requires an actual discrete graph, not soft adjacency.")
    if not torch.equal(adjacency, adjacency.transpose(1, 2)) or torch.any(adjacency.diagonal(dim1=1, dim2=2)):
        raise ValueError("Joint predictor requires symmetric loop-free adjacency.")
    if torch.any(adjacency * (~(mask.unsqueeze(1) & mask.unsqueeze(2))).to(adjacency.dtype)):
        raise ValueError("Edges incident to padded nodes are not allowed.")
    m = (degrees.sum(-1) / 2).long()
    if torch.any(m > vectorizer.max_edges):
        raise ValueError(f"Edge count exceeds joint DH-VAE support max_edges={vectorizer.max_edges}.")
    hist = torch.zeros((n.numel(), vectorizer.degree_dim), device=degrees.device, dtype=degrees.dtype)
    hist.scatter_add_(1, degrees.long(), mask.to(degrees.dtype))
    n_float = n.to(degrees.dtype)
    hist = hist / n_float.unsqueeze(1)
    features = torch.cat([n_float.unsqueeze(1) / max(vectorizer.max_nodes, 1), hist], -1)
    weighted = degrees * mask
    mean_degree = weighted.sum(-1) / n_float
    w2 = (degrees * (degrees - 1) / 2 * mask).sum(-1) / n_float
    w3 = (degrees * (degrees - 1) * (degrees - 2) / 6 * mask).sum(-1) / n_float
    totals = torch.stack([mean_degree, w2, w3], -1)
    targets = {
        "num_nodes": n - vectorizer.min_nodes,
        "num_nodes_count": n,
        "num_edges_count": m,
        "degree": hist,
        "mean_degree": mean_degree.unsqueeze(1),
    }
    return features, targets, totals


def degree_consistent_orbits(raw: torch.Tensor, totals: torch.Tensor) -> torch.Tensor:
    """Nonnegative 15-orbit parameterization satisfying nine necessary identities.

    o0=mean(d), o1=2o2, o2+o3=mean(C(d,2)), o4=o5, o6=3o7,
    o10=2o9, o11=o9, o12=o13, o7+o11+o13+o14=mean(C(d,3)).
    These are NOT a sufficient graph-realizability certificate. In particular
    integer subgraph counts, clustering compatibility, and other constraints
    are not enforced. The 15-logit head is retained for warm-start compatibility.
    """
    if raw.ndim != 2 or raw.shape[1] != 15 or totals.shape != (raw.shape[0], 3):
        raise ValueError("Expected nonnegative [B,15] orbit values and [B,3] degree totals.")
    eps = torch.finfo(raw.dtype).eps
    pair = raw[:, [2, 3]].clamp_min(eps)
    pair = pair / pair.sum(-1, keepdim=True) * totals[:, 1:2]
    open_wedge, triangle = pair.unbind(-1)
    path = (raw[:, 4] + raw[:, 5]) / 2
    neighbor_triples = torch.stack([
        (raw[:, 7] + raw[:, 6] / 3) / 2,
        (raw[:, 9] + raw[:, 10] / 2 + raw[:, 11]) / 3,
        (raw[:, 12] + raw[:, 13]) / 2,
        raw[:, 14],
    ], -1).clamp_min(eps)
    neighbor_triples = (neighbor_triples / neighbor_triples.sum(-1, keepdim=True)) * totals[:, 2:3]
    star, paw, diamond, clique = neighbor_triples.unbind(-1)
    return torch.stack([
        totals[:, 0], 2 * open_wedge, open_wedge, triangle,
        path, path, 3 * star, star, raw[:, 8], paw, 2 * paw,
        paw, diamond, diamond, clique,
    ], -1)


def orbit_identity_residual(orbits: torch.Tensor, totals: torch.Tensor) -> torch.Tensor:
    """Per-example maximum absolute error of the nine enforced identities."""
    o = orbits
    residuals = torch.stack([
        o[:, 0] - totals[:, 0], o[:, 1] - 2*o[:, 2],
        o[:, 2] + o[:, 3] - totals[:, 1], o[:, 4] - o[:, 5],
        o[:, 6] - 3*o[:, 7], o[:, 10] - 2*o[:, 9],
        o[:, 11] - o[:, 9], o[:, 12] - o[:, 13],
        o[:, 7] + o[:, 11] + o[:, 13] + o[:, 14] - totals[:, 2],
    ], -1)
    return residuals.abs().amax(-1)


class JointDegreeSpectralPredictor(TopologySpectralTransformerPredictor):
    """An embedded DH-VAE shares encoder AND decoder features with GraphER."""

    joint_degree_enabled = True

    def __init__(self, *, joint_degree_config: dict[str, Any], **spectral_kwargs: Any) -> None:
        super().__init__(**spectral_kwargs)
        cfg = deepcopy(joint_degree_config)
        if int(cfg.get("version", JOINT_DEGREE_VERSION)) != JOINT_DEGREE_VERSION:
            raise ValueError("Unsupported joint degree checkpoint version.")
        self.joint_degree_config = cfg
        degree_config = dict(cfg["model_config"])
        architecture = int(degree_config.pop("architecture_version", 4))
        if architecture not in (2, 3, 4):
            raise ValueError("Joint degree model requires a size-conditioned DH-VAE (v2--v4).")
        self.degree_model = DegreeHistogramVAE(**degree_config)
        self.degree_vectorizer = DegreeVectorizer(**cfg["vectorizer"])
        v = self.degree_vectorizer
        if (
            self.degree_model.input_dim != v.input_dim
            or self.degree_model.max_degree != v.max_degree
            or self.degree_model.min_nodes != v.min_nodes
            or self.degree_model.max_nodes != v.max_nodes
            or self.degree_model.max_edges != v.max_edges
        ):
            raise ValueError("Embedded degree model and vectorizer supports disagree.")
        self.orbit_consistency = str(cfg.get("orbit_consistency", "degree_identities"))
        if self.orbit_consistency not in {"none", "degree_identities"}:
            raise ValueError("orbit_consistency must be none or degree_identities.")
        hidden = int(cfg.get("conditioning_dim", self.spectral_dim))
        if hidden < 1:
            raise ValueError("conditioning_dim must be positive.")
        # Encoder features, posterior mean, decoder features, normalized n/m/
        # degree moments. This does not use independently sampled degree logits.
        condition_width = 2*self.degree_model.hidden_dim + self.degree_model.latent_dim + 5
        self.degree_conditioner = nn.Sequential(
            nn.Linear(condition_width, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, self.spectral_dim),
        )
        self._degree_frozen = False

    def set_degree_trainable(self, trainable: bool) -> None:
        self._degree_frozen = not bool(trainable)
        self.degree_model.requires_grad_(bool(trainable))
        self.degree_model.train(self.training and bool(trainable))

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_degree_frozen", False):
            self.degree_model.eval()
        return self

    def degree_condition(self, batch: TopologySpectralBatch) -> tuple[torch.Tensor, torch.Tensor]:
        features, targets, totals = exact_degree_inputs(batch, self.degree_vectorizer)
        encoded = self.degree_model.encoder(features)
        # Deterministic conditioning is deliberate: identical at training and
        # inference, works for supplied/empirical degrees, avoids latent history.
        mu = self.degree_model.mu(encoded)
        decoded = self.degree_model.decode(
            mu, targets["num_nodes_count"], targets["num_edges_count"], return_hidden=True,
        )
        v = self.degree_vectorizer
        scalars = torch.stack([
            targets["num_nodes_count"].float() / max(v.max_nodes, 1),
            targets["num_edges_count"].float() / max(v.max_edges, 1),
            totals[:, 0] / max(v.max_degree, 1),
            totals[:, 1] / max(v.max_degree*(v.max_degree-1)/2, 1),
            totals[:, 2] / max(v.max_degree*(v.max_degree-1)*(v.max_degree-2)/6, 1),
        ], -1)
        condition = self.degree_conditioner(torch.cat([
            encoded, mu, decoded["degree_hidden"], scalars,
        ], -1))
        return condition, totals

    def forward(self, batch: TopologySpectralBatch) -> dict[str, torch.Tensor]:
        context, totals = self.degree_condition(batch)
        graph_hidden = self._graph_context(batch) if self.use_graph_context else None
        outputs = self._spectral_outputs_from_graph_hidden(batch, graph_hidden, degree_context=context)
        outputs["degree_moment_totals"] = totals
        if self.predict_orbit_summary:
            raw = outputs["clean_orbit_summary"]
            outputs["unconstrained_orbit_summary"] = raw
            if self.orbit_consistency == "degree_identities":
                outputs["clean_orbit_summary"] = degree_consistent_orbits(raw, totals)
                outputs["clean_orbit_log_mean"] = torch.log1p(outputs["clean_orbit_summary"])
        return outputs

    def loss(self, batch: TopologySpectralBatch, *, loss_weights=None):
        """Structural loss only. Joint trainer adds degree loss ONCE per graph."""
        outputs = self(batch)
        loss, metrics = self._spectral_loss_from_outputs(batch, outputs, loss_weights=loss_weights)
        if self.predict_orbit_summary:
            with torch.no_grad():
                metrics["orbit_identity_max_abs"] = float(orbit_identity_residual(
                    outputs["clean_orbit_summary"], outputs["degree_moment_totals"],
                ).mean().cpu())
                metrics["unconstrained_orbit_identity_max_abs"] = float(orbit_identity_residual(
                    outputs["unconstrained_orbit_summary"], outputs["degree_moment_totals"],
                ).mean().cpu())
        return loss, metrics

    def degree_loss(
        self, batch: TopologySpectralBatch, *, beta: float, weights: dict[str, float],
        prior_distribution_sigma: float = 0.2,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        features, targets, _ = exact_degree_inputs(batch, self.degree_vectorizer)
        outputs, mu, logvar = self.degree_model(
            features, targets["num_nodes_count"], targets["num_edges_count"],
        )
        prior_outputs = None
        if float(weights.get("prior_distribution", 0.0)) > 0:
            # Do NOT use sample_outputs(): it is @no_grad in DH-VAE. Decoder
            # and selected GMM parameters remain differentiable here; mixture
            # component selection is discrete, as in the existing DH-VAE trainer.
            z = self.degree_model.sample_prior(
                targets["num_nodes_count"], edge_counts=targets["num_edges_count"],
            )
            prior_outputs = self.degree_model.decode(z, targets["num_nodes_count"], targets["num_edges_count"])
        return degree_vae_loss(
            outputs, targets, mu, logvar, beta=beta, weights=weights,
            prior_outputs=prior_outputs, prior_distribution_sigma=prior_distribution_sigma,
        )

    def model_config(self) -> dict[str, Any]:
        cfg = super().model_config()
        joint = deepcopy(self.joint_degree_config)
        joint["version"] = JOINT_DEGREE_VERSION
        joint["model_config"] = self.degree_model.model_config()
        joint["vectorizer"] = deepcopy(self.degree_vectorizer.__dict__)
        cfg["joint_degree_config"] = joint
        return cfg


def build_embedded_degree_sampler(model: JointDegreeSpectralPredictor, config: dict[str, Any], *, seed: int):
    """Use exactly the DH-VAE weights embedded in the selected GraphER model.

    Never silently fall back to an external pre-joint checkpoint after fine-tuning.
    """
    cfg = dict(config or {})
    if cfg.get("checkpoint_path") or cfg.get("checkpoint"):
        raise ValueError(
            "Joint generation uses its embedded DH-VAE. Set degree_generator.checkpoint_path=null; "
            "an external degree checkpoint would break the joint-model comparison."
        )
    if str(cfg.get("postprocess_policy", "reject_only")) != "reject_only" or str(cfg.get("fallback", "error")) != "error":
        raise ValueError("Joint degree generation requires reject_only sampling and fallback=error.")
    if not bool(cfg.get("enabled", True)):
        raise ValueError("Learned generation requires degree_generator.enabled=true.")
    cfg.update(checkpoint_path="<embedded_joint_degree>", device=str(next(model.parameters()).device),
               postprocess_policy="reject_only", fallback="error")
    return DegreeVAESampler.from_config(cfg, seed=seed, model=model.degree_model, vectorizer=model.degree_vectorizer)
