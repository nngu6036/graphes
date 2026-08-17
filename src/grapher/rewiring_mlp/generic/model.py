from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from grapher.properties.summary import SummaryConfig
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.data import TopologyGraphletBatch
from grapher.rewiring_mlp.generic.layers import TopologyMPNNLayer
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir
from grapher.utils.motifs import TOPOLOGY_CANONICALIZER_CONVENTION

TOPOLOGY_CHECKPOINT_FORMAT = "topology_structural_predictor_v2"


class TopologyGraphletPredictor(nn.Module):
    """Predict terminal graph-level structural summaries from an intermediate graph.

    The maintained targets are connected graphlet histograms, a clustering-
    coefficient histogram, and the standard 15-dimensional mean orbit-count
    vector for connected graphlets with two to four nodes. The model consumes
    only the current binary topology, indexed ordinary degrees, graph size, and
    rewiring time. It has no terminal node head, pair head, no-edge category,
    or typed-degree consistency objective.
    """

    def __init__(
        self,
        *,
        graphlet_slices: tuple[tuple[int, int], ...],
        clustering_width: int = 0,
        orbit_width: int = 0,
        hidden_dim: int = 128,
        edge_dim: int = 64,
        graph_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
        min_concentration: float = 0.05,
        max_concentration: float = 50.0,
    ) -> None:
        super().__init__()
        self.graphlet_slices = tuple(
            (int(start), int(stop)) for start, stop in graphlet_slices
        )
        if not self.graphlet_slices or any(
            stop <= start for start, stop in self.graphlet_slices
        ):
            raise ValueError(
                "graphlet_slices must contain non-empty contiguous blocks."
            )
        if any(
            right[0] != left[1]
            for left, right in zip(self.graphlet_slices, self.graphlet_slices[1:])
        ):
            raise ValueError("graphlet_slices must be contiguous.")
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.graph_dim = int(graph_dim)
        self.dropout_p = float(dropout)
        self.min_concentration = float(min_concentration)
        self.max_concentration = float(max_concentration)
        if self.max_concentration <= self.min_concentration:
            raise ValueError("max_concentration must exceed min_concentration.")
        self.clustering_width = int(clustering_width)
        self.orbit_width = int(orbit_width)
        if self.clustering_width < 0 or self.orbit_width < 0:
            raise ValueError("Structural target widths must be nonnegative.")

        # Normalized degree, graph size, time, and node mask.
        self.node_in = nn.Linear(4, self.hidden_dim)
        # Current adjacency, pair mask, time, symmetric degree statistics, size.
        self.edge_in = nn.Linear(7, self.edge_dim)
        self.layers = nn.ModuleList(
            [
                TopologyMPNNLayer(self.hidden_dim, self.edge_dim)
                for _ in range(int(num_layers))
            ]
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.graph_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim + 2 * self.edge_dim + 5, self.graph_dim),
            nn.SiLU(),
            nn.Linear(self.graph_dim, self.graph_dim),
            nn.SiLU(),
        )
        self.graphlet_heads = nn.ModuleList(
            [
                nn.Linear(self.graph_dim, stop - start)
                for start, stop in self.graphlet_slices
            ]
        )
        self.graphlet_mass_head = nn.Linear(
            self.graph_dim,
            2 * len(self.graphlet_slices),
        )
        self.clustering_head = (
            nn.Linear(self.graph_dim, self.clustering_width)
            if self.clustering_width > 0
            else None
        )
        # This head predicts log(1 + mean orbit count), which is substantially
        # better conditioned than raw orbit counts across graph sizes.
        self.orbit_head = (
            nn.Linear(self.graph_dim, self.orbit_width)
            if self.orbit_width > 0
            else None
        )

    @property
    def graphlet_width(self) -> int:
        return self.graphlet_slices[-1][1]

    def _bounded_concentration(self, raw: torch.Tensor) -> torch.Tensor:
        return self.min_concentration + (
            self.max_concentration - self.min_concentration
        ) * torch.sigmoid(raw)

    @staticmethod
    def _masked_pair_pool(
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(values.dtype)
        return (values * weights).sum(dim=(1, 2)) / weights.sum(
            dim=(1, 2)
        ).clamp_min(1.0)

    def forward(self, batch: TopologyGraphletBatch) -> dict[str, torch.Tensor]:
        adjacency = batch.adjacency.bool()
        node_mask = batch.node_mask.bool()
        pair_mask = batch.pair_mask.bool()
        batch_size, node_count = node_mask.shape
        size_feature = batch.graph_size / (batch.graph_size + 1.0).clamp_min(1.0)

        node_features = torch.stack(
            [
                batch.degrees,
                size_feature.view(-1, 1).expand(batch_size, node_count),
                batch.time.view(-1, 1).expand(batch_size, node_count),
                node_mask.to(batch.degrees.dtype),
            ],
            dim=-1,
        )
        node_hidden = self.node_in(node_features)
        node_hidden = node_hidden * node_mask.unsqueeze(-1).to(node_hidden.dtype)

        degree_i = batch.degrees.unsqueeze(2).expand(
            batch_size, node_count, node_count
        )
        degree_j = batch.degrees.unsqueeze(1).expand(
            batch_size, node_count, node_count
        )
        edge_features = torch.stack(
            [
                adjacency.to(batch.degrees.dtype),
                pair_mask.to(batch.degrees.dtype),
                batch.time.view(-1, 1, 1).expand(
                    batch_size, node_count, node_count
                ),
                0.5 * (degree_i + degree_j),
                torch.abs(degree_i - degree_j),
                degree_i * degree_j,
                size_feature.view(-1, 1, 1).expand(
                    batch_size, node_count, node_count
                ),
            ],
            dim=-1,
        )
        edge_hidden = self.edge_in(edge_features)
        edge_hidden = 0.5 * (edge_hidden + edge_hidden.transpose(1, 2))
        edge_hidden = edge_hidden * pair_mask.unsqueeze(-1).to(edge_hidden.dtype)

        for layer in self.layers:
            node_hidden, edge_hidden = layer(
                node_hidden,
                edge_hidden,
                adjacency,
                node_mask,
            )
            node_hidden = self.dropout(node_hidden)
            edge_hidden = self.dropout(edge_hidden)

        node_weights = node_mask.unsqueeze(-1).to(node_hidden.dtype)
        node_pool = (node_hidden * node_weights).sum(dim=1) / node_weights.sum(
            dim=1
        ).clamp_min(1.0)
        upper = torch.triu(
            torch.ones(
                (node_count, node_count),
                dtype=torch.bool,
                device=node_mask.device,
            ),
            diagonal=1,
        ).view(1, node_count, node_count)
        upper_pairs = pair_mask & upper
        present_pool = self._masked_pair_pool(
            edge_hidden,
            upper_pairs & adjacency,
        )
        absent_pool = self._masked_pair_pool(
            edge_hidden,
            upper_pairs & ~adjacency,
        )

        degree_weights = node_mask.to(batch.degrees.dtype)
        degree_count = degree_weights.sum(dim=1).clamp_min(1.0)
        degree_mean = (batch.degrees * degree_weights).sum(dim=1) / degree_count
        degree_variance = (
            (batch.degrees - degree_mean.unsqueeze(1)).square() * degree_weights
        ).sum(dim=1) / degree_count
        degree_max = batch.degrees.masked_fill(~node_mask, 0.0).max(dim=1).values
        graph_hidden = self.graph_encoder(
            torch.cat(
                [
                    node_pool,
                    present_pool,
                    absent_pool,
                    degree_mean.unsqueeze(1),
                    torch.sqrt(degree_variance.clamp_min(0.0)).unsqueeze(1),
                    degree_max.unsqueeze(1),
                    batch.time.unsqueeze(1),
                    size_feature.unsqueeze(1),
                ],
                dim=-1,
            )
        )
        graphlet_alpha = torch.cat(
            [
                self._bounded_concentration(head(graph_hidden))
                for head in self.graphlet_heads
            ],
            dim=-1,
        )
        graphlet_mass_ab = self._bounded_concentration(
            self.graphlet_mass_head(graph_hidden)
        ).view(batch_size, len(self.graphlet_slices), 2)
        result = {
            "graphlet_alpha": graphlet_alpha,
            "graphlet_mass_ab": graphlet_mass_ab,
        }
        if self.clustering_head is not None:
            result["clustering_alpha"] = self._bounded_concentration(
                self.clustering_head(graph_hidden)
            )
        if self.orbit_head is not None:
            result["orbit_log_mean"] = F.softplus(self.orbit_head(graph_hidden))
        return result

    def graphlet_means(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = outputs["graphlet_alpha"]
        means = torch.cat(
            [
                alpha[:, start:stop]
                / alpha[:, start:stop].sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
                for start, stop in self.graphlet_slices
            ],
            dim=-1,
        )
        mass_ab = outputs["graphlet_mass_ab"]
        mass_mean = mass_ab[..., 0] / mass_ab.sum(dim=-1).clamp_min(1.0e-12)
        return means, mass_mean

    def structural_means(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        graphlet, mass = self.graphlet_means(outputs)
        if self.clustering_width > 0:
            alpha = outputs["clustering_alpha"]
            clustering = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        else:
            clustering = graphlet.new_zeros((graphlet.shape[0], 0))
        if self.orbit_width > 0:
            orbit = torch.expm1(outputs["orbit_log_mean"]).clamp_min(0.0)
        else:
            orbit = graphlet.new_zeros((graphlet.shape[0], 0))
        return graphlet, mass, clustering, orbit

    def loss(
        self,
        batch: TopologyGraphletBatch,
        *,
        loss_weights: dict[str, float] | None = None,
        target_epsilon: float = 1.0e-5,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        weights = loss_weights or {}
        outputs = self.forward(batch)
        zero = outputs["graphlet_alpha"].sum() * 0.0
        distribution_terms: list[torch.Tensor] = []
        mean_terms: list[torch.Tensor] = []
        valid_masks: list[torch.Tensor] = []
        for start, stop in self.graphlet_slices:
            target = batch.graphlet_target[:, start:stop]
            valid = target.sum(dim=-1) > 0.0
            valid_masks.append(valid)
            if not torch.any(valid):
                continue
            smoothed = target[valid] + float(target_epsilon)
            smoothed = smoothed / smoothed.sum(dim=-1, keepdim=True)
            alpha = outputs["graphlet_alpha"][valid, start:stop]
            distribution_terms.append(
                -torch.distributions.Dirichlet(alpha).log_prob(smoothed).mean()
            )
            mean = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
            mean_terms.append(
                -(smoothed * torch.log(mean.clamp_min(1.0e-12)))
                .sum(dim=-1)
                .mean()
            )
        graphlet_distribution_loss = (
            torch.stack(distribution_terms).mean() if distribution_terms else zero
        )
        graphlet_mean_loss = torch.stack(mean_terms).mean() if mean_terms else zero

        mass_terms: list[torch.Tensor] = []
        for block_index, valid in enumerate(valid_masks):
            if not torch.any(valid):
                continue
            target_mass = batch.graphlet_mass_target[valid, block_index].clamp(
                min=float(target_epsilon),
                max=1.0 - float(target_epsilon),
            )
            ab = outputs["graphlet_mass_ab"][valid, block_index]
            mass_terms.append(
                -torch.distributions.Beta(ab[:, 0], ab[:, 1])
                .log_prob(target_mass)
                .mean()
            )
        graphlet_mass_loss = torch.stack(mass_terms).mean() if mass_terms else zero

        if batch.clustering_target.shape[-1] != self.clustering_width:
            raise ValueError(
                "Clustering target width does not match the predictor head: "
                f"{batch.clustering_target.shape[-1]} != {self.clustering_width}."
            )
        clustering_distribution_loss = zero
        clustering_mean_loss = zero
        if self.clustering_width > 0:
            target = batch.clustering_target
            valid = target.sum(dim=-1) > 0.0
            if torch.any(valid):
                smoothed = target[valid] + float(target_epsilon)
                smoothed = smoothed / smoothed.sum(dim=-1, keepdim=True)
                alpha = outputs["clustering_alpha"][valid]
                clustering_distribution_loss = -torch.distributions.Dirichlet(
                    alpha
                ).log_prob(smoothed).mean()
                mean = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
                clustering_mean_loss = -(
                    smoothed * torch.log(mean.clamp_min(1.0e-12))
                ).sum(dim=-1).mean()

        if batch.orbit_target.shape[-1] != self.orbit_width:
            raise ValueError(
                "Orbit target width does not match the predictor head: "
                f"{batch.orbit_target.shape[-1]} != {self.orbit_width}."
            )
        orbit_loss = zero
        if self.orbit_width > 0:
            orbit_log_target = torch.log1p(batch.orbit_target.clamp_min(0.0))
            orbit_loss = F.smooth_l1_loss(
                outputs["orbit_log_mean"],
                orbit_log_target,
            )
        total = (
            float(weights.get("graphlet_mean", weights.get("graphlet", 1.0)))
            * graphlet_mean_loss
            + float(weights.get("graphlet_distribution", 0.1))
            * graphlet_distribution_loss
            + float(weights.get("graphlet_mass", 0.0)) * graphlet_mass_loss
            + float(weights.get("clustering_mean", 1.0))
            * clustering_mean_loss
            + float(weights.get("clustering_distribution", 0.1))
            * clustering_distribution_loss
            + float(weights.get("orbit", 1.0)) * orbit_loss
        )

        with torch.no_grad():
            (
                predicted,
                predicted_mass,
                predicted_clustering,
                predicted_orbit,
            ) = self.structural_means(outputs)
            graphlet_errors: list[torch.Tensor] = []
            mass_errors: list[torch.Tensor] = []
            for block_index, ((start, stop), valid) in enumerate(
                zip(self.graphlet_slices, valid_masks)
            ):
                if not torch.any(valid):
                    continue
                graphlet_errors.append(
                    torch.abs(
                        predicted[valid, start:stop]
                        - batch.graphlet_target[valid, start:stop]
                    ).mean()
                )
                mass_errors.append(
                    torch.abs(
                        predicted_mass[valid, block_index]
                        - batch.graphlet_mass_target[valid, block_index]
                    ).mean()
                )
            graphlet_mae = (
                torch.stack(graphlet_errors).mean() if graphlet_errors else zero
            )
            graphlet_mass_mae = (
                torch.stack(mass_errors).mean() if mass_errors else zero
            )
            clustering_mae = (
                torch.abs(predicted_clustering - batch.clustering_target).mean()
                if self.clustering_width > 0
                else zero
            )
            orbit_mae = (
                torch.abs(predicted_orbit - batch.orbit_target).mean()
                if self.orbit_width > 0
                else zero
            )
            orbit_log_mae = (
                torch.abs(
                    outputs["orbit_log_mean"]
                    - torch.log1p(batch.orbit_target.clamp_min(0.0))
                ).mean()
                if self.orbit_width > 0
                else zero
            )
        metrics = {
            "loss": float(total.detach().cpu()),
            "graphlet_mean_loss": float(graphlet_mean_loss.detach().cpu()),
            "graphlet_distribution_loss": float(
                graphlet_distribution_loss.detach().cpu()
            ),
            "graphlet_mass_loss": float(graphlet_mass_loss.detach().cpu()),
            "graphlet_mae": float(graphlet_mae.detach().cpu()),
            "graphlet_mass_mae": float(graphlet_mass_mae.detach().cpu()),
        }
        if self.clustering_width > 0:
            metrics.update(
                {
                    "clustering_mean_loss": float(
                        clustering_mean_loss.detach().cpu()
                    ),
                    "clustering_distribution_loss": float(
                        clustering_distribution_loss.detach().cpu()
                    ),
                    "clustering_mae": float(clustering_mae.detach().cpu()),
                }
            )
        if self.orbit_width > 0:
            metrics.update(
                {
                    "orbit_loss": float(orbit_loss.detach().cpu()),
                    "orbit_mae": float(orbit_mae.detach().cpu()),
                    "orbit_log_mae": float(orbit_log_mae.detach().cpu()),
                }
            )
        return total, metrics

    def model_config(self) -> dict[str, Any]:
        return {
            "graphlet_slices": [list(value) for value in self.graphlet_slices],
            "clustering_width": self.clustering_width,
            "orbit_width": self.orbit_width,
            "hidden_dim": self.hidden_dim,
            "edge_dim": self.edge_dim,
            "graph_dim": self.graph_dim,
            "num_layers": len(self.layers),
            "dropout": self.dropout_p,
            "min_concentration": self.min_concentration,
            "max_concentration": self.max_concentration,
        }


def save_topology_checkpoint(
    model: TopologyGraphletPredictor,
    path: str | Path,
    *,
    graphlet_basis: TopologyGraphletBasis,
    summary_config: SummaryConfig,
    config: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "format": TOPOLOGY_CHECKPOINT_FORMAT,
            "pipeline_mode": "topology",
            "topology_canonicalizer": TOPOLOGY_CANONICALIZER_CONVENTION,
            "model_state_dict": model.state_dict(),
            "model_config": model.model_config(),
            "graphlet_basis": graphlet_basis.to_dict(),
            "summary_config": dict(summary_config.__dict__),
            "config": config or {},
            "report": report or {},
        },
        path,
    )


def load_topology_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "auto",
) -> tuple[
    TopologyGraphletPredictor,
    TopologyGraphletBasis,
    SummaryConfig,
    dict[str, Any],
]:
    resolved_device = (
        resolve_torch_device(device) if isinstance(device, str) else device
    )
    checkpoint = torch.load(Path(path), map_location=resolved_device)
    if checkpoint.get("format") != TOPOLOGY_CHECKPOINT_FORMAT:
        raise ValueError(
            "Checkpoint is not a decoupled topology structural predictor "
            f"({TOPOLOGY_CHECKPOINT_FORMAT}). Endpoint checkpoints require the "
            "legacy attributed pipeline and cannot be migrated without retraining."
        )
    if checkpoint.get("topology_canonicalizer") != (
        TOPOLOGY_CANONICALIZER_CONVENTION
    ):
        raise ValueError(
            "Topology checkpoint canonicalizer convention is missing or "
            "incompatible; retrain it with this topology implementation."
        )
    graphlet_basis = TopologyGraphletBasis.from_dict(checkpoint["graphlet_basis"])
    if graphlet_basis.attributed:
        raise ValueError("A topology checkpoint cannot contain attributed graphlets.")
    model_config = dict(checkpoint["model_config"])
    stored_slices = tuple(
        (int(value[0]), int(value[1]))
        for value in model_config.get("graphlet_slices", [])
    )
    if stored_slices != tuple(graphlet_basis.slices):
        raise ValueError("Checkpoint graphlet coordinates do not match its basis.")
    model = TopologyGraphletPredictor(**model_config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    summary_config = SummaryConfig.from_dict(checkpoint.get("summary_config", {}) or {})
    return model, graphlet_basis, summary_config, checkpoint
