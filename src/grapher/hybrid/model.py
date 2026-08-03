from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from grapher.hybrid.data import (
    GraphCategoryVocabulary,
    GraphletBasis,
    HybridEndpointBatch,
)
from grapher.hybrid.layers import EdgeAwareMPNNLayer
from grapher.properties.summary import SummaryConfig
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir

CHECKPOINT_FORMAT = "hybrid_endpoint_graphlet_v1"


class HybridEndpointPredictor(nn.Module):
    """Predict clean categorical endpoint states and a graphlet law.

    Unlike CatFlow, the predicted endpoint probabilities are not converted into
    a continuous velocity.  They are sampled (or used directly in an ablation)
    by a discrete rewiring selector.  Every accepted state update remains a
    valid double-edge swap.
    """

    def __init__(
        self,
        *,
        num_node_categories: int,
        num_edge_categories: int,
        graphlet_slices: tuple[tuple[int, int], ...],
        hidden_dim: int = 128,
        edge_dim: int = 64,
        graph_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
        min_concentration: float = 0.05,
        max_concentration: float = 50.0,
    ):
        super().__init__()
        if int(num_node_categories) <= 0:
            raise ValueError("num_node_categories must be positive.")
        if int(num_edge_categories) < 2:
            raise ValueError(
                "num_edge_categories must include no-edge and one edge category."
            )
        self.num_node_categories = int(num_node_categories)
        self.num_edge_categories = int(num_edge_categories)
        self.graphlet_slices = tuple(
            (int(start), int(stop)) for start, stop in graphlet_slices
        )
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.graph_dim = int(graph_dim)
        self.dropout_p = float(dropout)
        self.min_concentration = float(min_concentration)
        self.max_concentration = float(max_concentration)
        if self.max_concentration <= self.min_concentration:
            raise ValueError(
                "max_concentration must be greater than min_concentration."
            )

        # node one-hot, fixed labelled degree, time, real-node mask
        self.node_in = nn.Linear(self.num_node_categories + 3, self.hidden_dim)
        # edge one-hot, adjacency, pair mask, time, degree sum, degree difference
        self.edge_in = nn.Linear(self.num_edge_categories + 5, self.edge_dim)
        self.layers = nn.ModuleList(
            [
                EdgeAwareMPNNLayer(self.hidden_dim, self.edge_dim)
                for _ in range(int(num_layers))
            ]
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.node_head = nn.Linear(self.hidden_dim, self.num_node_categories)
        self.edge_head = nn.Linear(self.edge_dim, self.num_edge_categories)

        self.graph_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim + self.edge_dim + 1, self.graph_dim),
            nn.SiLU(),
            nn.Linear(self.graph_dim, self.graph_dim),
            nn.SiLU(),
        )
        self.graphlet_heads = nn.ModuleList(
            [
                nn.Linear(self.graph_dim, max(stop - start, 0))
                for start, stop in self.graphlet_slices
            ]
        )
        self.graphlet_mass_head = nn.Linear(
            self.graph_dim,
            2 * len(self.graphlet_slices),
        )

    def _bounded_concentration(self, raw: torch.Tensor) -> torch.Tensor:
        """Bound density sharpness so likelihood cannot improve without limit."""

        return self.min_concentration + (
            self.max_concentration - self.min_concentration
        ) * torch.sigmoid(raw)

    @property
    def graphlet_width(self) -> int:
        return max((stop for _, stop in self.graphlet_slices), default=0)

    def _pool_graph(
        self,
        node_hidden: torch.Tensor,
        edge_hidden: torch.Tensor,
        node_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        node_weight = node_mask.unsqueeze(-1).float()
        node_pool = (node_hidden * node_weight).sum(dim=1) / node_weight.sum(
            dim=1
        ).clamp_min(1.0)

        _, node_count, _ = pair_mask.shape
        upper = torch.triu(
            torch.ones(
                (node_count, node_count),
                dtype=torch.bool,
                device=pair_mask.device,
            ),
            diagonal=1,
        )
        upper_mask = pair_mask.bool() & upper.view(1, node_count, node_count)
        edge_weight = upper_mask.unsqueeze(-1).float()
        edge_pool = (edge_hidden * edge_weight).sum(dim=(1, 2)) / edge_weight.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        return self.graph_encoder(
            torch.cat([node_pool, edge_pool, time.view(-1, 1)], dim=-1)
        )

    def forward(self, batch: HybridEndpointBatch) -> dict[str, torch.Tensor]:
        node_mask = batch.node_mask.bool()
        pair_mask = batch.pair_mask.bool()
        current_nodes = batch.current_node_labels.clamp(
            min=0,
            max=self.num_node_categories - 1,
        )
        current_edges = batch.current_edge_labels.clamp(
            min=0,
            max=self.num_edge_categories - 1,
        )
        batch_size, node_count = current_nodes.shape

        node_onehot = torch.nn.functional.one_hot(
            current_nodes,
            self.num_node_categories,
        ).float()
        node_features = torch.cat(
            [
                node_onehot,
                batch.degrees.unsqueeze(-1),
                batch.time.view(batch_size, 1, 1).expand(
                    batch_size,
                    node_count,
                    1,
                ),
                node_mask.unsqueeze(-1).float(),
            ],
            dim=-1,
        )
        node_hidden = self.node_in(node_features)
        node_hidden = node_hidden * node_mask.unsqueeze(-1).float()

        edge_onehot = torch.nn.functional.one_hot(
            current_edges,
            self.num_edge_categories,
        ).float()
        adjacency = (current_edges > 0) & pair_mask
        degree_i = batch.degrees.unsqueeze(2).expand(
            batch_size,
            node_count,
            node_count,
        )
        degree_j = batch.degrees.unsqueeze(1).expand(
            batch_size,
            node_count,
            node_count,
        )
        edge_features = torch.cat(
            [
                edge_onehot,
                adjacency.unsqueeze(-1).float(),
                pair_mask.unsqueeze(-1).float(),
                batch.time.view(batch_size, 1, 1, 1).expand(
                    batch_size,
                    node_count,
                    node_count,
                    1,
                ),
                (0.5 * (degree_i + degree_j)).unsqueeze(-1),
                torch.abs(degree_i - degree_j).unsqueeze(-1),
            ],
            dim=-1,
        )
        edge_hidden = self.edge_in(edge_features)
        edge_hidden = 0.5 * (edge_hidden + edge_hidden.transpose(1, 2))
        edge_hidden = edge_hidden * pair_mask.unsqueeze(-1).float()

        for layer in self.layers:
            node_hidden, edge_hidden = layer(
                node_hidden,
                edge_hidden,
                adjacency,
                node_mask,
            )
            node_hidden = self.dropout(node_hidden)
            edge_hidden = self.dropout(edge_hidden)

        node_logits = self.node_head(node_hidden)
        edge_logits = self.edge_head(edge_hidden)
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))

        graph_hidden = self._pool_graph(
            node_hidden,
            edge_hidden,
            node_mask,
            pair_mask,
            batch.time,
        )
        alpha_blocks = [
            self._bounded_concentration(head(graph_hidden))
            for head in self.graphlet_heads
        ]
        graphlet_alpha = (
            torch.cat(alpha_blocks, dim=-1)
            if alpha_blocks
            else graph_hidden.new_zeros((batch_size, 0))
        )
        graphlet_mass_ab = self._bounded_concentration(
            self.graphlet_mass_head(graph_hidden)
        ).view(batch_size, len(self.graphlet_slices), 2)
        return {
            "node_logits": node_logits,
            "edge_logits": edge_logits,
            "graphlet_alpha": graphlet_alpha,
            "graphlet_mass_ab": graphlet_mass_ab,
        }

    def endpoint_probabilities(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temperature = max(float(temperature), 1.0e-8)
        node_probabilities = torch.softmax(
            outputs["node_logits"] / temperature,
            dim=-1,
        )
        edge_probabilities = torch.softmax(
            outputs["edge_logits"] / temperature,
            dim=-1,
        )
        edge_probabilities = 0.5 * (
            edge_probabilities + edge_probabilities.transpose(1, 2)
        )
        return node_probabilities, edge_probabilities

    def graphlet_means(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = outputs["graphlet_alpha"]
        blocks = []
        for start, stop in self.graphlet_slices:
            block = alpha[:, start:stop]
            blocks.append(block / block.sum(dim=-1, keepdim=True).clamp_min(1.0e-12))
        means = (
            torch.cat(blocks, dim=-1)
            if blocks
            else alpha.new_zeros((alpha.shape[0], 0))
        )
        mass_ab = outputs["graphlet_mass_ab"]
        mass_mean = mass_ab[..., 0] / mass_ab.sum(dim=-1).clamp_min(1.0e-12)
        return means, mass_mean

    @staticmethod
    def _upper_pair_mask(pair_mask: torch.Tensor) -> torch.Tensor:
        _, node_count, _ = pair_mask.shape
        upper = torch.triu(
            torch.ones(
                (node_count, node_count),
                dtype=torch.bool,
                device=pair_mask.device,
            ),
            diagonal=1,
        )
        return pair_mask.bool() & upper.view(1, node_count, node_count)

    def loss(
        self,
        batch: HybridEndpointBatch,
        *,
        loss_weights: dict[str, float] | None = None,
        edge_class_weights: list[float] | None = None,
        target_epsilon: float = 1.0e-5,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        weights = loss_weights or {}
        outputs = self.forward(batch)

        if self.num_node_categories > 1:
            node_loss = torch.nn.functional.cross_entropy(
                outputs["node_logits"][batch.node_mask.bool()],
                batch.target_node_labels[batch.node_mask.bool()],
            )
        else:
            node_loss = outputs["node_logits"].sum() * 0.0

        upper = self._upper_pair_mask(batch.pair_mask)
        class_weight_tensor = None
        if edge_class_weights is not None:
            if len(edge_class_weights) != self.num_edge_categories:
                raise ValueError(
                    "edge_class_weights must have one value per edge category."
                )
            class_weight_tensor = torch.as_tensor(
                edge_class_weights,
                dtype=outputs["edge_logits"].dtype,
                device=outputs["edge_logits"].device,
            )
        edge_loss = torch.nn.functional.cross_entropy(
            outputs["edge_logits"][upper],
            batch.target_edge_labels[upper],
            weight=class_weight_tensor,
        )

        graphlet_dirichlet_terms: list[torch.Tensor] = []
        graphlet_mean_terms: list[torch.Tensor] = []
        graphlet_valid_masks: list[torch.Tensor] = []
        for start, stop in self.graphlet_slices:
            target = batch.graphlet_target[:, start:stop]
            valid = target.sum(dim=-1) > 0.0
            graphlet_valid_masks.append(valid)
            if not torch.any(valid):
                continue
            smoothed = target[valid] + float(target_epsilon)
            smoothed = smoothed / smoothed.sum(dim=-1, keepdim=True)
            alpha = outputs["graphlet_alpha"][valid, start:stop]
            distribution = torch.distributions.Dirichlet(alpha)
            graphlet_dirichlet_terms.append(-distribution.log_prob(smoothed).mean())
            mean = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
            graphlet_mean_terms.append(
                -(smoothed * torch.log(mean.clamp_min(1.0e-12))).sum(dim=-1).mean()
            )
        zero = outputs["edge_logits"].sum() * 0.0
        graphlet_dirichlet_loss = (
            torch.stack(graphlet_dirichlet_terms).mean()
            if graphlet_dirichlet_terms
            else zero
        )
        graphlet_mean_loss = (
            torch.stack(graphlet_mean_terms).mean() if graphlet_mean_terms else zero
        )

        mass_terms: list[torch.Tensor] = []
        for block_index, valid in enumerate(graphlet_valid_masks):
            if not torch.any(valid):
                continue
            target_mass = batch.graphlet_mass_target[valid, block_index].clamp(
                min=float(target_epsilon),
                max=1.0 - float(target_epsilon),
            )
            ab = outputs["graphlet_mass_ab"][valid, block_index]
            distribution = torch.distributions.Beta(ab[:, 0], ab[:, 1])
            mass_terms.append(-distribution.log_prob(target_mass).mean())
        graphlet_mass_loss = torch.stack(mass_terms).mean() if mass_terms else zero

        total = (
            float(weights.get("node", 1.0)) * node_loss
            + float(weights.get("edge", 1.0)) * edge_loss
            + float(weights.get("graphlet_mean", weights.get("graphlet", 1.0)))
            * graphlet_mean_loss
            + float(weights.get("graphlet_distribution", 0.1)) * graphlet_dirichlet_loss
            + float(weights.get("graphlet_mass", 0.25)) * graphlet_mass_loss
        )

        with torch.no_grad():
            edge_prediction = torch.argmax(outputs["edge_logits"], dim=-1)
            edge_accuracy = (
                (edge_prediction[upper] == batch.target_edge_labels[upper])
                .float()
                .mean()
            )
            present = batch.target_edge_labels[upper] > 0
            present_recall = (
                (edge_prediction[upper][present] > 0).float().mean()
                if torch.any(present)
                else edge_accuracy.new_tensor(1.0)
            )
            graphlet_mean, mass_mean = self.graphlet_means(outputs)
            graphlet_mae = (
                torch.abs(graphlet_mean - batch.graphlet_target).mean()
                if graphlet_mean.numel()
                else edge_accuracy.new_tensor(0.0)
            )
            graphlet_mass_mae = (
                torch.abs(mass_mean - batch.graphlet_mass_target).mean()
                if mass_mean.numel()
                else edge_accuracy.new_tensor(0.0)
            )
        metrics = {
            "loss": float(total.detach().cpu()),
            "node_loss": float(node_loss.detach().cpu()),
            "edge_loss": float(edge_loss.detach().cpu()),
            "graphlet_mean_loss": float(graphlet_mean_loss.detach().cpu()),
            "graphlet_distribution_loss": float(graphlet_dirichlet_loss.detach().cpu()),
            "graphlet_mass_loss": float(graphlet_mass_loss.detach().cpu()),
            "edge_accuracy": float(edge_accuracy.detach().cpu()),
            "present_edge_recall": float(present_recall.detach().cpu()),
            "graphlet_mae": float(graphlet_mae.detach().cpu()),
            "graphlet_mass_mae": float(graphlet_mass_mae.detach().cpu()),
        }
        return total, metrics

    def model_config(self) -> dict[str, Any]:
        return {
            "num_node_categories": self.num_node_categories,
            "num_edge_categories": self.num_edge_categories,
            "graphlet_slices": [list(value) for value in self.graphlet_slices],
            "hidden_dim": self.hidden_dim,
            "edge_dim": self.edge_dim,
            "graph_dim": self.graph_dim,
            "num_layers": len(self.layers),
            "dropout": self.dropout_p,
            "min_concentration": self.min_concentration,
            "max_concentration": self.max_concentration,
        }


def save_hybrid_endpoint_checkpoint(
    model: HybridEndpointPredictor,
    path: str | Path,
    *,
    vocabulary: GraphCategoryVocabulary,
    graphlet_basis: GraphletBasis,
    summary_config: SummaryConfig,
    config: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "model_state_dict": model.state_dict(),
            "model_config": model.model_config(),
            "vocabulary": vocabulary.to_dict(),
            "graphlet_basis": graphlet_basis.to_dict(),
            "summary_config": {
                key: value for key, value in summary_config.__dict__.items()
            },
            "config": config or {},
            "report": report or {},
        },
        path,
    )


def load_hybrid_endpoint_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "auto",
) -> tuple[
    HybridEndpointPredictor,
    GraphCategoryVocabulary,
    GraphletBasis,
    SummaryConfig,
    dict[str, Any],
]:
    resolved_device = (
        resolve_torch_device(device) if isinstance(device, str) else device
    )
    checkpoint = torch.load(Path(path), map_location=resolved_device)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            "Checkpoint is not a hybrid endpoint+graphlet predictor "
            f"({CHECKPOINT_FORMAT})."
        )
    vocabulary = GraphCategoryVocabulary.from_dict(checkpoint["vocabulary"])
    graphlet_basis = GraphletBasis.from_dict(checkpoint["graphlet_basis"])
    model_cfg = dict(checkpoint["model_config"])
    expected_slices = tuple(graphlet_basis.slices)
    stored_slices = tuple(
        (int(value[0]), int(value[1])) for value in model_cfg.get("graphlet_slices", [])
    )
    if stored_slices != expected_slices:
        raise ValueError(
            "Checkpoint graphlet coordinates do not match its graphlet basis."
        )
    if int(model_cfg["num_node_categories"]) != vocabulary.num_node_categories:
        raise ValueError("Checkpoint node vocabulary/model mismatch.")
    if int(model_cfg["num_edge_categories"]) != vocabulary.num_edge_categories:
        raise ValueError("Checkpoint edge vocabulary/model mismatch.")
    model = HybridEndpointPredictor(**model_cfg).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    summary_config = SummaryConfig.from_dict(checkpoint.get("summary_config", {}) or {})
    return model, vocabulary, graphlet_basis, summary_config, checkpoint
