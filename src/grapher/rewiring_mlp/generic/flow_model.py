from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from grapher.properties.summary import SummaryConfig
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.flow_data import TopologyFlowGraphletBatch
from grapher.rewiring_mlp.generic.layers import TopologyMPNNLayer
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir


TOPOLOGY_FLOW_GRAPHLET_CHECKPOINT_FORMAT = "topology_flow_graphlet_predictor_v1"


class TopologyFlowGraphletPredictor(nn.Module):
    """Joint degree-tangent edge-flow and clean-graphlet estimator.

    The model consumes a continuous soft adjacency state P_t together with the
    fixed source adjacency and normalized time. A shared permutation-equivariant
    MPNN produces node/pair states and a permutation-invariant graph context.

    The graphlet branch predicts clean per-order graphlet simplexes. Its logits
    are embedded back into a global graphlet context used by the pair decoder,
    so the edge velocity is explicitly conditioned on the predicted higher-order
    destination rather than being an unrelated auxiliary head.

    The raw symmetric pair velocity is projected analytically onto the tangent
    space

        sum_j V_ij = 0  for every valid node i,

    matching the invariant preserved by double-edge swaps.
    """

    def __init__(
        self,
        *,
        graphlet_block_widths: Sequence[int],
        hidden_dim: int = 128,
        edge_dim: int = 64,
        graph_dim: int = 128,
        num_layers: int = 4,
        graphlet_dim: int = 256,
        graphlet_context_dim: int = 128,
        pair_dim: int = 256,
        dropout: float = 0.05,
        graphlet_dropout: float | None = None,
        project_degree_tangent: bool = True,
        flow_changed_pair_weight: float = 4.0,
    ) -> None:
        super().__init__()
        self.graphlet_block_widths = tuple(int(value) for value in graphlet_block_widths)
        if not self.graphlet_block_widths or any(value <= 1 for value in self.graphlet_block_widths):
            raise ValueError("graphlet_block_widths must contain simplex widths >= 2.")
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.graph_dim = int(graph_dim)
        self.num_layers = int(num_layers)
        self.graphlet_dim = int(graphlet_dim)
        self.graphlet_context_dim = int(graphlet_context_dim)
        self.pair_dim = int(pair_dim)
        self.dropout_p = float(dropout)
        self.graphlet_dropout_p = float(
            dropout if graphlet_dropout is None else graphlet_dropout
        )
        self.project_degree_tangent = bool(project_degree_tangent)
        self.flow_changed_pair_weight = float(flow_changed_pair_weight)
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if min(
            self.hidden_dim,
            self.edge_dim,
            self.graph_dim,
            self.graphlet_dim,
            self.graphlet_context_dim,
            self.pair_dim,
        ) <= 0:
            raise ValueError("All model dimensions must be positive.")
        if self.flow_changed_pair_weight <= 0.0:
            raise ValueError("flow_changed_pair_weight must be positive.")

        # Node features: normalized indexed degree, graph size, time, valid mask.
        self.node_in = nn.Linear(4, self.hidden_dim)
        # Pair features: current probability, source edge, pair mask, time,
        # average degree, degree difference, degree product, graph size.
        self.edge_in = nn.Linear(8, self.edge_dim)
        self.layers = nn.ModuleList(
            [TopologyMPNNLayer(self.hidden_dim, self.edge_dim) for _ in range(self.num_layers)]
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.graph_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim + 2 * self.edge_dim + 5, self.graph_dim),
            nn.SiLU(),
            nn.Linear(self.graph_dim, self.graph_dim),
            nn.SiLU(),
        )

        self.graphlet_slices = self._make_slices(self.graphlet_block_widths)
        self.graphlet_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.graph_dim, self.graphlet_dim),
                    nn.SiLU(),
                    nn.Dropout(self.graphlet_dropout_p),
                    nn.Linear(self.graphlet_dim, self.graphlet_dim),
                    nn.SiLU(),
                    nn.Linear(self.graphlet_dim, width),
                )
                for width in self.graphlet_block_widths
            ]
        )
        self.graphlet_to_flow = nn.Sequential(
            nn.Linear(sum(self.graphlet_block_widths), self.graphlet_context_dim),
            nn.SiLU(),
            nn.Linear(self.graphlet_context_dim, self.graphlet_context_dim),
            nn.SiLU(),
        )

        pair_scalar_width = 7
        pair_input_width = (
            2 * self.hidden_dim
            + self.edge_dim
            + self.graphlet_context_dim
            + pair_scalar_width
        )
        self.flow_pair_head = nn.Sequential(
            nn.Linear(pair_input_width, self.pair_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.pair_dim, self.pair_dim),
            nn.SiLU(),
            nn.Linear(self.pair_dim, 1),
        )

    @staticmethod
    def _make_slices(widths: Sequence[int]) -> tuple[tuple[int, int], ...]:
        result: list[tuple[int, int]] = []
        start = 0
        for width in widths:
            stop = start + int(width)
            result.append((start, stop))
            start = stop
        return tuple(result)

    @property
    def graphlet_width(self) -> int:
        return sum(self.graphlet_block_widths)

    @staticmethod
    def _weighted_pair_pool(
        values: torch.Tensor,
        pair_mask: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        valid = pair_mask.to(values.dtype) * weights.to(values.dtype)
        expanded = valid.unsqueeze(-1)
        return (values * expanded).sum(dim=(1, 2)) / expanded.sum(
            dim=(1, 2)
        ).clamp_min(1.0e-8)

    def _encode(
        self,
        batch: TopologyFlowGraphletBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = batch.current_edge_probabilities
        source = batch.source_adjacency
        node_mask = batch.node_mask.bool()
        pair_mask = batch.pair_mask.bool()
        batch_size, node_count = node_mask.shape
        dtype = current.dtype
        size_feature = batch.graph_size / (batch.graph_size + 1.0).clamp_min(1.0)

        node_features = torch.stack(
            [
                batch.degrees,
                size_feature.view(-1, 1).expand(batch_size, node_count),
                batch.time.view(-1, 1).expand(batch_size, node_count),
                node_mask.to(dtype),
            ],
            dim=-1,
        )
        node_hidden = self.node_in(node_features)
        node_hidden = node_hidden * node_mask.unsqueeze(-1).to(node_hidden.dtype)

        degree_i = batch.degrees.unsqueeze(2).expand(batch_size, node_count, node_count)
        degree_j = batch.degrees.unsqueeze(1).expand(batch_size, node_count, node_count)
        edge_features = torch.stack(
            [
                current,
                source,
                pair_mask.to(dtype),
                batch.time.view(-1, 1, 1).expand(batch_size, node_count, node_count),
                0.5 * (degree_i + degree_j),
                torch.abs(degree_i - degree_j),
                degree_i * degree_j,
                size_feature.view(-1, 1, 1).expand(batch_size, node_count, node_count),
            ],
            dim=-1,
        )
        edge_hidden = self.edge_in(edge_features)
        edge_hidden = 0.5 * (edge_hidden + edge_hidden.transpose(1, 2))
        edge_hidden = edge_hidden * pair_mask.unsqueeze(-1).to(edge_hidden.dtype)

        # Continuous message passing: P_ij weights the pair message. This is the
        # direct soft-graph analogue of binary topology message passing.
        for layer in self.layers:
            node_hidden, edge_hidden = layer(
                node_hidden,
                edge_hidden,
                current,
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
                (node_count, node_count), dtype=torch.bool, device=node_mask.device
            ),
            diagonal=1,
        ).view(1, node_count, node_count)
        upper_pairs = pair_mask & upper
        present_pool = self._weighted_pair_pool(
            edge_hidden,
            upper_pairs,
            current.clamp_min(0.0),
        )
        absent_pool = self._weighted_pair_pool(
            edge_hidden,
            upper_pairs,
            (1.0 - current).clamp_min(0.0),
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
        return node_hidden, edge_hidden, graph_hidden

    def _graphlet_outputs(
        self,
        batch: TopologyFlowGraphletBatch,
        graph_hidden: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        coordinate_mask = batch.graphlet_coordinate_mask.bool()
        if coordinate_mask.shape[1] != self.graphlet_width:
            raise ValueError(
                f"Expected {self.graphlet_width} graphlet coordinates, "
                f"received {coordinate_mask.shape[1]}."
            )
        logit_blocks: list[torch.Tensor] = []
        probability_blocks: list[torch.Tensor] = []
        for (start, stop), head in zip(self.graphlet_slices, self.graphlet_heads):
            block_mask = coordinate_mask[:, start:stop]
            block_valid = block_mask.any(dim=1, keepdim=True)
            logits = head(graph_hidden)
            logits = logits - logits.mean(dim=-1, keepdim=True)
            logits = logits * block_valid.to(logits.dtype)
            probability = torch.softmax(logits, dim=-1) * block_valid.to(logits.dtype)
            logit_blocks.append(logits)
            probability_blocks.append(probability)
        logits = torch.cat(logit_blocks, dim=-1)
        probabilities = torch.cat(probability_blocks, dim=-1)
        return {
            "clean_graphlet_logits": logits,
            "clean_graphlet_probabilities": probabilities,
            "graphlet_mask": coordinate_mask,
        }

    @staticmethod
    def project_symmetric_degree_tangent(
        raw_velocity: torch.Tensor,
        node_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Orthogonally remove node-wise degree drift from a symmetric velocity.

        For a graph with n>2, write V_ij = R_ij - a_i - a_j (i != j).
        Solving row_sum(V)=0 gives

          S = sum_i row_sum(R)_i / (2(n-1)),
          a_i = (row_sum(R)_i - S) / (n-2).

        The n<=2 degree-preserving tangent space is zero-dimensional.
        """

        mask = pair_mask.bool()
        node_valid = node_mask.bool()
        raw = 0.5 * (raw_velocity + raw_velocity.transpose(1, 2))
        raw = raw * mask.to(raw.dtype)
        n = node_valid.sum(dim=1).to(raw.dtype)
        row_sum = raw.sum(dim=2)
        denom_s = (2.0 * (n - 1.0)).clamp_min(1.0)
        total_potential = row_sum.sum(dim=1) / denom_s
        denom_a = (n - 2.0).clamp_min(1.0)
        potential = (row_sum - total_potential.unsqueeze(1)) / denom_a.unsqueeze(1)
        projected = raw - potential.unsqueeze(2) - potential.unsqueeze(1)
        projected = projected * mask.to(projected.dtype)
        active = (n > 2.0).view(-1, 1, 1)
        projected = torch.where(active, projected, torch.zeros_like(projected))
        projected = 0.5 * (projected + projected.transpose(1, 2))
        return projected * mask.to(projected.dtype)

    def _flow_outputs(
        self,
        batch: TopologyFlowGraphletBatch,
        node_hidden: torch.Tensor,
        edge_hidden: torch.Tensor,
        graphlet_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, node_count, hidden_dim = node_hidden.shape
        left = node_hidden.unsqueeze(2).expand(
            batch_size, node_count, node_count, hidden_dim
        )
        right = node_hidden.unsqueeze(1).expand(
            batch_size, node_count, node_count, hidden_dim
        )
        node_sum = left + right
        node_difference = torch.abs(left - right)
        graphlet_context = self.graphlet_to_flow(graphlet_logits)
        graphlet_pair = graphlet_context.view(
            batch_size, 1, 1, self.graphlet_context_dim
        ).expand(batch_size, node_count, node_count, self.graphlet_context_dim)

        degree_i = batch.degrees.unsqueeze(2).expand(batch_size, node_count, node_count)
        degree_j = batch.degrees.unsqueeze(1).expand(batch_size, node_count, node_count)
        size_feature = batch.graph_size / (batch.graph_size + 1.0).clamp_min(1.0)
        scalars = torch.stack(
            [
                batch.current_edge_probabilities,
                batch.source_adjacency,
                batch.time.view(-1, 1, 1).expand(batch_size, node_count, node_count),
                0.5 * (degree_i + degree_j),
                torch.abs(degree_i - degree_j),
                degree_i * degree_j,
                size_feature.view(-1, 1, 1).expand(batch_size, node_count, node_count),
            ],
            dim=-1,
        )
        pair_features = torch.cat(
            [node_sum, node_difference, edge_hidden, graphlet_pair, scalars], dim=-1
        )
        raw_velocity = self.flow_pair_head(pair_features).squeeze(-1)
        raw_velocity = 0.5 * (raw_velocity + raw_velocity.transpose(1, 2))
        raw_velocity = raw_velocity * batch.pair_mask.to(raw_velocity.dtype)
        if self.project_degree_tangent:
            velocity = self.project_symmetric_degree_tangent(
                raw_velocity,
                batch.node_mask,
                batch.pair_mask,
            )
        else:
            velocity = raw_velocity
        return {
            "flow_velocity": velocity,
            "raw_flow_velocity": raw_velocity,
            "graphlet_flow_context": graphlet_context,
        }

    def forward(self, batch: TopologyFlowGraphletBatch) -> dict[str, torch.Tensor]:
        node_hidden, edge_hidden, graph_hidden = self._encode(batch)
        graphlet_outputs = self._graphlet_outputs(batch, graph_hidden)
        flow_outputs = self._flow_outputs(
            batch,
            node_hidden,
            edge_hidden,
            graphlet_outputs["clean_graphlet_logits"],
        )
        return {**graphlet_outputs, **flow_outputs}

    def loss(
        self,
        batch: TopologyFlowGraphletBatch,
        *,
        loss_weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        weights = dict(loss_weights or {})
        outputs = self.forward(batch)
        predicted_flow = outputs["flow_velocity"]
        target_flow = batch.flow_target
        pair_mask = batch.pair_mask.bool()
        node_count = pair_mask.shape[1]
        upper = torch.triu(
            torch.ones(
                (node_count, node_count), dtype=torch.bool, device=pair_mask.device
            ),
            diagonal=1,
        ).view(1, node_count, node_count)
        valid = pair_mask & upper
        changed = valid & (torch.abs(target_flow) > 0.5)
        pair_weight = valid.to(predicted_flow.dtype)
        changed_weight = float(
            weights.get("flow_changed_pair_weight", self.flow_changed_pair_weight)
        )
        pair_weight = pair_weight * (
            1.0 + (changed_weight - 1.0) * changed.to(predicted_flow.dtype)
        )
        flow_count = pair_weight.sum().clamp_min(1.0)
        flow_element = F.smooth_l1_loss(
            predicted_flow,
            target_flow,
            reduction="none",
        )
        flow_loss = (flow_element * pair_weight).sum() / flow_count

        target_logits = batch.clean_graphlet_logits_target
        target_probabilities = batch.clean_graphlet_probabilities_target
        predicted_logits = outputs["clean_graphlet_logits"]
        predicted_probabilities = outputs["clean_graphlet_probabilities"]
        graphlet_mask = batch.graphlet_coordinate_mask.bool()
        mask_weight = graphlet_mask.to(predicted_logits.dtype)
        coordinate_count = mask_weight.sum().clamp_min(1.0)
        graphlet_logit_loss = (
            F.smooth_l1_loss(
                predicted_logits,
                target_logits,
                reduction="none",
            )
            * mask_weight
        ).sum() / coordinate_count

        probability_terms: list[torch.Tensor] = []
        probability_mae_terms: list[torch.Tensor] = []
        for start, stop in self.graphlet_slices:
            block_mask = graphlet_mask[:, start:stop]
            valid_rows = block_mask.any(dim=1)
            if not torch.any(valid_rows):
                continue
            pred = predicted_probabilities[valid_rows, start:stop].clamp_min(1.0e-8)
            target = target_probabilities[valid_rows, start:stop].clamp_min(0.0)
            target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            target_safe = target.clamp_min(1.0e-8)
            probability_terms.append(
                (target * (torch.log(target_safe) - torch.log(pred))).sum(dim=-1).mean()
            )
            probability_mae_terms.append(torch.abs(pred - target).mean())
        graphlet_probability_loss = (
            torch.stack(probability_terms).mean()
            if probability_terms
            else predicted_logits.sum() * 0.0
        )
        graphlet_probability_mae = (
            torch.stack(probability_mae_terms).mean()
            if probability_mae_terms
            else predicted_logits.sum() * 0.0
        )

        total = (
            float(weights.get("flow", 1.0)) * flow_loss
            + float(weights.get("graphlet_logit", 1.0)) * graphlet_logit_loss
            + float(weights.get("graphlet_probability", 0.25))
            * graphlet_probability_loss
        )

        with torch.no_grad():
            valid_float = valid.to(predicted_flow.dtype)
            valid_count = valid_float.sum().clamp_min(1.0)
            delta = (predicted_flow - target_flow) * valid_float
            flow_mae = torch.abs(delta).sum() / valid_count
            flow_rmse = torch.sqrt(delta.square().sum() / valid_count)
            changed_float = changed.to(predicted_flow.dtype)
            changed_count = changed_float.sum().clamp_min(1.0)
            changed_delta = (predicted_flow - target_flow) * changed_float
            changed_rmse = torch.sqrt(changed_delta.square().sum() / changed_count)
            sign_correct = (
                (torch.sign(predicted_flow) == torch.sign(target_flow)).to(predicted_flow.dtype)
                * changed_float
            ).sum() / changed_count
            degree_residual = (
                predicted_flow.sum(dim=2) * batch.node_mask.to(predicted_flow.dtype)
            )
            degree_tangent_mae = torch.abs(degree_residual).sum() / batch.node_mask.to(
                predicted_flow.dtype
            ).sum().clamp_min(1.0)
            graphlet_logit_rmse = torch.sqrt(
                ((predicted_logits - target_logits).square() * mask_weight).sum()
                / coordinate_count
            )
            graphlet_logit_mae = (
                torch.abs(predicted_logits - target_logits) * mask_weight
            ).sum() / coordinate_count

        metrics = {
            "loss": float(total.detach().cpu()),
            "flow_loss": float(flow_loss.detach().cpu()),
            "flow_mae": float(flow_mae.detach().cpu()),
            "flow_rmse": float(flow_rmse.detach().cpu()),
            "flow_changed_rmse": float(changed_rmse.detach().cpu()),
            "flow_changed_sign_accuracy": float(sign_correct.detach().cpu()),
            "flow_degree_tangent_mae": float(degree_tangent_mae.detach().cpu()),
            "graphlet_logit_loss": float(graphlet_logit_loss.detach().cpu()),
            "graphlet_probability_loss": float(
                graphlet_probability_loss.detach().cpu()
            ),
            "graphlet_logit_rmse": float(graphlet_logit_rmse.detach().cpu()),
            "graphlet_logit_mae": float(graphlet_logit_mae.detach().cpu()),
            "graphlet_probability_mae": float(
                graphlet_probability_mae.detach().cpu()
            ),
        }
        return total, metrics

    def model_config(self) -> dict[str, Any]:
        return {
            "graphlet_block_widths": list(self.graphlet_block_widths),
            "hidden_dim": self.hidden_dim,
            "edge_dim": self.edge_dim,
            "graph_dim": self.graph_dim,
            "num_layers": self.num_layers,
            "graphlet_dim": self.graphlet_dim,
            "graphlet_context_dim": self.graphlet_context_dim,
            "pair_dim": self.pair_dim,
            "dropout": self.dropout_p,
            "graphlet_dropout": self.graphlet_dropout_p,
            "project_degree_tangent": self.project_degree_tangent,
            "flow_changed_pair_weight": self.flow_changed_pair_weight,
        }


def save_topology_flow_graphlet_checkpoint(
    model: TopologyFlowGraphletPredictor,
    path: str | Path,
    *,
    graphlet_basis: TopologyGraphletBasis,
    summary_config: SummaryConfig | None = None,
    config: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if tuple(graphlet_basis.simplex_block_widths) != tuple(model.graphlet_block_widths):
        raise ValueError("Graphlet basis simplex widths do not match the flow predictor.")
    torch.save(
        {
            "format": TOPOLOGY_FLOW_GRAPHLET_CHECKPOINT_FORMAT,
            "pipeline_mode": "topology",
            "guidance_mode": "flow_graphlet",
            "predictor_type": "flow_graphlet",
            "model_state_dict": model.state_dict(),
            "model_config": model.model_config(),
            "graphlet_basis": graphlet_basis.to_dict(),
            "summary_config": (
                dict(summary_config.__dict__) if summary_config is not None else {}
            ),
            "time_parameterization": "normalized_flow_progress_0_source_1_clean",
            "config": config or {},
            "report": report or {},
        },
        path,
    )


def load_topology_flow_graphlet_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "auto",
) -> tuple[
    TopologyFlowGraphletPredictor,
    TopologyGraphletBasis,
    SummaryConfig,
    dict[str, Any],
]:
    resolved_device = resolve_torch_device(device) if isinstance(device, str) else device
    checkpoint = torch.load(Path(path), map_location=resolved_device)
    if checkpoint.get("format") != TOPOLOGY_FLOW_GRAPHLET_CHECKPOINT_FORMAT:
        raise ValueError(
            "Checkpoint is not a topology flow+graphlet predictor "
            f"({TOPOLOGY_FLOW_GRAPHLET_CHECKPOINT_FORMAT})."
        )
    basis = TopologyGraphletBasis.from_dict(
        dict(checkpoint.get("graphlet_basis", {}) or {})
    )
    model = TopologyFlowGraphletPredictor(
        **dict(checkpoint.get("model_config", {}) or {})
    ).to(resolved_device)
    if tuple(model.graphlet_block_widths) != tuple(basis.simplex_block_widths):
        raise ValueError("Checkpoint graphlet model width does not match its stored basis.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    summary_config = SummaryConfig.from_dict(
        checkpoint.get("summary_config", {}) or {}
    )
    return model, basis, summary_config, checkpoint
