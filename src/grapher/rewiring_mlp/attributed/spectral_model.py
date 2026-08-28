from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from grapher.properties.summary import SummaryConfig
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary, GraphletBasis
from grapher.rewiring_mlp.attributed.layers import EdgeAwareMPNNLayer
from grapher.rewiring_mlp.attributed.spectral_data import AttributedSpectralBatch
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir

ATTRIBUTED_SPECTRAL_GRAPHLET_CHECKPOINT_FORMAT = (
    "attributed_spectral_graphlet_transformer_v1"
)


class AttributedSpectralGraphletTransformerPredictor(nn.Module):
    """Joint clean dual-spectrum and attributed-graphlet CLR predictor.

    The fixed typed source graph supplies categorical/invariant context. The
    current diffusion state is continuous: two variable-length spectrum
    channels and fixed-width attributed graphlet CLR blocks. All clean endpoint
    summaries are predicted in one forward pass.
    """

    def __init__(
        self,
        *,
        num_node_categories: int,
        num_edge_categories: int,
        graphlet_block_widths: Sequence[int],
        hidden_dim: int = 128,
        edge_dim: int = 64,
        graph_dim: int = 192,
        num_layers: int = 4,
        spectral_dim: int = 192,
        spectral_layers: int = 4,
        spectral_heads: int = 6,
        spectral_ff_dim: int = 512,
        graphlet_dim: int = 384,
        dropout: float = 0.05,
        graphlet_dropout: float = 0.05,
        min_gap: float = 1.0e-6,
        input_normalization: str = "mean_degree",
        graphlet_logit_epsilon: float = 1.0e-4,
    ) -> None:
        super().__init__()
        self.num_node_categories = int(num_node_categories)
        self.num_edge_categories = int(num_edge_categories)
        self.graphlet_block_widths = tuple(int(value) for value in graphlet_block_widths)
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.graph_dim = int(graph_dim)
        self.num_layers = int(num_layers)
        self.spectral_dim = int(spectral_dim)
        self.spectral_layers = int(spectral_layers)
        self.spectral_heads = int(spectral_heads)
        self.spectral_ff_dim = int(spectral_ff_dim)
        self.graphlet_dim = int(graphlet_dim)
        self.dropout_p = float(dropout)
        self.graphlet_dropout_p = float(graphlet_dropout)
        self.min_gap = float(min_gap)
        self.input_normalization = str(input_normalization).lower()
        self.graphlet_logit_epsilon = float(graphlet_logit_epsilon)
        if self.num_node_categories <= 0 or self.num_edge_categories < 2:
            raise ValueError("Attributed model requires node categories and no-edge + edge categories.")
        if not self.graphlet_block_widths or any(width <= 1 for width in self.graphlet_block_widths):
            raise ValueError("graphlet_block_widths must contain simplex widths >= 2.")
        if self.spectral_dim % self.spectral_heads != 0:
            raise ValueError("spectral_dim must be divisible by spectral_heads.")
        if self.input_normalization not in {
            "mean_degree", "average_degree", "avg_degree", "trace", "degree_sum", "none", "raw"
        }:
            raise ValueError("input_normalization must be mean_degree, trace, or none.")

        typed_width = self.num_edge_categories - 1
        self.node_in = nn.Linear(
            self.num_node_categories + typed_width + 4,
            self.hidden_dim,
        )
        self.edge_in = nn.Linear(
            self.num_edge_categories + 6 + 2 * typed_width,
            self.edge_dim,
        )
        self.layers = nn.ModuleList(
            [EdgeAwareMPNNLayer(self.hidden_dim, self.edge_dim) for _ in range(self.num_layers)]
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.graph_encoder = nn.Sequential(
            nn.Linear(
                self.hidden_dim
                + 2 * self.edge_dim
                + self.num_node_categories
                + self.num_edge_categories
                + typed_width
                + 5,
                self.graph_dim,
            ),
            nn.SiLU(),
            nn.Linear(self.graph_dim, self.graph_dim),
            nn.SiLU(),
        )

        # [current lambda, source lambda, rank, progress, size, mask]
        self.spectral_token_in = nn.Sequential(
            nn.Linear(6, self.spectral_dim),
            nn.SiLU(),
            nn.Linear(self.spectral_dim, self.spectral_dim),
        )
        self.channel_embedding = nn.Embedding(2, self.spectral_dim)
        self.graph_to_spectral = nn.Linear(self.graph_dim, self.spectral_dim)
        spectral_layer = nn.TransformerEncoderLayer(
            d_model=self.spectral_dim,
            nhead=self.spectral_heads,
            dim_feedforward=self.spectral_ff_dim,
            dropout=self.dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spectral_transformer = nn.TransformerEncoder(
            spectral_layer,
            num_layers=self.spectral_layers,
            enable_nested_tensor=False,
        )
        self.spectral_norm = nn.LayerNorm(self.spectral_dim)
        self.gap_head = nn.Sequential(
            nn.Linear(self.spectral_dim, self.spectral_dim),
            nn.SiLU(),
            nn.Linear(self.spectral_dim, 1),
        )

        self.graphlet_slices = self._make_slices(self.graphlet_block_widths)
        self.graphlet_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.graph_dim + 2 * width, self.graphlet_dim),
                    nn.SiLU(),
                    nn.Dropout(self.graphlet_dropout_p),
                    nn.Linear(self.graphlet_dim, self.graphlet_dim),
                    nn.SiLU(),
                    nn.Linear(self.graphlet_dim, width),
                )
                for width in self.graphlet_block_widths
            ]
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
    def _masked_pair_pool(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.unsqueeze(-1).to(values.dtype)
        return (values * weight).sum(dim=(1, 2)) / weight.sum(dim=(1, 2)).clamp_min(1.0)

    def _graph_context(self, batch: AttributedSpectralBatch) -> torch.Tensor:
        node_mask = batch.node_mask.bool()
        pair_mask = batch.pair_mask.bool()
        node_labels = batch.source_node_labels.clamp(0, self.num_node_categories - 1)
        edge_labels = batch.source_edge_labels.clamp(0, self.num_edge_categories - 1)
        adjacency = (edge_labels > 0) & pair_mask
        batch_size, node_count = node_labels.shape
        typed_width = self.num_edge_categories - 1
        typed_scale = (batch.graph_size - 1.0).clamp_min(1.0).view(-1, 1, 1)
        typed = batch.typed_degrees / typed_scale
        size_feature = batch.graph_size / (batch.graph_size + 1.0).clamp_min(1.0)

        node_onehot = F.one_hot(node_labels, self.num_node_categories).float()
        node_features = torch.cat(
            [
                node_onehot,
                batch.degrees.unsqueeze(-1),
                typed,
                size_feature.view(batch_size, 1, 1).expand(batch_size, node_count, 1),
                batch.time.view(batch_size, 1, 1).expand(batch_size, node_count, 1),
                node_mask.unsqueeze(-1).float(),
            ],
            dim=-1,
        )
        node_hidden = self.node_in(node_features) * node_mask.unsqueeze(-1).float()

        edge_onehot = F.one_hot(edge_labels, self.num_edge_categories).float()
        degree_i = batch.degrees.unsqueeze(2).expand(batch_size, node_count, node_count)
        degree_j = batch.degrees.unsqueeze(1).expand(batch_size, node_count, node_count)
        typed_i = typed.unsqueeze(2).expand(batch_size, node_count, node_count, typed_width)
        typed_j = typed.unsqueeze(1).expand(batch_size, node_count, node_count, typed_width)
        edge_features = torch.cat(
            [
                edge_onehot,
                adjacency.unsqueeze(-1).float(),
                pair_mask.unsqueeze(-1).float(),
                batch.time.view(batch_size, 1, 1, 1).expand(batch_size, node_count, node_count, 1),
                (0.5 * (degree_i + degree_j)).unsqueeze(-1),
                torch.abs(degree_i - degree_j).unsqueeze(-1),
                typed_i + typed_j,
                torch.abs(typed_i - typed_j),
                size_feature.view(batch_size, 1, 1, 1).expand(batch_size, node_count, node_count, 1),
            ],
            dim=-1,
        )
        edge_hidden = self.edge_in(edge_features)
        edge_hidden = 0.5 * (edge_hidden + edge_hidden.transpose(1, 2))
        edge_hidden = edge_hidden * pair_mask.unsqueeze(-1).float()
        for layer in self.layers:
            node_hidden, edge_hidden = layer(node_hidden, edge_hidden, adjacency, node_mask)
            node_hidden = self.dropout(node_hidden)
            edge_hidden = self.dropout(edge_hidden)

        node_weight = node_mask.unsqueeze(-1).float()
        node_pool = (node_hidden * node_weight).sum(dim=1) / node_weight.sum(dim=1).clamp_min(1.0)
        upper = torch.triu(
            torch.ones((node_count, node_count), dtype=torch.bool, device=node_mask.device),
            diagonal=1,
        ).view(1, node_count, node_count)
        upper_pairs = pair_mask & upper
        present_pool = self._masked_pair_pool(edge_hidden, upper_pairs & adjacency)
        absent_pool = self._masked_pair_pool(edge_hidden, upper_pairs & ~adjacency)
        node_category_pool = (node_onehot * node_weight).sum(dim=1) / node_weight.sum(dim=1).clamp_min(1.0)
        pair_weight = upper_pairs.unsqueeze(-1).float()
        edge_probability_pool = (
            edge_onehot * pair_weight
        ).sum(dim=(1, 2)) / pair_weight.sum(dim=(1, 2)).clamp_min(1.0)
        typed_pool = (typed * node_weight).sum(dim=1) / node_weight.sum(dim=1).clamp_min(1.0)
        degree_weight = node_mask.float()
        degree_count = degree_weight.sum(dim=1).clamp_min(1.0)
        degree_mean = (batch.degrees * degree_weight).sum(dim=1) / degree_count
        degree_var = (
            (batch.degrees - degree_mean.unsqueeze(1)).square() * degree_weight
        ).sum(dim=1) / degree_count
        degree_max = batch.degrees.masked_fill(~node_mask, 0.0).max(dim=1).values
        return self.graph_encoder(
            torch.cat(
                [
                    node_pool,
                    present_pool,
                    absent_pool,
                    node_category_pool,
                    edge_probability_pool,
                    typed_pool,
                    degree_mean.unsqueeze(1),
                    torch.sqrt(degree_var.clamp_min(0.0)).unsqueeze(1),
                    degree_max.unsqueeze(1),
                    batch.time.unsqueeze(1),
                    size_feature.unsqueeze(1),
                ],
                dim=-1,
            )
        )

    def _spectral_trace(self, batch: AttributedSpectralBatch) -> torch.Tensor:
        topology_trace = (batch.source_edge_labels > 0).to(batch.current_spectra.dtype).sum(dim=(1, 2))
        bond_trace = batch.source_edge_weights.to(batch.current_spectra.dtype).sum(dim=(1, 2))
        return torch.stack([topology_trace, bond_trace], dim=1)

    def _spectrum_scale(self, batch: AttributedSpectralBatch) -> torch.Tensor:
        trace = self._spectral_trace(batch)
        if self.input_normalization in {"none", "raw"}:
            return torch.ones_like(trace)
        if self.input_normalization in {"trace", "degree_sum"}:
            return trace.clamp_min(1.0)
        return (trace / batch.graph_size.unsqueeze(1).clamp_min(1.0)).clamp_min(1.0e-8)

    def _constrained_spectrum(
        self,
        raw_scores: torch.Tensor,
        mask: torch.Tensor,
        graph_size: torch.Tensor,
        trace: torch.Tensor,
    ) -> torch.Tensor:
        # raw_scores: [B, 2, N]
        width = raw_scores.shape[-1]
        token_index = torch.arange(width, device=raw_scores.device).view(1, 1, -1)
        gap_mask = mask.unsqueeze(1) & (token_index > 0)
        gaps = (F.softplus(raw_scores) + self.min_gap) * gap_mask.to(raw_scores.dtype)
        multiplicity = (
            graph_size.view(-1, 1, 1) - token_index.to(raw_scores.dtype)
        ).clamp_min(0.0)
        denominator = (gaps * multiplicity).sum(dim=-1)
        scale = torch.where(
            denominator > 0,
            trace / denominator.clamp_min(1.0e-12),
            torch.zeros_like(denominator),
        )
        spectrum = torch.cumsum(gaps * scale.unsqueeze(-1), dim=-1)
        return spectrum * mask.unsqueeze(1).to(spectrum.dtype)

    def _spectral_outputs(
        self,
        batch: AttributedSpectralBatch,
        graph_hidden: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask = batch.spectrum_mask.bool()
        batch_size, width = mask.shape
        scale = self._spectrum_scale(batch)
        current = batch.current_spectra / scale.unsqueeze(-1)
        source = batch.source_spectra / scale.unsqueeze(-1)
        indices = torch.arange(width, device=mask.device, dtype=current.dtype)
        rank = indices.view(1, 1, -1) / (batch.graph_size - 1.0).clamp_min(1.0).view(-1, 1, 1)
        rank = rank.expand(batch_size, 2, width)
        progress = batch.time.view(-1, 1, 1).expand(batch_size, 2, width)
        size_feature = (batch.graph_size / (batch.graph_size + 1.0)).view(-1, 1, 1).expand(batch_size, 2, width)
        token_mask = mask.unsqueeze(1).expand(batch_size, 2, width)
        features = torch.stack(
            [current, source, rank, progress, size_feature, token_mask.to(current.dtype)],
            dim=-1,
        )
        tokens = self.spectral_token_in(features)
        channel_ids = torch.arange(2, device=mask.device).view(1, 2, 1).expand(batch_size, 2, width)
        tokens = tokens + self.channel_embedding(channel_ids)
        tokens = tokens + self.graph_to_spectral(graph_hidden).view(batch_size, 1, 1, -1)
        tokens = tokens.reshape(batch_size, 2 * width, self.spectral_dim)
        flat_mask = token_mask.reshape(batch_size, 2 * width)
        encoded = self.spectral_transformer(tokens, src_key_padding_mask=~flat_mask)
        encoded = self.spectral_norm(encoded).reshape(batch_size, 2, width, self.spectral_dim)
        raw_scores = self.gap_head(encoded).squeeze(-1)
        clean = self._constrained_spectrum(
            raw_scores,
            mask,
            batch.graph_size,
            self._spectral_trace(batch),
        )
        return {
            "clean_spectra": clean,
            "raw_gap_scores": raw_scores,
            "spectrum_mask": mask,
        }

    def _graphlet_outputs(
        self,
        batch: AttributedSpectralBatch,
        graph_hidden: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        current = batch.current_graphlet_logits
        source = batch.source_graphlet_logits
        mask = batch.graphlet_coordinate_mask.bool()
        if current.shape[1] != self.graphlet_width:
            raise ValueError("Graphlet input width does not match the attributed basis.")
        clean_blocks: list[torch.Tensor] = []
        probability_blocks: list[torch.Tensor] = []
        residual_blocks: list[torch.Tensor] = []
        for (start, stop), head in zip(self.graphlet_slices, self.graphlet_heads):
            block_valid = mask[:, start:stop].any(dim=1, keepdim=True)
            residual = head(
                torch.cat(
                    [graph_hidden, current[:, start:stop], source[:, start:stop]],
                    dim=-1,
                )
            )
            clean = current[:, start:stop] + residual
            clean = clean - clean.mean(dim=-1, keepdim=True)
            clean = clean * block_valid.to(clean.dtype)
            probability = torch.softmax(clean, dim=-1) * block_valid.to(clean.dtype)
            clean_blocks.append(clean)
            probability_blocks.append(probability)
            residual_blocks.append(residual * block_valid.to(residual.dtype))
        return {
            "clean_graphlet_logits": torch.cat(clean_blocks, dim=-1),
            "clean_graphlet_probabilities": torch.cat(probability_blocks, dim=-1),
            "raw_graphlet_logit_residual": torch.cat(residual_blocks, dim=-1),
            "graphlet_mask": mask,
        }

    def forward(self, batch: AttributedSpectralBatch) -> dict[str, torch.Tensor]:
        graph_hidden = self._graph_context(batch)
        outputs = self._spectral_outputs(batch, graph_hidden)
        outputs.update(self._graphlet_outputs(batch, graph_hidden))
        return outputs

    def loss(
        self,
        batch: AttributedSpectralBatch,
        *,
        loss_weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        weights = dict(loss_weights or {})
        outputs = self.forward(batch)
        predicted = outputs["clean_spectra"]
        target = batch.clean_spectra_target
        mask = batch.spectrum_mask.bool().unsqueeze(1)
        valid = mask.to(predicted.dtype)
        scale = self._spectrum_scale(batch).unsqueeze(-1)
        normalized_delta = (predicted - target) / scale
        channel_count = valid.sum(dim=-1).clamp_min(1.0)
        channel_loss = (
            F.smooth_l1_loss(predicted / scale, target / scale, reduction="none") * valid
        ).sum(dim=-1) / channel_count
        topology_loss = channel_loss[:, 0].mean()
        bond_loss = channel_loss[:, 1].mean()
        spectral_loss = (
            float(weights.get("topology_spectrum", 1.0)) * topology_loss
            + float(weights.get("bond_spectrum", 1.0)) * bond_loss
        )

        low_k = int(weights.get("low_frequency_k", 0))
        low_loss = predicted.sum() * 0.0
        if low_k > 0:
            index = torch.arange(predicted.shape[-1], device=predicted.device).view(1, 1, -1)
            low_mask = mask & (index > 0) & (index <= low_k)
            low_weight = low_mask.to(predicted.dtype)
            if torch.any(low_mask):
                low_loss = (
                    F.smooth_l1_loss(predicted / scale, target / scale, reduction="none")
                    * low_weight
                ).sum() / low_weight.sum().clamp_min(1.0)
        # Ordinary degree preservation fixes the topology second moment,
        # while the bond-weighted second moment can change under cross-type
        # bond reassignment.  This term is therefore best interpreted as an
        # auxiliary clean-spectrum reconstruction constraint rather than a
        # hard invariant of both channels.  The output decoder still enforces
        # both traces because global bond-type counts are preserved.
        graph_size = batch.graph_size.unsqueeze(1).clamp_min(1.0)
        second_scale = scale.squeeze(-1).square() * graph_size
        predicted_moment2 = (predicted.square() * valid).sum(dim=-1)
        target_moment2 = (target.square() * valid).sum(dim=-1)
        moment2_loss = F.smooth_l1_loss(
            predicted_moment2 / second_scale.clamp_min(1.0e-8),
            target_moment2 / second_scale.clamp_min(1.0e-8),
        )
        spectral_total = (
            float(weights.get("spectrum", 1.0)) * spectral_loss
            + float(weights.get("low_frequency", 0.0)) * low_loss
            + float(weights.get("moment2", 0.0)) * moment2_loss
        )

        predicted_logits = outputs["clean_graphlet_logits"]
        predicted_prob = outputs["clean_graphlet_probabilities"]
        target_logits = batch.clean_graphlet_logits_target
        target_prob = batch.clean_graphlet_probabilities_target
        graphlet_mask = batch.graphlet_coordinate_mask.bool()
        graphlet_weight = graphlet_mask.to(predicted_logits.dtype)
        coordinate_count = graphlet_weight.sum().clamp_min(1.0)
        graphlet_logit_loss = (
            F.smooth_l1_loss(predicted_logits, target_logits, reduction="none")
            * graphlet_weight
        ).sum() / coordinate_count
        kl_terms: list[torch.Tensor] = []
        probability_mae_terms: list[torch.Tensor] = []
        for start, stop in self.graphlet_slices:
            valid_rows = graphlet_mask[:, start:stop].any(dim=1)
            if not torch.any(valid_rows):
                continue
            pred = predicted_prob[valid_rows, start:stop].clamp_min(1.0e-12)
            truth = target_prob[valid_rows, start:stop]
            truth = truth / truth.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
            truth_safe = truth.clamp_min(1.0e-12)
            kl_terms.append((truth * (torch.log(truth_safe) - torch.log(pred))).sum(dim=-1).mean())
            probability_mae_terms.append(torch.abs(pred - truth).mean())
        graphlet_probability_loss = (
            torch.stack(kl_terms).mean() if kl_terms else predicted_logits.sum() * 0.0
        )
        probability_mae = (
            torch.stack(probability_mae_terms).mean()
            if probability_mae_terms
            else predicted_logits.sum() * 0.0
        )
        graphlet_total = (
            float(weights.get("graphlet_logit", 1.0)) * graphlet_logit_loss
            + float(weights.get("graphlet_probability", 0.25)) * graphlet_probability_loss
        )
        total = spectral_total + graphlet_total

        with torch.no_grad():
            channel_rmse = torch.sqrt(
                (normalized_delta.square() * valid).sum(dim=-1) / channel_count
            ).mean(dim=0)
            channel_mae = (
                torch.abs(normalized_delta) * valid
            ).sum(dim=-1) / channel_count
            channel_mae = channel_mae.mean(dim=0)
            graphlet_rmse = torch.sqrt(
                ((predicted_logits - target_logits).square() * graphlet_weight).sum()
                / coordinate_count
            )
            graphlet_mae = (
                torch.abs(predicted_logits - target_logits) * graphlet_weight
            ).sum() / coordinate_count
            trace_error = torch.abs(
                (predicted * valid).sum(dim=-1) - (target * valid).sum(dim=-1)
            ).mean(dim=0)
            moment2_relative_error = (
                torch.abs(predicted_moment2 - target_moment2)
                / target_moment2.abs().clamp_min(1.0e-8)
            ).mean(dim=0)

        metrics = {
            "loss": float(total.detach().cpu()),
            "spectral_component_loss": float(spectral_total.detach().cpu()),
            "topology_spectrum_loss": float(topology_loss.detach().cpu()),
            "bond_spectrum_loss": float(bond_loss.detach().cpu()),
            "low_frequency_loss": float(low_loss.detach().cpu()),
            "moment2_loss": float(moment2_loss.detach().cpu()),
            "topology_spectral_nrmse": float(channel_rmse[0].detach().cpu()),
            "bond_spectral_nrmse": float(channel_rmse[1].detach().cpu()),
            "topology_spectral_nmae": float(channel_mae[0].detach().cpu()),
            "bond_spectral_nmae": float(channel_mae[1].detach().cpu()),
            "topology_trace_mae": float(trace_error[0].detach().cpu()),
            "bond_trace_mae": float(trace_error[1].detach().cpu()),
            "topology_moment2_relative_error": float(
                moment2_relative_error[0].detach().cpu()
            ),
            "bond_moment2_relative_error": float(
                moment2_relative_error[1].detach().cpu()
            ),
            "graphlet_component_loss": float(graphlet_total.detach().cpu()),
            "graphlet_logit_loss": float(graphlet_logit_loss.detach().cpu()),
            "graphlet_probability_loss": float(graphlet_probability_loss.detach().cpu()),
            "graphlet_logit_rmse": float(graphlet_rmse.detach().cpu()),
            "graphlet_logit_mae": float(graphlet_mae.detach().cpu()),
            "graphlet_probability_mae": float(probability_mae.detach().cpu()),
        }
        return total, metrics

    def model_config(self) -> dict[str, Any]:
        return {
            "num_node_categories": self.num_node_categories,
            "num_edge_categories": self.num_edge_categories,
            "graphlet_block_widths": list(self.graphlet_block_widths),
            "hidden_dim": self.hidden_dim,
            "edge_dim": self.edge_dim,
            "graph_dim": self.graph_dim,
            "num_layers": self.num_layers,
            "spectral_dim": self.spectral_dim,
            "spectral_layers": self.spectral_layers,
            "spectral_heads": self.spectral_heads,
            "spectral_ff_dim": self.spectral_ff_dim,
            "graphlet_dim": self.graphlet_dim,
            "dropout": self.dropout_p,
            "graphlet_dropout": self.graphlet_dropout_p,
            "min_gap": self.min_gap,
            "input_normalization": self.input_normalization,
            "graphlet_logit_epsilon": self.graphlet_logit_epsilon,
        }


def save_attributed_spectral_graphlet_checkpoint(
    model: AttributedSpectralGraphletTransformerPredictor,
    path: str | Path,
    *,
    vocabulary: GraphCategoryVocabulary,
    graphlet_basis: GraphletBasis,
    summary_config: SummaryConfig | None = None,
    config: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if tuple(model.graphlet_block_widths) != tuple(graphlet_basis.simplex_block_widths):
        raise ValueError("Attributed graphlet basis widths do not match the model.")
    torch.save(
        {
            "format": ATTRIBUTED_SPECTRAL_GRAPHLET_CHECKPOINT_FORMAT,
            "pipeline_mode": "attributed",
            "guidance_mode": "spectral_graphlet",
            "model_state_dict": model.state_dict(),
            "model_config": model.model_config(),
            "vocabulary": vocabulary.to_dict(),
            "graphlet_basis": graphlet_basis.to_dict(),
            "summary_config": dict(summary_config.__dict__) if summary_config is not None else {},
            "time_parameterization": "normalized_diffusion_progress_0_source_1_clean",
            "config": config or {},
            "report": report or {},
        },
        path,
    )


def load_attributed_spectral_graphlet_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "auto",
) -> tuple[
    AttributedSpectralGraphletTransformerPredictor,
    GraphCategoryVocabulary,
    GraphletBasis,
    SummaryConfig,
    dict[str, Any],
]:
    resolved = resolve_torch_device(device) if isinstance(device, str) else device
    checkpoint = torch.load(Path(path), map_location=resolved)
    if checkpoint.get("format") != ATTRIBUTED_SPECTRAL_GRAPHLET_CHECKPOINT_FORMAT:
        raise ValueError(
            "Checkpoint is not an attributed spectral+graphlet predictor "
            f"({ATTRIBUTED_SPECTRAL_GRAPHLET_CHECKPOINT_FORMAT})."
        )
    vocabulary = GraphCategoryVocabulary.from_dict(dict(checkpoint.get("vocabulary", {}) or {}))
    basis = GraphletBasis.from_dict(dict(checkpoint.get("graphlet_basis", {}) or {}))
    model = AttributedSpectralGraphletTransformerPredictor(
        **dict(checkpoint.get("model_config", {}) or {})
    ).to(resolved)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    summary = SummaryConfig.from_dict(checkpoint.get("summary_config", {}) or {})
    return model, vocabulary, basis, summary, checkpoint


__all__ = [
    "ATTRIBUTED_SPECTRAL_GRAPHLET_CHECKPOINT_FORMAT",
    "AttributedSpectralGraphletTransformerPredictor",
    "load_attributed_spectral_graphlet_checkpoint",
    "save_attributed_spectral_graphlet_checkpoint",
]
