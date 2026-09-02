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
        invariant_summary_enabled: bool = False,
        invariant_summary_dim: int = 256,
        invariant_summary_layers: int = 2,
        invariant_summary_dropout: float = 0.05,
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
        self.invariant_summary_enabled = bool(invariant_summary_enabled)
        self.invariant_summary_dim = int(invariant_summary_dim)
        self.invariant_summary_layers = int(invariant_summary_layers)
        self.invariant_summary_dropout_p = float(invariant_summary_dropout)
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

        # Optional source-summary estimator conditioned only on the hard
        # attributed rewiring invariant used by the cross-type molecular
        # kernel.  It deliberately does NOT consume adjacency topology or
        # per-node typed degree because the latter is allowed to change during
        # bond-reassigning swaps.  Node tokens contain atom category + ordinary
        # degree; global bond-category counts are pooled separately.
        if self.invariant_summary_enabled:
            if self.invariant_summary_dim <= 0 or self.invariant_summary_layers <= 0:
                raise ValueError("Invariant-summary dimensions/layers must be positive.")
            invariant_token_width = self.num_node_categories + 3
            self.invariant_token_in = nn.Sequential(
                nn.Linear(invariant_token_width, self.invariant_summary_dim),
                nn.SiLU(),
                nn.Linear(self.invariant_summary_dim, self.invariant_summary_dim),
                nn.SiLU(),
            )
            self.invariant_token_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(self.invariant_summary_dim),
                        nn.Linear(self.invariant_summary_dim, self.invariant_summary_dim),
                        nn.SiLU(),
                        nn.Dropout(self.invariant_summary_dropout_p),
                        nn.Linear(self.invariant_summary_dim, self.invariant_summary_dim),
                    )
                    for _ in range(self.invariant_summary_layers)
                ]
            )
            present_edge_types = self.num_edge_categories - 1
            invariant_graph_width = (
                2 * self.invariant_summary_dim
                + self.num_node_categories
                + 2 * present_edge_types
                + 4
            )
            self.invariant_graph_encoder = nn.Sequential(
                nn.Linear(invariant_graph_width, self.invariant_summary_dim),
                nn.SiLU(),
                nn.Linear(self.invariant_summary_dim, self.invariant_summary_dim),
                nn.SiLU(),
            )
            self.invariant_spectral_token_in = nn.Sequential(
                nn.Linear(3, self.invariant_summary_dim),
                nn.SiLU(),
                nn.Linear(self.invariant_summary_dim, self.invariant_summary_dim),
            )
            self.invariant_channel_embedding = nn.Embedding(2, self.invariant_summary_dim)
            self.invariant_spectral_head = nn.Sequential(
                nn.LayerNorm(self.invariant_summary_dim),
                nn.Linear(self.invariant_summary_dim, self.invariant_summary_dim),
                nn.SiLU(),
                nn.Linear(self.invariant_summary_dim, 1),
            )
            self.invariant_graphlet_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.invariant_summary_dim, self.graphlet_dim),
                        nn.SiLU(),
                        nn.Dropout(self.invariant_summary_dropout_p),
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

    def _invariant_summary_outputs(
        self, batch: AttributedSpectralBatch
    ) -> dict[str, torch.Tensor]:
        if not self.invariant_summary_enabled:
            return {}
        node_mask = batch.node_mask.bool()
        batch_size, node_count = node_mask.shape
        dtype = batch.degrees.dtype
        node_labels = batch.source_node_labels.clamp(0, self.num_node_categories - 1)
        node_onehot = F.one_hot(node_labels, self.num_node_categories).to(dtype)
        size_feature = batch.graph_size / (batch.graph_size + 1.0).clamp_min(1.0)
        token_features = torch.cat(
            [
                node_onehot,
                batch.degrees.unsqueeze(-1),
                size_feature.view(batch_size, 1, 1).expand(batch_size, node_count, 1),
                node_mask.unsqueeze(-1).to(dtype),
            ],
            dim=-1,
        )
        hidden = self.invariant_token_in(token_features)
        weights = node_mask.unsqueeze(-1).to(hidden.dtype)
        hidden = hidden * weights
        for layer in self.invariant_token_layers:
            hidden = (hidden + layer(hidden)) * weights
        count = weights.sum(dim=1).clamp_min(1.0)
        mean_pool = hidden.sum(dim=1) / count
        neg_inf = torch.finfo(hidden.dtype).min
        max_pool = hidden.masked_fill(~node_mask.unsqueeze(-1), neg_inf).max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        # Permutation-invariant hard-invariant statistics.  Bond counts are
        # extracted from the upper triangle only, then represented both as
        # counts-per-node and proportions among present bonds.
        node_prop = (node_onehot * weights).sum(dim=1) / count
        upper = torch.triu(
            torch.ones((node_count, node_count), dtype=torch.bool, device=node_mask.device),
            diagonal=1,
        ).view(1, node_count, node_count)
        pair_mask = batch.pair_mask.bool() & upper
        edge_labels = batch.source_edge_labels
        bond_count_blocks = []
        for category in range(1, self.num_edge_categories):
            bond_count_blocks.append(
                ((edge_labels == category) & pair_mask).to(dtype).sum(dim=(1, 2))
            )
        bond_counts = torch.stack(bond_count_blocks, dim=1)
        total_bonds = bond_counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        bond_prop = bond_counts / total_bonds
        bond_per_node = bond_counts / batch.graph_size.unsqueeze(1).clamp_min(1.0)

        degree_mask = node_mask.to(dtype)
        degree_count = degree_mask.sum(dim=1).clamp_min(1.0)
        degree_mean = (batch.degrees * degree_mask).sum(dim=1) / degree_count
        degree_var = (
            (batch.degrees - degree_mean.unsqueeze(1)).square() * degree_mask
        ).sum(dim=1) / degree_count
        degree_max = batch.degrees.masked_fill(~node_mask, 0.0).max(dim=1).values
        graph_hidden = self.invariant_graph_encoder(
            torch.cat(
                [
                    mean_pool,
                    max_pool,
                    node_prop,
                    bond_prop,
                    bond_per_node,
                    degree_mean.unsqueeze(1),
                    torch.sqrt(degree_var.clamp_min(0.0)).unsqueeze(1),
                    degree_max.unsqueeze(1),
                    size_feature.unsqueeze(1),
                ],
                dim=-1,
            )
        )

        spectrum_mask = batch.spectrum_mask.bool()
        width = spectrum_mask.shape[1]
        index = torch.arange(width, device=node_mask.device, dtype=dtype)
        rank = index.view(1, 1, -1) / (batch.graph_size - 1.0).clamp_min(1.0).view(-1, 1, 1)
        rank = rank.expand(batch_size, 2, width)
        size = size_feature.view(-1, 1, 1).expand(batch_size, 2, width)
        token_mask = spectrum_mask.unsqueeze(1).expand(batch_size, 2, width)
        spectral_features = torch.stack([rank, size, token_mask.to(dtype)], dim=-1)
        spectral_hidden = self.invariant_spectral_token_in(spectral_features)
        channel_ids = torch.arange(2, device=node_mask.device).view(1, 2, 1).expand(batch_size, 2, width)
        spectral_hidden = spectral_hidden + self.invariant_channel_embedding(channel_ids)
        spectral_hidden = spectral_hidden + graph_hidden.view(batch_size, 1, 1, -1)
        raw_gap = self.invariant_spectral_head(spectral_hidden).squeeze(-1)
        clean_spectra = self._constrained_spectrum(
            raw_gap,
            spectrum_mask,
            batch.graph_size,
            self._spectral_trace(batch),
        )

        graphlet_mask = batch.graphlet_coordinate_mask.bool()
        clean_blocks: list[torch.Tensor] = []
        probability_blocks: list[torch.Tensor] = []
        for (start, stop), head in zip(self.graphlet_slices, self.invariant_graphlet_heads):
            valid = graphlet_mask[:, start:stop].any(dim=1, keepdim=True)
            block = head(graph_hidden)
            block = block - block.mean(dim=-1, keepdim=True)
            block = block * valid.to(block.dtype)
            prob = torch.softmax(block, dim=-1) * valid.to(block.dtype)
            clean_blocks.append(block)
            probability_blocks.append(prob)
        return {
            "invariant_clean_spectra": clean_spectra,
            "invariant_clean_graphlet_logits": torch.cat(clean_blocks, dim=-1),
            "invariant_clean_graphlet_probabilities": torch.cat(probability_blocks, dim=-1),
        }

    def invariant_summary(self, batch: AttributedSpectralBatch) -> dict[str, torch.Tensor]:
        if not self.invariant_summary_enabled:
            raise ValueError("This checkpoint has no hard-invariant-conditioned summary estimator.")
        return self._invariant_summary_outputs(batch)

    def forward(self, batch: AttributedSpectralBatch) -> dict[str, torch.Tensor]:
        graph_hidden = self._graph_context(batch)
        outputs = self._spectral_outputs(batch, graph_hidden)
        outputs.update(self._graphlet_outputs(batch, graph_hidden))
        outputs.update(self._invariant_summary_outputs(batch))
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

        invariant_summary_weight = float(weights.get("invariant_summary", 0.0))
        invariant_summary_total = predicted_logits.sum() * 0.0
        invariant_metrics: dict[str, float] = {}
        if self.invariant_summary_enabled and invariant_summary_weight > 0.0:
            invariant_spectra = outputs["invariant_clean_spectra"]
            inv_normalized = (invariant_spectra - target) / scale
            inv_channel_loss = (
                F.smooth_l1_loss(invariant_spectra / scale, target / scale, reduction="none")
                * valid
            ).sum(dim=-1) / channel_count
            inv_topology_loss = inv_channel_loss[:, 0].mean()
            inv_bond_loss = inv_channel_loss[:, 1].mean()
            inv_spectral_loss = (
                float(weights.get("topology_spectrum", 1.0)) * inv_topology_loss
                + float(weights.get("bond_spectrum", 1.0)) * inv_bond_loss
            )
            inv_low_loss = invariant_spectra.sum() * 0.0
            if low_k > 0:
                index = torch.arange(invariant_spectra.shape[-1], device=invariant_spectra.device).view(1, 1, -1)
                low_mask = mask & (index > 0) & (index <= low_k)
                low_weight = low_mask.to(invariant_spectra.dtype)
                if torch.any(low_mask):
                    inv_low_loss = (
                        F.smooth_l1_loss(invariant_spectra / scale, target / scale, reduction="none")
                        * low_weight
                    ).sum() / low_weight.sum().clamp_min(1.0)
            inv_pred_moment2 = (invariant_spectra.square() * valid).sum(dim=-1)
            inv_moment2_loss = F.smooth_l1_loss(
                inv_pred_moment2 / second_scale.clamp_min(1.0e-8),
                target_moment2 / second_scale.clamp_min(1.0e-8),
            )
            inv_spectral_total = (
                float(weights.get("spectrum", 1.0)) * inv_spectral_loss
                + float(weights.get("low_frequency", 0.0)) * inv_low_loss
                + float(weights.get("moment2", 0.0)) * inv_moment2_loss
            )

            inv_logits = outputs["invariant_clean_graphlet_logits"]
            inv_prob = outputs["invariant_clean_graphlet_probabilities"]
            inv_logit_loss = (
                F.smooth_l1_loss(inv_logits, target_logits, reduction="none") * graphlet_weight
            ).sum() / coordinate_count
            inv_kl_terms: list[torch.Tensor] = []
            for start, stop in self.graphlet_slices:
                valid_rows = graphlet_mask[:, start:stop].any(dim=1)
                if not torch.any(valid_rows):
                    continue
                pred = inv_prob[valid_rows, start:stop].clamp_min(1.0e-12)
                truth = target_prob[valid_rows, start:stop]
                truth = truth / truth.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
                truth_safe = truth.clamp_min(1.0e-12)
                inv_kl_terms.append(
                    (truth * (torch.log(truth_safe) - torch.log(pred))).sum(dim=-1).mean()
                )
            inv_probability_loss = (
                torch.stack(inv_kl_terms).mean() if inv_kl_terms else inv_logits.sum() * 0.0
            )
            inv_graphlet_total = (
                float(weights.get("graphlet_logit", 1.0)) * inv_logit_loss
                + float(weights.get("graphlet_probability", 0.25)) * inv_probability_loss
            )
            invariant_summary_total = inv_spectral_total + inv_graphlet_total
            total = total + invariant_summary_weight * invariant_summary_total
            with torch.no_grad():
                inv_channel_rmse = torch.sqrt(
                    (inv_normalized.square() * valid).sum(dim=-1) / channel_count
                ).mean(dim=0)
            invariant_metrics = {
                "invariant_summary_loss": float(invariant_summary_total.detach().cpu()),
                "invariant_summary_spectral_loss": float(inv_spectral_total.detach().cpu()),
                "invariant_summary_topology_nrmse": float(inv_channel_rmse[0].detach().cpu()),
                "invariant_summary_bond_nrmse": float(inv_channel_rmse[1].detach().cpu()),
                "invariant_summary_graphlet_logit_loss": float(inv_logit_loss.detach().cpu()),
                "invariant_summary_graphlet_probability_loss": float(inv_probability_loss.detach().cpu()),
            }

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
        metrics.update(invariant_metrics)
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
            "invariant_summary_enabled": self.invariant_summary_enabled,
            "invariant_summary_dim": self.invariant_summary_dim,
            "invariant_summary_layers": self.invariant_summary_layers,
            "invariant_summary_dropout": self.invariant_summary_dropout_p,
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
