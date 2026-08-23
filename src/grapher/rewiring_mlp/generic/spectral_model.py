from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from grapher.properties.summary import SummaryConfig
from grapher.rewiring_mlp.generic.layers import TopologyMPNNLayer
from grapher.rewiring_mlp.generic.spectral_data import TopologySpectralBatch
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir

TOPOLOGY_SPECTRAL_CHECKPOINT_FORMAT = "topology_spectral_transformer_v1"


class TopologySpectralTransformerPredictor(nn.Module):
    """Predict the complete clean Laplacian spectrum in one forward pass.

    The graph encoder provides permutation-invariant context.  One spectral
    token is constructed for each *ordered* current eigenvalue and all tokens
    are processed jointly by a Transformer encoder.  The shared token head
    predicts eigenvalue gaps in parallel.  Positive gaps plus a trace
    normalization guarantee:

      lambda_1 = 0,
      lambda_1 <= ... <= lambda_n,
      sum_i lambda_i = 2m.

    Padding is masked, so the same checkpoint handles variable graph sizes.
    Eigenvectors are intentionally neither predicted nor constrained.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        edge_dim: int = 64,
        graph_dim: int = 128,
        num_layers: int = 4,
        spectral_dim: int = 128,
        spectral_layers: int = 3,
        spectral_heads: int = 4,
        spectral_ff_dim: int = 256,
        dropout: float = 0.0,
        min_gap: float = 1.0e-6,
        input_normalization: str = "mean_degree",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.graph_dim = int(graph_dim)
        self.num_layers = int(num_layers)
        self.spectral_dim = int(spectral_dim)
        self.spectral_layers = int(spectral_layers)
        self.spectral_heads = int(spectral_heads)
        self.spectral_ff_dim = int(spectral_ff_dim)
        self.dropout_p = float(dropout)
        self.min_gap = float(min_gap)
        self.input_normalization = str(input_normalization).lower()

        if self.spectral_dim <= 0 or self.spectral_heads <= 0:
            raise ValueError("spectral_dim and spectral_heads must be positive.")
        if self.spectral_dim % self.spectral_heads != 0:
            raise ValueError("spectral_dim must be divisible by spectral_heads.")
        if self.spectral_layers <= 0 or self.num_layers <= 0:
            raise ValueError("Graph and spectral layer counts must be positive.")
        if self.spectral_ff_dim <= 0:
            raise ValueError("spectral_ff_dim must be positive.")
        if self.min_gap < 0.0:
            raise ValueError("min_gap must be nonnegative.")
        if self.input_normalization not in {
            "mean_degree",
            "average_degree",
            "avg_degree",
            "trace",
            "degree_sum",
            "none",
            "raw",
        }:
            raise ValueError(
                "input_normalization must be mean_degree, trace, or none."
            )

        # Same topology-state encoder family as the maintained structural model.
        self.node_in = nn.Linear(4, self.hidden_dim)
        self.edge_in = nn.Linear(7, self.edge_dim)
        self.layers = nn.ModuleList(
            [
                TopologyMPNNLayer(self.hidden_dim, self.edge_dim)
                for _ in range(self.num_layers)
            ]
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.graph_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim + 2 * self.edge_dim + 5, self.graph_dim),
            nn.SiLU(),
            nn.Linear(self.graph_dim, self.graph_dim),
            nn.SiLU(),
        )

        # [normalized current lambda_i, normalized rank i/(n-1), time,
        #  normalized graph size, valid-token flag]
        self.spectral_token_in = nn.Sequential(
            nn.Linear(5, self.spectral_dim),
            nn.SiLU(),
            nn.Linear(self.spectral_dim, self.spectral_dim),
        )
        self.graph_to_spectral = nn.Linear(self.graph_dim, self.spectral_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.spectral_dim,
            nhead=self.spectral_heads,
            dim_feedforward=self.spectral_ff_dim,
            dropout=self.dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spectral_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.spectral_layers,
            enable_nested_tensor=False,
        )
        self.spectral_norm = nn.LayerNorm(self.spectral_dim)
        self.gap_head = nn.Sequential(
            nn.Linear(self.spectral_dim, self.spectral_dim),
            nn.SiLU(),
            nn.Linear(self.spectral_dim, 1),
        )

    @staticmethod
    def _masked_pair_pool(
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(values.dtype)
        return (values * weights).sum(dim=(1, 2)) / weights.sum(
            dim=(1, 2)
        ).clamp_min(1.0)

    def _graph_context(self, batch: TopologySpectralBatch) -> torch.Tensor:
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
        return self.graph_encoder(
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

    def _spectrum_scale(self, batch: TopologySpectralBatch) -> torch.Tensor:
        # adjacency is symmetric, hence its total sum is exactly 2m.
        trace = batch.adjacency.to(batch.current_spectrum.dtype).sum(dim=(1, 2))
        if self.input_normalization in {"none", "raw"}:
            return torch.ones_like(trace)
        if self.input_normalization in {"trace", "degree_sum"}:
            return trace.clamp_min(1.0)
        return (trace / batch.graph_size.clamp_min(1.0)).clamp_min(1.0e-8)

    def _constrained_spectrum(
        self,
        raw_gap_scores: torch.Tensor,
        batch: TopologySpectralBatch,
    ) -> torch.Tensor:
        """Convert parallel gap scores into a sorted trace-preserving spectrum."""

        mask = batch.spectrum_mask.bool()
        batch_size, width = mask.shape
        token_index = torch.arange(width, device=mask.device).view(1, -1)
        # lambda_1 is fixed to zero, so only tokens 1..n-1 carry gaps.
        gap_mask = mask & (token_index > 0)
        positive_gap = F.softplus(raw_gap_scores) + float(self.min_gap)
        positive_gap = positive_gap * gap_mask.to(positive_gap.dtype)

        n = batch.graph_size.to(raw_gap_scores.dtype).view(-1, 1)
        # Gap at zero-based token j contributes to lambda_{j+1},...,lambda_n:
        # multiplicity n-j.
        multiplicity = (n - token_index.to(raw_gap_scores.dtype)).clamp_min(0.0)
        denominator = (positive_gap * multiplicity).sum(dim=1)
        trace = batch.adjacency.to(raw_gap_scores.dtype).sum(dim=(1, 2))
        scale = torch.where(
            denominator > 0.0,
            trace / denominator.clamp_min(1.0e-12),
            torch.zeros_like(denominator),
        )
        gaps = positive_gap * scale.unsqueeze(1)
        spectrum = torch.cumsum(gaps, dim=1)
        spectrum = spectrum * mask.to(spectrum.dtype)
        return spectrum

    def forward(self, batch: TopologySpectralBatch) -> dict[str, torch.Tensor]:
        graph_hidden = self._graph_context(batch)
        mask = batch.spectrum_mask.bool()
        batch_size, width = mask.shape
        scale = self._spectrum_scale(batch)
        normalized_lambda = batch.current_spectrum / scale.unsqueeze(1)

        indices = torch.arange(width, device=mask.device, dtype=normalized_lambda.dtype)
        denom = (batch.graph_size - 1.0).clamp_min(1.0).unsqueeze(1)
        rank = indices.view(1, -1).expand(batch_size, -1) / denom
        size_feature = batch.graph_size / (batch.graph_size + 1.0).clamp_min(1.0)
        token_features = torch.stack(
            [
                normalized_lambda,
                rank,
                batch.time.unsqueeze(1).expand(batch_size, width),
                size_feature.unsqueeze(1).expand(batch_size, width),
                mask.to(normalized_lambda.dtype),
            ],
            dim=-1,
        )
        tokens = self.spectral_token_in(token_features)
        tokens = tokens + self.graph_to_spectral(graph_hidden).unsqueeze(1)
        tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        encoded = self.spectral_transformer(
            tokens,
            src_key_padding_mask=~mask,
        )
        encoded = self.spectral_norm(encoded)
        raw_gap_scores = self.gap_head(encoded).squeeze(-1)
        clean_spectrum = self._constrained_spectrum(raw_gap_scores, batch)
        return {
            "clean_spectrum": clean_spectrum,
            "raw_gap_scores": raw_gap_scores,
            "spectral_mask": mask,
        }

    def loss(
        self,
        batch: TopologySpectralBatch,
        *,
        loss_weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        weights = dict(loss_weights or {})
        outputs = self.forward(batch)
        predicted = outputs["clean_spectrum"]
        target = batch.clean_spectrum_target
        mask = batch.spectrum_mask.bool()
        scale = self._spectrum_scale(batch).unsqueeze(1)
        valid_weight = mask.to(predicted.dtype)
        count = valid_weight.sum().clamp_min(1.0)
        normalized_delta = (predicted - target) / scale

        spectrum_loss = (
            F.smooth_l1_loss(
                predicted / scale,
                target / scale,
                reduction="none",
            )
            * valid_weight
        ).sum() / count

        predicted_m2 = (predicted.square() * valid_weight).sum(dim=1)
        target_m2 = (target.square() * valid_weight).sum(dim=1)
        m2_scale = target_m2.abs().clamp_min(1.0)
        moment2_loss = F.smooth_l1_loss(
            predicted_m2 / m2_scale,
            target_m2 / m2_scale,
        )

        # Optional extra emphasis on low-frequency eigenvalues after lambda_1.
        low_k = int(weights.get("low_frequency_k", 0))
        low_frequency_loss = predicted.sum() * 0.0
        if low_k > 0:
            token_index = torch.arange(mask.shape[1], device=mask.device).view(1, -1)
            low_mask = mask & (token_index > 0) & (token_index <= low_k)
            if torch.any(low_mask):
                low_weight = low_mask.to(predicted.dtype)
                low_frequency_loss = (
                    F.smooth_l1_loss(
                        predicted / scale,
                        target / scale,
                        reduction="none",
                    )
                    * low_weight
                ).sum() / low_weight.sum().clamp_min(1.0)

        total = (
            float(weights.get("spectrum", 1.0)) * spectrum_loss
            + float(weights.get("moment2", 0.1)) * moment2_loss
            + float(weights.get("low_frequency", 0.0)) * low_frequency_loss
        )

        with torch.no_grad():
            abs_delta = torch.abs(predicted - target) * valid_weight
            spectral_mae = abs_delta.sum() / count
            spectral_rmse = torch.sqrt(
                (normalized_delta.square() * valid_weight).sum() / count
            )
            normalized_mae = (
                torch.abs(normalized_delta) * valid_weight
            ).sum() / count
            trace_pred = (predicted * valid_weight).sum(dim=1)
            trace_target = (target * valid_weight).sum(dim=1)
            trace_error = torch.mean(torch.abs(trace_pred - trace_target))
            moment2_relative_error = torch.mean(
                torch.abs(predicted_m2 - target_m2) / m2_scale
            )
            if predicted.shape[1] > 1:
                pair_mask = mask[:, 1:] & mask[:, :-1]
                monotonic_violations = (
                    ((predicted[:, 1:] + 1.0e-7) < predicted[:, :-1]) & pair_mask
                ).to(predicted.dtype).sum()
            else:
                monotonic_violations = predicted.sum() * 0.0
            lambda1_error = torch.mean(torch.abs(predicted[:, 0]))

        metrics = {
            "loss": float(total.detach().cpu()),
            "spectrum_loss": float(spectrum_loss.detach().cpu()),
            "moment2_loss": float(moment2_loss.detach().cpu()),
            "low_frequency_loss": float(low_frequency_loss.detach().cpu()),
            "spectral_mae": float(spectral_mae.detach().cpu()),
            "spectral_normalized_mae": float(normalized_mae.detach().cpu()),
            "spectral_normalized_rmse": float(spectral_rmse.detach().cpu()),
            "spectral_trace_mae": float(trace_error.detach().cpu()),
            "spectral_moment2_relative_error": float(
                moment2_relative_error.detach().cpu()
            ),
            "spectral_lambda1_mae": float(lambda1_error.detach().cpu()),
            "spectral_monotonic_violations": float(
                monotonic_violations.detach().cpu()
            ),
        }
        return total, metrics

    def model_config(self) -> dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "edge_dim": self.edge_dim,
            "graph_dim": self.graph_dim,
            "num_layers": self.num_layers,
            "spectral_dim": self.spectral_dim,
            "spectral_layers": self.spectral_layers,
            "spectral_heads": self.spectral_heads,
            "spectral_ff_dim": self.spectral_ff_dim,
            "dropout": self.dropout_p,
            "min_gap": self.min_gap,
            "input_normalization": self.input_normalization,
        }


def save_topology_spectral_checkpoint(
    model: TopologySpectralTransformerPredictor,
    path: str | Path,
    *,
    summary_config: SummaryConfig | None = None,
    config: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "format": TOPOLOGY_SPECTRAL_CHECKPOINT_FORMAT,
            "pipeline_mode": "topology",
            "guidance_mode": "spectral",
            "predictor_type": "spectral_transformer",
            "model_state_dict": model.state_dict(),
            "model_config": model.model_config(),
            "summary_config": (
                dict(summary_config.__dict__) if summary_config is not None else {}
            ),
            "config": config or {},
            "report": report or {},
        },
        path,
    )


def load_topology_spectral_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "auto",
) -> tuple[
    TopologySpectralTransformerPredictor,
    SummaryConfig,
    dict[str, Any],
]:
    resolved_device = (
        resolve_torch_device(device) if isinstance(device, str) else device
    )
    checkpoint = torch.load(Path(path), map_location=resolved_device)
    if checkpoint.get("format") != TOPOLOGY_SPECTRAL_CHECKPOINT_FORMAT:
        raise ValueError(
            "Checkpoint is not a topology Spectral Transformer predictor "
            f"({TOPOLOGY_SPECTRAL_CHECKPOINT_FORMAT})."
        )
    model = TopologySpectralTransformerPredictor(
        **dict(checkpoint.get("model_config", {}) or {})
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    summary_config = SummaryConfig.from_dict(
        checkpoint.get("summary_config", {}) or {}
    )
    return model, summary_config, checkpoint
