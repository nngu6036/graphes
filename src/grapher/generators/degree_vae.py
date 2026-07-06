from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from grapher.generators.degree_vectorizer import DegreeVectorizer
from grapher.utils.device import resolve_torch_device


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int | None = None, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        dim = int(input_dim)
        for _ in range(max(int(num_layers), 1)):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            dim = int(hidden_dim)
        if output_dim is not None:
            layers.append(nn.Linear(dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DegreeHistogramVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        head_dims: dict[str, int],
        *,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.head_dims = dict(head_dims)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.encoder = MLP(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = MLP(latent_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.num_nodes_head = nn.Linear(hidden_dim, head_dims["num_nodes"])
        self.degree_head = nn.Linear(hidden_dim, head_dims["degree"])
        self.edge_scalar_head = nn.Linear(hidden_dim, head_dims.get("edge_scalar", 1))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(min=-10.0, max=10.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.decoder(z)
        return {
            "num_nodes_logits": self.num_nodes_head(h),
            "degree_logits": self.degree_head(h),
            "edge_scalar": F.softplus(self.edge_scalar_head(h)),
        }

    def forward(self, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample_outputs(self, num_samples: int, *, device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(int(num_samples), self.latent_dim, device=device)
        return self.decode(z)

    def model_config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "head_dims": self.head_dims,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


def soft_histogram_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))


def degree_vae_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 5.0e-3,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or {}
    n_loss = F.cross_entropy(outputs["num_nodes_logits"], targets["num_nodes"].long())
    degree_loss = soft_histogram_ce(outputs["degree_logits"], targets["degree"])
    edge_loss = F.mse_loss(outputs["edge_scalar"], targets["edge_scalar"])
    kld = kl_loss(mu, logvar)
    total = (
        float(weights.get("num_nodes", 1.0)) * n_loss
        + float(weights.get("degree", 5.0)) * degree_loss
        + float(weights.get("edge_scalar", 0.1)) * edge_loss
        + float(beta) * kld
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "num_nodes_loss": float(n_loss.detach().cpu()),
        "degree_loss": float(degree_loss.detach().cpu()),
        "edge_scalar_loss": float(edge_loss.detach().cpu()),
        "kl_loss": float(kld.detach().cpu()),
    }
    return total, metrics


def build_degree_vae(vectorizer: DegreeVectorizer, *, latent_dim: int = 32, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.0) -> DegreeHistogramVAE:
    return DegreeHistogramVAE(
        input_dim=vectorizer.input_dim,
        latent_dim=int(latent_dim),
        hidden_dim=int(hidden_dim),
        head_dims=vectorizer.head_dims(),
        num_layers=int(num_layers),
        dropout=float(dropout),
    )


def save_degree_vae_checkpoint(
    path: str | Path,
    model: DegreeHistogramVAE,
    vectorizer: DegreeVectorizer,
    *,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.model_config(),
            "vectorizer": vectorizer.__dict__,
            "config": config or {},
            "metrics": metrics or {},
        },
        path,
    )


def load_degree_vae_checkpoint(path: str | Path, *, device: torch.device | str = "auto") -> tuple[DegreeHistogramVAE, DegreeVectorizer, dict[str, Any]]:
    resolved_device = resolve_torch_device(device)
    checkpoint = torch.load(path, map_location=resolved_device)
    vectorizer = DegreeVectorizer(**checkpoint["vectorizer"])
    model = DegreeHistogramVAE(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, vectorizer, checkpoint
