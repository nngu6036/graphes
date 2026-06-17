from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-12


def encode_degree_sequence(degree_sequence: Sequence[int], max_class: int) -> torch.Tensor:
    """Encode a degree sequence as a DH-VAE histogram.

    The revised degree prior follows the paper's size-conditioned DH-VAE: for a
    graph with n nodes, the histogram bins are degree values 0, ..., n-1.  For
    batching we pad histograms to ``max_class`` bins, so this helper returns a
    vector whose index is the degree value itself.  Zero-degree nodes are kept;
    connectedness filtering is handled by the caller when needed.
    """

    max_class = int(max_class)
    if max_class <= 0:
        raise ValueError("max_class must be positive.")
    hist = torch.zeros(max_class, dtype=torch.float32)
    for degree in degree_sequence:
        d = int(degree)
        if d < 0 or d >= max_class:
            raise ValueError(
                f"Invalid degree {d} for histogram width {max_class}. "
                "Expected degree values in [0, max_class - 1]."
            )
        hist[d] += 1.0
    return hist


def decode_degree_sequence(histogram: torch.Tensor | Sequence[int | float]) -> list[int]:
    """Expand a degree histogram into a sorted non-increasing degree sequence."""

    tensor = torch.as_tensor(histogram).detach().cpu().reshape(-1)
    sequence: list[int] = []
    for degree, count in enumerate(tensor):
        c = int(round(float(count.item())))
        if c > 0:
            sequence.extend([int(degree)] * c)
    return sorted(sequence, reverse=True)


def degree_histogram_from_sequence(degree_sequence: Sequence[int], max_nodes: int) -> torch.Tensor:
    """Alias with paper terminology: h_D=(m_0,...,m_{N-1})."""

    return encode_degree_sequence(degree_sequence, max_nodes)


@dataclass(frozen=True)
class DegreeHistogramBatch:
    """Fixed-width degree histograms paired with graph sizes."""

    histograms: torch.Tensor
    sizes: torch.Tensor


def make_degree_histogram_batch(sequences: Sequence[Sequence[int]], max_nodes: int | None = None) -> DegreeHistogramBatch:
    """Create a DH-VAE batch from degree sequences.

    Histograms use degree-value bins 0..max_nodes-1.  When max_nodes is not
    supplied, it is inferred from the largest graph size and maximum degree.
    """

    if not sequences:
        raise ValueError("No degree sequences were provided.")
    inferred_max_nodes = max(len(seq) for seq in sequences)
    inferred_max_degree = max(max(int(d) for d in seq) for seq in sequences)
    width = int(max_nodes or max(inferred_max_nodes, inferred_max_degree + 1, 1))
    rows = [degree_histogram_from_sequence(seq, width) for seq in sequences]
    sizes = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    return DegreeHistogramBatch(histograms=torch.stack(rows, dim=0).float(), sizes=sizes)


@dataclass(frozen=True)
class DHVAELoss:
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    kl_loss: torch.Tensor


class SizeConditionedHistogramEncoder(nn.Module):
    """Encoder q_phi(z | h_D, n)."""

    def __init__(self, histogram_dim: int, hidden_dim: int, latent_dim: int, size_embedding_dim: int, max_nodes: int):
        super().__init__()
        self.size_embedding = nn.Embedding(int(max_nodes) + 1, int(size_embedding_dim))
        self.net = nn.Sequential(
            nn.Linear(int(histogram_dim) + int(size_embedding_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(int(hidden_dim), int(latent_dim))
        self.logvar_layer = nn.Linear(int(hidden_dim), int(latent_dim))

    def forward(self, histogram: torch.Tensor, sizes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sizes = sizes.long().clamp(min=0, max=self.size_embedding.num_embeddings - 1)
        size_emb = self.size_embedding(sizes)
        h = self.net(torch.cat([histogram.float(), size_emb], dim=-1))
        return self.mean_layer(h), self.logvar_layer(h)


class SizeConditionedHistogramDecoder(nn.Module):
    """Decoder p_theta(h_D | z, n) via Multinomial(n, pi_theta(.|z,n))."""

    def __init__(self, histogram_dim: int, hidden_dim: int, latent_dim: int, size_embedding_dim: int, max_nodes: int):
        super().__init__()
        self.histogram_dim = int(histogram_dim)
        self.max_nodes = int(max_nodes)
        self.size_embedding = nn.Embedding(int(max_nodes) + 1, int(size_embedding_dim))
        self.net = nn.Sequential(
            nn.Linear(int(latent_dim) + int(size_embedding_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(histogram_dim)),
        )

    def valid_degree_mask(self, sizes: torch.Tensor) -> torch.Tensor:
        """Mask valid degree bins k=0,...,n-1 for each graph size n."""

        sizes = sizes.long().clamp(min=0, max=self.max_nodes)
        degree_values = torch.arange(self.histogram_dim, device=sizes.device).view(1, -1)
        return degree_values < sizes.view(-1, 1)

    def forward(self, z: torch.Tensor, sizes: torch.Tensor, *, mask_invalid: bool = True) -> torch.Tensor:
        sizes = sizes.long().clamp(min=0, max=self.size_embedding.num_embeddings - 1)
        size_emb = self.size_embedding(sizes)
        logits = self.net(torch.cat([z.float(), size_emb], dim=-1))
        if mask_invalid:
            mask = self.valid_degree_mask(sizes)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 4)
        return logits


class DHVAE(nn.Module):
    """Size-conditioned Degree-Histogram VAE.

    * input: degree histogram h_D with bins 0..N-1 and graph size n;
    * encoder: q_phi(z | h_D, n);
    * decoder: categorical pi_theta(k | z, n) over valid degree values;
    * likelihood: Multinomial(n, pi_theta(. | z, n));
    * generation: sample n from the empirical size distribution, then sample a
      complete histogram from the multinomial so sum_k m_k = n by construction.
    """

    architecture = "size_conditioned_dhvae"

    def __init__(
        self,
        max_nodes: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        size_embedding_dim: int = 32,
        histogram_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.max_nodes = int(max_nodes)
        self.histogram_dim = int(histogram_dim or self.max_nodes)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.size_embedding_dim = int(size_embedding_dim)
        if self.max_nodes <= 0:
            raise ValueError("max_nodes must be positive.")
        if self.histogram_dim < self.max_nodes:
            raise ValueError("histogram_dim must be at least max_nodes so degree n-1 is representable.")

        self.encoder = SizeConditionedHistogramEncoder(
            histogram_dim=self.histogram_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            size_embedding_dim=self.size_embedding_dim,
            max_nodes=self.max_nodes,
        )
        self.decoder = SizeConditionedHistogramDecoder(
            histogram_dim=self.histogram_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            size_embedding_dim=self.size_embedding_dim,
            max_nodes=self.max_nodes,
        )
        # Index i stores P(n=i).  It is overwritten from training data before
        # checkpointing.  The default puts all mass at max_nodes so ad hoc model
        # construction still has a deterministic fallback.
        size_probs = torch.zeros(self.max_nodes + 1, dtype=torch.float32)
        size_probs[self.max_nodes] = 1.0
        self.register_buffer("size_probs", size_probs)

    def set_size_distribution(self, sizes_or_counts: Mapping[int, int] | Iterable[int] | torch.Tensor, *, values_are_counts: bool = False) -> None:
        """Store the empirical graph-size prior p_data(n).

        By default ``sizes_or_counts`` is interpreted as a list of observed graph
        sizes.  Pass ``values_are_counts=True`` when supplying an already binned
        vector whose index is the node count.
        """

        if isinstance(sizes_or_counts, Mapping):
            counts = torch.zeros(self.max_nodes + 1, dtype=torch.float32)
            for raw_size, raw_count in sizes_or_counts.items():
                n = int(raw_size)
                if 0 <= n <= self.max_nodes:
                    counts[n] += float(raw_count)
            values = counts
            values_are_counts = True
        else:
            values = torch.as_tensor(list(sizes_or_counts) if not torch.is_tensor(sizes_or_counts) else sizes_or_counts, dtype=torch.float32)
        if values.numel() == 0:
            raise ValueError("Cannot set an empty size distribution.")
        if values_are_counts:
            if values.numel() != self.max_nodes + 1:
                raise ValueError(f"Size-count vector must have length {self.max_nodes + 1}.")
            counts = values.clone().float()
        else:
            counts = torch.zeros(self.max_nodes + 1, dtype=torch.float32)
            for size in values.reshape(-1).tolist():
                n = int(size)
                if 0 <= n <= self.max_nodes:
                    counts[n] += 1.0
        counts[0] = 0.0
        if float(counts.sum().item()) <= 0.0:
            counts.zero_()
            counts[self.max_nodes] = 1.0
        probs = counts / counts.sum().clamp_min(_EPS)
        self.size_probs.detach().copy_(probs.to(device=self.size_probs.device, dtype=self.size_probs.dtype))

    def empirical_size_distribution(self) -> dict[str, list[int] | list[float]]:
        """Return nonzero empirical size probabilities in checkpoint-friendly form."""

        probs = self.size_probs.detach().cpu().float()
        values = [int(i) for i, p in enumerate(probs.tolist()) if i > 0 and p > 0.0]
        return {"values": values, "probs": [float(probs[i].item()) for i in values]}

    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(torch.sum(1.0 + logvar - mean.pow(2) - logvar.exp(), dim=1))


    def sample_sizes(self, num_samples: int, *, device: torch.device | str | None = None) -> torch.Tensor:
        device = torch.device(device or self.size_probs.device)
        probs = self.size_probs.to(device=device)
        if float(probs.sum().item()) <= 0.0:
            probs = torch.zeros_like(probs)
            probs[self.max_nodes] = 1.0
        return torch.multinomial(probs / probs.sum().clamp_min(_EPS), int(num_samples), replacement=True)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def forward(self, histogram: torch.Tensor, sizes: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        histogram = histogram.float()
        if sizes is None:
            sizes = histogram.sum(dim=-1).round().long()
        sizes = sizes.long().to(histogram.device)
        mean, logvar = self.encoder(histogram, sizes)
        z = self.reparameterize(mean, logvar)
        logits = self.decoder(z, sizes, mask_invalid=True)
        return logits, mean, logvar

    def loss(self, histogram: torch.Tensor, sizes: torch.Tensor | None = None, *, beta_kl: float = 1e-3) -> DHVAELoss:
        """Multinomial negative log-likelihood plus beta-weighted KL."""

        histogram = histogram.float()
        if sizes is None:
            sizes = histogram.sum(dim=-1).round().long()
        sizes = sizes.long().to(histogram.device)
        logits, mean, logvar = self.forward(histogram, sizes)
        log_probs = F.log_softmax(logits, dim=-1)
        # Eq. 3 in the paper sums over degree multiplicities.  We keep that
        # scale, rather than averaging over n, so larger graphs contribute in
        # proportion to the number of node-degree observations they contain.
        reconstruction = -(histogram * log_probs).sum(dim=-1).mean()
        kl = -0.5 * torch.mean(torch.sum(1.0 + logvar - mean.pow(2) - logvar.exp(), dim=1))
        return DHVAELoss(loss=reconstruction + float(beta_kl) * kl, reconstruction_loss=reconstruction, kl_loss=kl)

    def reconstruction_loss(
        self,
        logits: torch.Tensor,
        histogram: torch.Tensor,
        sizes: torch.Tensor | None = None,
        *,
        normalize_by_n: bool = False,
        normalize_by_size: bool | None = None,
    ) -> torch.Tensor:
        """Paper Eq. 3 multinomial NLL: -sum_k m_k log pi(k|z,n)."""

        histogram = histogram.float()
        if normalize_by_size is not None:
            normalize_by_n = bool(normalize_by_size)
        if sizes is None:
            sizes = histogram.sum(dim=-1).round().long()
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -(histogram * log_probs).sum(dim=-1)
        if normalize_by_n:
            nll = nll / sizes.to(nll.device).float().clamp_min(1.0)
        return nll.mean()

    def degree_probabilities(self, sizes: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
        """Return pi_theta(k | z, n) for z sampled from the standard normal prior."""

        device = next(self.parameters()).device
        sizes = sizes.long().to(device)
        z = torch.randn((sizes.numel(), self.latent_dim), device=device)
        temp = max(float(temperature), 1e-6)
        logits = self.decoder(z, sizes, mask_invalid=True) / temp
        return F.softmax(logits, dim=-1)

    def sample_histograms(self, sizes: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
        """Sample complete histograms h_D ~ Multinomial(n, pi_theta(.|z,n))."""

        device = next(self.parameters()).device
        sizes = sizes.long().to(device)
        probs = self.degree_probabilities(sizes, temperature=temperature)
        histograms = torch.zeros((sizes.numel(), self.histogram_dim), device=device, dtype=torch.float32)
        for i, n_tensor in enumerate(sizes):
            n = int(n_tensor.item())
            if n <= 0:
                continue
            draws = torch.multinomial(probs[i], num_samples=n, replacement=True)
            histograms[i].scatter_add_(0, draws, torch.ones_like(draws, dtype=histograms.dtype))
        return histograms

    def generate(
        self,
        num_samples: int,
        *,
        sizes: Sequence[int] | torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> list[list[int]]:
        """Generate degree sequences from the empirical-size DH-VAE prior."""

        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if sizes is None:
                sampled_sizes = self.sample_sizes(int(num_samples), device=device)
            else:
                sampled_sizes = torch.as_tensor(sizes, dtype=torch.long, device=device)
                if sampled_sizes.numel() != int(num_samples):
                    raise ValueError("When sizes are supplied, len(sizes) must equal num_samples.")
                if torch.any(sampled_sizes < 1) or torch.any(sampled_sizes > self.max_nodes):
                    raise ValueError(f"Sizes must be in [1, {self.max_nodes}].")
            histograms = self.sample_histograms(sampled_sizes, temperature=temperature)
            return [decode_degree_sequence(hist) for hist in histograms]

    def save_model(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path: str) -> None:
        self.load_state_dict(torch.load(file_path, map_location="cpu"))
        self.eval()


DegreeHistogramVAE = DHVAE
SizeConditionedDHVAE = DHVAE


__all__ = [
    "DHVAE",
    "DegreeHistogramVAE",
    "SizeConditionedDHVAE",
    "DegreeHistogramBatch",
    "DHVAELoss",
    "SizeConditionedHistogramEncoder",
    "SizeConditionedHistogramDecoder",
    "encode_degree_sequence",
    "decode_degree_sequence",
    "degree_histogram_from_sequence",
    "make_degree_histogram_batch",
]
