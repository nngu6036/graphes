"""Equivariant node/pair denoiser with scalar adjacency regression.

There is no categorical edge softmax, predicted eigenbasis, spectral transformer,
separate spectral output head, or spectral stochastic state. Ordered weighted
adjacency eigenvalues are used by the loss, not to reconstruct a second graph.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .diffusion import pair_mask, symmetrize


def mlp(input_dim, hidden, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, output_dim))


def time_embedding(t, dim):
    half = dim // 2
    freq = torch.exp(-math.log(10000.) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half-1, 1))
    phase = t.float()[:, None] * freq[None]
    value = torch.cat((phase.sin(), phase.cos()), -1)
    return F.pad(value, (0, dim-value.shape[-1]))


def mean_nodes(value, mask):
    return (value * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


def mean_pairs(value, active):
    return (value * active[..., None]).sum((1, 2)) / active.sum((1, 2))[:, None].clamp_min(1)


class NodePairBlock(nn.Module):
    def __init__(self, hidden, heads, ff_dim, dropout):
        super().__init__()
        self.pair_update = mlp(4*hidden, hidden, hidden)
        self.message = mlp(2*hidden, hidden, hidden)
        self.node_update = mlp(3*hidden, hidden, hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(hidden, ff_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(ff_dim, hidden))
        self.pair_norm = nn.LayerNorm(hidden)
        self.node_norm = nn.LayerNorm(hidden)
        self.attention_norm = nn.LayerNorm(hidden)
        self.ff_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, pairs, mask):
        active = pair_mask(mask)
        hi, hj = nodes[:, :, None], nodes[:, None, :]
        global_state = mean_nodes(nodes, mask)
        global_pairs = global_state[:, None, None].expand_as(pairs)
        pairs = self.pair_norm(pairs + self.dropout(self.pair_update(
            torch.cat((pairs, hi+hj, (hi-hj).abs(), global_pairs), -1)))) * active[..., None]
        messages = self.message(torch.cat((pairs, hj.expand_as(pairs)), -1)) * active[..., None]
        aggregated = messages.sum(2) / active.sum(2)[..., None].clamp_min(1)
        nodes = self.node_norm(nodes + self.dropout(self.node_update(torch.cat((
            nodes, aggregated, global_state[:, None].expand_as(nodes)), -1)))) * mask[..., None]
        attended, _ = self.attention(nodes, nodes, nodes, key_padding_mask=~mask, need_weights=False)
        nodes = self.attention_norm(nodes + self.dropout(attended)) * mask[..., None]
        nodes = self.ff_norm(nodes + self.dropout(self.ff(nodes))) * mask[..., None]
        return nodes, pairs


class OptionCDenoiser(nn.Module):
    def __init__(self, *, node_classes, max_nodes, hidden_dim, num_layers, num_heads,
                 ff_dim, dropout, graphlet_orders, graphlet_block_sizes, clustering_bins):
        super().__init__()
        self.node_classes, self.max_nodes, self.hidden_dim = node_classes, max_nodes, hidden_dim
        self.graphlet_orders = tuple(graphlet_orders)
        self.graphlet_block_sizes = tuple(graphlet_block_sizes)
        self.time = mlp(hidden_dim, hidden_dim, hidden_dim)
        self.node_input = nn.Linear(node_classes+3, hidden_dim)
        self.pair_input = nn.Linear(2, hidden_dim)
        self.blocks = nn.ModuleList([NodePairBlock(hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.node_head = mlp(hidden_dim, hidden_dim, node_classes)
        self.adjacency_head = mlp(hidden_dim+2*node_classes, hidden_dim, 1)
        self.summary = mlp(2*hidden_dim+node_classes+2, hidden_dim, hidden_dim)
        self.graphlet_heads = nn.ModuleDict({str(k): mlp(hidden_dim, hidden_dim, width)
                                            for k, width in zip(self.graphlet_orders, self.graphlet_block_sizes)})
        self.mass_head = mlp(hidden_dim, hidden_dim, len(self.graphlet_orders))
        self.clustering_head = mlp(hidden_dim, hidden_dim, clustering_bins)
        self.orbit_head = mlp(hidden_dim, hidden_dim, 4)

    def forward(self, x, weighted, t, mask, total_steps):
        active = pair_mask(mask)
        weighted = symmetrize(weighted, mask)
        n = mask.sum(1).to(weighted.dtype)
        onehot = F.one_hot(x.long(), self.node_classes).to(weighted.dtype)
        strength = weighted.sum(2) / (n-1).clamp_min(1)[:, None]
        squared = weighted.square().sum(2) / (n-1).clamp_min(1)[:, None]
        size = (n / self.max_nodes)[:, None].expand_as(strength)
        time = self.time(time_embedding(t.float() / total_steps * 1000., self.hidden_dim))
        nodes = (self.node_input(torch.cat((onehot, strength[..., None], squared[..., None], size[..., None]), -1))
                 + time[:, None]) * mask[..., None]
        pairs = (self.pair_input(torch.stack((weighted, weighted.square()), -1))
                 + time[:, None, None]) * active[..., None]
        for block in self.blocks:
            nodes, pairs = block(nodes, pairs, mask)
        node_logits = self.node_head(nodes) * mask[..., None]
        node_probs = node_logits.softmax(-1) * mask[..., None]
        pi, pj = node_probs[:, :, None], node_probs[:, None, :]
        clean = self.adjacency_head(torch.cat((pairs, pi+pj, (pi-pj).abs()), -1)).squeeze(-1)
        clean = symmetrize(clean, mask)  # unbounded real regression during training
        pooled = self.summary(torch.cat((mean_nodes(nodes, mask), mean_pairs(pairs, active),
                                        mean_nodes(node_probs, mask),
                                        mean_pairs(torch.stack((clean, clean.square()), -1), active)), -1))
        return {"node_logits": node_logits, "clean_adjacency": clean,
                "graphlet_logits": torch.cat([self.graphlet_heads[str(k)](pooled) for k in self.graphlet_orders], -1),
                "graphlet_mass": self.mass_head(pooled).sigmoid(),
                "clustering_logits": self.clustering_head(pooled),
                "orbit_log_mean": F.softplus(self.orbit_head(pooled))}


def model_config(cfg, schema, basis):
    return {**cfg["model"], "node_classes": len(schema["node_marginal"]),
            "graphlet_orders": list(basis.orders), "graphlet_block_sizes": list(basis.block_sizes),
            "clustering_bins": cfg["graphlets"]["clustering_bins"]}


def prediction_targets(pred, index, n, basis, *, scaled_clean=None):
    weighted = pred["clean_adjacency"][index, :n, :n].detach().cpu().numpy() if scaled_clean is None else scaled_clean
    return {"weighted_adjacency": weighted,
            "histogram": torch.cat([h.softmax(-1) for h in pred["graphlet_logits"][index].split(basis.block_sizes)]).detach().cpu().numpy(),
            "mass": pred["graphlet_mass"][index].detach().cpu().numpy(),
            "clustering": pred["clustering_logits"][index].softmax(-1).detach().cpu().numpy(),
            "orbit": pred["orbit_log_mean"][index].detach().cpu().numpy()}
