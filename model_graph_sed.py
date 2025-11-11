# model_graph_sed.py
# Graph-level representation learning where Euclidean distance between graph
# embeddings is proportional to the symmetric edit distance (SED) between graphs.
#

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Iterable, Optional
import toml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
import torch_geometric
import networkx as nx
from itertools import combinations

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils import from_networkx

from utils import *
# ------------------------------
# Small MLP helper
# ------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int = 2, dropout: float = 0.0, act=nn.ReLU):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(d, hidden), act(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # noqa
        return self.net(x)

# ------------------------------
# Graph-level Encoder (GIN)
# ------------------------------
class GraphSEDEncoder(nn.Module):
    """
    GIN-based encoder that outputs:
      - z: node embeddings [N, D]
      - h: graph embedding [B, D] (mean pool), or [1, D] if batch is None

    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        proj_dim: int | None = None,
        use_ln: bool = True,
        dropout: float = 0.0,
        l2_normalize: bool = True,
        norm_z: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = proj_dim if proj_dim is not None else hidden_dim
        self.l2_normalize = l2_normalize
        self.norm_z = norm_z

        def gin_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

        self.gin = nn.ModuleList([
            GINConv(gin_mlp(in_channels if i == 0 else hidden_dim, hidden_dim))
            for i in range(num_layers)
        ])

        self.proj_z = nn.Linear(hidden_dim, self.out_dim) if proj_dim is not None else None
        self.proj_h = nn.Linear(hidden_dim, self.out_dim) if proj_dim is not None else None

        self.ln_z = nn.LayerNorm(self.out_dim) if use_ln else nn.Identity()
        self.ln_h = nn.LayerNorm(self.out_dim) if use_ln else nn.Identity()

        # learnable scale for normalized outputs (stabilizes distances)
        self.scale = nn.Parameter(torch.tensor(1.0)) if l2_normalize else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None = None):
        # Node embedding stack
        z = x
        for conv in self.gin:
            z = conv(z, edge_index)  # [N, H]

        # Projection + norm
        if self.proj_z is not None:
            z = self.proj_z(z)
        z = self.ln_z(z)

        # Graph embedding: mean pool
        h = z.mean(dim=0, keepdim=True) if batch is None else global_mean_pool(z, batch)
        if self.proj_h is not None:
            h = self.proj_h(h)
        h = self.ln_h(h)

        # Optional L2 normalization (recommended for distance learning)
        if self.l2_normalize:
            h = F.normalize(h, dim=-1)
            if self.norm_z:
                z = F.normalize(z, dim=-1)
            if self.scale is not None:
                h = self.scale * h
                if self.norm_z:
                    z = self.scale * z

        return z, h  # node, graph


# ------------------------------
# Graph-level SED Decoder (distance regressor)
# ------------------------------
class GraphSEDDecoder(nn.Module):
    """
    Predict symmetric edit distance (SED) between two graphs from their embeddings.

    """

    def __init__(self, hidden_dim: int, mlp_hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.reg = MLP(in_dim=hidden_dim, out_dim=1, hidden=mlp_hidden, depth=2, dropout=dropout)

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(h_i - h_j)          # [B,H]
        d_hat = self.reg(diff).squeeze(-1)
        return torch.relu(d_hat)


# ------------------------------
# Combined model
# ------------------------------
class GraphSEDModel(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 128, num_layers: int = 4, proj_dim: Optional[int] = 128,
                 enc_dropout: float = 0.0, l2_normalize: bool = True, dec_hidden: int = 128,
                 dec_dropout: float = 0.0):
        super().__init__()
        self.encoder = GraphSEDEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            proj_dim=proj_dim,
            dropout=enc_dropout,
            l2_normalize=l2_normalize,
        )
        out_dim = proj_dim if proj_dim is not None else hidden_dim
        self.decoder = GraphSEDDecoder(hidden_dim=out_dim, mlp_hidden=dec_hidden, dropout=dec_dropout)

    def embed(self, batch: Data) -> torch.Tensor:
        _, h = self.encoder(batch.x, batch.edge_index, batch.batch)
        return h

    def forward(self, batch1: Data, batch2: Data) -> torch.Tensor:
        h1 = self.embed(batch1)
        h2 = self.embed(batch2)
        return self.decoder(h1, h2)

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path, map_location="cpu"))
        self.eval()


# ------------------------------
# Loss utilities (optional)
# ------------------------------

def sed_regression_loss(d_hat: torch.Tensor, d_true: torch.Tensor, zero_margin: float = 0.0) -> torch.Tensor:
    """
    L2 regression + optional small margin for exact-zero pairs.
    d_hat, d_true: [B]
    """
    loss = F.mse_loss(d_hat, d_true)
    if zero_margin > 0:
        zero = (d_true == 0)
        if zero.any():
            loss = loss + 0.01 * torch.relu(d_hat[zero] - zero_margin).mean()
    return loss

def compact_relabel(G: nx.Graph, H: nx.Graph) -> Tuple[nx.Graph, nx.Graph]:
    """Relabel two graphs so they share the same compact node set 0..n-1."""
    nodes = sorted(set(G.nodes()) | set(H.nodes()))
    mapping = {u: i for i, u in enumerate(nodes)}
    return nx.relabel_nodes(G, mapping, copy=True), nx.relabel_nodes(H, mapping, copy=True)


def symmetric_edit_distance(G: nx.Graph, H: nx.Graph) -> int:
    """SED as |E(G) Δ E(H)| for simple undirected graphs."""
    Gc, Hc = compact_relabel(G, H)
    eG = {tuple(sorted(e)) for e in Gc.edges()}
    eH = {tuple(sorted(e)) for e in Hc.edges()}
    return len(eG.symmetric_difference(eH))


def to_pyg_data(G: nx.Graph) -> Data:
    """Convert NetworkX graph to PyG Data with simple node features (degree as scalar)."""
    Gc = nx.convert_node_labels_to_integers(G, ordering="sorted")
    degs = torch.tensor([Gc.degree(i) for i in range(Gc.number_of_nodes())], dtype=torch.float32).unsqueeze(-1)
    data = from_networkx(Gc)
    data.x = degs  # [N,1]
    return data


# ------------------------------
# Pair dataset
# ------------------------------
@dataclass
class PairItem:
    G1: nx.Graph
    G2: nx.Graph
    sed_raw: int
    n_nodes: int


class GraphPairSEDataset(Dataset):
    def __init__(self, graphs: List[nx.Graph], pairs_per_graph: int = 4, rewires_per_pair: int = 1,
                 include_hh: bool = True, seed: int = 0):
        super().__init__()
        g_list = [g.copy() for g in graphs if g.number_of_nodes() >= 2]
        rng = nx.utils.create_random_state(seed)
        items: List[PairItem] = []
        for G in g_list:
            n = G.number_of_nodes()
            if include_hh:
                H = havel_hakimi_construction(G)
                d = symmetric_edit_distance(G, H)
                items.append(PairItem(G, H, d, n))
            # Random rewiring based pairs
            for _ in range(pairs_per_graph):
                H = G.copy()
                for _ in range(max(1, rewires_per_pair)):
                    if H.number_of_edges() < 2:
                        break
                    # pick two edges uniformly and try a valid rewiring
                    e_list = list(H.edges())
                    e1 = e_list[rng.randint(0, len(e_list))]
                    e2 = e_list[rng.randint(0, len(e_list))]
                    if len(set(e1 + e2)) < 4:
                        continue
                    u, v = e1
                    x, y = e2
                    candidates = [(u, x, v, y), (u, y, v, x)]
                    a, b, c, d_ = candidates[rng.randint(0, 2)]
                    if not H.has_edge(a, b) and not H.has_edge(c, d_) and a != b and c != d_ and a != c and b != d_:
                        H.remove_edge(u, v)
                        H.remove_edge(x, y)
                        H.add_edge(a, b)
                        H.add_edge(c, d_)
                d = symmetric_edit_distance(G, H)
                items.append(PairItem(G, H, d, n))
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        G1, G2 = item.G1, item.G2
        n = item.n_nodes
        max_possible = n * (n - 1) // 2
        d_raw = item.sed_raw
        d_norm = d_raw / max_possible if max_possible > 0 else 0.0
        data1 = to_pyg_data(G1)
        data2 = to_pyg_data(G2)
        return data1, data2, torch.tensor(d_norm, dtype=torch.float32), torch.tensor(d_raw, dtype=torch.float32)

# ------------------------------
# Training / Evaluation
# ------------------------------

def train_epoch(model: GraphSEDModel, loader: DataLoader, lr, log_every: int = 100):
    opt = Adam(model.parameters(), lr=lr)
    model.train()
    total, total_raw = 0.0, 0.0
    n_batches = 0
    for step, (d1, d2, d_norm, d_raw) in enumerate(loader, 1):
        d1 = d1
        d2 = d2
        d_norm = d_norm
        pred = model(d1, d2)
        loss = sed_regression_loss(pred, d_norm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
        total_raw += torch.mean(torch.abs(pred.detach() - d_norm)).item()
        n_batches += 1
        if log_every and step % log_every == 0:
            print(f"  step {step:04d} | loss={loss.item():.4f} | L1(norm)={total_raw/n_batches:.4f}")
    return total / max(1, n_batches)


@torch.no_grad()
def eval_epoch(model: GraphSEDModel, loader: DataLoader


    ):
    model.eval()
    total_mse, total_l1, total_raw_l1 = 0.0, 0.0, 0.0
    n_batches = 0
    for d1, d2, d_norm, d_raw in loader:
        d1 = d1
        d2 = d2
        d_norm = d_norm
        pred = model(d1, d2)
        total_mse += F.mse_loss(pred, d_norm).item()
        total_l1 += torch.mean(torch.abs(pred - d_norm)).item()
        n_batches += 1
    return {
        "mse": total_mse / max(1, n_batches),
        "l1_norm": total_l1 / max(1, n_batches),
    }


def main(args):
    config_dir = Path("configs")
    dataset_dir = Path("datasets") / args.dataset_dir
    model_dir = Path("models")
    config = toml.load(config_dir / args.config)
    graphs, max_node = load_graph_from_directory(dataset_dir)
    print(f"Loading graphs dataset {len(graphs)}")

    dataset = GraphPairSEDataset(
        graphs,
        pairs_per_graph=config['training']['pairs_per_graph'],
        rewires_per_pair=config['training']['rewires_per_pair'],
        include_hh=True
    )
    # Split 80/20
    n_total = len(dataset)
    n_train = max(1, int(0.8 * n_total))
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_total - n_train])

    def pyg_collate(batch):
        d1_list, d2_list, d_norm_list, d_raw_list = [], [], [], []
        for d1, d2, d_norm, d_raw in batch:
            d1_list.append(d1)
            d2_list.append(d2)
            d_norm_list.append(d_norm)
            d_raw_list.append(d_raw)
        return (
            torch_geometric.data.Batch.from_data_list(d1_list),
            torch_geometric.data.Batch.from_data_list(d2_list),
            torch.stack(d_norm_list, dim=0),
            torch.stack(d_raw_list, dim=0),
        )

    # PyG provides a Batch class, but to avoid extra imports we can leverage its loader directly
    # We therefore use two DataLoaders in parallel via a small custom dataset that already emits pairs
    train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config['training']['batch_size'], shuffle=False)

    model = GraphSEDModel(
        in_channels=config['training']['in_channels'],
        hidden_dim=config['training']['hidden_dim'],
        num_layers=config['training']['num_layers'],
        proj_dim=config['training']['proj_dim'],
        enc_dropout=0.1,
        l2_normalize=True,
        dec_hidden=config['training']['dec_hidden'],
        dec_dropout=0.1,
    )

    if args.input_model:
        model.load_model(model_dir / args.input_model)
        print(f"Model Graph-SED loaded from {args.input_model}")
    else:
        num_epochs = config['training']['num_epochs']
        learning_rate = config['training']['learning_rate']
        best_val = float("inf")
        for ep in range(1, num_epochs + 1):
            loss = train_epoch(model, train_loader, learning_rate)
            metrics = eval_epoch(model, val_loader)
            print(f"Epoch {ep:03d} | train/mse={loss:.4f} | val/mse={metrics['mse']:.4f} | val/L1(norm)={metrics['l1_norm']:.4f}")
    if args.output_model:
        model.save_model(model_dir / args.output_model)
        print(f"Model saved to {args.output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GRAPH-ER Model')
    parser.add_argument('--dataset-dir', type=str, required=True,help='Path to the directory containing graph files')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file in TOML format of Graph-ER')
    parser.add_argument('--output-model', type=str, help='Path to save the trained model')
    parser.add_argument('--input-model', type=str, help='Path to load a pre-trained model')
    args = parser.parse_args()
    main(args)