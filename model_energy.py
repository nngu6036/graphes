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

class EnergyMLP(nn.Module):
    """
    E_theta(G) = MLP(z(G))  -> scalar energy (lower = preferred)

    Inputs:
      z: Tensor[*, d]   graph embedding from your shared encoder (GIN/spectral/etc.)

    Outputs:
      energy: Tensor[*]  scalar energy (no sigmoid; raw score)
    """
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()

        act_layer = nn.GELU()

        layers = []
        in_dim = emb_dim
        for i in range(num_layers - 1):
            out_dim = hidden_dim if i < num_layers - 2 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 2:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(act_layer)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.net = nn.Sequential(*layers)

        # Nice-to-have: initialize last layer small so energies start near 0
        last = None
        for m in reversed(self.net):
            if isinstance(m, nn.Linear):
                last = m
                break
        if last is not None:
            nn.init.zeros_(last.bias)
            nn.init.normal_(last.weight, mean=0.0, std=1e-3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # returns shape: [batch] (or [] if scalar)
        e = self.net(z).squeeze(-1)
        return e

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path, map_location="cpu"))
        self.eval()

