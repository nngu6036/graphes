from __future__ import annotations

import networkx as nx
import torch

from grapher.molecular.constants import BOND_SINGLE
from grapher.molecular.attribute_flow import collate_molecular_graphs
from grapher.molecular.mixture_catflow import TopologyConditionalMixtureCatFlow


def _toy_molecule():
    g = nx.Graph()
    g.add_node(0, atomic_num=6, atom_type=6)
    g.add_node(1, atomic_num=8, atom_type=8)
    g.add_node(2, atomic_num=9, atom_type=9)
    g.add_edge(0, 1, bond_type=BOND_SINGLE)
    g.add_edge(0, 2, bond_type=BOND_SINGLE)
    return g


def test_mixture_catflow_loss_and_sample():
    graphs = [_toy_molecule(), _toy_molecule()]
    batch = collate_molecular_graphs(graphs)
    model = TopologyConditionalMixtureCatFlow(hidden_dim=32, edge_dim=16, num_layers=2, num_mixtures=2)
    loss, stats = model.loss(batch, device=torch.device("cpu"))
    assert torch.isfinite(loss)
    assert stats["node_loss"] > 0
    assert stats["edge_loss"] > 0

    topology = nx.Graph()
    topology.add_nodes_from([0, 1, 2])
    topology.add_edges_from([(0, 1), (0, 2)])
    out = model.sample_attributes(topology, steps=2, device="cpu", seed=0)
    assert out.number_of_nodes() == 3
    assert out.number_of_edges() == 2
    for _, data in out.nodes(data=True):
        assert "atomic_num" in data
    for _, _, data in out.edges(data=True):
        assert "bond_type" in data
