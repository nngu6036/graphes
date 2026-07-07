from __future__ import annotations

import networkx as nx
import torch

from grapher.molecular.attribute_flow import TopologyConditionalAttributeFlow, collate_molecular_graphs
from grapher.molecular.constants import BOND_SINGLE


def _toy_mol(n=4):
    g = nx.path_graph(n)
    for i in g.nodes():
        g.nodes[i]["atomic_num"] = 6 if i % 2 == 0 else 7
        g.nodes[i]["atom_type"] = g.nodes[i]["atomic_num"]
    for u, v in g.edges():
        g.edges[u, v]["bond_type"] = BOND_SINGLE
    return g


def test_attribute_flow_forward_loss_and_sample():
    graphs = [_toy_mol(4), _toy_mol(5)]
    batch = collate_molecular_graphs(graphs)
    model = TopologyConditionalAttributeFlow(hidden_dim=16, edge_dim=8, num_layers=2)
    loss, stats = model.loss(batch, device=torch.device("cpu"))
    assert torch.isfinite(loss)
    assert stats["node_loss"] >= 0
    assert stats["edge_loss"] >= 0
    sampled = model.sample_attributes(nx.path_graph(4), steps=2, device="cpu", seed=0)
    assert sampled.number_of_nodes() == 4
    assert sampled.number_of_edges() == 3
    for _, data in sampled.nodes(data=True):
        assert "atomic_num" in data
    for _, _, data in sampled.edges(data=True):
        assert "bond_type" in data
