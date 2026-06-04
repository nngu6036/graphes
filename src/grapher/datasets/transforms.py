from __future__ import annotations
import networkx as nx

def to_adjacency_tensor(graph: nx.Graph):
    return nx.to_numpy_array(graph, dtype=float)
