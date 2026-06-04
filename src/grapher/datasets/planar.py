from __future__ import annotations

import itertools

import networkx as nx

from grapher.datasets.base import BaseDatasetBuilder


class PlanarDatasetBuilder(BaseDatasetBuilder):
    def build(self) -> dict[str, list[nx.Graph]]:
        num_graphs = int(self.config.get("num_graphs", 1024))
        num_nodes = int(self.config.get("num_nodes", 64))
        target_edges = self.config.get("target_edges")
        max_edges = 3 * num_nodes - 6 if num_nodes >= 3 else max(num_nodes - 1, 0)
        target = min(int(target_edges) if target_edges is not None else max_edges, max_edges)
        rng = self.rng
        graphs = []
        for _ in range(num_graphs):
            graph = nx.random_tree(num_nodes, seed=int(rng.integers(0, 2**31 - 1)))
            candidates = list(itertools.combinations(range(num_nodes), 2))
            rng.shuffle(candidates)
            for u, v in candidates:
                if graph.number_of_edges() >= target:
                    break
                if graph.has_edge(u, v):
                    continue
                graph.add_edge(u, v)
                if not nx.check_planarity(graph)[0]:
                    graph.remove_edge(u, v)
            graph.graph["source_dataset"] = "planar"
            graphs.append(graph)
        return self.finalize(graphs, shuffle=True)
