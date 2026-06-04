from __future__ import annotations

import networkx as nx

from grapher.datasets.base import BaseDatasetBuilder


class SBMDatasetBuilder(BaseDatasetBuilder):
    def build(self) -> dict[str, list[nx.Graph]]:
        num_graphs = int(self.config.get("num_graphs", 1024))
        num_nodes = int(self.config.get("num_nodes", 64))
        num_blocks = int(self.config.get("num_blocks", 4))
        p_in = float(self.config.get("p_in", 0.25))
        p_out = float(self.config.get("p_out", 0.02))

        base = num_nodes // num_blocks
        sizes = [base] * num_blocks
        for i in range(num_nodes - base * num_blocks):
            sizes[i % num_blocks] += 1
        probs = [[p_in if i == j else p_out for j in range(num_blocks)] for i in range(num_blocks)]

        graphs = []
        for i in range(num_graphs):
            graph = nx.stochastic_block_model(sizes, probs, seed=self.seed + i, selfloops=False)
            graph.graph["source_dataset"] = "sbm"
            graphs.append(graph)
        return self.finalize(graphs, shuffle=True)
