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

        # New options for GraphER-compatible SBM.
        # GraphER is defined over connected degree-constrained graphs, so for
        # the main GraphER benchmark we should sample SBM conditioned on
        # connectedness instead of training/evaluating against disconnected
        # raw SBM graphs.
        require_connected = bool(self.config.get("require_connected", True))
        reject_zero_degree = bool(self.config.get("reject_zero_degree", True))
        max_attempts_per_graph = int(self.config.get("max_attempts_per_graph", 30))

        if num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive, got {num_nodes}.")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}.")
        if num_blocks > num_nodes:
            raise ValueError(
                f"num_blocks={num_blocks} cannot exceed num_nodes={num_nodes}."
            )
        if not (0.0 <= p_in <= 1.0):
            raise ValueError(f"p_in must be in [0, 1], got {p_in}.")
        if not (0.0 <= p_out <= 1.0):
            raise ValueError(f"p_out must be in [0, 1], got {p_out}.")
        if max_attempts_per_graph <= 0:
            raise ValueError(
                f"max_attempts_per_graph must be positive, got {max_attempts_per_graph}."
            )

        base = num_nodes // num_blocks
        sizes = [base] * num_blocks
        for i in range(num_nodes - base * num_blocks):
            sizes[i % num_blocks] += 1

        probs = [
            [p_in if i == j else p_out for j in range(num_blocks)]
            for i in range(num_blocks)
        ]

        def graph_is_acceptable(graph: nx.Graph) -> bool:
            if require_connected and graph.number_of_nodes() > 1:
                if not nx.is_connected(graph):
                    return False

            if reject_zero_degree and graph.number_of_nodes() > 1:
                if any(degree == 0 for _, degree in graph.degree()):
                    return False

            return True

        graphs: list[nx.Graph] = []
        attempts = 0
        max_attempts = num_graphs * max_attempts_per_graph

        while len(graphs) < num_graphs and attempts < max_attempts:
            graph_seed = int(self.seed + attempts)
            graph = nx.stochastic_block_model(
                sizes,
                probs,
                seed=graph_seed,
                selfloops=False,
            )

            # Keep labels compact and deterministic.
            graph = nx.convert_node_labels_to_integers(
                nx.Graph(graph),
                first_label=0,
                ordering="sorted",
            )

            attempts += 1

            if not graph_is_acceptable(graph):
                continue

            graph.graph["source_dataset"] = "sbm"
            graph.graph["conditioned_connected"] = bool(require_connected)
            graph.graph["reject_zero_degree"] = bool(reject_zero_degree)
            graph.graph["sbm_seed"] = graph_seed
            graph.graph["sbm_attempt_index"] = attempts - 1
            graph.graph["num_blocks"] = num_blocks
            graph.graph["block_sizes"] = list(sizes)
            graph.graph["p_in"] = p_in
            graph.graph["p_out"] = p_out

            graphs.append(graph)

        if len(graphs) < num_graphs:
            raise RuntimeError(
                "Could not generate enough acceptable SBM graphs. "
                f"Requested {num_graphs}, generated {len(graphs)}, "
                f"attempts {attempts}/{max_attempts}. "
                "Try increasing max_attempts_per_graph, increasing p_in/p_out, "
                "or disabling require_connected/reject_zero_degree."
            )

        return self.finalize(graphs, shuffle=True)