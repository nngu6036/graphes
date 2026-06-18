from __future__ import annotations

import random
import unittest

import networkx as nx

from grapher.generation.rewiring import (
    action_structural_delta,
    enumerate_rewire_actions,
    rewire_action,
)


class StructuralRewiringDeltaTest(unittest.TestCase):
    def test_deltas_match_explicit_graph_recomputation(self) -> None:
        rng = random.Random(0)
        checked = 0
        for seed in range(20):
            graph = nx.erdos_renyi_graph(20, 0.25, seed=seed)
            if not nx.is_connected(graph):
                continue
            actions = enumerate_rewire_actions(
                graph,
                ensure_connected=True,
                k_hop=None,
                max_candidates=40,
                rng=rng,
                shuffle=True,
            )
            for action in actions:
                before_triangles = sum(nx.triangles(graph).values()) // 3
                before_clustering = nx.average_clustering(graph)
                delta = action_structural_delta(graph, action)
                output = rewire_action(graph, action, ensure_connected=True)
                self.assertIsNotNone(output)
                candidate = output[0]
                after_triangles = sum(nx.triangles(candidate).values()) // 3
                after_clustering = nx.average_clustering(candidate)
                self.assertEqual(after_triangles - before_triangles, delta.triangle_delta)
                self.assertAlmostEqual(
                    after_clustering - before_clustering,
                    delta.average_clustering_delta,
                    places=12,
                )
                checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
