"""Real MILP regressions for connected exact-degree realization."""
from __future__ import annotations

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np

from grapher.models.grapher import degree_constrained_realization as realization


class ConnectedDegreeRealizationTests(unittest.TestCase):
    seed = 417

    @staticmethod
    def config(*, cut_rounds: int, connected: bool = True) -> dict:
        return {
            "require_connected": connected,
            "max_connectivity_cut_rounds": cut_rounds,
            "spectral_weight": 1.0,
            "gumbel_temperature": 0.1,
            "solver_time_limit_seconds": 15.0,
            "solver_mip_rel_gap": 0.0,
        }

    @staticmethod
    def triangle_scores(n: int) -> np.ndarray:
        left, right = realization._edge_index(n)
        # Separate triangle groups dominate the initial unconstrained optimum.
        return np.where(left // 3 == right // 3, 10.0, 0.0)

    def solve(self, n: int, scores: np.ndarray, config: dict):
        degrees = np.full(n, 2, dtype=np.int64)
        before = degrees.copy()
        rng = np.random.default_rng(self.seed)
        graph, diagnostics = realization._solve_exact_degree_realization(
            degrees, scores, rng=rng, config=config,
        )
        np.testing.assert_array_equal(degrees, before)
        self.assertEqual(list(graph.nodes()), list(range(n)))
        self.assertEqual([graph.degree(node) for node in range(n)], degrees.tolist())
        self.assertEqual(graph.number_of_edges(), n)
        self.assertEqual(nx.number_of_selfloops(graph), 0)
        return graph, diagnostics, rng

    def objective(self, scores: np.ndarray, config: dict) -> tuple[np.ndarray, dict]:
        rng = np.random.default_rng(self.seed)
        uniform = rng.uniform(
            low=np.finfo(np.float64).eps,
            high=1.0 - np.finfo(np.float64).eps,
            size=len(scores),
        )
        gumbel = -np.log(-np.log(uniform))
        coefficients = -(
            config["spectral_weight"] * scores + config["gumbel_temperature"] * gumbel
        )
        return coefficients, rng.bit_generator.state

    @staticmethod
    def graph_objective(graph: nx.Graph, coefficients: np.ndarray) -> float:
        left, right = realization._edge_index(graph.number_of_nodes())
        return float(sum(
            value for u, v, value in zip(left, right, coefficients)
            if graph.has_edge(int(u), int(v))
        ))

    def exhaustive_connected_optimum(self, coefficients: np.ndarray) -> float:
        # A connected simple 2-regular graph on six indexed nodes is a six-cycle.
        objectives = []
        for tail in itertools.permutations(range(1, 6)):
            if tail[0] > tail[-1]:
                continue
            cycle = nx.Graph()
            cycle.add_nodes_from(range(6))
            nx.add_cycle(cycle, [0, *tail])
            objectives.append(self.graph_objective(cycle, coefficients))
        self.assertEqual(len(objectives), 60)
        return min(objectives)

    def test_zero_cut_cap_uses_flow_and_preserves_exact_perturbed_optimum(self) -> None:
        config = self.config(cut_rounds=0)
        scores = self.triangle_scores(6)
        coefficients, expected_rng_state = self.objective(scores, config)
        graph, diagnostics, rng = self.solve(6, scores, config)
        self.assertTrue(nx.is_connected(graph))
        self.assertTrue(diagnostics["connectivity_flow_used"])
        self.assertEqual(diagnostics["connectivity_method"], "single_commodity_flow")
        self.assertEqual(diagnostics["connectivity_cut_rounds"], 0)
        optimum = self.exhaustive_connected_optimum(coefficients)
        self.assertAlmostEqual(self.graph_objective(graph, coefficients), optimum, places=7)
        self.assertAlmostEqual(diagnostics["objective"], optimum, places=7)
        self.assertEqual(rng.bit_generator.state, expected_rng_state)

    def test_cut_path_and_flow_use_the_same_objective_and_one_gumbel_draw(self) -> None:
        scores = self.triangle_scores(6)
        flow_graph, flow_diagnostics, flow_rng = self.solve(
            6, scores, self.config(cut_rounds=0)
        )
        cut_graph, cut_diagnostics, cut_rng = self.solve(
            6, scores, self.config(cut_rounds=16)
        )
        self.assertTrue(nx.is_connected(cut_graph))
        self.assertFalse(cut_diagnostics["connectivity_flow_used"])
        self.assertEqual(cut_diagnostics["connectivity_method"], "component_cuts")
        self.assertGreater(cut_diagnostics["connectivity_cut_rounds"], 0)
        self.assertAlmostEqual(
            flow_diagnostics["objective"], cut_diagnostics["objective"], places=7
        )
        self.assertEqual(set(flow_graph.edges()), set(cut_graph.edges()))
        self.assertEqual(flow_rng.bit_generator.state, cut_rng.bit_generator.state)

    def test_one_cut_round_can_finish_without_using_flow(self) -> None:
        graph, diagnostics, _ = self.solve(
            6, self.triangle_scores(6), self.config(cut_rounds=1)
        )
        self.assertTrue(nx.is_connected(graph))
        self.assertEqual(diagnostics["connectivity_cut_rounds"], 1)
        self.assertFalse(diagnostics["connectivity_flow_used"])

    def test_eighteen_nodes_switch_to_flow_after_one_insufficient_cut_round(self) -> None:
        n = 18
        left, right = realization._edge_index(n)
        scores = self.triangle_scores(n)
        # After each triangle receives crossing edges, the preferred solution
        # still consists of three disconnected six-node groups. Connectivity
        # requires edges between these groups with their original lower scores.
        same_triangle_pair = (left // 6 == right // 6) & (left // 3 != right // 3)
        scores[same_triangle_pair] = 4.0
        config = self.config(cut_rounds=1)
        graph, diagnostics, rng = self.solve(n, scores, config)
        self.assertTrue(nx.is_connected(graph))
        self.assertTrue(diagnostics["connectivity_flow_used"])
        self.assertEqual(diagnostics["connectivity_method"], "single_commodity_flow")
        self.assertEqual(diagnostics["connectivity_cut_rounds"], 1)
        _, expected_rng_state = self.objective(scores, config)
        self.assertEqual(rng.bit_generator.state, expected_rng_state)

    def test_disconnected_protocol_retains_preferred_triangles_without_flow(self) -> None:
        scores = self.triangle_scores(6)
        graph, diagnostics, rng = self.solve(
            6, scores, self.config(cut_rounds=0, connected=False)
        )
        self.assertEqual(sorted(map(len, nx.connected_components(graph))), [3, 3])
        self.assertFalse(diagnostics["connectivity_flow_used"])
        self.assertEqual(diagnostics["connectivity_cut_rounds"], 0)
        _, expected_rng_state = self.objective(
            scores, self.config(cut_rounds=0, connected=False)
        )
        self.assertEqual(rng.bit_generator.state, expected_rng_state)

    def test_impossible_connected_degree_sequence_remains_an_error(self) -> None:
        degrees = np.ones(4, dtype=np.int64)
        scores = np.zeros(6, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        with self.assertRaisesRegex(RuntimeError, "MILP|infeasible"):
            realization._solve_exact_degree_realization(
                degrees, scores, rng=rng, config=self.config(cut_rounds=0),
            )
        np.testing.assert_array_equal(degrees, np.ones(4, dtype=np.int64))
        _, expected_rng_state = self.objective(scores, self.config(cut_rounds=0))
        self.assertEqual(rng.bit_generator.state, expected_rng_state)

    def test_flow_timeout_still_fails_without_retrying_or_resampling(self) -> None:
        scores = self.triangle_scores(6)
        config = self.config(cut_rounds=0)
        rng = np.random.default_rng(self.seed)
        timeout = SimpleNamespace(
            success=False, x=None, status=1, message="Time limit reached",
        )
        with patch.object(realization, "_solve_connected_flow", return_value=timeout) as flow:
            with self.assertRaisesRegex(RuntimeError, "MILP failed.*Time limit reached"):
                realization._solve_exact_degree_realization(
                    np.full(6, 2, dtype=np.int64), scores, rng=rng, config=config,
                )
        flow.assert_called_once()
        _, expected_rng_state = self.objective(scores, config)
        self.assertEqual(rng.bit_generator.state, expected_rng_state)


if __name__ == "__main__":
    unittest.main()
