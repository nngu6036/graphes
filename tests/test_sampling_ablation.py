"""CPU contract tests for the added generation-only experiments.

Run with: PYTHONPATH=src python -m unittest discover -s tests -p test_sampling_ablation.py
These are synthetic fixtures, not benchmark results.
"""
from __future__ import annotations

import copy
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np
import torch

from grapher.models.grapher import attributed_diversification as molecular
from grapher.models.grapher import basis_sampling_ablation as bank
from grapher.models.grapher import eigenbasis_perturbation as perturbation
from grapher.models.grapher import sampling_ablation as sampling
from grapher.models.grapher.degree_constrained_realization import resolve_config
from grapher.models.grapher.graphlet_stage3 import default_fixed_target_graphlet_refinement_options
from grapher.models.grapher.matched_ablation import _sha
from grapher.properties.summary import SummaryConfig
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary, GraphletBasis
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.graphlets import extract_topology_graphlet_target


def attributed_cycle():
    graph = nx.cycle_graph(6)
    for i, atom in enumerate((6, 6, 7, 8, 6, 7)):
        graph.nodes[i]["atomic_num"] = atom
    nx.set_edge_attributes(graph, 1, "bond_type")
    return graph


class BasisTests(unittest.TestCase):
    def test_nested_reservoirs_and_node_count_matching(self):
        sizes = [4, 4, 4, 5, 5, 6]
        parents = [0, 3, 5, 1, 4, 2]
        small = bank.make_schedule(parents, sizes, seed=41, cap_per_size=1, donor_offset=1)
        large = bank.make_schedule(parents, sizes, seed=41, cap_per_size=2, donor_offset=1)
        for n, indices in small["design"]["reservoir_original_indices"].items():
            self.assertTrue(set(indices) <= set(large["design"]["reservoir_original_indices"][n]))
        self.assertEqual(small["expected_node_counts"], [sizes[i] for i in parents])
        np.testing.assert_array_equal(bank.resolve_schedule({"basis_ablation": small}, sizes, len(parents)), small["indices"])
        original = bank.make_schedule(parents, sizes, seed=41)
        self.assertEqual(original["indices"], parents)
        invalid = copy.deepcopy(small); invalid["indices"][0] = 5
        with self.assertRaisesRegex(ValueError, "node-count"):
            bank.resolve_schedule({"basis_ablation": invalid}, sizes, len(parents))

    def test_schedule_preserves_other_rng_consumption(self):
        a, b = np.random.default_rng(7), np.random.default_rng(7)
        expected = a.integers(0, 5, size=3)
        actual = bank.select_indices(b, 5, 3, 0, np.array([1, 2, 3]))
        np.testing.assert_array_equal(actual, [1, 2, 3])
        self.assertEqual(a.random(), b.random())
        self.assertEqual(expected.shape, (3,))

    def test_repeated_eigenspace_and_sign_preserve_operator(self):
        graph = nx.complete_graph(4)
        adjacency = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float64)
        lap = torch.diag(adjacency.sum(1)) - adjacency
        values, basis = torch.linalg.eigh(lap)
        for method in ("givens", "sign"):
            cfg = perturbation.resolve_config({"eigenbasis_perturbation": {
                "enabled": True, "stage": "reverse", "pairing": "repeated", "method": method,
                "sigma": 0.5, "num_rotations": 8}})
            changed, rows = perturbation.perturb_laplacian_basis(
                basis[None], torch.tensor([4]), config=cfg, rng=np.random.default_rng(41), eigenvalues=values[None])
            np.testing.assert_allclose(changed[0].T @ changed[0], torch.eye(4), atol=1e-12)
            np.testing.assert_allclose(changed[0, :, 0], basis[:, 0], atol=0)
            self.assertLess(rows[0]["donor_operator_relative_change"], 1e-12)
        self.assertEqual(rows[0]["repeated_pairs_available"], 3)

    def test_repeated_rotation_does_not_rotate_distinct_modes(self):
        graph = nx.path_graph(5)
        adjacency = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float64)
        values, basis = torch.linalg.eigh(torch.diag(adjacency.sum(1)) - adjacency)
        cfg = perturbation.resolve_config({"eigenbasis_perturbation": {
            "enabled": True, "pairing": "repeated", "sigma": 1.0, "num_rotations": 8}})
        changed, rows = perturbation.perturb_laplacian_basis(
            basis[None], torch.tensor([5]), config=cfg, rng=np.random.default_rng(41), eigenvalues=values[None])
        self.assertTrue(torch.equal(changed[0], basis))
        self.assertEqual(rows[0]["applied_rotations"], 0)
        with self.assertRaisesRegex(ValueError, "donor eigenvalues"):
            perturbation.perturb_laplacian_basis(basis[None], torch.tensor([5]), config=cfg, rng=np.random.default_rng(41))


class DecoderTests(unittest.TestCase):
    def test_fixed_degrees_and_repeatable_common_noise(self):
        scores = np.random.default_rng(31).normal(size=(6, 6))
        config = resolve_config({"degree_constrained_realization": {"enabled": True, "require_connected": True}})
        for temperature in (0.0, 0.1, 0.4, 0.8):
            graphs = [sampling.decode_temperature(scores, [2] * 6, config, seed=113, temperature=temperature)[0]
                      for _ in range(2)]
            self.assertEqual(set(graphs[0].edges()), set(graphs[1].edges()))
            self.assertEqual([graphs[0].degree(i) for i in range(6)], [2] * 6)
            self.assertTrue(nx.is_connected(graphs[0]))

    def test_matched_budget_draws(self):
        for i in range(50):
            b2 = sampling.sample_budget(41, i, 1, 2)
            b8 = sampling.sample_budget(41, i, 1, 8)
            self.assertEqual(b2, 1 + (b8 - 1) // 4)
            self.assertEqual(sampling.sample_budget(41, i, 0, 0), 0)


class MolecularMaskTests(unittest.TestCase):
    def test_relaxed_mask_admits_only_additional_same_bond_candidates(self):
        graph = attributed_cycle()
        signature = molecular._neighbor_atom_bond_signature(graph, node_attribute="atomic_num", edge_attribute="bond_type")
        valence = molecular._weighted_bond_valence_vector(graph, edge_attribute="bond_type")
        outcomes = {}
        for strict in (False, True):
            cfg = molecular.resolve_options({"enabled": True, "rdkit_candidate_filter": False,
                                             "preserve_neighbor_atom_type_by_bond_type": strict})
            outcomes[strict] = list(molecular._candidate_graphs(
                graph, edge_attribute="bond_type", node_attribute="atomic_num",
                neighbor_atom_bond_invariant=signature, bond_valence_invariant=valence,
                cfg=cfg, rng=np.random.default_rng(4), seen=set(), stats=Counter()))
            for _, candidate, _ in outcomes[strict]:
                self.assertEqual(dict(candidate.degree()), dict(graph.degree()))
                self.assertEqual(molecular._weighted_bond_valence_vector(candidate, edge_attribute="bond_type"), valence)
                self.assertTrue(nx.is_connected(candidate))
        self.assertTrue({row[2] for row in outcomes[True]} <= {row[2] for row in outcomes[False]})
        self.assertTrue(any(molecular._neighbor_atom_bond_signature(candidate, node_attribute="atomic_num", edge_attribute="bond_type") != signature
                            for _, candidate, _ in outcomes[False]))

    def test_full_same_bond_and_strict_trajectories(self):
        graph = attributed_cycle()
        summary = SummaryConfig(graphlet_history=True, graphlet_k_min=3, graphlet_k_max=3,
                                graphlet_backend="exact", clustering_summary=False)
        top_basis = TopologyGraphletBasis.from_config(summary)
        vocab = GraphCategoryVocabulary.from_graphs([graph], {"node_attribute": "atomic_num", "edge_attribute": "bond_type"})
        typed_basis = GraphletBasis.fit_from_graphs([graph], summary, vocabulary=vocab, attributed=True)
        top_hist, top_mass = extract_topology_graphlet_target(graph, graphlet_basis=top_basis, summary_config=summary)
        typed_hist, typed_mass = molecular._typed_graphlet_summary(graph, typed_basis, summary)
        for strict in (False, True):
            cfg = {"enabled": True, "rdkit_candidate_filter": False,
                   "preserve_neighbor_atom_type_by_bond_type": strict,
                   "random_steps": {"min_steps": 1, "max_steps": 1},
                   "guidance": {"topology_orbit_weight": 0.0},
                   "target_trust_region": {"enabled": False},
                   "soft_selection": {"max_target_worsening": 1e9, "max_relative_target_worsening": 1e9}}
            result, row = molecular.diversify_same_bond_graph(
                graph, topology_prediction={"graphlet_histogram": top_hist, "graphlet_mass": top_mass},
                attribute_prediction={"typed_graphlet_histogram": typed_hist, "typed_graphlet_mass": typed_mass},
                topology_basis=top_basis, typed_basis=typed_basis, topology_summary_config=summary,
                typed_summary_config=summary, vocabulary=vocab, config=cfg, rng=np.random.default_rng(5))
            self.assertEqual(dict(result.degree()), dict(graph.degree()))
            self.assertEqual(row["preserve_neighbor_atom_type_by_bond_type"], strict)
            self.assertEqual(row["requested_steps"], 1)
            self.assertTrue(row["typed_degree_preserved"])
            if strict:
                self.assertTrue(row["neighbor_atom_bond_signature_preserved"])


class SavedEndpointTests(unittest.TestCase):
    def test_saved_populations_budget_temperature_export_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); parent = directory / "parent"; parent.mkdir()
            graphs = [nx.cycle_graph(6), nx.cycle_graph(6)]
            summary = SummaryConfig(graphlet_history=True, graphlet_backend="exact", clustering_summary=False)
            basis = TopologyGraphletBasis.from_config(summary)
            predictions = []
            for i, graph in enumerate(graphs):
                hist, mass = extract_topology_graphlet_target(graph, graphlet_basis=basis, summary_config=summary)
                predictions.append({"graph_index": i, "graphlet_histogram": hist, "graphlet_mass": mass})
            checkpoint = directory / "checkpoint.pt"
            torch.save({"structure_summary": {"enabled": True, "graphlet_basis": basis.to_dict(), "summary_config": asdict(summary)}}, checkpoint)
            payloads = {"pre_realization_graphs": graphs, "pre_refinement_graphs": graphs,
                        "predicted_structure_summaries": predictions,
                        "continuous_adjacencies": [nx.to_numpy_array(graph) for graph in graphs],
                        "projected_realization_degree_sequences": [[2] * 6] * 2}
            generation = {}
            for key, value in payloads.items():
                path = parent / (key + ".pkl")
                path.write_bytes(pickle.dumps(value))
                generation[key] = {"path": path.name, "sha256": _sha(path)}
            (parent / "manifest.json").write_text(json.dumps(generation))
            cfg = {"sample": {"degree_constrained_realization": {"enabled": True}},
                   "graphlet_refinement": default_fixed_target_graphlet_refinement_options()}
            context = {"checkpoint_sha256": _sha(checkpoint), "seed": 41}
            args = SimpleNamespace(study="rewiring_budget", generation_dir=parent, output_root=directory / "budget",
                                   seed=41, num_samples=2, device="cpu", perturbation_sigma=.5, rotations=8,
                                   repeated_atol=1e-6, repeated_rtol=1e-5)
            arms = [{"name": f"{d}_B{b}", "decoder": d, "budget_min": b, "budget_max": b}
                    for d in ("T", "D") for b in (0, 1)]
            result = sampling.run_study(args, cfg, generation, context, checkpoint, arms)
            self.assertEqual(result["failed_count"], 0)
            before = _sha(parent / "pre_realization_graphs.pkl")
            self.assertEqual(sampling.run_study(args, cfg, generation, context, checkpoint, arms)["failed_count"], 0)
            self.assertEqual(before, _sha(parent / "pre_realization_graphs.pkl"))
            args.study = "gumbel_temperature"; args.output_root = directory / "temperature"
            with patch.object(sampling, "decode_temperature", wraps=sampling.decode_temperature) as decoded:
                result = sampling.run_study(args, cfg, generation, context, checkpoint,
                                            [{"name": "D_tau0p4", "temperature": .4, "budget": 0}])
                self.assertEqual(decoded.call_count, 2)  # no second realization for DR
            self.assertEqual(result["failed_count"], 0)
            branch = json.loads((args.output_root / "D_tau0p4" / "manifest.json").read_text())
            rewired = json.loads((args.output_root / "DR_tau0p4" / "manifest.json").read_text())
            self.assertEqual(rewired["paired_decoder_artifact"]["sha256"], branch["base_graphs"]["sha256"])
            self.assertEqual(branch["requested_sample_ids"], ["41:0", "41:1"])
            self.assertGreaterEqual(branch["time_seconds"], 0)
            (args.output_root / "D_tau0p4" / "base_graphs.pkl").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                sampling.run_study(args, cfg, generation, context, checkpoint,
                                   [{"name": "D_tau0p4", "temperature": .4, "budget": 0}])
            args.output_root = directory / "temperature_failure"
            with patch.object(sampling, "decode_temperature", side_effect=[RuntimeError("synthetic solver failure"), (graphs[1], {})]) as decoded:
                result = sampling.run_study(args, cfg, generation, context, checkpoint,
                                            [{"name": "D_tau0p4", "temperature": .4, "budget": 0}])
                self.assertEqual(decoded.call_count, 2)
            self.assertEqual(result["failed_count"], 2)
            rewired = json.loads((args.output_root / "DR_tau0p4" / "manifest.json").read_text())
            self.assertEqual(rewired["sample_ids"], ["41:1"])
            self.assertTrue(rewired["failures"][0]["inherited_decoder_failure"])
            self.assertFalse(rewired["failures"][0]["rewiring_attempted"])


class BasisRunnerTests(unittest.TestCase):
    def fixture(self, root):
        run = root / "baselines" / "grapher" / "community_small" / "synthetic"
        parent = run / "generations" / "parent"; parent.mkdir(parents=True)
        checkpoint = root / "synthetic_checkpoint.pt"
        graphs = [nx.cycle_graph(4), nx.complete_graph(4)]
        torch.save({"basis_num_nodes": np.array([4, 4]),
                    "basis_adjacencies": np.stack([nx.to_numpy_array(g) for g in graphs])}, checkpoint)
        path = parent / "sampled_basis_indices.pkl"; path.write_bytes(pickle.dumps([0, 1, 0, 1]))
        generation = {"sampled_basis_indices": {"path": path.name, "sha256": _sha(path)}}
        (parent / "manifest.json").write_text(json.dumps(generation))
        args = SimpleNamespace(study="basis_reservoir", dataset="community_small", kind="matched",
                               run_dir=run, generation_dir=parent, output_root=root / "basis",
                               seed=41, num_samples=4, device="cpu", perturbation_sigma=.5, rotations=8,
                               repeated_atol=1e-6, repeated_rtol=1e-5)
        context = {"checkpoint_sha256": _sha(checkpoint), "seed": 41}
        cfg = {"sample": {}, "generation_batch_size": 2,
               "graphlet_refinement": {"enabled": True}}
        return args, cfg, generation, context, checkpoint

    def test_basis_runner_publication_and_interrupted_export_resume(self):
        from grapher.models.grapher import vanilla_gsdm
        with tempfile.TemporaryDirectory() as tmp:
            args, cfg, generation, context, checkpoint = self.fixture(Path(tmp))
            arms = [{"name": "B_reservoir_1", "control": "reservoir", "cap_per_size": 1}]
            def fake_generate(wrapper, request):
                directory = request.run.layout.generation_dir(request.generation_id); directory.mkdir()
                schedule = request.options["sample"]["basis_ablation"]
                vanilla_gsdm.sample_batch(donor_adjacencies=torch.zeros(4, 4, 4))
                records = {}
                for key, value in {"base_graphs": [nx.cycle_graph(4) for _ in range(4)],
                                   "pre_refinement_graphs": [nx.cycle_graph(4) for _ in range(4)],
                                   "sampled_basis_indices": schedule["indices"]}.items():
                    path = directory / (key + ".pkl"); path.write_bytes(pickle.dumps(value))
                    records[key] = {"path": path.name, "sha256": _sha(path)}
                manifest = {**records, "checkpoint": {"sha256": _sha(request.checkpoint_path)},
                            "num_generated": 4, "generation_seed": 41, "duration_seconds": .1,
                            "effective_sample_config": request.options["sample"]}
                (directory / "manifest.json").write_text(json.dumps(manifest))
            with (patch("grapher.models.grapher.wrapper.GraphERWrapper.generate", autospec=True, side_effect=fake_generate) as mocked,
                  patch.object(vanilla_gsdm, "sample_batch", return_value=None),
                  patch.object(sampling, "_build_basis_refiner", return_value=lambda i, graph: (graph.copy(), {"requested_steps": 1, "accepted_steps": 0}))):
                result = sampling.run_study(args, cfg, generation, context, checkpoint, arms)
                self.assertEqual(result["failed_count"], 0)
                self.assertEqual(mocked.call_count, 1)
                branch = json.loads((args.output_root / "B_reservoir_1_TR" / "manifest.json").read_text())
                self.assertEqual(branch["reverse_measurement"]["attempted_reverse_samples"], 4)
                self.assertTrue(branch["shared_context"]["reverse_diffusion_rerun"])
                saved = _sha(args.output_root / "B_reservoir_1_TR" / "base_graphs.pkl")
                shutil.rmtree(args.output_root / "B_reservoir_1_TR")
                result = sampling.run_study(args, cfg, generation, context, checkpoint, arms)
                self.assertEqual(result["failed_count"], 0)
                self.assertEqual(mocked.call_count, 1)
                self.assertEqual(saved, _sha(args.output_root / "B_reservoir_1_TR" / "base_graphs.pkl"))

    def test_basis_abort_counts_actual_attempts_without_redraw(self):
        from grapher.models.grapher import vanilla_gsdm
        with tempfile.TemporaryDirectory() as tmp:
            args, cfg, generation, context, checkpoint = self.fixture(Path(tmp))
            def fail_generate(wrapper, request):
                vanilla_gsdm.sample_batch(donor_adjacencies=torch.zeros(2, 4, 4))
                raise RuntimeError("synthetic downstream failure")
            with (patch("grapher.models.grapher.wrapper.GraphERWrapper.generate", autospec=True, side_effect=fail_generate) as mocked,
                  patch.object(vanilla_gsdm, "sample_batch", return_value=None)):
                result = sampling.run_study(args, cfg, generation, context, checkpoint,
                                            [{"name": "B_reservoir_1", "control": "reservoir", "cap_per_size": 1}])
                self.assertEqual(mocked.call_count, 1)
                self.assertEqual(result["failed_count"], 8)  # four missing outputs in each T/TR view
                branch = json.loads((args.output_root / "B_reservoir_1_TR" / "manifest.json").read_text())
                self.assertEqual(branch["attempted_count"], 2)
                self.assertEqual([r["reverse_attempted"] for r in branch["failures"]], [True, True, False, False])


class CLIValidationTests(unittest.TestCase):
    def command(self, root):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_grapher_sampling_ablation.py"
        return [sys.executable, str(script), "--study", "gumbel_temperature", "--config", str(root / "config.yaml"),
                "--run-dir", str(root / "run"), "--generation-dir", str(root / "run/generations/parent"),
                "--dataset", "community_small", "--serialized-dataset", "sbm", "--dataset-root", str(root / "data"),
                "--seed", "41", "--num-samples", "2", "--output-root", str(root / "result")]

    def test_dry_run_lists_real_paired_variants_without_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subprocess.run(self.command(root) + ["--dry-run"], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertIn("D_tau0p4", plan["exported_variants"])
            self.assertIn("DR_tau0p4", plan["exported_variants"])
            self.assertEqual(len(list(root.iterdir())), 0)

    def test_old_epoch_budget_is_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.yaml").write_text("grapher:\n  train:\n    epochs: 10000\n")
            result = subprocess.run(self.command(root), text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("200-epoch checkpoint", result.stderr)
            self.assertFalse((root / "result").exists())


if __name__ == "__main__":
    unittest.main()
