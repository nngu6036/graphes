"""Saved-endpoint matched export regressions; every artifact is temporary."""
from __future__ import annotations

import copy
import importlib.util
import io
import hashlib
import json
import pickle
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
import torch
import yaml

from grapher.models.grapher import matched_ablation as exporter


class MatchedExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="matched-export-regression-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "run" / "generations" / "paper_seed_41"
        self.directory.mkdir(parents=True)
        self.checkpoint = self.root / "run" / "train" / "checkpoints" / "grapher.pt"
        self.checkpoint.parent.mkdir(parents=True)
        torch.save({"structure_summary": {
            "enabled": True, "graphlet_basis": {}, "summary_config": {},
        }}, self.checkpoint)
        self.cfg = {
            "variant": "vanilla_laplacian_loggap_graphlet",
            "sde": {"eps": 1e-5},
            "generation_batch_size": 2,
            "sample": {
                "threshold": 0.5,
                "matched_ablation": {"enabled": True, "include_d0": True, "include_budget_sweep": True},
                "degree_constrained_realization": {"enabled": True, "require_connected": True},
            },
            "graphlet_refinement": {
                "enabled": True, "mode": "stochastic_diversification",
                "target_source": "joint_structure_summary_head",
                "random_steps": {"enabled": True, "distribution": "uniform_integer", "min_steps": 1, "max_steps": 8},
                "refiner": {"steps": 8, "preserve_connectivity": True},
            },
        }
        self.context = {
            "seed": 41, "num_samples": 2, "run_id": "run",
            "generation_id": self.directory.name,
            "checkpoint_sha256": self.sha(self.checkpoint),
        }
        self.manifest = {
            "model_id": "grapher", "run_id": "run", "generation_id": self.directory.name,
            "generation_seed": 41, "num_requested": 2, "num_generated": 2,
            "variant": self.cfg["variant"], "duration_seconds": 123.0,
            "checkpoint": {"path": str(self.checkpoint), "sha256": self.sha(self.checkpoint)},
            "sampling": {
                "threshold": 0.5,
                "stochastic_diversification": {
                    key: copy.deepcopy(self.cfg["graphlet_refinement"][key])
                    for key in ("random_steps", "refiner", "target_source")
                },
            },
        }
        self.save("pre_realization_graphs", [nx.path_graph(4), nx.cycle_graph(4)])
        self.save("pre_refinement_graphs", [nx.cycle_graph(4), nx.cycle_graph(4)])
        self.save("base_graphs", [nx.cycle_graph(4), nx.cycle_graph(4)])
        self.save("projected_realization_degree_sequences", [[2] * 4, [2] * 4])
        self.save("predicted_structure_summaries", [
            {"graph_index": index, "graphlet_histogram": [0.0], "graphlet_mass": [0.0]}
            for index in range(2)
        ])
        self.save("sampled_basis_indices", [7, 13])
        self.save("graphlet_refinement_diagnostics", [
            {"graph_index": 0, "requested_steps": 2},
            {"graph_index": 1, "requested_steps": 7},
        ])
        self.save("continuous_adjacencies", [nx.to_numpy_array(nx.cycle_graph(4)) for _ in range(2)])
        self.write_manifest()
        self.calls = []

        def refine(graph, prediction, payload, cfg, seed, index, budget):
            self.calls.append((index, budget, seed))
            return graph.copy(), {"graph_index": index, "requested_steps": budget}

        self.refine = patch.object(exporter, "_refine_one", side_effect=refine)
        self.mock_refine = self.refine.start()
        self.addCleanup(self.refine.stop)
        self.decode = patch.object(exporter, "_decode_zero", side_effect=lambda *args: (nx.cycle_graph(4), {}))
        self.mock_decode = self.decode.start()
        self.addCleanup(self.decode.stop)

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def save(self, key, value):
        path = self.directory / f"{key}.pkl"
        path.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        self.manifest[key] = {"path": path.name, "sha256": self.sha(path)}
        return path

    def write_manifest(self):
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def snapshot(self):
        return {path.relative_to(self.root).as_posix(): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}

    def export(self):
        return exporter.export_matched(self.directory, self.cfg, self.context, self.checkpoint)

    def common(self):
        return json.loads((self.directory / exporter.COMMON).read_text())

    def test_export_preserves_all_parent_bytes_and_exact_copy_branches(self):
        before = self.snapshot()
        result = self.export()
        self.assertEqual(result, self.directory / exporter.COMMON)
        for path, content in before.items():
            self.assertEqual((self.root / path).read_bytes(), content, path)
        common = self.common()
        self.assertEqual(common, json.loads((self.directory / "ablations/ablation_manifest.json").read_text()))
        self.assertEqual(common["sample_ids"], ["41:0", "41:1"])
        self.assertEqual(set(common["branches"]), {"T", "D", "TR", "DR", "D0", "R1", "R12"})
        self.assertNotIn("checkpoint", common["provenance"]["source_artifacts"])
        for variant, source in (("T", "pre_realization_graphs"), ("D", "pre_refinement_graphs"), ("DR", "base_graphs")):
            child = common["branches"][variant]
            original = self.directory / self.manifest[source]["path"]
            copied = self.directory / "ablations" / variant / "base_graphs.pkl"
            self.assertEqual(copied.read_bytes(), original.read_bytes())
            self.assertEqual(child["identity_source"], exporter.IDENTITY)
            self.assertEqual(child["source_artifact"], self.manifest[source])

    def test_dr_actual_budgets_are_preserved_and_controls_are_isolated(self):
        self.export()
        common = self.common()
        self.assertEqual(common["saved_dr_requested_steps"], [2, 7])
        self.assertEqual(common["configured_dr_budget"]["max_steps"], 8)
        self.assertEqual([budget for _, budget, _ in self.calls[:2]], [2, 7])
        self.assertEqual([budget for _, budget, _ in self.calls[2:4]], [1, 1])
        self.assertTrue(all(budget in {1, 2} for _, budget, _ in self.calls[4:]))
        self.assertNotEqual(self.calls[0][2], self.calls[1][2])
        for variant in ("TR", "R1", "R12", "D0"):
            with (self.directory / "ablations" / variant / "base_graphs.pkl").open("rb") as stream:
                graphs = pickle.load(stream)
            self.assertEqual([graph.graph["sample_id"] for graph in graphs], ["41:0", "41:1"])

    def test_valid_existing_export_is_reused_without_deriving_again(self):
        self.export()
        before = self.snapshot()
        self.mock_refine.reset_mock()
        self.mock_decode.reset_mock()
        self.export()
        self.assertEqual(self.snapshot(), before)
        self.mock_refine.assert_not_called()
        self.mock_decode.assert_not_called()

    def test_report_only_failure_remnants_survive_byte_for_byte(self):
        report = self.directory / "ablations/T/evaluation_test/primary/evaluation_status.json"
        report.parent.mkdir(parents=True)
        report.write_bytes(b'{"status":"failed","exit_code":2}\n')
        before = report.read_bytes()
        self.export()
        self.assertEqual(report.read_bytes(), before)

    def test_publication_failure_restores_existing_report_tree(self):
        report = self.directory / "ablations/TR/evaluation_test/primary/evaluation_status.json"
        report.parent.mkdir(parents=True)
        report.write_bytes(b"existing report bytes")
        before = self.snapshot()
        with patch.object(exporter.os, "link", side_effect=OSError("publication failed")):
            with self.assertRaisesRegex(OSError, "publication failed"):
                self.export()
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(list(self.directory.glob(".matched_export_*")))

    def test_corrupt_parent_hash_is_rejected_before_derivation(self):
        (self.directory / self.manifest["base_graphs"]["path"]).write_bytes(b"corrupted")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.export()
        self.assertEqual(self.snapshot(), before)
        self.mock_refine.assert_not_called()

    def test_conflicting_embedded_parent_identity_is_rejected(self):
        graphs = [nx.cycle_graph(4), nx.cycle_graph(4)]
        graphs[0].graph["sample_id"] = "41:1"
        self.save("base_graphs", graphs)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "ID/order differs"):
            self.export()

    def test_prediction_order_is_verified(self):
        self.save("predicted_structure_summaries", [{"graph_index": 1}, {"graph_index": 0}])
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Prediction graph_index differs"):
            self.export()

    def test_parent_projected_degrees_are_verified(self):
        self.save("base_graphs", [nx.path_graph(4), nx.cycle_graph(4)])
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "DR degree target differs"):
            self.export()

    def test_d0_solver_failure_is_explicit_partial_without_replacements(self):
        self.mock_decode.side_effect = [(nx.cycle_graph(4), {}), RuntimeError("MILP timeout")]
        self.export()
        child = self.common()["branches"]["D0"]
        self.assertEqual(child["status"], "partial")
        self.assertEqual(child["sample_ids"], ["41:0"])
        self.assertEqual(child["completed_count"], 1)
        self.assertEqual(child["failed_count"], 1)
        self.assertEqual(child["failures"][0]["sample_id"], "41:1")
        self.assertFalse(child["failures"][0]["replacement_generated"])
        self.assertEqual(self.mock_decode.call_count, 2)
        self.export()
        self.assertEqual(self.mock_decode.call_count, 2)

    def test_controls_can_be_disabled(self):
        self.cfg["sample"]["matched_ablation"].update(include_d0=False, include_budget_sweep=False)
        self.export()
        self.assertEqual(set(self.common()["branches"]), {"T", "D", "TR", "DR"})
        self.mock_decode.assert_not_called()

    def test_conflicting_existing_branch_artifacts_are_not_overwritten(self):
        path = self.directory / "ablations/T/base_graphs.pkl"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"previous artifact")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "conflicting ablation artifacts"):
            self.export()
        self.assertEqual(self.snapshot(), before)

    def test_existing_export_hash_corruption_is_rejected(self):
        self.export()
        path = self.directory / "ablations/TR/base_graphs.pkl"
        path.write_bytes(b"damaged child")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.export()

    def test_checkpoint_identity_is_verified_before_loading(self):
        self.checkpoint.write_bytes(b"changed checkpoint")
        with self.assertRaisesRegex(ValueError, "Checkpoint hash differs"):
            self.export()

    def test_derived_rewiring_cannot_change_indexed_degrees(self):
        self.mock_refine.side_effect = lambda *args: (nx.empty_graph(4), {"graph_index": args[6]})
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Indexed degrees changed"):
            self.export()
        self.assertEqual(self.snapshot(), before)

    def test_resolved_producer_defaults_are_allowed_but_declared_settings_remain_strict(self):
        self.manifest["sampling"]["stochastic_diversification"]["refiner"]["extra_resolved_default"] = 123
        self.write_manifest()
        self.export()
        self.manifest["sampling"]["stochastic_diversification"]["refiner"]["steps"] = 999
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Saved setting differs"):
            self.export()

    def test_checker_prepare_seal_and_resume_accept_actual_export_format(self):
        script = Path(__file__).resolve().parents[1] / "scripts/check_grapher_ablation_artifacts.py"
        spec = importlib.util.spec_from_file_location("checker_export_integration", script)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        self.cfg["train"] = {"epochs": 200, "batch_size": 128, "lr": 0.01}
        config = self.root / "config.yaml"
        config.write_text(yaml.safe_dump({"grapher": self.cfg}), encoding="utf-8")
        prepared = self.root / "prepared/sbm"
        prepared.mkdir(parents=True)
        for split in ("train", "val", "test"):
            (prepared / f"{split}.pkl").write_bytes(f"synthetic split {split}".encode())
        split_hashes = {split: self.sha(prepared / f"{split}.pkl") for split in ("train", "val", "test")}
        train_manifest = {
            "model_id": "grapher", "run_id": "run", "train_seed": 41,
            "options": self.cfg,
            "checkpoint": {"path": "checkpoints/grapher.pt", "sha256": self.sha(self.checkpoint)},
            "checkpoint_selection": {"kind": "final_configured_epoch", "epoch": 200},
            "dataset": {
                "benchmark_id": "community_small", "serialized_id": "sbm", "split_sha256": split_hashes,
                "fingerprint": checker.digest({split: {"path": f"{split}.pkl", "sha256": split_hashes[split]} for split in sorted(split_hashes)}),
            },
        }
        (self.root / "run/train/manifest.json").write_text(json.dumps(train_manifest))
        self.manifest["sampling"].update(
            spectral_operator="combinatorial_laplacian",
            degree_constrained_realization={"enabled": True},
        )
        self.write_manifest()
        report = self.directory / "ablations/T/evaluation_test/primary/evaluation_status.json"
        report.parent.mkdir(parents=True)
        report.write_bytes(b'{"status":"failed","exit_code":2}')
        args = Namespace(
            run_dir=self.root / "run", config=config, dataset="community_small",
            serialized_dataset="sbm", dataset_root=self.root / "prepared", seed=41,
            generation_dir=self.directory, num_samples=2, kind="matched",
        )
        _, context = checker.invocation_context(args, self.cfg)
        intent = self.directory.parent / f"{self.directory.name}.ablation_generation_intent.json"
        intent.write_text(json.dumps({"context": context, "config_file_sha256": self.sha(config)}))
        before = self.snapshot()
        checker.prepare_matched(args, self.cfg)
        cli = [
            "--run-dir", str(args.run_dir), "--config", str(config),
            "--dataset", args.dataset, "--serialized-dataset", args.serialized_dataset,
            "--dataset-root", str(args.dataset_root), "--seed", "41",
            "--generation-dir", str(self.directory), "--num-samples", "2", "--kind", "matched",
        ]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(checker.main(["seal", *cli]), 0)
        self.assertFalse(intent.exists())
        receipt = checker.read_json(self.directory / checker.RECEIPT)
        self.assertEqual(receipt["matched_ablation_manifest_sha256"], self.sha(self.directory / exporter.COMMON))
        self.assertEqual(receipt["context"], context)
        generation, _, receipt = checker.verify_generation(args, self.cfg)
        snapshots = checker.verify_branches(args, self.cfg, generation, receipt, ["T", "D", "TR", "DR", "D0", "R1", "R12"])
        self.assertEqual(len(snapshots), 7)
        self.mock_refine.reset_mock()
        self.mock_decode.reset_mock()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(checker.main(["generation", *cli]), 0)
            self.assertEqual(checker.main(["branches", *cli, "--variants", "T", "D", "TR", "DR"]), 0)
        self.mock_refine.assert_not_called()
        self.mock_decode.assert_not_called()
        for path, content in before.items():
            if self.root / path != intent:
                self.assertEqual((self.root / path).read_bytes(), content, path)

    def test_real_fixed_summary_refinement_and_zero_temperature_milp_smoke(self):
        from grapher.models.grapher.graphlet_stage3 import _summary_config
        from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
        from grapher.rewiring_mlp.generic.graphlets import extract_topology_graphlet_target

        self.refine.stop()
        self.decode.stop()
        cfg = copy.deepcopy(self.cfg)
        cfg["graphlet_refinement"]["mode"] = "refinement"
        summary = _summary_config(cfg["graphlet_refinement"])
        basis = TopologyGraphletBasis.from_config(summary)
        source = nx.cycle_graph(4)
        histogram, mass = extract_topology_graphlet_target(source, graphlet_basis=basis, summary_config=summary)
        payload = {"enabled": True, "graphlet_basis": basis.to_dict(), "summary_config": {}}
        prediction = {"graph_index": 0, "graphlet_histogram": histogram, "graphlet_mass": mass}
        graph, diagnostic = exporter._refine_one(source, prediction, payload, cfg, 123, 0, 1)
        self.assertEqual([graph.degree(i) for i in range(4)], [2] * 4)
        self.assertTrue(nx.is_connected(graph))
        self.assertEqual(diagnostic["requested_steps"], 1)
        graph, _ = exporter._decode_zero(nx.to_numpy_array(source), [2] * 4, exporter.realization.resolve_config(cfg["sample"]), 123)
        self.assertEqual([graph.degree(i) for i in range(4)], [2] * 4)
        self.assertTrue(nx.is_connected(graph))


if __name__ == "__main__":
    unittest.main()
