"""Paired molecular export tests using only temporary trusted graph artifacts."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import pickle
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import yaml

from grapher.models.grapher import molecular_ablation as exporter


class MolecularExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="molecular-export-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "run/generations/qm9_seed_41"
        self.directory.mkdir(parents=True)
        self.cfg = {
            "variant": "vanilla_laplacian_loggap_hybrid_categorical_attributed_ppgn",
            "sde": {"eps": 1e-5}, "sample": {"predictor": "euler"},
            "generation_batch_size": 128,
            "attributed": {
                "node_attribute": "atomic_num", "edge_attribute": "bond_type",
                "node_categories": [6, 7, 8, 9], "edge_categories": [1, 2, 3],
                "decode": {"constraint_mode": "atom_bond_valence"},
            },
            "attributed_diversification": {
                "enabled": True, "strict_same_bond": True,
                "preserve_neighbor_atom_type_by_bond_type": True,
                "verify_bond_valence_preservation": True, "preserve_connectivity": True,
            },
        }
        self.context = {
            "seed": 41, "num_samples": 2, "run_id": "run",
            "generation_id": self.directory.name, "dataset": "qm9",
            "serialized_dataset": "qm9_attributed", "checkpoint_sha256": "a" * 64,
        }
        self.manifest = {
            "model_id": "grapher", "variant": self.cfg["variant"], "run_id": "run",
            "generation_id": self.directory.name, "generation_seed": 41,
            "num_requested": 2, "num_generated": 2, "duration_seconds": 45.0,
            "checkpoint": {"path": "unavailable-model.pt", "sha256": "a" * 64},
        }
        first = self.molecule(nx.cycle_graph(4))
        second = self.molecule(nx.path_graph(3), [6, 7, 8])
        second[1][2]["bond_type"] = 2
        first.graph["attribute_diffusion"] = {"topology_state_sha256": "endpoint-0", "constraint_mode": "atom_bond_valence"}
        second.graph["attribute_diffusion"] = {"topology_state_sha256": "endpoint-1", "constraint_mode": "atom_bond_valence"}
        self.before = [first, second]
        self.after = copy.deepcopy(self.before)
        # A valid same-bond swap changes the endpoint while preserving every
        # indexed chemical invariant; all four atoms in this example are C.
        self.after[0].remove_edges_from([(0, 1), (2, 3)])
        self.after[0].add_edges_from([(0, 2, {"bond_type": 1}), (1, 3, {"bond_type": 1})])
        self.after[0].graph["attributed_diversification"] = {"accepted_steps": 1}
        self.save("pre_refinement_graphs", self.before)
        self.save("base_graphs", self.after)
        self.save("sampled_basis_indices", [17, 29])
        self.write_manifest()

    @staticmethod
    def molecule(graph, atoms=None):
        atoms = atoms or [6] * len(graph)
        for node, atom in enumerate(atoms):
            graph.nodes[node]["atomic_num"] = atom
        nx.set_edge_attributes(graph, 1, "bond_type")
        return graph

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

    def update_populations(self):
        self.save("pre_refinement_graphs", self.before)
        self.save("base_graphs", self.after)
        self.write_manifest()

    def snapshot(self):
        return {path.relative_to(self.root).as_posix(): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}

    def export(self):
        return exporter.export_molecular(self.directory, self.cfg, self.context)

    def common(self):
        return json.loads((self.directory / exporter.COMMON).read_text(encoding="utf-8"))

    def test_export_copies_parent_bytes_and_embeds_only_manifest_identities(self):
        before = self.snapshot()
        self.assertEqual(self.export(), self.directory / exporter.COMMON)
        for path, content in before.items():
            self.assertEqual((self.root / path).read_bytes(), content, path)
        common = self.common()
        self.assertEqual(common["format"], "graphes_molecular_ablation_export_v1")
        self.assertEqual(common, json.loads((self.directory / "ablations/ablation_manifest.json").read_text()))
        self.assertEqual(common["sample_ids"], ["41:0", "41:1"])
        self.assertEqual(set(common["branches"]), {"T", "TR"})
        self.assertEqual(set(common["provenance"]["source_artifacts"]), {"pre_refinement_graphs", "base_graphs", "sampled_basis_indices"})
        self.assertEqual(common["provenance"]["source_generation_manifest_sha256"], self.sha(self.directory / "manifest.json"))
        for variant, source in exporter.SOURCES.items():
            branch = self.directory / "ablations" / variant
            self.assertEqual((branch / "base_graphs.pkl").read_bytes(), (self.directory / self.manifest[source]["path"]).read_bytes())
            child = json.loads((branch / "manifest.json").read_text())
            self.assertEqual(child, common["branches"][variant])
            self.assertEqual(child["identity_source"], exporter.IDENTITY)
            self.assertEqual(child["source_artifact"], self.manifest[source])
            self.assertEqual(child["provenance"]["source_sha256"], self.manifest[source]["sha256"])
            self.assertEqual(child["shared_context"], common["provenance"])
            with (branch / "base_graphs.pkl").open("rb") as stream:
                self.assertTrue(all("sample_id" not in graph.graph for graph in pickle.load(stream)))
        self.assertEqual(common["invariants"]["num_checked"], 2)

    def test_export_never_deserializes_a_model_or_runs_generic_controls(self):
        with patch("torch.load", side_effect=AssertionError("model loading forbidden")) as load:
            with patch("grapher.models.grapher.matched_ablation.export_matched", side_effect=AssertionError("generic sampling forbidden")) as controls:
                self.export()
        load.assert_not_called()
        controls.assert_not_called()

    def test_existing_valid_export_is_reused_read_only(self):
        self.export()
        before = self.snapshot()
        self.export()
        self.assertEqual(self.snapshot(), before)

    def test_existing_report_only_directories_are_preserved(self):
        report = self.directory / "ablations/TR/evaluation_test/rewiring/evaluation_status.json"
        report.parent.mkdir(parents=True)
        report.write_bytes(b'{"status":"failed","exit_code":2}\n')
        content = report.read_bytes()
        self.export()
        self.assertEqual(report.read_bytes(), content)

    def test_publication_failure_rolls_back_without_changing_reports_or_parents(self):
        report = self.directory / "ablations/T/evaluation_test/rewiring/report.txt"
        report.parent.mkdir(parents=True)
        report.write_bytes(b"existing molecular report")
        before = self.snapshot()
        with patch.object(exporter.os, "link", side_effect=OSError("publish failed")):
            with self.assertRaisesRegex(OSError, "publish failed"):
                self.export()
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(list(self.directory.glob(".molecular_export_*")))

    def test_parent_hash_corruption_is_rejected_without_writes(self):
        (self.directory / "base_graphs.pkl").write_bytes(b"corrupted")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.export()
        self.assertEqual(self.snapshot(), before)

    def test_conflicting_embedded_ids_and_duplicate_ids_are_rejected(self):
        for ids in (["41:1", "41:0"], ["41:0", "41:0"]):
            with self.subTest(ids=ids):
                for graph, sample_id in zip(self.after, ids):
                    graph.graph["sample_id"] = sample_id
                self.update_populations()
                with self.assertRaisesRegex(ValueError, "ID/order differs"):
                    self.export()

    def test_correct_existing_embedded_ids_are_retained_unchanged(self):
        for population in (self.before, self.after):
            for index, graph in enumerate(population):
                graph.graph["sample_id"] = f"41:{index}"
        self.update_populations()
        before = self.snapshot()
        self.export()
        for path, content in before.items():
            self.assertEqual((self.root / path).read_bytes(), content)

    def test_population_count_mismatch_is_rejected(self):
        self.save("base_graphs", self.after[:1])
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Population count differs"):
            self.export()

    def test_basis_count_mismatch_is_rejected(self):
        self.save("sampled_basis_indices", [17])
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "sampled-basis count/order differs"):
            self.export()

    def test_indexed_atom_attribute_mismatch_is_rejected(self):
        self.after[0].nodes[0]["atomic_num"] = 7
        self.update_populations()
        with self.assertRaisesRegex(ValueError, "indexed atom attributes differ"):
            self.export()

    def test_bond_label_count_mismatch_is_rejected(self):
        u, v = next(iter(self.after[0].edges()))
        self.after[0][u][v]["bond_type"] = 2
        self.update_populations()
        with self.assertRaisesRegex(ValueError, "bond-label counts differ"):
            self.export()

    def test_indexed_typed_degree_mismatch_is_rejected_even_when_histogram_matches(self):
        source = self.molecule(nx.cycle_graph(4))
        source[0][1]["bond_type"] = 2
        target = self.molecule(nx.cycle_graph(4))
        target[1][2]["bond_type"] = 2
        self.before[0], self.after[0] = source, target
        self.update_populations()
        with self.assertRaisesRegex(ValueError, "indexed typed degrees differ"):
            self.export()

    def test_atom_bond_environment_mismatch_is_rejected_when_typed_degrees_match(self):
        source = self.molecule(nx.cycle_graph(4), [6, 6, 7, 7])
        source[0][1]["bond_type"] = source[2][3]["bond_type"] = 2
        target = copy.deepcopy(source)
        for u, v in target.edges():
            target[u][v]["bond_type"] = 3 - source[u][v]["bond_type"]
        self.before[0], self.after[0] = source, target
        self.update_populations()
        with self.assertRaisesRegex(ValueError, "neighbor atom/bond signatures differ"):
            self.export()

    def test_endpoint_provenance_order_mismatch_is_rejected(self):
        self.after[0].graph["attribute_diffusion"]["topology_state_sha256"] = "endpoint-other"
        self.update_populations()
        with self.assertRaisesRegex(ValueError, "endpoint provenance/order differs"):
            self.export()

    def test_zinc_decode_none_and_non_qm9_atoms_are_supported(self):
        self.cfg["attributed"]["decode"]["constraint_mode"] = "none"
        self.cfg["attributed"]["node_categories"] = [6, 7, 8, 9, 15, 16, 17, 35, 53]
        self.context.update(dataset="zinc", serialized_dataset="zinc")
        for population in (self.before, self.after):
            nx.set_node_attributes(population[0], 15, "atomic_num")
            population[0].graph["attribute_diffusion"]["constraint_mode"] = "none"
        self.update_populations()
        self.export()
        self.assertEqual(self.common()["invariants"]["decode_constraint_mode"], "none")

    def test_raw_invalid_population_is_retained_without_chemical_filtering(self):
        invalid = self.molecule(nx.complete_graph(6))
        self.before[0] = invalid
        self.after[0] = copy.deepcopy(invalid)
        self.update_populations()
        self.export()
        self.assertEqual(self.common()["branches"]["TR"]["completed_count"], 2)
        self.assertFalse(self.common()["invariants"]["chemical_validity_filter_applied"])

    def test_disconnected_raw_sources_are_preserved_without_connectivity_repair(self):
        disconnected = self.molecule(nx.empty_graph(3))
        self.before[0] = disconnected
        self.after[0] = copy.deepcopy(disconnected)
        self.update_populations()
        self.export()
        self.assertEqual(self.common()["invariants"]["connected_sources_checked"], 1)

    def test_partial_existing_branch_is_rejected_without_overwrite(self):
        path = self.directory / "ablations/T/base_graphs.pkl"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"pre-existing branch")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "conflicting ablation artifacts"):
            self.export()
        self.assertEqual(self.snapshot(), before)

    def test_partial_existing_common_index_is_rejected_without_overwrite(self):
        path = self.directory / exporter.COMMON
        path.write_text("{}")
        before = self.snapshot()
        with self.assertRaises((ValueError, OSError)):
            self.export()
        self.assertEqual(self.snapshot(), before)

    def test_corrupted_exported_branch_is_rejected_on_reuse(self):
        self.export()
        (self.directory / "ablations/TR/base_graphs.pkl").write_bytes(b"corrupted child")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.export()

    def test_mismatched_checkpoint_context_is_rejected(self):
        self.context["checkpoint_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint differs"):
            self.export()

    def test_source_path_escape_is_rejected(self):
        self.manifest["base_graphs"]["path"] = "../../outside.pkl"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "escapes directory"):
            self.export()

    def test_actual_checker_prepare_seal_and_resume_accepts_molecular_export(self):
        script = Path(__file__).resolve().parents[1] / "scripts/check_grapher_ablation_artifacts.py"
        spec = importlib.util.spec_from_file_location("molecular_checker_integration", script)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        self.cfg["train"] = {"epochs": 200, "batch_size": 128, "lr": 0.001}
        self.cfg["sample"]["separate_attribute_rng"] = True
        self.cfg["attribute_diffusion"] = {"steps": 100, "sample_steps": 100}
        config = self.root / "configuration.yaml"
        config.write_text(yaml.safe_dump({"grapher": self.cfg}), encoding="utf-8")
        checkpoint = self.root / "run/train/checkpoints/grapher.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic checkpoint must not be deserialized")
        prepared = self.root / "prepared/qm9_attributed"
        prepared.mkdir(parents=True)
        for split in ("train", "val", "test"):
            (prepared / f"{split}.pkl").write_bytes(f"synthetic {split}".encode())
        hashes = {split: self.sha(prepared / f"{split}.pkl") for split in ("train", "val", "test")}
        training = {
            "model_id": "grapher", "run_id": "run", "train_seed": 41,
            "options": self.cfg,
            "checkpoint": {"path": "checkpoints/grapher.pt", "sha256": self.sha(checkpoint)},
            "checkpoint_selection": {"kind": "final_configured_epoch", "epoch": 200},
            "dataset": {
                "benchmark_id": "qm9", "serialized_id": "qm9_attributed", "split_sha256": hashes,
                "fingerprint": checker.digest({split: {"path": f"{split}.pkl", "sha256": hashes[split]} for split in sorted(hashes)}),
            },
        }
        (self.root / "run/train/manifest.json").write_text(json.dumps(training))
        self.manifest["checkpoint"] = {"path": str(checkpoint), "sha256": self.sha(checkpoint)}
        self.manifest["sampling"] = {
            "generation_batch_size": 128, "separate_attribute_rng": True,
            "attributed_diversification": self.cfg["attributed_diversification"],
            "constraint_mode": "atom_bond_valence",
            "attribute_diffusion_steps": 100, "attribute_sampling_steps": 100,
            "posthoc_repair": False, "filtering": False, "replacement_sampling": False,
        }
        self.write_manifest()
        args = Namespace(
            run_dir=self.root / "run", config=config, dataset="qm9",
            serialized_dataset="qm9_attributed", dataset_root=self.root / "prepared",
            seed=41, generation_dir=self.directory, num_samples=2, kind="molecular",
        )
        _, context = checker.invocation_context(args, self.cfg)
        intent = self.directory.parent / f"{self.directory.name}.ablation_generation_intent.json"
        intent.write_text(json.dumps({"context": context, "config_file_sha256": self.sha(config)}))
        report = self.directory / "ablations/T/evaluation_test/rewiring/evaluation_status.json"
        report.parent.mkdir(parents=True)
        report.write_bytes(b'{"status":"failed","exit_code":2}')
        before = self.snapshot()
        cli = [
            "--run-dir", str(args.run_dir), "--config", str(config),
            "--dataset", "qm9", "--serialized-dataset", "qm9_attributed",
            "--dataset-root", str(args.dataset_root), "--seed", "41",
            "--generation-dir", str(self.directory), "--num-samples", "2", "--kind", "molecular",
        ]
        with patch("torch.load", side_effect=AssertionError("model loading forbidden")):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main(["prepare", *cli]), 0)
                self.assertEqual(checker.main(["seal", *cli]), 0)
                self.assertEqual(checker.main(["generation", *cli]), 0)
                self.assertEqual(checker.main(["branches", *cli, "--variants", "T", "TR"]), 0)
        self.assertFalse(intent.exists())
        receipt = checker.read_json(self.directory / checker.RECEIPT)
        self.assertEqual(receipt["context"], context)
        self.assertEqual(receipt["molecular_ablation_manifest_sha256"], self.sha(self.directory / exporter.COMMON))
        self.assertEqual(receipt["ablation_manifest_sha256"], self.sha(self.directory / "ablations/ablation_manifest.json"))
        self.assertEqual(set(receipt["branch_manifest_sha256"]), {"T", "TR"})
        for path, content in before.items():
            if self.root / path != intent:
                self.assertEqual((self.root / path).read_bytes(), content, path)


if __name__ == "__main__":
    unittest.main()
