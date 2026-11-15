"""Verify sealed legacy molecular references without weakening new sample IDs."""
from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import networkx as nx

from grapher.experiments.learning_ablation import sha256, verify_attribute_population


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_pickle(path: Path, value: object) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


class LegacyPopulationSampleIdsTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.parent = root / "parent"
        self.child = root / "control"
        self.parent.mkdir()
        self.child.mkdir()
        self.n = 3

        # Distinct edge sets make graph reordering detectable independently of
        # any graph-level labels.
        edge_sets = (((0, 1), (1, 2)), ((1, 2), (2, 3)), ((0, 3), (0, 2)))
        self.parent_graphs = []
        self.child_graphs = []
        for index, edges in enumerate(edge_sets):
            parent = nx.Graph()
            parent.add_nodes_from(range(4))
            parent.add_edges_from(edges)
            child = parent.copy()
            child.graph["sample_id"] = f"41:{index}"
            self.parent_graphs.append(parent)
            self.child_graphs.append(child)
        self.final_graphs = [graph.copy() for graph in self.child_graphs]
        for graph in self.final_graphs:
            nx.set_node_attributes(graph, "C", "atomic_num")
            nx.set_edge_attributes(graph, "single", "bond_type")

        pairing = {
            "format": "gdsm_hybrid_attributed_topology_pairing_v1",
            "hash_inputs": ["sample_x", "soft_adjacency", "binary_topology"],
            "per_graph_sha256": [f"topology-state-{i}" for i in range(self.n)],
            "batch_size": 2,
            "generation_seed": 41,
        }
        for directory in (self.parent, self.child):
            _write_json(directory / "topology_pairing.json", pairing)
            _write_pickle(directory / "sampled_basis_indices.pkl", [0, 2, 1])
        self._save_and_seal()

    def _save_and_seal(self) -> None:
        _write_pickle(self.parent / "topology_graphs.pkl", self.parent_graphs)
        _write_pickle(self.child / "topology_graphs.pkl", self.child_graphs)
        _write_pickle(self.child / "base_graphs.pkl", self.final_graphs)
        for directory in (self.parent, self.child):
            names = ("topology_graphs", "topology_pairing", "sampled_basis_indices")
            if directory == self.child:
                names = (*names, "base_graphs")
            files = {
                name: {
                    "path": f"{name}.{'json' if name == 'topology_pairing' else 'pkl'}",
                    "sha256": sha256(directory / f"{name}.{'json' if name == 'topology_pairing' else 'pkl'}"),
                }
                for name in names
            }
            _write_json(directory / "manifest.json", {"files": files})
        self.reference = {
            "generation_dir": str(self.parent),
            "generation_manifest_sha256": sha256(self.parent / "manifest.json"),
            "sampled_basis_indices_sha256": sha256(self.parent / "sampled_basis_indices.pkl"),
            "generation_seed": 41,
            "num_requested": self.n,
            "checkpoint_sha256": "synthetic-checkpoint-sha256",
        }

    def _verify(self) -> dict:
        return verify_attribute_population(self.child, self.reference)

    def test_fully_unlabelled_legacy_parent_passes(self) -> None:
        result = self._verify()
        self.assertTrue(result["fixed_topology_verified"])
        self.assertTrue(result["ordered_sample_ids_verified"])
        self.assertTrue(result["legacy_reference_sample_ids_inferred_from_order"])
        self.assertEqual(result["reference_sample_id_policy"], "manifest_sealed_implicit_order")
        self.assertEqual(result["num_matched"], 3)

    def test_modern_reference_with_explicit_ids_passes(self) -> None:
        for i, graph in enumerate(self.parent_graphs):
            graph.graph["sample_id"] = f"41:{i}"
        self._save_and_seal()
        result = self._verify()
        self.assertFalse(result["legacy_reference_sample_ids_inferred_from_order"])
        self.assertEqual(result["reference_sample_id_policy"], "explicit_graph_metadata")

    def test_partially_missing_parent_ids_rejected(self) -> None:
        for i, graph in enumerate(self.parent_graphs):
            if i:
                graph.graph["sample_id"] = f"41:{i}"
        self._save_and_seal()
        with self.assertRaisesRegex(ValueError, "topology_graphs.pkl at index 0"):
            self._verify()

    def test_wrong_parent_id_rejected(self) -> None:
        for i, graph in enumerate(self.parent_graphs):
            graph.graph["sample_id"] = f"41:{i}"
        self.parent_graphs[1].graph["sample_id"] = "41:999"
        self._save_and_seal()
        with self.assertRaisesRegex(ValueError, "topology_graphs.pkl at index 1"):
            self._verify()

    def test_missing_child_topology_id_rejected(self) -> None:
        self.child_graphs[0].graph.pop("sample_id")
        self._save_and_seal()
        with self.assertRaisesRegex(ValueError, "topology_graphs.pkl at index 0"):
            self._verify()

    def test_duplicate_child_id_rejected(self) -> None:
        self.child_graphs[1].graph["sample_id"] = "41:0"
        self._save_and_seal()
        with self.assertRaisesRegex(ValueError, "topology_graphs.pkl at index 1"):
            self._verify()

    def test_missing_final_graph_id_rejected(self) -> None:
        self.final_graphs[2].graph.pop("sample_id")
        self._save_and_seal()
        with self.assertRaisesRegex(ValueError, "base_graphs.pkl at index 2"):
            self._verify()

    def test_changed_topology_rejected_even_with_correct_new_ids(self) -> None:
        self.child_graphs[1].remove_edge(2, 3)
        self.child_graphs[1].add_edge(0, 3)
        self.final_graphs[1] = self.child_graphs[1].copy()
        self._save_and_seal()
        with self.assertRaisesRegex(ValueError, "changed at least one fixed topology"):
            self._verify()

    def test_modified_parent_pickle_rejected_without_manifest_update(self) -> None:
        # The reference manifest is immutable, so an in-place repair must
        # never be silently accepted, even if its graph labels look correct.
        self.parent_graphs[0].graph["sample_id"] = "41:0"
        _write_pickle(self.parent / "topology_graphs.pkl", self.parent_graphs)
        with self.assertRaisesRegex(ValueError, "Missing or modified generated artifact"):
            self._verify()

    def test_mismatched_ordered_state_hashes_rejected(self) -> None:
        pairing = json.loads((self.child / "topology_pairing.json").read_text())
        pairing["per_graph_sha256"][1] = "different-topology-state"
        _write_json(self.child / "topology_pairing.json", pairing)
        self._save_and_seal()
        with self.assertRaisesRegex(ValueError, "did not preserve reference topology evidence"):
            self._verify()

    def test_mismatched_donor_indices_rejected(self) -> None:
        _write_pickle(self.child / "sampled_basis_indices.pkl", [1, 2, 0])
        # Update the child manifest to reflect the changed on-disk artifact.
        child_manifest = json.loads((self.child / "manifest.json").read_text())
        child_manifest["files"]["sampled_basis_indices"]["sha256"] = sha256(
            self.child / "sampled_basis_indices.pkl"
        )
        _write_json(self.child / "manifest.json", child_manifest)
        with self.assertRaisesRegex(ValueError, "ordered donor IDs"):
            self._verify()


if __name__ == "__main__":
    unittest.main()
