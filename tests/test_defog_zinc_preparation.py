"""Exercise DeFoG conversion of the canonical ZINC categorical projection."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
import torch
import yaml
from rdkit import Chem


REPOSITORY = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preparer = load_module("defog_test_zinc_preparer", REPOSITORY / "scripts/prepare_zinc_dataset.py")
worker = load_module(
    "defog_test_molecular_worker",
    REPOSITORY / "src/grapher/models/defog/workers/prepare_molecular_dataset.py",
)
PROJECTION_POLICY = "audit_source_metadata_to_categorical_graph_state_v1"


class DefogZincPreparationTests(unittest.TestCase):
    smiles = (
        "CCO",                          # neutral control
        "CC(=O)[O-]",                   # negative charge
        "C[N+](C)(C)C",                 # positive charge
        "C[N+](=O)[O-]",                # charge-separated nitro group
        "c1cc[nH+]cc1",                 # aromatic cation; charge needed for kekulization
        "C[N+](C)(C)C.[Cl-]",           # both fragments belong to the graph
        "F[C@H](Cl)Br",                 # source chirality projected out
        "F/C=C/F",                      # source bond stereo projected out
        "c1ccccc1",                     # neutral aromatic control
    )

    @classmethod
    def setUpClass(cls) -> None:
        config = yaml.safe_load((REPOSITORY / "configs/datasets/zinc.yaml").read_text(encoding="utf-8"))
        cls.protocol = preparer.ZincProtocol.from_config(config)

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="defog-zinc-preparation-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.graphs = [
            preparer.smiles_to_zinc_graph(smiles, self.protocol, source_index=index)
            for index, smiles in enumerate(self.smiles)
        ]

    def convert(self, graphs: list[nx.Graph] | None = None):
        source = self.root / "source.pkl"
        destination = self.root / "processed" / "train.pt"
        model_view = self.root / "model_view" / "train.pkl"
        source.write_bytes(pickle.dumps(
            self.graphs if graphs is None else graphs, protocol=pickle.HIGHEST_PROTOCOL,
        ))
        original = source.read_bytes()
        record = worker._convert_split(
            source, destination, model_view_destination=model_view,
            dataset="zinc", split="train",
        )
        self.assertEqual(source.read_bytes(), original)
        data, slices = torch.load(destination, map_location="cpu", weights_only=False)
        models = pickle.loads(model_view.read_bytes())
        return record, data, slices, models

    def assert_rejected_before_publication(self, graph: nx.Graph, pattern: str) -> None:
        source = self.root / "invalid.pkl"
        source.write_bytes(pickle.dumps([self.graphs[0], graph], protocol=pickle.HIGHEST_PROTOCOL))
        original = source.read_bytes()
        destination = self.root / "failed" / "train.pt"
        model_view = self.root / "failed_view" / "train.pkl"
        with self.assertRaisesRegex(ValueError, pattern):
            worker._convert_split(
                source, destination, model_view_destination=model_view,
                dataset="zinc", split="train",
            )
        self.assertEqual(source.read_bytes(), original)
        self.assertFalse(destination.exists())
        self.assertFalse(model_view.exists())

    def test_charged_aromatic_stereo_and_salt_graphs_are_all_preserved(self) -> None:
        record, data, slices, models = self.convert()
        self.assertEqual(record["graph_count"], len(self.graphs))
        self.assertEqual(record["source_to_processed_index"], "identity")
        self.assertEqual(len(models), len(self.graphs))
        self.assertEqual(data.x.shape, (sum(len(graph) for graph in self.graphs), 9))
        self.assertEqual(data.edge_attr.shape, (sum(graph.number_of_edges() for graph in self.graphs) * 2, 4))
        self.assertEqual(data.idx.tolist(), list(range(len(self.graphs))))
        np.testing.assert_array_equal(
            slices["x"].numpy(), np.cumsum([0, *[len(graph) for graph in self.graphs]])
        )
        atom_classes = {number: index for index, number in enumerate(worker.ATOM_VOCABULARIES["zinc"])}
        expected_classes = [
            atom_classes[graph.nodes[node]["atomic_num"]]
            for graph in self.graphs for node in graph.nodes()
        ]
        self.assertEqual(data.x.argmax(dim=1).tolist(), expected_classes)
        self.assertEqual(set(data.edge_attr.argmax(dim=1).tolist()), {1, 2})
        for index, (source, model) in enumerate(zip(self.graphs, models)):
            self.assertEqual(model.graph["source_graph_index"], index)
            self.assertEqual(model.graph["source_smiles"], self.smiles[index])
            self.assertEqual(model.graph["zinc_source_state_projection_policy"], PROJECTION_POLICY)
            self.assertEqual(dict(model.degree()), dict(source.degree()))
            self.assertEqual(
                [model.nodes[node]["atomic_num"] for node in model],
                [source.nodes[node]["atomic_num"] for node in source],
            )
            self.assertEqual(
                {(min(u, v), max(u, v), row["bond_type"]) for u, v, row in model.edges(data=True)},
                {(min(u, v), max(u, v), row["bond_type"]) for u, v, row in source.edges(data=True)},
            )
            self.assertTrue(all("formal_charge" not in row for _, row in model.nodes(data=True)))
        self.assertEqual(nx.number_connected_components(models[5]), 2)
        self.assertEqual(models[5].number_of_nodes(), 6)

    def test_projection_audit_records_original_charge_stereo_and_fragments(self) -> None:
        record, _, _, models = self.convert()
        expected_charges = (
            [], [[3, -1]], [[1, 1]], [[1, 1], [3, -1]], [[3, 1]],
            [[1, 1], [5, -1]], [], [], [],
        )
        for model, charges in zip(models, expected_charges):
            self.assertEqual(model.graph["projected_formal_charge_atoms"], charges)
        self.assertEqual(models[6].graph["projected_chiral_atoms"], [1])
        self.assertEqual(models[7].graph["projected_stereo_bonds"], [1])
        self.assertEqual([model.graph["source_fragment_count"] for model in models], [1, 1, 1, 1, 1, 2, 1, 1, 1])
        projection = record["zinc_source_state_projection"]
        self.assertEqual(projection["policy"], PROJECTION_POLICY)
        self.assertEqual(projection["formal_charge_graphs"], 5)
        self.assertEqual(projection["formal_charge_atoms"], 7)
        self.assertEqual(projection["stereochemistry_graphs"], 2)
        self.assertEqual(projection["chiral_atoms"], 1)
        self.assertEqual(projection["stereo_bonds"], 1)
        self.assertEqual(projection["fragmented_graphs"], 1)
        self.assertEqual(projection["source_fragments"], 10)

    def test_aromatic_charged_source_is_kekulized_without_neutralizing_it(self) -> None:
        source = Chem.MolFromSmiles("c1cc[nH+]cc1")
        graph = nx.Graph(source_smiles="c1cc[nH+]cc1")
        for atom in source.GetAtoms():
            graph.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum(), atom_type=atom.GetAtomicNum())
        for bond in source.GetBonds():
            graph.add_edge(
                bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=4, bond_order=1.5,
            )
        original = pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL)
        _, _, _, models = self.convert([graph])
        self.assertEqual(pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL), original)
        self.assertEqual(models[0].graph["projected_formal_charge_atoms"], [[3, 1]])
        self.assertEqual(sorted(row["bond_type"] for _, _, row in models[0].edges(data=True)), [1, 1, 1, 2, 2, 2])

    def test_rdkit_source_preparation_preserves_real_charge_and_all_fragments(self) -> None:
        molecule = worker._prepared_zinc_molecule("C[N+](C)(C)C.[Cl-]", Chem)
        self.assertEqual(molecule.GetNumAtoms(), 6)
        self.assertEqual(len(Chem.GetMolFrags(molecule)), 2)
        self.assertEqual(
            [(atom.GetIdx(), atom.GetFormalCharge()) for atom in molecule.GetAtoms()
             if atom.GetFormalCharge()],
            [(1, 1), (5, -1)],
        )

    def test_nonconsecutive_node_labels_follow_insertion_order_not_numeric_sort(self) -> None:
        graph = nx.relabel_nodes(self.graphs[0], {0: 20, 1: 4, 2: 99}, copy=True)
        self.assertEqual(list(graph), [20, 4, 99])
        _, data, _, models = self.convert([graph])
        self.assertEqual(data.x.argmax(dim=1).tolist(), [0, 0, 2])
        self.assertEqual(list(models[0]), [0, 1, 2])
        self.assertEqual(set(models[0].edges()), {(0, 1), (1, 2)})

    def test_mismatched_source_atom_identity_rejects_without_outputs(self) -> None:
        graph = copy.deepcopy(self.graphs[0])
        graph.graph["source_smiles"] = "CCN"
        self.assert_rejected_before_publication(graph, "atom order")

    def test_mismatched_source_bonds_reject_without_outputs(self) -> None:
        graph = copy.deepcopy(self.graphs[0])
        graph.graph["source_smiles"] = "CC=O"
        self.assert_rejected_before_publication(graph, "bonds.*reproduce")

    def test_missing_source_smiles_rejects_without_outputs(self) -> None:
        graph = copy.deepcopy(self.graphs[0])
        graph.graph.pop("source_smiles")
        self.assert_rejected_before_publication(graph, "source_smiles")

    def test_nonzero_charge_in_actual_categorical_node_state_is_still_rejected(self) -> None:
        graph = copy.deepcopy(self.graphs[2])
        graph.nodes[1]["formal_charge"] = 1
        self.assert_rejected_before_publication(graph, "non-zero formal_charge")

    def test_qm9_default_graph_validation_still_rejects_disconnected_graphs(self) -> None:
        with self.assertRaisesRegex(ValueError, "disconnected"):
            worker._validate_graph(self.graphs[5], graph_label="qm9_test[0]")

    def test_worker_cli_converts_every_split_and_reports_projection_policy(self) -> None:
        paths = {}
        for split, indices in (("train", (1, 2, 5)), ("val", (4, 6)), ("test", (0, 3, 7, 8))):
            path = self.root / f"{split}.pkl"
            path.write_bytes(pickle.dumps([self.graphs[index] for index in indices], protocol=pickle.HIGHEST_PROTOCOL))
            paths[split] = path
        original = {split: path.read_bytes() for split, path in paths.items()}
        output = self.root / "converted"
        manifest_path = self.root / "conversion_manifest.json"
        arguments = [
            "prepare_molecular_dataset.py", "--dataset", "zinc", "--output-root", str(output),
            "--manifest", str(manifest_path),
        ]
        for split, path in paths.items():
            arguments.extend([f"--{split}", str(path)])
        with patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
            worker.main()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["graphs_dropped"], 0)
        self.assertTrue(manifest["split_order_preserved"])
        self.assertEqual(manifest["unsupported_state_policy"]["formal_charge"], "audit_source_metadata_projected_at_zinc_preparation")
        self.assertEqual(manifest["unsupported_state_policy"]["serialized_nonzero_formal_charge"], "reject")
        for split, expected_count in (("train", 3), ("val", 2), ("test", 4)):
            self.assertEqual(manifest["splits"][split]["graph_count"], expected_count)
            self.assertEqual(manifest["splits"][split]["zinc_source_state_projection"]["policy"], PROJECTION_POLICY)
            self.assertTrue((output / "full" / "processed" / f"{split}.pt").is_file())
            self.assertEqual(paths[split].read_bytes(), original[split])


if __name__ == "__main__":
    unittest.main()
