from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "scripts" / "defog_prepare_molecular_dataset_worker.py"
SPEC = importlib.util.spec_from_file_location("defog_molecular_worker", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


class DeFoGMolecularWorkerContractTest(unittest.TestCase):
    def test_qm9_vocabularies_match_defog(self) -> None:
        self.assertEqual(
            [WORKER._atom_class("qm9", value) for value in (6, 7, 8, 9)],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            [WORKER._edge_class("qm9", value) for value in (1, 2, 3, 4)],
            [1, 2, 3, 4],
        )

    def test_zinc_vocabularies_match_defog(self) -> None:
        atomic_numbers = (6, 7, 8, 9, 15, 16, 17, 35, 53)
        self.assertEqual(
            [WORKER._atom_class("zinc", value) for value in atomic_numbers],
            list(range(9)),
        )
        self.assertEqual(
            [WORKER._edge_class("zinc", value) for value in (1, 2, 3)],
            [1, 2, 3],
        )
        with self.assertRaises(ValueError):
            WORKER._edge_class("zinc", 4)

    def test_processed_paths_match_upstream_dataset_classes(self) -> None:
        root = Path("native")
        self.assertEqual(
            WORKER._processed_path(root, "qm9", "train"),
            root / "processed" / "proc_tr_no_h.pt",
        )
        self.assertEqual(
            WORKER._processed_path(root, "zinc", "test"),
            root / "full" / "processed" / "test.pt",
        )

    def test_placeholder_sets_prevent_upstream_download(self) -> None:
        self.assertEqual(
            set(WORKER._raw_placeholder_names("qm9")),
            {"gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt"},
        )
        self.assertEqual(len(WORKER._raw_placeholder_names("zinc")), 6)

    def test_unrepresented_charge_and_stereo_fail_closed(self) -> None:
        WORKER._validate_supported_node_state(
            {"atomic_num": 6, "formal_charge": 0}, graph_label="graph[0]"
        )
        with self.assertRaisesRegex(ValueError, "formal_charge"):
            WORKER._validate_supported_node_state(
                {"atomic_num": 7, "formal_charge": 1}, graph_label="graph[0]"
            )
        with self.assertRaisesRegex(ValueError, "stereo"):
            WORKER._bond_type(
                {"bond_type": 1, "bond_order": 1.0, "stereo": "E"},
                graph_label="graph[0]",
            )

    def test_qm9_projection_provenance_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "Regenerate qm9_attributed"):
            WORKER._qm9_projection_metadata(
                SimpleNamespace(graph={}),
                graph_label="train[0]",
            )

        graph = SimpleNamespace(
            graph={
                "qm9_source_state_projection_policy": WORKER.QM9_PROJECTION_POLICY,
                "projected_formal_charge_atoms": [[2, 1], [3, -1]],
                "projected_chiral_atoms": [4],
                "projected_stereo_bonds": [1],
            }
        )
        self.assertEqual(
            WORKER._qm9_projection_metadata(graph, graph_label="train[0]"),
            ([[2, 1], [3, -1]], [4], [1]),
        )


if __name__ == "__main__":
    unittest.main()
