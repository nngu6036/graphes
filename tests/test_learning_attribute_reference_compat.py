"""Regression tests for modern and sealed legacy molecular A-control parents."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from grapher.experiments.learning_ablation import require_attribute_reference, sha256


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


class AttributeReferenceCompatibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = Path(self.tmp.name) / "seed_41_qm9_grapher_final_B1to4_g345_v2"
        self.generation_dir = self.run_dir / "generations" / "paper_ablation_v1_seed_41_n_3"
        self.generation_dir.mkdir(parents=True)
        self.checkpoint = self.run_dir / "train/checkpoints/grapher.pt"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"synthetic checkpoint")
        self.checkpoint_sha = sha256(self.checkpoint)
        self.train_manifest = self.run_dir / "train/manifest.json"
        write_json(self.train_manifest, {
            "checkpoint": {"sha256": self.checkpoint_sha},
            "dataset": {"benchmark_id": "qm9", "fingerprint": "synthetic-dataset"},
            "train_seed": 41,
            "checkpoint_selection": {"epoch": 200},
            "options": {"train": {"epochs": 200}},
        })
        self.sample = {"separate_attribute_rng": True, "use_ema": False, "threshold": 0.5}
        self._make_parent(legacy=True)

    def _make_parent(self, *, legacy: bool, recorded_rng: bool = True) -> None:
        for name, value in (
            ("topology_graphs.pkl", b"topologies"),
            ("sampled_basis_indices.pkl", b"donor indices"),
        ):
            (self.generation_dir / name).write_bytes(value)
        write_json(self.generation_dir / "topology_pairing.json", {
            "checkpoint_sha256": self.checkpoint_sha,
            "per_graph_sha256": ["a", "b", "c"],
            "batch_size": 1,
        })
        files = {
            name: {"path": name, "sha256": sha256(self.generation_dir / name)}
            for name in ("topology_graphs.pkl", "sampled_basis_indices.pkl", "topology_pairing.json")
        }
        files = {
            "topology_graphs": files["topology_graphs.pkl"],
            "sampled_basis_indices": files["sampled_basis_indices.pkl"],
            "topology_pairing": files["topology_pairing.json"],
        }
        generation = {
            "generation_seed": 41,
            "num_generated": 3,
            "sampling": {"separate_attribute_rng": recorded_rng, "threshold": 0.5},
            "files": files,
        }
        if not legacy:
            generation["effective_sample_config"] = dict(self.sample)
        write_json(self.generation_dir / "manifest.json", generation)
        if not legacy:
            return
        context = {
            "kind": "molecular",
            "dataset": "qm9",
            "seed": 41,
            "num_samples": 3,
            "run_id": self.run_dir.name,
            "generation_id": self.generation_dir.name,
            "checkpoint_sha256": self.checkpoint_sha,
            "training_manifest_sha256": sha256(self.train_manifest),
            "config_semantic_sha256": "synthetic-config",
        }
        index_path = self.generation_dir / "molecular_ablation_manifest.json"
        write_json(index_path, {
            "format": "graphes_molecular_ablation_export_v1",
            "provenance": {
                **context,
                "sample_config": dict(self.sample),
                "source_generation_manifest_sha256": sha256(self.generation_dir / "manifest.json"),
            },
        })
        write_json(self.generation_dir / "ablation_reuse_receipt.json", {
            "format": "graphes_ablation_reuse_receipt_v1",
            "context": context,
            "generation_manifest_sha256": sha256(self.generation_dir / "manifest.json"),
            "molecular_ablation_manifest_sha256": sha256(index_path),
        })

    def _reference(self) -> dict:
        return require_attribute_reference(
            self.run_dir, self.generation_dir, dataset="qm9", seed=41, n=3
        )

    def test_sealed_legacy_sample_config_is_recovered(self) -> None:
        record = self._reference()
        self.assertEqual(record["effective_sample_config"], self.sample)
        self.assertEqual(record["effective_sample_config_source"], "sealed_molecular_ablation_export")
        self.assertEqual(len(record["effective_sample_config_provenance_sha256"]), 2)

    def test_modern_sample_config_is_used_directly(self) -> None:
        self._make_parent(legacy=False)
        record = self._reference()
        self.assertEqual(record["effective_sample_config_source"], "generation_manifest")
        self.assertEqual(record["effective_sample_config_provenance_sha256"], {})

    def test_missing_sealed_receipt_fails(self) -> None:
        (self.generation_dir / "ablation_reuse_receipt.json").unlink()
        with self.assertRaisesRegex(ValueError, "sealed molecular export/receipt"):
            self._reference()

    def test_false_saved_rng_fails_even_with_legacy_export(self) -> None:
        self._make_parent(legacy=True, recorded_rng=False)
        with self.assertRaisesRegex(ValueError, "sampling.separate_attribute_rng"):
            self._reference()

    def test_altered_export_fails_receipt_hash(self) -> None:
        path = self.generation_dir / "molecular_ablation_manifest.json"
        record = json.loads(path.read_text())
        record["provenance"]["sample_config"]["threshold"] = 0.9
        write_json(path, record)
        with self.assertRaisesRegex(ValueError, "hashes no longer match"):
            self._reference()

    def test_receipt_and_export_context_must_agree(self) -> None:
        path = self.generation_dir / "molecular_ablation_manifest.json"
        record = json.loads(path.read_text())
        record["provenance"]["seed"] = 42
        write_json(path, record)
        receipt_path = self.generation_dir / "ablation_reuse_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["molecular_ablation_manifest_sha256"] = sha256(path)
        write_json(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, "disagree on generation provenance"):
            self._reference()

    def test_explicit_ema_policy_is_required(self) -> None:
        self.sample["use_ema"] = True
        self._make_parent(legacy=True)
        with self.assertRaisesRegex(ValueError, "sample.use_ema=false"):
            self._reference()


if __name__ == "__main__":
    unittest.main()
