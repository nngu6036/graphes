"""Verify matched reuse without rewriting hash-anchored parent graph pickles."""
from __future__ import annotations

import copy
import importlib.util
import pickle
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import networkx as nx
import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "matched_reuse_checker", REPOSITORY / "scripts/check_grapher_ablation_artifacts.py"
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class MatchedPopulationReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="matched-ablation-reuse-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "runs" / "seed_41_community_small"
        self.train = self.run / "train"
        checkpoint = self.train / "checkpoints" / "grapher.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic final 200-epoch checkpoint")
        dataset_root = self.root / "prepared"
        dataset = dataset_root / "sbm"
        dataset.mkdir(parents=True)
        for split in ("train", "val", "test"):
            (dataset / f"{split}.pkl").write_bytes(f"synthetic {split}".encode())
        self.options = {
            "variant": "vanilla_laplacian_loggap_graphlet",
            "train": {"epochs": 200, "batch_size": 2},
            "model": {"hidden_dim": 4},
            "sde": {"eps": 1e-5},
            "sample": {
                "threshold": 0.5,
                "matched_ablation": {"enabled": True},
            },
            "generation_batch_size": 2,
            "graphlet_refinement": {
                "enabled": True,
                "random_steps": {"enabled": True, "min_steps": 1, "max_steps": 4},
                "refiner": {"steps": 4},
                "target_source": "joint_structure_summary_head",
            },
        }
        self.config = self.root / "configuration.yaml"
        self.config.write_text(
            yaml.safe_dump({"grapher": self.options}), encoding="utf-8"
        )
        split_hashes = {
            split: checker.sha256(dataset / f"{split}.pkl")
            for split in ("train", "val", "test")
        }
        training = {
            "model_id": "grapher",
            "run_id": self.run.name,
            "train_seed": 41,
            "checkpoint": {
                "path": "checkpoints/grapher.pt",
                "sha256": checker.sha256(checkpoint),
            },
            "options": copy.deepcopy(self.options),
            "checkpoint_selection": {"kind": "final_configured_epoch", "epoch": 200},
            "dataset": {
                "benchmark_id": "community_small",
                "serialized_id": "sbm",
                "split_sha256": split_hashes,
                "fingerprint": checker.digest({
                    split: {"path": f"{split}.pkl", "sha256": split_hashes[split]}
                    for split in sorted(split_hashes)
                }),
            },
        }
        checker.write_json(self.train / "manifest.json", training)
        self.generation = self.run / "generations" / "paper_ablation_seed_41_n_2"
        self.generation.mkdir(parents=True)
        self.args = Namespace(
            run_dir=self.run, config=self.config, dataset="community_small",
            serialized_dataset="sbm", dataset_root=dataset_root, seed=41,
            generation_dir=self.generation, num_samples=2, kind="matched",
        )
        self.ids = ["41:0", "41:1"]
        self.sources = {
            "T": "pre_realization_graphs",
            "D": "pre_refinement_graphs",
            "DR": "base_graphs",
        }
        records = {}
        for name in self.sources.values():
            path = self.generation / f"{name}.pkl"
            self.dump_graphs(path, [nx.path_graph(3), nx.cycle_graph(4)])
            records[name] = {"path": path.name, "sha256": checker.sha256(path)}
        basis_path = self.generation / "sampled_basis_indices.pkl"
        basis_path.write_bytes(pickle.dumps([0, 1]))
        self.manifest = {
            "model_id": "grapher", "run_id": self.run.name,
            "generation_id": self.generation.name, "generation_seed": 41,
            "num_requested": 2, "num_generated": 2,
            "variant": self.options["variant"],
            "checkpoint": {"sha256": checker.sha256(checkpoint)},
            "sampled_basis_indices": {
                "path": basis_path.name, "sha256": checker.sha256(basis_path),
            },
            "effective_sample_config": copy.deepcopy(self.options["sample"]),
            "sde": copy.deepcopy(self.options["sde"]),
            "generation_batch_size": 2,
            "sampling": {
                "stochastic_diversification": copy.deepcopy(
                    self.options["graphlet_refinement"]
                ),
            },
            **records,
        }
        self.write_generation()
        _, context = checker.invocation_context(self.args, self.options)
        self.common = {
            "format": "graphes_matched_ablation_export_v1",
            "enabled": True,
            "sample_ids": self.ids,
            "generation_seed": 41,
            "num_requested": 2,
            "provenance": {
                **context,
                "sample_config": copy.deepcopy(self.options["sample"]),
                "sde": copy.deepcopy(self.options["sde"]),
                "generation_batch_size": 2,
                "source_generation_manifest_sha256": checker.sha256(
                    self.generation / "manifest.json"
                ),
                "identity_scheme": "hash_anchored_parent_list_order",
                "source_artifacts": {
                    **copy.deepcopy(records),
                    "sampled_basis_indices": copy.deepcopy(
                        self.manifest["sampled_basis_indices"]
                    ),
                },
            },
            "branches": {},
        }
        for variant, source in self.sources.items():
            source_path = self.generation / self.manifest[source]["path"]
            child_dir = self.generation / "ablations" / variant
            child_dir.mkdir(parents=True)
            child_graphs = child_dir / "base_graphs.pkl"
            shutil.copyfile(source_path, child_graphs)
            child = {
                "generation_seed": 41, "requested_count": 2,
                "requested_sample_ids": self.ids, "sample_ids": self.ids,
                "completed_count": 2, "failed_count": 0, "failures": [],
                "status": "complete",
                "identity_source": "hash_anchored_parent_list_order",
                "source_artifact": copy.deepcopy(self.manifest[source]),
                "base_graphs": {
                    "path": "base_graphs.pkl", "sha256": checker.sha256(child_graphs),
                },
            }
            self.write_child(variant, child)
        self.write_common()

    @staticmethod
    def dump_graphs(path: Path, graphs: list[nx.Graph]) -> None:
        path.write_bytes(pickle.dumps(graphs, protocol=pickle.HIGHEST_PROTOCOL))

    def write_generation(self) -> None:
        checker.write_json(self.generation / "manifest.json", self.manifest)

    def write_child(self, variant: str, child: dict) -> None:
        checker.write_json(self.generation / "ablations" / variant / "manifest.json", child)
        self.common["branches"][variant] = {
            **copy.deepcopy(child), "manifest_path": f"{variant}/manifest.json",
        }

    def write_common(self) -> None:
        for path in (
            self.generation / "matched_ablation_manifest.json",
            self.generation / "ablations" / "ablation_manifest.json",
        ):
            checker.write_json(path, self.common)

    def rewrite_parent(self, variant: str, graphs: list[nx.Graph]) -> None:
        """Keep hashes consistent so tests reach the identity/count checks."""
        source = self.sources[variant]
        record = self.manifest[source]
        path = self.generation / record["path"]
        self.dump_graphs(path, graphs)
        record["sha256"] = checker.sha256(path)
        self.write_generation()
        self.common["provenance"]["source_generation_manifest_sha256"] = (
            checker.sha256(self.generation / "manifest.json")
        )
        self.common["provenance"]["source_artifacts"][source] = copy.deepcopy(record)
        child = copy.deepcopy(self.common["branches"][variant])
        child.pop("manifest_path", None)
        child["source_artifact"] = copy.deepcopy(record)
        child["base_graphs"]["sha256"] = record["sha256"]
        shutil.copyfile(path, self.generation / "ablations" / variant / "base_graphs.pkl")
        self.write_child(variant, child)
        self.write_common()

    def verify(self, variants: list[str] | None = None) -> None:
        generation, _, receipt = checker.verify_generation(self.args, self.options)
        checker.verify_branches(
            self.args, self.options, generation, receipt,
            variants if variants is not None else ["T", "D", "DR"],
        )

    def snapshot(self) -> dict[str, bytes]:
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*") if path.is_file()
        }

    def test_unlabelled_parent_lists_reuse_without_modifying_artifacts(self) -> None:
        before = self.snapshot()
        self.verify()
        self.assertEqual(before, self.snapshot())

    def test_missing_matched_sidecar_does_not_authorize_positional_ids(self) -> None:
        (self.generation / "matched_ablation_manifest.json").unlink()
        with self.assertRaises((ValueError, OSError)):
            self.verify()

    def test_wrong_identity_scheme_is_rejected(self) -> None:
        self.common["provenance"]["identity_scheme"] = "unverified_list_order"
        self.write_common()
        with self.assertRaisesRegex(ValueError, "identity mapping"):
            self.verify()

    def test_conflicting_embedded_parent_id_is_rejected_despite_valid_hashes(self) -> None:
        graphs = [nx.path_graph(3), nx.cycle_graph(4)]
        graphs[0].graph["sample_id"] = "41:99"
        self.rewrite_parent("DR", graphs)
        with self.assertRaisesRegex(ValueError, "ID/order"):
            self.verify()

    def test_labelled_parent_order_is_checked_despite_valid_hashes(self) -> None:
        graphs = [nx.path_graph(3), nx.cycle_graph(4)]
        graphs[0].graph["sample_id"] = "41:1"
        graphs[1].graph["sample_id"] = "41:0"
        self.rewrite_parent("DR", graphs)
        with self.assertRaisesRegex(ValueError, "ID/order"):
            self.verify()

    def test_parent_count_is_checked_despite_valid_hashes(self) -> None:
        self.rewrite_parent("DR", [nx.path_graph(3)])
        with self.assertRaisesRegex(ValueError, "population count"):
            self.verify()

    def test_changed_parent_file_fails_hash_validation(self) -> None:
        (self.generation / "base_graphs.pkl").write_bytes(b"changed parent bytes")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.verify()

    def test_sidecar_source_generation_digest_is_checked(self) -> None:
        self.common["provenance"]["source_generation_manifest_sha256"] = "wrong digest"
        self.write_common()
        with self.assertRaisesRegex(ValueError, "source generation manifest"):
            self.verify()

    def test_sidecar_training_context_is_checked(self) -> None:
        self.common["provenance"]["training_manifest_sha256"] = "wrong training hash"
        self.write_common()
        with self.assertRaisesRegex(ValueError, "training_manifest_sha256"):
            self.verify()

    def test_sidecar_configuration_digest_is_checked(self) -> None:
        self.common["provenance"]["config_semantic_sha256"] = "wrong config digest"
        self.write_common()
        with self.assertRaisesRegex(ValueError, "config_semantic_sha256"):
            self.verify()

    def test_sidecar_and_branch_index_must_agree(self) -> None:
        changed = copy.deepcopy(self.common)
        changed["generation_seed"] = 42
        checker.write_json(self.generation / "ablations" / "ablation_manifest.json", changed)
        with self.assertRaises(ValueError):
            self.verify()

    def test_parent_copy_requires_matching_source_record(self) -> None:
        child = copy.deepcopy(self.common["branches"]["T"])
        child.pop("manifest_path", None)
        child["source_artifact"] = copy.deepcopy(self.manifest["pre_refinement_graphs"])
        self.write_child("T", child)
        self.write_common()
        with self.assertRaises(ValueError):
            self.verify()

    def test_derived_branch_requires_embedded_sample_ids(self) -> None:
        child_dir = self.generation / "ablations" / "TR"
        child_dir.mkdir()
        path = child_dir / "base_graphs.pkl"
        self.dump_graphs(path, [nx.complete_graph(3), nx.path_graph(4)])
        child = copy.deepcopy(self.common["branches"]["T"])
        child.pop("manifest_path", None)
        child.pop("source_artifact")
        child.pop("identity_source")
        child["base_graphs"] = {"path": path.name, "sha256": checker.sha256(path)}
        self.write_child("TR", child)
        self.write_common()
        with self.assertRaises(ValueError):
            self.verify(["TR"])

    def test_derived_branch_with_correct_embedded_ids_reuses(self) -> None:
        child_dir = self.generation / "ablations" / "TR"
        child_dir.mkdir()
        path = child_dir / "base_graphs.pkl"
        graphs = [nx.complete_graph(3), nx.path_graph(4)]
        for graph, sample_id in zip(graphs, self.ids):
            graph.graph["sample_id"] = sample_id
        self.dump_graphs(path, graphs)
        child = copy.deepcopy(self.common["branches"]["T"])
        child.pop("manifest_path", None)
        child.pop("source_artifact")
        child.pop("identity_source")
        child["base_graphs"] = {"path": path.name, "sha256": checker.sha256(path)}
        self.write_child("TR", child)
        self.write_common()
        self.verify(["TR"])

    def test_prepare_requires_original_intent_and_leaves_artifacts_unchanged(self) -> None:
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "original generation intent"):
            checker.prepare_matched(self.args, self.options)
        self.assertEqual(before, self.snapshot())

    def test_prepare_rejects_changed_original_intent_context(self) -> None:
        _, context = checker.invocation_context(self.args, self.options)
        context["seed"] = 42
        path = self.generation.parent / (
            f"{self.generation.name}.ablation_generation_intent.json"
        )
        checker.write_json(path, {
            "context": context, "config_file_sha256": checker.sha256(self.config),
        })
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "intent.*provenance differs"):
            checker.prepare_matched(self.args, self.options)
        self.assertEqual(before, self.snapshot())

    def test_prepare_rejects_changed_configuration_file_hash(self) -> None:
        _, context = checker.invocation_context(self.args, self.options)
        path = self.generation.parent / (
            f"{self.generation.name}.ablation_generation_intent.json"
        )
        checker.write_json(path, {
            "context": context, "config_file_sha256": "changed file digest",
        })
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "configuration file changed"):
            checker.prepare_matched(self.args, self.options)
        self.assertEqual(before, self.snapshot())


if __name__ == "__main__":
    unittest.main()
