"""Recover and reuse trusted molecular endpoints without modifying parent files."""
from __future__ import annotations

import copy
import importlib.util
import io
import os
import pickle
import shutil
import subprocess
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "molecular_reuse_checker", REPOSITORY / "scripts/check_grapher_ablation_artifacts.py"
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class MolecularPopulationReuseTests(unittest.TestCase):
    dataset_id = "qm9"
    serialized_id = "qm9_attributed"
    seed = 44

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="molecular-ablation-reuse-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "runs" / f"seed_{self.seed}_{self.dataset_id}_grapher_final_B1to4_g345_v2"
        self.train = self.run / "train"
        checkpoint = self.train / "checkpoints" / "grapher.pt"
        checkpoint.parent.mkdir(parents=True)
        # Export and verification must never need to load a neural checkpoint.
        checkpoint.write_bytes(b"synthetic final 200-epoch checkpoint; not a torch payload")
        dataset_root = self.root / "prepared"
        dataset = dataset_root / self.serialized_id
        dataset.mkdir(parents=True)
        for split in ("train", "val", "test"):
            (dataset / f"{split}.pkl").write_bytes(f"synthetic {split}".encode())
        self.options = {
            "variant": "vanilla_laplacian_loggap_hybrid_categorical_attributed_ppgn",
            "train": {"epochs": 200, "batch_size": 2},
            "model": {"hidden_dim": 4},
            "sde": {"eps": 1e-5, "spectral_parameterization": "laplacian_log_gap"},
            "sample": {
                "threshold": 0.5, "separate_attribute_rng": True,
                "degree_constrained_realization": {"enabled": False},
                "matched_ablation": {"enabled": False},
            },
            "attributed": {"decode": {
                "node_mode": "sample", "edge_mode": "sample",
                "node_temperature": 1.0, "edge_temperature": 0.6,
                "constraint_mode": "atom_bond_valence", "edge_order": "random",
                "infeasible_policy": "retain",
            }},
            "attribute_diffusion": {"enabled": True, "steps": 100, "sample_steps": 100},
            "attributed_diversification": {
                "enabled": True, "mode": "stochastic_same_bond_diversification",
                "strict_same_bond": True,
                "random_steps": {"min_steps": 1, "max_steps": 4},
            },
            "generation_batch_size": 2,
        }
        self.config = self.root / "configuration.yaml"
        self.config.write_text(
            yaml.safe_dump({"grapher": self.options}), encoding="utf-8"
        )
        split_hashes = {
            split: checker.sha256(dataset / f"{split}.pkl")
            for split in ("train", "val", "test")
        }
        self.training_manifest = {
            "model_id": "grapher", "run_id": self.run.name, "train_seed": self.seed,
            "checkpoint": {
                "path": "checkpoints/grapher.pt", "sha256": checker.sha256(checkpoint),
            },
            "options": copy.deepcopy(self.options),
            "checkpoint_selection": {"kind": "final_configured_epoch", "epoch": 200},
            "dataset": {
                "benchmark_id": self.dataset_id, "serialized_id": self.serialized_id,
                "split_sha256": split_hashes,
                "fingerprint": checker.digest({
                    split: {"path": f"{split}.pkl", "sha256": split_hashes[split]}
                    for split in sorted(split_hashes)
                }),
            },
        }
        checker.write_json(self.train / "manifest.json", self.training_manifest)
        self.generation = self.run / "generations" / f"paper_ablation_v1_seed_{self.seed}_n_2"
        self.generation.mkdir(parents=True)
        self.args = Namespace(
            run_dir=self.run, config=self.config, dataset=self.dataset_id,
            serialized_dataset=self.serialized_id, dataset_root=dataset_root, seed=self.seed,
            generation_dir=self.generation, num_samples=2, kind="molecular",
        )
        self.ids = [f"{self.seed}:0", f"{self.seed}:1"]
        self.parent_records = {}
        for name, stage in (("pre_refinement_graphs", "T"), ("base_graphs", "TR")):
            path = self.generation / f"{name}.pkl"
            self.dump_graphs(path, self.molecules(stage))
            self.parent_records[name] = {
                "path": path.name, "sha256": checker.sha256(path),
            }
        basis_path = self.generation / "sampled_basis_indices.pkl"
        basis_path.write_bytes(pickle.dumps([0, 1]))
        self.parent_records["sampled_basis_indices"] = {
            "path": basis_path.name, "sha256": checker.sha256(basis_path),
        }
        decode = self.options["attributed"]["decode"]
        self.manifest = {
            "model_id": "grapher", "run_id": self.run.name,
            "generation_id": self.generation.name, "generation_seed": self.seed,
            "num_requested": 2, "num_generated": 2,
            "variant": self.options["variant"],
            "checkpoint": {"sha256": checker.sha256(checkpoint)},
            "sampling": {
                "spectral_operator": "combinatorial_laplacian", "threshold": 0.5,
                "generation_batch_size": 2, "separate_attribute_rng": True,
                "node_decode_mode": decode["node_mode"],
                "edge_decode_mode": decode["edge_mode"],
                **{key: value for key, value in decode.items()
                   if key not in {"node_mode", "edge_mode"}},
                "attribute_diffusion_steps": 100, "attribute_sampling_steps": 100,
                "attributed_diversification": copy.deepcopy(
                    self.options["attributed_diversification"]
                ),
            },
            **copy.deepcopy(self.parent_records),
        }
        # Reproduce the old molecular producer: no effective sampler, SDE or
        # top-level batch fields. Only the original begin intent proves them.
        self.write_generation()
        _, self.context = checker.invocation_context(self.args, self.options)
        self.intent = self.generation.parent / (
            f"{self.generation.name}.ablation_generation_intent.json"
        )
        checker.write_json(self.intent, {
            "context": self.context, "config_file_sha256": checker.sha256(self.config),
        })

    @staticmethod
    def molecules(stage: str) -> list[nx.Graph]:
        output = []
        for atomic_numbers in ((6, 8), (6, 7, 8)):
            graph = nx.path_graph(len(atomic_numbers))
            for node, atomic_number in enumerate(atomic_numbers):
                graph.nodes[node]["atomic_num"] = atomic_number
            for left, right in graph.edges():
                graph.edges[left, right]["bond_type"] = 1
            graph.graph["saved_stage"] = stage
            output.append(graph)
        return output

    @staticmethod
    def dump_graphs(path: Path, graphs: list[nx.Graph]) -> None:
        path.write_bytes(pickle.dumps(graphs, protocol=pickle.HIGHEST_PROTOCOL))

    def write_generation(self) -> None:
        checker.write_json(self.generation / "manifest.json", self.manifest)

    def rewrite_parent(self, name: str, graphs: list[nx.Graph]) -> None:
        record = self.manifest[name]
        path = self.generation / record["path"]
        self.dump_graphs(path, graphs)
        record["sha256"] = checker.sha256(path)
        self.write_generation()

    def invoke(self, mode: str, *extra: str) -> tuple[int, str]:
        arguments = [
            mode, "--run-dir", str(self.run), "--config", str(self.config),
            "--dataset", self.dataset_id, "--serialized-dataset", self.serialized_id,
            "--dataset-root", str(self.args.dataset_root), "--seed", str(self.seed),
            "--generation-dir", str(self.generation), "--num-samples", "2",
            "--kind", "molecular", *extra,
        ]
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = checker.main(arguments)
        return result, output.getvalue()

    def snapshot(self) -> dict[str, bytes]:
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*") if path.is_file()
        }

    def prepare(self) -> None:
        result, output = self.invoke("prepare")
        self.assertEqual(result, 0, output)

    def verify(self) -> None:
        generation, _, receipt = checker.verify_generation(
            self.args, self.options, allow_unsealed=True,
        )
        checker.verify_branches(self.args, self.options, generation, receipt, ["T", "TR"])

    def write_common(self, common: dict) -> None:
        checker.write_json(self.generation / "molecular_ablation_manifest.json", common)
        checker.write_json(self.generation / "ablations" / "ablation_manifest.json", common)

    def label_parent_populations(self) -> None:
        for source, stage in (("pre_refinement_graphs", "T"), ("base_graphs", "TR")):
            graphs = self.molecules(stage)
            for graph, sample_id in zip(graphs, self.ids):
                graph.graph["sample_id"] = sample_id
            self.rewrite_parent(source, graphs)

    def seal_and_restore_original_intent(self) -> None:
        original_intent = self.intent.read_bytes()
        result, output = self.invoke("seal")
        self.assertEqual(result, 0, output)
        self.intent.write_bytes(original_intent)

    def assert_direct_seal_preserves_parents(self) -> None:
        original = self.snapshot()
        self.assertFalse((self.generation / "ablations").exists())
        with patch("torch.load", side_effect=AssertionError("seal loaded a neural checkpoint")):
            result, output = self.invoke("seal")
        self.assertEqual(result, 0, output)
        self.assertFalse(self.intent.exists())
        common = checker.read_json(self.generation / "molecular_ablation_manifest.json")
        self.assertEqual(common["sample_ids"], self.ids)
        self.assertEqual(
            common, checker.read_json(self.generation / "ablations" / "ablation_manifest.json")
        )
        self.assertTrue((self.generation / checker.RECEIPT).is_file())
        result, output = self.invoke("branches", "--variants", "T", "TR")
        self.assertEqual(result, 0, output)
        after = self.snapshot()
        for path, content in original.items():
            if not path.endswith(".ablation_generation_intent.json"):
                self.assertEqual(after[path], content, path)

    def test_seal_exports_missing_molecular_views_without_a_prepare_call(self) -> None:
        self.assert_direct_seal_preserves_parents()

    def test_zinc_seed_41_seal_exports_missing_views_without_a_prepare_call(self) -> None:
        zinc = MolecularPopulationReuseTests(
            "test_seal_exports_missing_molecular_views_without_a_prepare_call"
        )
        zinc.dataset_id = "zinc"
        zinc.serialized_id = "zinc"
        zinc.seed = 41
        self.addCleanup(zinc.doCleanups)
        zinc.setUp()
        zinc.assert_direct_seal_preserves_parents()

    def test_labelled_zinc_producer_seals_missing_views_with_recorded_sampler(self) -> None:
        zinc = MolecularPopulationReuseTests(
            "test_seal_exports_missing_molecular_views_without_a_prepare_call"
        )
        zinc.dataset_id = "zinc"
        zinc.serialized_id = "zinc"
        zinc.seed = 41
        self.addCleanup(zinc.doCleanups)
        zinc.setUp()
        zinc.label_parent_populations()
        zinc.manifest.update({
            "effective_sample_config": copy.deepcopy(zinc.options["sample"]),
            "sde": copy.deepcopy(zinc.options["sde"]),
            "generation_batch_size": zinc.options["generation_batch_size"],
        })
        zinc.write_generation()
        zinc.assert_direct_seal_preserves_parents()

    def test_seal_without_original_intent_rejects_before_export_writes(self) -> None:
        self.intent.unlink()
        original = self.snapshot()
        result, output = self.invoke("seal")
        self.assertEqual(result, 2, output)
        self.assertIn("intent", output.lower())
        self.assertEqual(original, self.snapshot())

    def test_seal_with_wrong_intent_context_rejects_before_export_writes(self) -> None:
        intent = checker.read_json(self.intent)
        intent["context"]["config_semantic_sha256"] = "different original configuration"
        checker.write_json(self.intent, intent)
        original = self.snapshot()
        result, output = self.invoke("seal")
        self.assertEqual(result, 2, output)
        self.assertEqual(original, self.snapshot())

    def test_seal_with_wrong_configuration_file_hash_rejects_before_export_writes(self) -> None:
        intent = checker.read_json(self.intent)
        intent["config_file_sha256"] = "different original file"
        checker.write_json(self.intent, intent)
        original = self.snapshot()
        result, output = self.invoke("seal")
        self.assertEqual(result, 2, output)
        self.assertEqual(original, self.snapshot())

    def test_seal_rejects_an_incomplete_existing_export_without_rewriting_it(self) -> None:
        self.prepare()
        index = self.generation / "ablations" / "ablation_manifest.json"
        index.unlink()
        original = self.snapshot()
        result, output = self.invoke("seal")
        self.assertEqual(result, 2, output)
        self.assertFalse(index.exists())
        self.assertEqual(original, self.snapshot())

    def test_seal_rejects_a_corrupt_existing_index_without_rewriting_it(self) -> None:
        self.prepare()
        index = self.generation / "ablations" / "ablation_manifest.json"
        index.write_text("{invalid existing index", encoding="utf-8")
        original = self.snapshot()
        result, output = self.invoke("seal")
        self.assertEqual(result, 2, output)
        self.assertEqual(original, self.snapshot())

    def test_seal_keeps_an_existing_complete_export_unchanged(self) -> None:
        self.prepare()
        original = {
            str(path.relative_to(self.root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in self.root.rglob("*") if path.is_file() and path != self.intent
        }
        result, output = self.invoke("seal")
        self.assertEqual(result, 0, output)
        for name, (content, modified_ns) in original.items():
            path = self.root / name
            self.assertEqual(path.read_bytes(), content, name)
            self.assertEqual(path.stat().st_mtime_ns, modified_ns, name)

    def test_completed_receipt_and_leftover_intent_only_remove_the_intent(self) -> None:
        self.label_parent_populations()
        self.seal_and_restore_original_intent()
        original = {
            str(path.relative_to(self.root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in self.root.rglob("*") if path.is_file() and path != self.intent
        }
        result, output = self.invoke("seal")
        self.assertEqual(result, 0, output)
        self.assertFalse(self.intent.exists())
        for name, (content, modified_ns) in original.items():
            path = self.root / name
            self.assertEqual(path.read_bytes(), content, name)
            self.assertEqual(path.stat().st_mtime_ns, modified_ns, name)

    def test_completed_receipt_never_authorizes_reconstruction_of_deleted_indices(self) -> None:
        self.label_parent_populations()
        self.seal_and_restore_original_intent()
        (self.generation / "molecular_ablation_manifest.json").unlink()
        (self.generation / "ablations" / "ablation_manifest.json").unlink()
        original = self.snapshot()
        for mode in ("prepare", "seal"):
            with self.subTest(mode=mode):
                result, output = self.invoke(mode)
                self.assertEqual(result, 2, output)
                self.assertEqual(original, self.snapshot())

    def test_completed_receipt_rejects_changed_branch_manifest_without_rewriting_it(self) -> None:
        self.label_parent_populations()
        self.seal_and_restore_original_intent()
        path = self.generation / "ablations" / "T" / "manifest.json"
        child = checker.read_json(path)
        child["unrecorded_extra_field"] = "changed after receipt"
        checker.write_json(path, child)
        original = self.snapshot()
        result, output = self.invoke("seal")
        self.assertEqual(result, 2, output)
        self.assertIn("Branch manifest changed after receipt", output)
        self.assertEqual(original, self.snapshot())

    def test_original_intent_recovers_and_seals_unlabelled_population(self) -> None:
        original = self.snapshot()
        self.prepare()
        self.assertTrue(self.intent.is_file())
        common = checker.read_json(self.generation / "molecular_ablation_manifest.json")
        self.assertEqual(common["format"], "graphes_molecular_ablation_export_v1")
        self.assertEqual(common["sample_ids"], self.ids)
        self.assertEqual(
            common, checker.read_json(self.generation / "ablations" / "ablation_manifest.json")
        )
        for branch, source in (("T", "pre_refinement_graphs"), ("TR", "base_graphs")):
            child = checker.read_json(self.generation / "ablations" / branch / "manifest.json")
            self.assertEqual(child["source_artifact"], self.manifest[source])
            self.assertEqual(child["base_graphs"]["sha256"], self.manifest[source]["sha256"])
        result, output = self.invoke("seal")
        self.assertEqual(result, 0, output)
        self.assertFalse(self.intent.exists())
        for mode, extra in (("generation", ()), ("branches", ("--variants", "T", "TR"))):
            result, output = self.invoke(mode, *extra)
            self.assertEqual(result, 0, output)
        after = self.snapshot()
        for path, content in original.items():
            if not path.endswith(".ablation_generation_intent.json"):
                self.assertEqual(after[path], content, path)

    def test_prepare_without_original_intent_rejects_without_writes(self) -> None:
        self.intent.unlink()
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "original generation intent"):
            checker.prepare_molecular(self.args, self.options)
        self.assertEqual(before, self.snapshot())

    def test_unlabelled_parent_has_no_positional_permission_without_sidecar(self) -> None:
        with self.assertRaisesRegex(ValueError, "ID/order"):
            checker.verify_generation(self.args, self.options, allow_unsealed=True)

    def test_prepare_rejects_changed_configuration_file(self) -> None:
        self.config.write_text(self.config.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "configuration file changed"):
            checker.prepare_molecular(self.args, self.options)
        self.assertEqual(before, self.snapshot())

    def test_prepare_rejects_changed_intent_context(self) -> None:
        intent = checker.read_json(self.intent)
        intent["context"]["seed"] = 45
        checker.write_json(self.intent, intent)
        with self.assertRaisesRegex(ValueError, "intent.*provenance differs"):
            checker.prepare_molecular(self.args, self.options)

    def test_changed_parent_hash_is_rejected(self) -> None:
        (self.generation / "base_graphs.pkl").write_bytes(b"altered parent bytes")
        with self.assertRaisesRegex((ValueError, RuntimeError), "hash"):
            checker.prepare_molecular(self.args, self.options)

    def test_conflicting_embedded_parent_id_is_rejected_with_valid_hashes(self) -> None:
        graphs = self.molecules("TR")
        graphs[0].graph["sample_id"] = "44:99"
        self.rewrite_parent("base_graphs", graphs)
        with self.assertRaisesRegex((ValueError, RuntimeError), "ID|identity|sample_id"):
            checker.prepare_molecular(self.args, self.options)

    def test_reordered_labelled_parent_is_rejected_with_valid_hashes(self) -> None:
        graphs = self.molecules("TR")
        for graph, sample_id in zip(graphs, reversed(self.ids)):
            graph.graph["sample_id"] = sample_id
        self.rewrite_parent("base_graphs", graphs)
        with self.assertRaisesRegex((ValueError, RuntimeError), "ID|identity|sample_id"):
            checker.prepare_molecular(self.args, self.options)

    def test_parent_count_must_match_requested_population(self) -> None:
        self.rewrite_parent("base_graphs", self.molecules("TR")[:1])
        with self.assertRaisesRegex((ValueError, RuntimeError), "count|population"):
            checker.prepare_molecular(self.args, self.options)

    def test_recorded_molecular_sampler_settings_remain_strict(self) -> None:
        original = copy.deepcopy(self.manifest)
        for key, value in (
            ("generation_batch_size", 1), ("separate_attribute_rng", False),
            ("node_decode_mode", "argmax"), ("edge_decode_mode", "argmax"),
            ("node_temperature", 0.5), ("edge_temperature", 0.9),
            ("constraint_mode", "none"), ("attribute_diffusion_steps", 99),
            ("attribute_sampling_steps", 99),
        ):
            with self.subTest(setting=key):
                self.manifest = copy.deepcopy(original)
                self.manifest["sampling"][key] = value
                self.write_generation()
                with self.assertRaises(ValueError):
                    checker.prepare_molecular(self.args, self.options)
                self.assertFalse((self.generation / "molecular_ablation_manifest.json").exists())

    def replace_branch_source(self, variant: str, wrong_source: str) -> None:
        common = checker.read_json(self.generation / "molecular_ablation_manifest.json")
        directory = self.generation / "ablations" / variant
        child = checker.read_json(directory / "manifest.json")
        wrong_record = self.manifest[wrong_source]
        shutil.copyfile(self.generation / wrong_record["path"], directory / "base_graphs.pkl")
        child["source_artifact"] = copy.deepcopy(wrong_record)
        child["base_graphs"]["sha256"] = wrong_record["sha256"]
        child.setdefault("provenance", {})["source_sha256"] = wrong_record["sha256"]
        checker.write_json(directory / "manifest.json", child)
        common["branches"][variant] = copy.deepcopy(child)
        self.write_common(common)

    def test_wrong_t_parent_is_rejected_even_when_index_and_child_agree(self) -> None:
        self.prepare()
        self.replace_branch_source("T", "base_graphs")
        with self.assertRaises(ValueError):
            self.verify()

    def test_wrong_tr_parent_is_rejected_even_when_index_and_child_agree(self) -> None:
        self.prepare()
        self.replace_branch_source("TR", "pre_refinement_graphs")
        with self.assertRaises(ValueError):
            self.verify()

    def test_wrong_sidecar_training_context_is_rejected(self) -> None:
        self.prepare()
        common = checker.read_json(self.generation / "molecular_ablation_manifest.json")
        common["provenance"]["training_manifest_sha256"] = "changed training identity"
        self.write_common(common)
        with self.assertRaisesRegex(ValueError, "training_manifest_sha256"):
            self.verify()

    def test_sealed_sidecar_tampering_is_rejected(self) -> None:
        self.prepare()
        result, output = self.invoke("seal")
        self.assertEqual(result, 0, output)
        path = self.generation / "molecular_ablation_manifest.json"
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "receipt|changed"):
            self.verify()

    def test_legacy_10000_epoch_training_remains_incompatible(self) -> None:
        self.training_manifest["options"]["train"]["epochs"] = 10000
        self.training_manifest["checkpoint_selection"]["epoch"] = 10000
        checker.write_json(self.train / "manifest.json", self.training_manifest)
        with self.assertRaisesRegex(ValueError, "training.train.epochs"):
            checker.prepare_molecular(self.args, self.options)


class MolecularCommandRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        candidates = [shutil.which("bash")]
        if os.name == "nt":
            candidates.insert(0, str(
                Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
            ))
        cls.bash = next((path for path in candidates if path and Path(path).is_file()), None)
        if cls.bash is None:
            raise unittest.SkipTest("Bash is required for the real wrapper recovery check")

    def test_completed_qm9_wrapper_prepares_and_seals_with_generation_disabled(self) -> None:
        # Keep shell writes in the workspace so the sandbox also covers the
        # external Bash process; all fixture paths are removed on completion.
        with tempfile.TemporaryDirectory(
            prefix=".molecular-wrapper-recovery-", dir=REPOSITORY,
        ) as temporary:
            root = Path(temporary)
            self.assertTrue(root.resolve().is_relative_to(REPOSITORY.resolve()))
            relative = root.relative_to(REPOSITORY)
            output_root = root / "outputs"
            run = output_root / "grapher" / "qm9" / "seed_44_qm9_grapher_final_B1to4_g345_v2"
            generation = run / "generations" / "paper_ablation_v1_seed_44_n_10000"
            generation.mkdir(parents=True)
            intent = generation.parent / f"{generation.name}.ablation_generation_intent.json"
            intent.write_text("{}", encoding="utf-8")
            stub = root / "python-stub.sh"
            stub.write_text(
                '#!/usr/bin/env bash\n'
                'printf "%s\\n" "$*" >> "$MOLECULAR_WRAPPER_LOG"\n'
                'case "$1" in\n'
                '  scripts/run_grapher_baseline.py) exit 97 ;;\n'
                '  scripts/check_grapher_ablation_artifacts.py|scripts/summarize_grapher_ablation.py) exit 0 ;;\n'
                '  *) exit 98 ;;\n'
                'esac\n',
                encoding="utf-8", newline="\n",
            )
            stub.chmod(0o755)
            environment = os.environ.copy()
            for key in (
                "TRAIN_EPOCHS", "RUN_TEMPLATE", "GENERATION_TAG", "S0_GENERATION_TAG",
                "DO_S0_TRAIN", "DO_S0_GENERATE", "RUN_S0", "ABLATION_STUDY", "BASH_ENV",
            ):
                environment.pop(key, None)
            environment.update({
                "SEEDS": "44", "N": "10000", "DO_TRAIN": "0", "DO_GENERATE": "0",
                "DO_EVALUATE": "0", "REUSE": "1", "DRY_RUN": "0", "OVERWRITE": "0",
                "REFERENCE_SPLIT": "test", "DEVICE": "cpu",
                "OUTPUT_ROOT": (relative / "outputs").as_posix(),
                "REPORT_ROOT": (relative / "reports").as_posix(),
                "DATASET_ROOT": (relative / "prepared").as_posix(),
                "PYTHON_BIN": (relative / "python-stub.sh").as_posix(),
                "MOLECULAR_WRAPPER_LOG": (relative / "invocations.log").as_posix(),
            })
            # The isolated test does not exercise locking. Git Bash lacks
            # flock, so supply a no-op function while preserving the real flow.
            result = subprocess.run(
                [self.bash, "-c", 'flock() { return 0; }; export -f flock; exec bash "$1"',
                 "molecular-recovery-test", "commands/ablations/qm9/rewiring.sh"],
                cwd=REPOSITORY, env=environment, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            invocations = (root / "invocations.log").read_text(encoding="utf-8").splitlines()
            checks = [
                line.split()[1] for line in invocations
                if line.startswith("scripts/check_grapher_ablation_artifacts.py ")
            ]
            self.assertEqual(checks, ["prepare", "seal", "generation", "branches"])
            self.assertFalse(any("run_grapher_baseline.py" in line for line in invocations))
            self.assertFalse(any("export_grapher_ablation_views.py" in line for line in invocations))
            self.assertEqual(sum("summarize_grapher_ablation.py" in line for line in invocations), 1)


if __name__ == "__main__":
    unittest.main()
