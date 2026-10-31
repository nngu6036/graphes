"""Regression checks for strict 200-epoch checkpoint reuse and study commands."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_grapher_ablation_artifacts.py"
SPEC = importlib.util.spec_from_file_location("check_grapher_ablation_artifacts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class TrainingEpochReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="grapher-ablation-check-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "runs" / "seed_41_community_small"
        self.train = self.run / "train"
        self.checkpoint = self.train / "checkpoints" / "grapher.pt"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"trusted synthetic checkpoint for hash checks")
        self.dataset_root = self.root / "prepared"
        dataset = self.dataset_root / "sbm"
        dataset.mkdir(parents=True)
        for split in ("train", "val", "test"):
            (dataset / f"{split}.pkl").write_bytes(f"synthetic {split} split".encode())
        self.config = self.root / "configuration.yaml"
        self.options = {
            "variant": "vanilla_laplacian_loggap_graphlet",
            "train": {"epochs": 200, "batch_size": 128, "lr": 0.01},
            "model": {"hidden_dim": 32},
            "sde": {"eps": 1e-5},
            "sample": {"steps": 10},
        }
        self.config.write_text(yaml.safe_dump({"grapher": self.options}), encoding="utf-8")
        recorded = copy.deepcopy(self.options)
        split_hashes = {
            split: checker.sha256(dataset / f"{split}.pkl")
            for split in ("train", "val", "test")
        }
        self.manifest = {
            "model_id": "grapher",
            "run_id": self.run.name,
            "train_seed": 41,
            "checkpoint": {
                "path": "checkpoints/grapher.pt",
                "sha256": checker.sha256(self.checkpoint),
            },
            "options": recorded,
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
        self.write_manifest()

    def write_manifest(self) -> None:
        (self.train / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def snapshot(self) -> dict[str, bytes]:
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*") if path.is_file()
        }

    def invoke(self, *extra: str, mode: str = "training") -> tuple[int, str]:
        arguments = [
            mode, "--run-dir", str(self.run), "--config", str(self.config),
            "--dataset", "community_small", "--serialized-dataset", "sbm",
            "--dataset-root", str(self.dataset_root), "--seed", "41", *extra,
        ]
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            try:
                result = checker.main(arguments)
            except SystemExit as error:
                result = error.code
        return int(result), output.getvalue()

    def assert_training_result(self, expected: int, *extra: str) -> str:
        before = self.snapshot()
        actual, output = self.invoke(*extra)
        self.assertEqual(actual, expected, output)
        self.assertEqual(self.snapshot(), before, "Training verification modified saved artifacts")
        return output

    def test_recorded_200_epoch_checkpoint_reuses_without_modifying_artifacts(self) -> None:
        self.assert_training_result(0)
        self.assertEqual(checker.config_options(self.config)["train"]["epochs"], 200)

    def test_main_and_ablation_yaml_budgets_remain_200_epochs(self) -> None:
        repository = SCRIPT.parents[1]
        for folder in ("grapher_final", "grapher_paper_ablation"):
            configurations = sorted((repository / "configs/experiments" / folder).glob("*.yaml"))
            self.assertTrue(configurations, f"Missing {folder} configurations")
            for path in configurations:
                with self.subTest(config=path.name):
                    self.assertEqual(checker.config_options(path)["train"]["epochs"], 200)

    def test_legacy_10000_epoch_checkpoint_is_rejected(self) -> None:
        self.manifest["options"]["train"]["epochs"] = 10000
        self.manifest["checkpoint_selection"]["epoch"] = 10000
        self.write_manifest()
        output = self.assert_training_result(2)
        self.assertIn("training.train.epochs", output)

    def test_nonfinal_epoch_is_rejected(self) -> None:
        self.manifest["checkpoint_selection"]["epoch"] = 199
        self.write_manifest()
        self.assert_training_result(2)

    def test_wrong_checkpoint_selection_kind_is_rejected(self) -> None:
        self.manifest["checkpoint_selection"]["kind"] = "best_validation_epoch"
        self.write_manifest()
        self.assert_training_result(2)

    def test_other_training_configuration_remains_strict(self) -> None:
        self.manifest["options"]["train"]["lr"] = 0.5
        self.write_manifest()
        output = self.assert_training_result(2)
        self.assertIn("training.train.lr", output)

    def test_model_configuration_remains_strict(self) -> None:
        self.manifest["options"]["model"]["hidden_dim"] = 64
        self.write_manifest()
        output = self.assert_training_result(2)
        self.assertIn("training.model.hidden_dim", output)

    def test_checkpoint_hash_corruption_is_rejected(self) -> None:
        self.checkpoint.write_bytes(b"changed checkpoint")
        output = self.assert_training_result(2)
        self.assertIn("hash mismatch", output)

    def test_changed_dataset_split_is_rejected(self) -> None:
        (self.dataset_root / "sbm" / "train.pkl").write_bytes(b"different split")
        output = self.assert_training_result(2)
        self.assertIn("split hashes", output)

    def test_changed_dataset_identity_is_rejected(self) -> None:
        self.manifest["dataset"]["serialized_id"] = "ego_small"
        self.write_manifest()
        self.assert_training_result(2)

    def test_changed_dataset_fingerprint_is_rejected(self) -> None:
        self.manifest["dataset"]["fingerprint"] = "wrong fingerprint"
        self.write_manifest()
        self.assert_training_result(2)

    def test_epoch_override_is_not_accepted(self) -> None:
        for epochs in ("200", "10000", "0", "-1"):
            with self.subTest(epochs=epochs):
                output = self.assert_training_result(2, "--epochs", epochs)
                self.assertIn("unrecognized arguments", output)

    def test_begin_semantic_hash_uses_yaml_200_epoch_budget(self) -> None:
        generation = self.run / "generations" / "paper_seed_41"
        before = self.snapshot()
        result, output = self.invoke(
            "--generation-dir", str(generation),
            "--num-samples", "16", "--kind", "matched", mode="begin",
        )
        self.assertEqual(result, 0, output)
        intent_path = generation.parent / f"{generation.name}.ablation_generation_intent.json"
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
        self.assertEqual(intent["context"]["config_semantic_sha256"], checker.digest(self.options))
        self.assertEqual(intent["config_file_sha256"], checker.sha256(self.config))
        after = self.snapshot()
        for path, content in before.items():
            self.assertEqual(after[path], content, f"begin modified existing artifact {path}")
        self.assertFalse(generation.exists(), "begin must not create a generation directory")


class StudyCommandRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        candidates = [shutil.which("bash")]
        if os.name == "nt":
            candidates = [
                str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"),
                *candidates,
            ]
        cls.bash = next((path for path in candidates if path and Path(path).is_file()), None)
        if cls.bash is None:
            raise unittest.SkipTest("Bash is required for command dry-run regressions")
        cls.repository = SCRIPT.parents[1]

    def dry_run(self, entry: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        for key in (
            "TRAIN_EPOCHS", "RUN_TEMPLATE", "OUTPUT_ROOT", "DATASET_ROOT", "REPORT_ROOT",
            "GENERATION_TAG", "S0_GENERATION_TAG", "N", "SEEDS", "DO_TRAIN", "DO_GENERATE",
            "DO_EVALUATE", "DO_S0_TRAIN", "DO_S0_GENERATE", "RUN_S0", "OVERWRITE",
            "REFERENCE_SPLIT", "REUSE", "ABLATION_STUDY",
        ):
            environment.pop(key, None)
        environment.update(DRY_RUN="1", SEEDS="41", PYTHON_BIN="python")
        environment.update(overrides)
        return subprocess.run(
            [self.bash, entry.as_posix()], cwd=self.repository, env=environment,
            capture_output=True, text=True, timeout=30,
        )

    def test_all_ten_studies_select_current_main_v2_run_ids(self) -> None:
        entries = sorted((self.repository / "commands/ablations").glob("*/*.sh"))
        self.assertEqual(len(entries), 10)
        for entry in entries:
            with self.subTest(entry=entry.relative_to(self.repository)):
                result = self.dry_run(entry.relative_to(self.repository))
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                dataset = entry.parent.name
                profile = "tau0p4_B1to4" if dataset in {"community_small", "ego_small"} else "B1to4"
                run_id = f"seed_41_{dataset}_grapher_final_{profile}_g345_v2"
                self.assertIn(f"--run-id {run_id}", result.stdout)
                self.assertIn(f"--run-dir outputs/baselines/grapher/{dataset}/{run_id}", result.stdout)
                self.assertNotIn("_grapher_final_g345_v1", result.stdout)
                self.assertNotIn("--epochs", result.stdout)

    def test_stale_epoch_override_is_rejected_before_any_work(self) -> None:
        result = self.dry_run(
            Path("commands/ablations/community_small/primary.sh"), TRAIN_EPOCHS="10000",
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("TRAIN_EPOCHS", result.stderr)
        self.assertIn("200", result.stderr)
        self.assertNotIn("scripts/run_grapher_baseline.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
