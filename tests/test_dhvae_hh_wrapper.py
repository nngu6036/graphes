from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from grapher.models import (
    DatasetReference,
    GenerateRequest,
    RunSpec,
    TrainRequest,
    create_baseline,
)


class _GraphStub:
    """Pickle-safe graph-shaped object for dependency-light wrapper tests."""

    def __init__(self, index: int) -> None:
        self.index = int(index)
        self.graph: dict[str, object] = {"source_index": self.index}

    def nodes(self):
        return (0, 1)

    def edges(self):
        return ((0, 1),)


def _write_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


class DHVAEHHWrapperTests(unittest.TestCase):
    def _fixture(self, root: Path):
        dataset_root = root / "datasets"
        dataset_dir = dataset_root / "sbm"
        source_graphs = [_GraphStub(index) for index in range(4)]
        for split, graphs in {
            "train": source_graphs,
            "val": source_graphs[:1],
            "test": source_graphs[1:2],
        }.items():
            _write_pickle(dataset_dir / f"{split}.pkl", graphs)

        dataset_config = root / "community_small.yaml"
        dataset_config.write_text("name: sbm\ntype: sbm\n", encoding="utf-8")
        experiment_config = root / "dhvae.yaml"
        experiment_config.write_text(
            yaml.safe_dump(
                {
                    "experiment": "test_dhvae",
                    "seed": 0,
                    "dataset": {},
                    "summary": {"degree_hist_max_degree": "auto"},
                    "constructor": {"type": "havel_hakimi", "ensure_connected": True},
                    "degree_generator": {
                        "type": "degree_histogram_vae",
                        "epochs": 1,
                        "batch_size": 2,
                        "latent_dim": 2,
                        "hidden_dim": 4,
                        "num_layers": 1,
                        "dropout": 0.0,
                        "learning_rate": 0.001,
                        "weight_decay": 0.0,
                        "fallback": "error",
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        dataset = DatasetReference(
            "community_small",
            root=dataset_root,
            serialized_id="sbm",
            config_path=dataset_config,
        )
        run = RunSpec.for_seed(
            model_id="dhvae_hh",
            dataset_id="community_small",
            seed=17,
            output_root=root / "baselines",
        )
        return dataset, run, experiment_config, source_graphs

    @staticmethod
    def _fake_training_subprocess(*, config_path, log_path, seed, timeout):
        del seed, timeout
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        checkpoint = Path(config["degree_generator"]["checkpoint_path"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        (checkpoint.parent / "degree_vectorizer.json").write_text(
            "{}\n", encoding="utf-8"
        )
        (checkpoint.parent / "training_metrics.json").write_text(
            "{}\n", encoding="utf-8"
        )
        Path(log_path).write_text("training complete\n", encoding="utf-8")
        return ["python", "train_degree_generator.py", "--config", str(config_path)]

    @staticmethod
    def _fake_generate_batch(
        *, checkpoint_path, config, num_graphs, seed, max_attempts_per_graph
    ):
        del checkpoint_path, config, max_attempts_per_graph
        graphs = [_GraphStub(index) for index in range(num_graphs)]
        for index, graph in enumerate(graphs):
            graph.graph.update(
                {
                    "raw_index": index,
                    "generation_seed": seed,
                    "base_model": "dhvae_hh",
                }
            )
        return graphs, {
            "requested": num_graphs,
            "returned": num_graphs,
            "sample_order": "raw_index_0_to_n_minus_1",
            "dropped_after_acceptance": 0,
        }

    def test_training_publishes_unpaired_estimates_and_provenance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, run, config, source_graphs = self._fixture(root)
            wrapper = create_baseline("dhvae_hh")
            with (
                patch(
                    "grapher.models.dhvae_hh.wrapper._run_training_subprocess",
                    side_effect=self._fake_training_subprocess,
                ),
                patch(
                    "grapher.models.dhvae_hh.wrapper._normalize_checkpoint_config"
                ),
                patch(
                    "grapher.models.dhvae_hh.wrapper._generate_batch",
                    side_effect=self._fake_generate_batch,
                ),
            ):
                artifacts = wrapper.train(
                    TrainRequest(
                        run=run,
                        dataset=dataset,
                        config_path=config,
                    )
                )

            self.assertEqual(artifacts.checkpoint_path.read_bytes(), b"checkpoint")
            self.assertTrue(artifacts.manifest_path.is_file())
            self.assertTrue(artifacts.estimated_graphs_path.is_file())
            self.assertTrue(artifacts.ground_truth_graphs_path.is_file())
            estimate_manifest = json.loads(
                artifacts.training_estimates_manifest_path.read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(estimate_manifest["pairing"]["status"], "unpaired")
            self.assertEqual(
                estimate_manifest["semantics"],
                "independent_unconditional_sample_pool",
            )
            self.assertEqual(estimate_manifest["estimated_graphs"]["count"], 4)
            with artifacts.ground_truth_graphs_path.open("rb") as handle:
                copied_ground_truth = pickle.load(handle)
            self.assertEqual(
                [graph.index for graph in copied_ground_truth],
                [graph.index for graph in source_graphs],
            )
            manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset"]["benchmark_id"], "community_small")
            self.assertEqual(manifest["dataset"]["serialized_id"], "sbm")
            self.assertEqual(manifest["run_id"], "seed_17")
            self.assertEqual(manifest["train_seed"], 17)
            self.assertEqual(manifest["generator_type"], "degree_histogram_vae")

    def test_generation_reuses_managed_checkpoint_and_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, run, config, _source_graphs = self._fixture(root)
            wrapper = create_baseline("dhvae_hh")
            with (
                patch(
                    "grapher.models.dhvae_hh.wrapper._run_training_subprocess",
                    side_effect=self._fake_training_subprocess,
                ),
                patch(
                    "grapher.models.dhvae_hh.wrapper._normalize_checkpoint_config"
                ),
                patch(
                    "grapher.models.dhvae_hh.wrapper._generate_batch",
                    side_effect=self._fake_generate_batch,
                ),
            ):
                training = wrapper.train(
                    TrainRequest(
                        run=run,
                        dataset=dataset,
                        config_path=config,
                        options={"training_estimates": {"enabled": False}},
                    )
                )
                generation = wrapper.generate(
                    GenerateRequest(
                        run=run,
                        checkpoint_path=training.checkpoint_path,
                        num_graphs=3,
                        generation_seed=29,
                    )
                )

            self.assertEqual(generation.num_requested, 3)
            self.assertEqual(generation.num_generated, 3)
            with generation.graphs_path.open("rb") as handle:
                graphs = pickle.load(handle)
            self.assertEqual([graph.graph["raw_index"] for graph in graphs], [0, 1, 2])
            self.assertEqual(
                [graph.graph["generation_seed"] for graph in graphs], [29, 29, 29]
            )
            manifest = json.loads(generation.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["graphs"]["requested"], 3)
            self.assertEqual(manifest["graphs"]["returned"], 3)
            self.assertEqual(manifest["graphs"]["dropped"], 0)
            self.assertTrue(manifest["checkpoint"]["managed_training_run"])

    def test_unknown_options_fail_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset, run, config, _ = self._fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "Unknown DH-VAE\\+HH"):
                create_baseline("dhvae_hh").train(
                    TrainRequest(
                        run=run,
                        dataset=dataset,
                        config_path=config,
                        options={"misspelled": True},
                    )
                )


if __name__ == "__main__":
    unittest.main()
