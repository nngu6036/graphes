"""Exercise the evaluator arguments emitted by the generic ablation runner."""
from __future__ import annotations

import importlib.util
import io
import json
import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ablation_graph_evaluator", REPOSITORY / "scripts/evaluate_graph_generation_report.py"
)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


class AblationEvaluatorArgumentTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="ablation-evaluator-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset_root = self.root / "prepared splits"
        dataset = self.dataset_root / "synthetic"
        dataset.mkdir(parents=True)
        for split in ("train", "val", "test"):
            (dataset / f"{split}.pkl").write_bytes(
                pickle.dumps([nx.path_graph(3)], protocol=pickle.HIGHEST_PROTOCOL)
            )
        self.generated = self.root / "generation" / "ablations" / "DR"
        self.generated.mkdir(parents=True)
        self.graphs = [
            nx.path_graph(1), nx.path_graph(2),
            nx.disjoint_union(nx.path_graph(2), nx.path_graph(2)), nx.cycle_graph(4),
        ]
        (self.generated / "base_graphs.pkl").write_bytes(
            pickle.dumps(self.graphs, protocol=pickle.HIGHEST_PROTOCOL)
        )
        self.config = self.root / "evaluation.yaml"
        self.options = {
            "dataset": {
                "name": "synthetic", "benchmark": "synthetic",
                "root": str(self.root / "deliberately missing configured root"),
            },
            "evaluation": {
                "compute_orbit": False,
                "graphlet_k_min": 3, "graphlet_k_max": 3,
                "graphlet_backend": "sampled", "graphlet_num_samples": 16,
            },
        }
        self.write_config()

    def write_config(self) -> None:
        self.config.write_text(yaml.safe_dump(self.options), encoding="utf-8")

    def run_evaluation(self, *extra: str, override_root: bool = True) -> dict:
        output = self.root / "report"
        arguments = [
            "evaluate_graph_generation_report.py", "--config", str(self.config),
            "--generated-dir", str(self.generated),
            "--generated-graphs", str(self.generated / "base_graphs.pkl"),
            "--generated-stage", "DR", "--output-dir", str(output),
            "--generic-mmd-protocol", "graphrnn",
            "--skip-graded-diversity-diagnostics", *extra,
        ]
        if override_root:
            arguments.extend(["--dataset-root", str(self.dataset_root)])
        with patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
            evaluator.main()
        return json.loads((output / "graph_evaluation_report.json").read_text(encoding="utf-8"))

    def test_ablation_flags_apply_to_loading_provenance_and_validity(self) -> None:
        with (
            patch.object(evaluator, "validate_prepared_dataset_contract",
                         wraps=evaluator.validate_prepared_dataset_contract) as validate,
            patch.object(evaluator, "load_dataset_splits",
                         wraps=evaluator.load_dataset_splits) as load,
        ):
            result = self.run_evaluation(
                "--generic-validity-require-connected",
                "--generic-validity-min-nodes", "2",
                "--generic-validity-max-nodes", "3",
            )
        self.assertEqual(validate.call_args.kwargs["root"], str(self.dataset_root))
        self.assertEqual(load.call_args.kwargs["root"], str(self.dataset_root))
        self.assertEqual(result["dataset_provenance"]["root"], str(self.dataset_root))
        protocol = result["generic_vun_protocol"]["validity_contract"]
        self.assertTrue(protocol["require_connected"])
        self.assertEqual(protocol["min_nodes"], 2)
        self.assertEqual(protocol["max_nodes"], 3)
        self.assertEqual(protocol["cli_overrides"], {
            "require_connected": True, "min_nodes": 2, "max_nodes": 3,
        })
        population = next(row for row in result["generic_vun_metrics"] if row["population"] == "dr")
        self.assertEqual(population["validity_rate"], 0.25)
        final = next(row for row in result["metrics"] if row["comparison"] == "dr_to_test")
        self.assertEqual(final["validity_rate"], 0.25)

    def test_omitted_overrides_preserve_configured_root_and_validity_defaults(self) -> None:
        self.options["dataset"]["root"] = str(self.dataset_root)
        self.write_config()
        result = self.run_evaluation(override_root=False)
        protocol = result["generic_vun_protocol"]["validity_contract"]
        self.assertEqual(protocol["mode"], "prepared_dataset_contract")
        self.assertFalse(protocol["require_connected"])
        self.assertNotIn("min_nodes", protocol)
        self.assertNotIn("max_nodes", protocol)
        self.assertNotIn("cli_overrides", protocol)
        population = next(row for row in result["generic_vun_metrics"] if row["population"] == "dr")
        self.assertEqual(population["validity_rate"], 1.0)

    def test_invalid_bounds_fail_before_reading_config_or_generating_reports(self) -> None:
        invalid_options = [
            ["--generic-validity-min-nodes", "0"],
            ["--generic-validity-min-nodes", "-1"],
            ["--generic-validity-max-nodes", "0"],
            ["--generic-validity-max-nodes", "-1"],
            ["--generic-validity-min-nodes", "4", "--generic-validity-max-nodes", "3"],
        ]
        for options in invalid_options:
            with self.subTest(options=options):
                with (
                    patch.object(evaluator, "load_yaml") as load,
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as failure,
                ):
                    self.run_evaluation(*options)
                self.assertEqual(failure.exception.code, 2)
                load.assert_not_called()
                self.assertFalse((self.root / "report").exists())

    def test_existing_predicate_callers_keep_optional_bound_defaults(self) -> None:
        self.assertTrue(evaluator._generic_graph_is_valid(
            nx.path_graph(1), require_connected=False, reject_zero_degree=False,
        ))
        self.assertFalse(evaluator._generic_graph_is_valid(
            nx.path_graph(4), require_connected=False, reject_zero_degree=False,
            max_nodes=3,
        ))
        self.assertFalse(evaluator._generic_graph_is_valid(
            nx.path_graph(1), require_connected=False, reject_zero_degree=False,
            min_nodes=2,
        ))


if __name__ == "__main__":
    unittest.main()
