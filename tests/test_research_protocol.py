from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_research_protocol.py"
    spec = importlib.util.spec_from_file_location("run_research_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_dry_run_is_default_and_materializes_all_fixed_seed_variants(
    tmp_path: Path,
) -> None:
    protocol_path = tmp_path / "protocol.yaml"
    output_dir = tmp_path / "protocol_output"
    _write_yaml(
        protocol_path,
        {
            "format": runner.PROTOCOL_FORMAT,
            "name": "small-protocol",
            "seeds": [42, 43, 44],
            "ablations": [
                {
                    "name": "energy_only",
                    "argv": [
                        "python",
                        "run.py",
                        "--seed",
                        "{seed}",
                        "--variant",
                        "{variant}",
                    ],
                    "config": {"seed": "{seed}", "mode": "{variant}"},
                    "report": "reports/{variant}/{seed}.json",
                }
            ],
            "cost_sweeps": [
                {
                    "name": "budget_32",
                    "argv": ["python", "run.py", "--seed={seed}"],
                    "parameters": {"candidate_budget": 32},
                    "report": "reports/{variant}/{seed}.json",
                }
            ],
        },
    )

    manifest, report = runner.run_protocol(protocol_path, output_dir=output_dir)

    assert manifest["mode"] == "dry_run"
    assert manifest["status_counts"] == {"planned": 6}
    assert {run["seed"] for run in manifest["runs"]} == {42, 43, 44}
    energy_42 = next(
        run
        for run in manifest["runs"]
        if run["variant"] == "energy_only" and run["seed"] == 42
    )
    assert energy_42["argv"][-3:] == ["42", "--variant", "energy_only"]
    assert energy_42["config"]["content"] == {"mode": "energy_only", "seed": "42"}
    assert Path(energy_42["config"]["path"]).is_file()
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "report.json").is_file()
    assert report["success"] is True


@pytest.mark.parametrize(
    "bad_argv",
    ["python run.py --seed {seed}", ["bash", "-c", "python run.py"]],
)
def test_runner_refuses_shell_strings_and_shell_command_adapters(
    tmp_path: Path, bad_argv
) -> None:
    protocol_path = tmp_path / "bad.yaml"
    _write_yaml(
        protocol_path,
        {
            "seeds": [42, 43, 44],
            "ablations": [{"name": "bad", "argv": bad_argv}],
        },
    )
    protocol = runner.load_protocol(protocol_path)

    with pytest.raises((TypeError, ValueError), match="shell"):
        runner.materialize_protocol(
            protocol,
            protocol_path=protocol_path,
            output_dir=tmp_path / "out",
        )


def test_protocol_requires_exact_ordered_three_seeds(tmp_path: Path) -> None:
    protocol_path = tmp_path / "bad_seeds.yaml"
    _write_yaml(
        protocol_path,
        {
            "seeds": [44, 43, 42],
            "ablations": [{"name": "x", "argv": ["true"]}],
        },
    )

    with pytest.raises(ValueError, match="requires seeds"):
        runner.load_protocol(protocol_path)


def test_missing_external_adapter_calls_explicit_placeholder_and_logs(
    tmp_path: Path,
) -> None:
    protocol_path = tmp_path / "external.yaml"
    output_dir = tmp_path / "out"
    _write_yaml(
        protocol_path,
        {
            "seeds": [42, 43, 44],
            "external_baselines": [{"name": "unpublished_baseline"}],
        },
    )

    with pytest.raises(NotImplementedError, match="argv adapter"):
        runner.external_baseline_adapter_placeholder("unpublished_baseline")
    manifest, report = runner.run_protocol(protocol_path, output_dir=output_dir)

    assert manifest["status_counts"] == {"not_implemented": 3}
    assert all(Path(run["failure_log"]).is_file() for run in manifest["runs"])
    assert report["success"] is False
    assert len(report["failures"]) == 3


def _write_report_writer(path: Path) -> None:
    path.write_text(
        """\
import json
import sys
from pathlib import Path

seed = int(sys.argv[1])
variant = sys.argv[2]
output = Path(sys.argv[3])
output.parent.mkdir(parents=True, exist_ok=True)
values = {
    "baseline": (0.50, 2.0),
    "treatment": (0.30, 2.5),
    "cheap": (0.45, 1.0),
    "quality": (0.20, 5.0),
}
mmd, runtime = values[variant]
pipeline = {
    "predictor_nll": 1.0,
    "predictor_macro_f1": 0.8,
    "graphlet_error": mmd,
    "consistency_residual": 0.1,
    "invariant_feasible": True,
    "constructor_success": True,
    "candidate_proposals": 10,
    "candidate_passes": 5,
    "accepted_swaps": 2,
    "stopped": 1,
    "stop_opportunities": 1,
    "rejection_reasons": {"locality": 5},
    "runtime_seconds": runtime,
    "generation_attempts": 1,
    "generation_successes": 1,
    "fallback_used": False,
}
report = {
    "metrics": {"mmd": mmd + (seed - 43) * 0.01, "runtime_seconds": runtime},
    "pipeline_diagnostics": [pipeline],
    "ablation_records": [{
        "sample_id": "sample",
        "invariant_id": "invariant",
        "initial_graph_id": "source",
        "mmd": mmd,
    }],
    "stage_metrics": {
        "constructor": {"mmd": mmd + 0.1},
        "refined": {"mmd": mmd},
    },
}
output.write_text(json.dumps(report), encoding="utf-8")
""",
        encoding="utf-8",
    )


def test_execute_captures_runs_and_aggregates_ablation_pipeline_and_pareto(
    tmp_path: Path,
) -> None:
    writer = tmp_path / "writer.py"
    _write_report_writer(writer)
    protocol_path = tmp_path / "protocol.yaml"
    output_dir = tmp_path / "out"

    def variant(name: str, *, baseline: bool = False, runtime: float | None = None):
        entry = {
            "name": name,
            "argv": [
                sys.executable,
                str(writer),
                "{seed}",
                "{variant}",
                str(tmp_path / "reports" / "{variant}" / "{seed}.json"),
            ],
            "report": str(tmp_path / "reports" / "{variant}" / "{seed}.json"),
        }
        if baseline:
            entry["baseline"] = True
        if runtime is not None:
            entry["parameters"] = {"declared_runtime": runtime}
        return entry

    _write_yaml(
        protocol_path,
        {
            "format": runner.PROTOCOL_FORMAT,
            "seeds": [42, 43, 44],
            "ablations": [
                variant("baseline", baseline=True),
                variant("treatment"),
            ],
            "cost_sweeps": [
                variant("cheap", runtime=1.0),
                variant("quality", runtime=5.0),
            ],
            "ablation_comparison": {"metrics": ["mmd"]},
            "quality_cost": {
                "quality_keys": "mmd",
                "cost_keys": "runtime_seconds",
            },
        },
    )

    manifest, report = runner.run_protocol(
        protocol_path,
        output_dir=output_dir,
        execute=True,
    )

    assert manifest["status_counts"] == {"completed": 12}
    assert all(run["return_code"] == 0 for run in manifest["runs"])
    assert all(run["runtime_seconds"] > 0 for run in manifest["runs"])
    baseline = report["variant_aggregates"]["ablation.baseline"]
    assert baseline["metrics"]["aggregate"]["mmd"]["mean"] == pytest.approx(0.5)
    assert baseline["pipeline_diagnostics"]["metrics"]["candidate_pass_rate"] == 0.5
    assert report["ablation_comparisons"]["treatment"]["metrics"]["mmd"][
        "mean_improvement"
    ] == pytest.approx(0.2)
    assert report["quality_cost"]["frontier_ids"] == ["cheap", "quality"]
    assert set(report["stage_decompositions"]["ablation.baseline"]) == {
        "42",
        "43",
        "44",
    }
    assert report["success"] is True


def test_nonzero_return_code_records_failure_log(tmp_path: Path) -> None:
    protocol_path = tmp_path / "failure.yaml"
    output_dir = tmp_path / "out"
    _write_yaml(
        protocol_path,
        {
            "seeds": [42, 43, 44],
            "ablations": [
                {
                    "name": "failure",
                    "argv": [sys.executable, "-c", "raise SystemExit(7)"],
                }
            ],
        },
    )

    manifest, report = runner.run_protocol(
        protocol_path,
        output_dir=output_dir,
        execute=True,
    )

    assert manifest["status_counts"] == {"failed": 3}
    assert all(run["return_code"] == 7 for run in manifest["runs"])
    assert all(Path(run["failure_log"]).is_file() for run in manifest["runs"])
    assert report["success"] is False
