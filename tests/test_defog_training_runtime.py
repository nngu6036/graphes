from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from grapher.models.defog import DeFoGWrapper
from scripts import defog_train_worker


def _fake_lightning_module() -> tuple[ModuleType, type]:
    module = ModuleType("pytorch_lightning")

    class Trainer:
        init_calls: list[dict[str, object]] = []

        def __init__(self, *args, **kwargs) -> None:
            del args
            type(self).init_calls.append(dict(kwargs))

    module.Trainer = Trainer
    return module, Trainer


def test_single_gpu_training_rewrites_upstream_ddp_strategy(
    monkeypatch,
    capsys,
) -> None:
    lightning, trainer = _fake_lightning_module()
    monkeypatch.setitem(sys.modules, "pytorch_lightning", lightning)

    defog_train_worker._install_single_device_strategy()
    trainer(
        accelerator="gpu",
        devices=1,
        strategy="ddp_find_unused_parameters_true",
    )

    assert trainer.init_calls == [
        {
            "accelerator": "gpu",
            "devices": 1,
            "strategy": "auto",
        }
    ]
    assert trainer._grapher_single_device_strategy_patch is True
    output = capsys.readouterr().out
    assert "Disabled one-device DDP" in output
    assert "Effective Lightning runtime" in output


def test_single_device_strategy_patch_is_narrow_and_idempotent(
    monkeypatch,
) -> None:
    lightning, trainer = _fake_lightning_module()
    monkeypatch.setitem(sys.modules, "pytorch_lightning", lightning)

    defog_train_worker._install_single_device_strategy()
    first_patched_init = trainer.__init__
    defog_train_worker._install_single_device_strategy()

    assert trainer.__init__ is first_patched_init

    trainer(devices=2, strategy="ddp_find_unused_parameters_true")
    trainer(devices=1, strategy="auto")

    assert trainer.init_calls == [
        {"devices": 2, "strategy": "ddp_find_unused_parameters_true"},
        {"devices": 1, "strategy": "auto"},
    ]


def test_runtime_diagnostics_preserve_probe_failures_without_raising(
    monkeypatch,
) -> None:
    torch = ModuleType("torch")
    torch.__version__ = "2.4.0"
    torch.version = SimpleNamespace(cuda="11.8")

    class FailingCudaProbe:
        @staticmethod
        def is_available() -> bool:
            raise RuntimeError("NVML driver/library version mismatch")

    torch.cuda = FailingCudaProbe()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("GRAPHER_TEST_SECRET", "must-not-be-recorded")

    def fake_run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        assert kwargs["shell"] is False
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Failed to initialize NVML: Driver/library version mismatch",
        )

    monkeypatch.setattr(defog_train_worker.subprocess, "run", fake_run)

    diagnostics = defog_train_worker._collect_runtime_diagnostics("comm20", 1)

    assert diagnostics["format"] == "grapher_defog_runtime_diagnostics_v1"
    assert diagnostics["dataset"] == "comm20"
    assert diagnostics["requested_gpus"] == 1
    assert diagnostics["single_device_strategy_policy"] == "disable_ddp_use_auto"
    assert diagnostics["environment"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert "GRAPHER_TEST_SECRET" not in diagnostics["environment"]
    assert diagnostics["torch"]["version"] == "2.4.0"
    assert diagnostics["torch"]["compiled_cuda"] == "11.8"
    assert "driver/library version mismatch" in diagnostics["torch"][
        "cuda_probe_error"
    ].lower()
    assert diagnostics["torch"]["cuda_available"] is False
    assert diagnostics["nvidia_smi"]["returncode"] == 1
    assert "driver/library version mismatch" in diagnostics["nvidia_smi"][
        "stderr"
    ].lower()
    # The worker prints and optionally persists this object as JSON, so every
    # diagnostic value must remain serialization-safe.
    json.dumps(diagnostics)


def test_runtime_diagnostics_are_printed_and_atomically_persisted(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "diagnostics" / "runtime.json"
    monkeypatch.setenv("GRAPHER_DEFOG_DIAGNOSTICS_PATH", str(path))
    record = {
        "format": "grapher_defog_runtime_diagnostics_v1",
        "dataset": "comm20",
        "requested_gpus": 1,
    }

    defog_train_worker._publish_runtime_diagnostics(record)

    assert json.loads(path.read_text(encoding="utf-8")) == record
    assert not path.with_name(path.name + ".tmp").exists()
    output = capsys.readouterr().out
    assert "[GraphER/DeFoG] Runtime preflight:" in output
    assert "grapher_defog_runtime_diagnostics_v1" in output


def test_external_training_failure_surfaces_context_root_cause_and_action(
    monkeypatch,
    tmp_path: Path,
) -> None:
    wrapper = DeFoGWrapper()
    working_directory = tmp_path / "defog" / "src"
    working_directory.mkdir(parents=True)
    log_path = tmp_path / "artifacts" / "train.log"
    command = ["/opt/defog/bin/python", "train.py", "general.gpus=1"]
    root_cause = "nvmlInit_v2() failed: Driver/library version mismatch"

    def fake_run(argv, **kwargs):
        assert argv == command
        assert kwargs["shell"] is False
        kwargs["stdout"].write(root_cause + "\n")
        # The previous 60-line tail discarded the useful root cause in noisy
        # Hydra/Lightning failures. Keep enough trailing noise to regress that
        # behavior without producing a large test fixture.
        for index in range(75):
            kwargs["stdout"].write(f"secondary traceback line {index}\n")
        kwargs["stdout"].flush()
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr("grapher.models.defog.subprocess.run", fake_run)

    with pytest.raises(RuntimeError) as captured:
        wrapper._run_external(
            command,
            cwd=working_directory,
            environment={"GRAPHER_DEFOG_DATASET": "comm20"},
            log_path=log_path,
            timeout_seconds=None,
            label="DeFoG training",
        )

    message = str(captured.value)
    lower_message = message.lower()
    assert "DeFoG training" in message
    assert "exited with code 17" in message
    assert "cwd" in lower_message
    assert str(working_directory) in message
    assert "argv" in lower_message
    assert str(log_path) in message
    assert json.dumps(command) in message or all(part in message for part in command)
    assert root_cause in message
    assert "nvidia-smi" in lower_message
    assert "administrator" in lower_message or "reboot" in lower_message
    assert log_path.is_file()
