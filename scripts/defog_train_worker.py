#!/usr/bin/env python
"""Launch upstream DeFoG training with GraphER's artifact safeguards."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNTIME_DIAGNOSTICS_FORMAT = "grapher_defog_runtime_diagnostics_v1"


def _device_count(value: Any) -> int | None:
    """Return a concrete Lightning device count when one is declared."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _install_single_device_strategy() -> None:
    """Prevent upstream DeFoG from launching DDP for one local device.

    The reference DeFoG entrypoint hard-codes
    ``ddp_find_unused_parameters_true`` even though its configuration states
    that multi-GPU execution is unsupported.  Lightning consequently creates
    a distributed process group for a one-GPU run and makes training depend on
    NCCL and NVML.  A single device needs neither.  This process-local patch
    leaves multi-device declarations untouched and records every replacement
    in the subprocess log.
    """

    from pytorch_lightning import Trainer

    if getattr(Trainer, "_grapher_single_device_strategy_patch", False):
        return
    original_init = Trainer.__init__

    def init_with_single_device_strategy(self, *args, **kwargs):
        strategy = kwargs.get("strategy")
        devices = kwargs.get("devices")
        count = _device_count(devices)
        replaced = False
        if (
            count == 1
            and isinstance(strategy, str)
            and strategy.lower().startswith("ddp")
        ):
            kwargs["strategy"] = "auto"
            replaced = True
            print(
                "[GraphER/DeFoG] Disabled one-device DDP: "
                f"strategy={strategy!r} -> 'auto', devices={devices!r}.",
                flush=True,
            )
        result = original_init(self, *args, **kwargs)
        if replaced:
            effective_strategy = type(getattr(self, "strategy", None)).__name__
            effective_accelerator = type(
                getattr(self, "accelerator", None)
            ).__name__
            print(
                "[GraphER/DeFoG] Effective Lightning runtime: "
                f"strategy={effective_strategy}, "
                f"accelerator={effective_accelerator}.",
                flush=True,
            )
        return result

    Trainer.__init__ = init_with_single_device_strategy
    Trainer._grapher_single_device_strategy_patch = True


def _capture_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "argv": command,
            "resolved_executable": shutil.which(command[0]),
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "argv": command,
        "resolved_executable": shutil.which(command[0]),
        "available": True,
        "status": "ok" if completed.returncode == 0 else "error",
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _read_optional_text(path: Path) -> dict[str, Any]:
    try:
        return {
            "path": str(path),
            "available": True,
            "text": path.read_text(encoding="utf-8", errors="replace").strip(),
        }
    except OSError as exc:
        return {
            "path": str(path),
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _collect_runtime_diagnostics(
    dataset: str,
    requested_gpus: int,
) -> dict[str, Any]:
    """Collect a bounded, non-secret preflight record for failed GPU runs."""

    record: dict[str, Any] = {
        "format": RUNTIME_DIAGNOSTICS_FORMAT,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "requested_gpus": requested_gpus,
        "single_device_strategy_policy": "disable_ddp_use_auto",
        "process": {
            "pid": os.getpid(),
            "cwd": str(Path.cwd()),
            "argv": list(sys.argv),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "platform": platform.platform(),
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
                "CUDA_HOME",
                "NVIDIA_VISIBLE_DEVICES",
                "NVIDIA_DRIVER_CAPABILITIES",
                "CONDA_DEFAULT_ENV",
                "CONDA_PREFIX",
                "VIRTUAL_ENV",
                "PYTHONPATH",
                "PATH",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "NCCL_DEBUG",
                "NCCL_DEBUG_SUBSYS",
                "NCCL_P2P_DISABLE",
                "NCCL_IB_DISABLE",
                "NCCL_SOCKET_IFNAME",
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
                "TORCH_DISTRIBUTED_DEBUG",
            )
        },
        "nvidia_smi": _capture_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "nvidia_kernel_module": _read_optional_text(
            Path("/proc/driver/nvidia/version")
        ),
    }
    try:
        import torch

        torch_record: dict[str, Any] = {
            "import_ok": True,
            "version": str(torch.__version__),
            "compiled_cuda": str(torch.version.cuda),
        }
        try:
            torch_record["cudnn_version"] = (
                int(torch.backends.cudnn.version())
                if torch.backends.cudnn.is_available()
                else None
            )
        except Exception as exc:
            torch_record["cudnn_probe_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
        try:
            torch_record["distributed_available"] = bool(
                torch.distributed.is_available()
            )
        except Exception as exc:
            torch_record["distributed_probe_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
        try:
            torch_record["cuda_available"] = bool(torch.cuda.is_available())
            torch_record["cuda_device_count"] = int(torch.cuda.device_count())
            torch_record["cuda_current_device"] = (
                int(torch.cuda.current_device())
                if torch_record["cuda_available"]
                and torch_record["cuda_device_count"] > 0
                else None
            )
            torch_record["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": int(
                        torch.cuda.get_device_properties(index).total_memory
                    ),
                }
                for index in range(torch_record["cuda_device_count"])
            ]
        except Exception as exc:  # diagnostic collection must remain available
            torch_record["cuda_probe_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            torch_record.setdefault("cuda_available", False)
            torch_record.setdefault("cuda_device_count", 0)
        try:
            nccl_available = bool(torch.distributed.is_nccl_available())
            nccl_version = (
                torch.cuda.nccl.version() if nccl_available else None
            )
            if isinstance(nccl_version, tuple):
                nccl_version = list(nccl_version)
            torch_record["nccl"] = {
                "required_for_this_run": False,
                "available": nccl_available,
                "version": nccl_version,
            }
        except Exception as exc:
            torch_record["nccl"] = {
                "required_for_this_run": False,
                "probe_error": f"{type(exc).__name__}: {exc}",
            }
        record["torch"] = torch_record
    except Exception as exc:
        record["torch"] = {
            "import_error": f"{type(exc).__name__}: {exc}",
            "cuda_available": False,
            "cuda_device_count": 0,
        }
    return record


def _publish_runtime_diagnostics(record: dict[str, Any]) -> None:
    print(
        "[GraphER/DeFoG] Runtime preflight:\n"
        + json.dumps(record, indent=2, sort_keys=True),
        flush=True,
    )
    target_value = os.environ.get("GRAPHER_DEFOG_DIAGNOSTICS_PATH", "").strip()
    if not target_value:
        return
    target = Path(target_value).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _requested_gpus() -> int:
    raw = os.environ.get("GRAPHER_DEFOG_REQUESTED_GPUS", "1")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "GRAPHER_DEFOG_REQUESTED_GPUS must be the integer 0 or 1; "
            f"received {raw!r}."
        ) from exc
    if value not in {0, 1}:
        raise ValueError(
            "GRAPHER_DEFOG_REQUESTED_GPUS must be 0 or 1; "
            f"received {value}."
        )
    return value


def _progress_enabled() -> bool:
    return os.environ.get("GRAPHER_DEFOG_PROGRESS_ENABLED", "0") == "1"


def _epoch_progress_interval(max_epochs: int) -> int:
    """Return an explicit or bounded automatic epoch-reporting interval."""

    raw = os.environ.get("GRAPHER_DEFOG_EPOCH_PROGRESS_INTERVAL", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                "GRAPHER_DEFOG_EPOCH_PROGRESS_INTERVAL must be a positive "
                f"integer; received {raw!r}."
            ) from exc
        if value <= 0:
            raise ValueError(
                "GRAPHER_DEFOG_EPOCH_PROGRESS_INTERVAL must be positive; "
                f"received {value}."
            )
        return value
    # DeFoG experiments can use very large epoch counts.  About one hundred
    # stable progress lines is informative without turning train.log into a
    # second per-batch trace.
    return max(int(max_epochs) // 100, 1)


def _scalar_callback_metrics(trainer: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    raw_metrics = getattr(trainer, "callback_metrics", {})
    if not isinstance(raw_metrics, Mapping):
        return values
    for key, raw_value in raw_metrics.items():
        if len(values) >= 8:
            break
        try:
            if hasattr(raw_value, "detach"):
                raw_value = raw_value.detach()
            if hasattr(raw_value, "numel") and int(raw_value.numel()) != 1:
                continue
            if hasattr(raw_value, "item"):
                raw_value = raw_value.item()
            value = float(raw_value)
        except (TypeError, ValueError, RuntimeError):
            continue
        if value == value and abs(value) != float("inf"):
            values[str(key)] = value
    return values


def _install_final_checkpoint_policy() -> None:
    """Save one explicit checkpoint after a successful upstream ``fit``.

    DeFoG ties periodic checkpointing to validation cadence, so an arbitrary
    training horizon can end between saved epochs.  Patching ``Trainer.fit``
    leaves that upstream schedule untouched and writes exactly one additional
    ``grapher_final.ckpt`` after the final optimizer epoch.
    """

    from pathlib import Path

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint

    if getattr(Trainer, "_grapher_final_checkpoint_patch", False):
        return
    original_fit = Trainer.fit

    class GraphERProgressCallback(Callback):
        """Emit stable epoch-level progress independently of Lightning bars."""

        _grapher_defog_progress_callback = True

        def __init__(self, max_epochs: int) -> None:
            super().__init__()
            self.max_epochs = max(int(max_epochs), 1)
            self.interval = _epoch_progress_interval(self.max_epochs)

        def on_fit_start(self, trainer, pl_module) -> None:
            del pl_module
            estimated_steps = getattr(trainer, "estimated_stepping_batches", None)
            print(
                "[GraphER/DeFoG] Training loop started: "
                f"max_epochs={self.max_epochs}, "
                f"epoch_progress_interval={self.interval}, "
                f"estimated_stepping_batches={estimated_steps}.",
                flush=True,
            )

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            del pl_module
            epoch = int(getattr(trainer, "current_epoch", 0)) + 1
            if not (
                epoch == 1
                or epoch % self.interval == 0
                or epoch >= self.max_epochs
            ):
                return
            metrics = _scalar_callback_metrics(trainer)
            metric_text = (
                " metrics=" + json.dumps(metrics, sort_keys=True)
                if metrics
                else ""
            )
            print(
                "[GraphER/DeFoG] Training progress: "
                f"epoch={epoch}/{self.max_epochs}, "
                f"global_step={int(getattr(trainer, 'global_step', 0))}."
                f"{metric_text}",
                flush=True,
            )

        def on_fit_end(self, trainer, pl_module) -> None:
            del pl_module
            print(
                "[GraphER/DeFoG] Training loop finished: "
                f"completed_epochs="
                f"{int(trainer.fit_loop.epoch_progress.current.completed)}, "
                f"global_step={int(getattr(trainer, 'global_step', 0))}.",
                flush=True,
            )

    def fit_and_save_final(self, *args, **kwargs):
        if _progress_enabled() and not any(
            getattr(callback, "_grapher_defog_progress_callback", False)
            for callback in self.callbacks
        ):
            self.callbacks.append(GraphERProgressCallback(int(self.max_epochs)))
        result = original_fit(self, *args, **kwargs)
        checkpoint_callbacks = [
            callback
            for callback in self.callbacks
            if isinstance(callback, ModelCheckpoint)
        ]
        if len(checkpoint_callbacks) != 1:
            raise RuntimeError(
                "GraphER expected exactly one DeFoG ModelCheckpoint callback; "
                f"observed {len(checkpoint_callbacks)}."
            )
        directory = Path(checkpoint_callbacks[0].dirpath)
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = directory / "grapher_final.ckpt"
        completed_epochs = int(self.fit_loop.epoch_progress.current.completed)
        configured_epochs = int(self.max_epochs)
        if completed_epochs != configured_epochs:
            raise RuntimeError(
                "DeFoG fit returned before the configured horizon: "
                f"completed {completed_epochs} epochs, expected "
                f"{configured_epochs}."
            )
        self.save_checkpoint(str(checkpoint_path))
        record = {
            "format": "grapher_defog_final_checkpoint_v1",
            "checkpoint": checkpoint_path.name,
            "completed_epochs": completed_epochs,
            "configured_epochs": configured_epochs,
            "selected_epoch": completed_epochs - 1,
            "global_step": int(self.global_step),
        }
        record_path = checkpoint_path.with_suffix(".json")
        temporary = record_path.with_suffix(record_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(record_path)
        return result

    Trainer.fit = fit_and_save_final
    Trainer._grapher_final_checkpoint_patch = True


def main() -> None:
    dataset = os.environ.get("GRAPHER_DEFOG_DATASET", "").strip().lower()
    supported = {"comm20", "planar", "sbm", "tree", "qm9", "zinc"}
    if dataset not in supported:
        raise OSError(
            "GRAPHER_DEFOG_DATASET must name a supported DeFoG dataset; "
            f"received {dataset!r}."
        )
    requested_gpus = _requested_gpus()
    if _progress_enabled():
        print(
            "[GraphER/DeFoG] Training worker initializing: "
            f"dataset={dataset}, requested_gpus={requested_gpus}, "
            f"python={sys.executable}.",
            flush=True,
        )
    diagnostics = _collect_runtime_diagnostics(dataset, requested_gpus)
    _publish_runtime_diagnostics(diagnostics)
    torch_record = diagnostics.get("torch", {})
    if requested_gpus == 1 and not bool(torch_record.get("cuda_available")):
        print(
            "[GraphER/DeFoG] WARNING: one GPU was requested, but the isolated "
            "interpreter reported torch.cuda.is_available() == False. The "
            "upstream DeFoG entrypoint will use its documented CPU fallback. "
            "Review the runtime diagnostics if this was unexpected.",
            flush=True,
        )
    if dataset in {"qm9", "zinc"}:
        from defog_molecular_runtime import install_dataset_info_patch

        install_dataset_info_patch(dataset)
    _install_single_device_strategy()
    _install_final_checkpoint_policy()
    from main import main as upstream_main

    # Hydra consumes the original command line unchanged.
    if _progress_enabled():
        print(
            "[GraphER/DeFoG] Entering the upstream Hydra/Lightning training "
            "entrypoint.",
            flush=True,
        )
    upstream_main()


if __name__ == "__main__":
    main()
