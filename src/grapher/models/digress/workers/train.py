#!/usr/bin/env python
"""Train DiGress on GraphER-managed PyG artifacts in an isolated process.

The worker builds the upstream model directly rather than importing DiGress's
``src/main.py``. The attached main entrypoint imports graph-tool
unconditionally, enables DDP for a single GPU, runs expensive sample metrics
inside validation, and invokes a final test-generation pass. None of those
behaviours is required to train a frozen baseline checkpoint.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from common import (
    atomic_json,
    build_components,
    compose_config,
    install_discrete_model_runtime_patches,
    install_upstream_runtime_patches,
    seed_everything,
    status,
)

FORMAT = "grapher_digress_training_worker_v1"


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one GraphER-managed DiGress baseline checkpoint."
    )
    parser.add_argument("--digress-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dataset-datadir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=_nonnegative_int, required=True)
    parser.add_argument("--gpus", type=_nonnegative_int, choices=(0, 1), default=1)
    parser.add_argument("--n-epochs", type=_positive_int, default=None)
    parser.add_argument("--batch-size", type=_positive_int, default=None)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=None)
    parser.add_argument(
        "--check-val-every-n-epochs", type=_positive_int, default=None
    )
    parser.add_argument("--save-every-n-epochs", type=_positive_int, default=None)
    parser.add_argument("--epoch-progress-interval", type=_positive_int, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser


def _float_metric(metric: Any) -> Optional[float]:
    try:
        total = getattr(metric, "total_samples", None)
        if total is not None and float(total.detach().cpu().item()) <= 0:
            return None
        value = metric.compute()
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        return float(value)
    except Exception:
        return None


def _json_metrics(values: Mapping[str, Any]) -> str:
    normalized: dict[str, float] = {}
    for key, value in values.items():
        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
            normalized[str(key)] = float(value)
        except Exception:
            continue
    return json.dumps(normalized, sort_keys=True)


def _build_model_class() -> type[Any]:
    """Create a bounded-output subclass of the attached Lightning module."""

    from diffusion_model_discrete import DiscreteDenoisingDiffusion

    class GraphERDiGressModel(DiscreteDenoisingDiffusion):
        def on_fit_start(self) -> None:
            self.train_iterations = len(self.trainer.datamodule.train_dataloader())
            status(
                "Training loop started: "
                f"max_epochs={self.trainer.max_epochs}, "
                f"batches_per_epoch={self.train_iterations}, "
                f"device={self.device}."
            )

        def on_train_epoch_start(self) -> None:
            self.start_epoch_time = time.time()
            self.train_loss.reset()
            self.train_metrics.reset()

        def on_train_epoch_end(self) -> None:
            completed = int(self.current_epoch) + 1
            total = int(self.trainer.max_epochs)
            interval = int(
                os.environ.get(
                    "GRAPHER_DIGRESS_EPOCH_PROGRESS_INTERVAL",
                    max(total // 100, 1),
                )
            )
            should_report = completed == 1 or completed == total or completed % interval == 0
            if should_report:
                x_ce = _float_metric(self.train_loss.node_loss)
                e_ce = _float_metric(self.train_loss.edge_loss)
                elapsed = time.time() - float(self.start_epoch_time or time.time())
                metrics = {
                    "train_x_ce": x_ce,
                    "train_e_ce": e_ce,
                }
                compact = {
                    key: value for key, value in metrics.items() if value is not None
                }
                status(
                    "Training progress: "
                    f"epoch={completed}/{total}, global_step={self.global_step}, "
                    f"epoch_seconds={elapsed:.2f}, metrics={json.dumps(compact, sort_keys=True)}."
                )

        def on_validation_epoch_start(self) -> None:
            self.val_nll.reset()
            self.val_X_kl.reset()
            self.val_E_kl.reset()
            self.val_X_logp.reset()
            self.val_E_logp.reset()
            self.sampling_metrics.reset()

        def on_validation_epoch_end(self) -> None:
            values = {
                "val_nll": self.val_nll.compute(),
                "val_x_kl": self.val_X_kl.compute() * self.T,
                "val_e_kl": self.val_E_kl.compute() * self.T,
                "val_x_logp": self.val_X_logp.compute(),
                "val_e_logp": self.val_E_logp.compute(),
            }
            val_nll = values["val_nll"]
            try:
                numeric = float(val_nll.detach().cpu().item())
                self.best_val_nll = min(float(self.best_val_nll), numeric)
            except Exception:
                pass
            status(
                f"Validation completed at epoch={int(self.current_epoch) + 1}: "
                f"{_json_metrics(values)}."
            )
            # Deliberately do not call sample_batch here. The upstream hook
            # couples likelihood validation to expensive generation metrics.

        def on_test_epoch_end(self) -> None:
            # The GraphER worker never calls trainer.test(). Retain this guard
            # so an accidental call cannot trigger 10,000-sample generation.
            status("Test epoch completed; final sampling is managed separately.")

    return GraphERDiGressModel


def _trainer_kwargs(cfg: Any, *, output_dir: Path, callbacks: Sequence[Any]) -> dict[str, Any]:
    import torch
    from pytorch_lightning import Trainer

    use_gpu = int(cfg.general.gpus) > 0 and torch.cuda.is_available()
    parameters = inspect.signature(Trainer.__init__).parameters
    kwargs: dict[str, Any] = {
        "max_epochs": int(cfg.train.n_epochs),
        "accelerator": "gpu" if use_gpu else "cpu",
        "devices": 1,
        "callbacks": list(callbacks),
        "logger": False,
        "enable_progress_bar": False,
        "log_every_n_steps": max(int(cfg.general.log_every_steps), 1),
        "default_root_dir": str(output_dir),
        "deterministic": True,
        "num_sanity_val_steps": 0,
    }
    if cfg.train.clip_grad is not None:
        kwargs["gradient_clip_val"] = float(cfg.train.clip_grad)
    if "strategy" in parameters:
        kwargs["strategy"] = "auto"
    if "check_val_every_n_epoch" in parameters:
        kwargs["check_val_every_n_epoch"] = int(
            cfg.general.check_val_every_n_epochs
        )
    if "enable_model_summary" in parameters:
        kwargs["enable_model_summary"] = False
    return kwargs


def _periodic_checkpoint_callback(
    *, output_dir: Path, every_n_epochs: Optional[int]
) -> Any:
    import pytorch_lightning as pl

    class PeriodicCheckpoint(pl.Callback):
        def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            del pl_module
            if every_n_epochs is None:
                return
            completed = int(trainer.current_epoch) + 1
            if completed % int(every_n_epochs) != 0:
                return
            path = output_dir / "checkpoints" / f"epoch={completed - 1}.ckpt"
            path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(path))
            status(f"Periodic checkpoint saved: epoch={completed}, path={path}.")

    return PeriodicCheckpoint()


def main() -> None:
    args = _parser().parse_args()
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    root = args.digress_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_from = (
        None if args.resume_from is None else args.resume_from.expanduser().resolve()
    )
    if resume_from is not None and not resume_from.is_file():
        raise FileNotFoundError(f"Missing resume checkpoint: {resume_from}")

    seed_everything(args.seed)
    cfg = compose_config(
        digress_root=root,
        dataset=args.dataset,
        experiment=args.experiment,
        dataset_datadir=args.dataset_datadir,
        run_name=args.run_name,
        seed=args.seed,
        gpus=args.gpus,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        check_val_every_n_epochs=args.check_val_every_n_epochs,
        extra_overrides=args.override,
    )
    if int(cfg.model.type == "discrete") != 1:
        raise ValueError("DiGressWrapper supports only the discrete model.")
    if args.epoch_progress_interval is not None:
        os.environ["GRAPHER_DIGRESS_EPOCH_PROGRESS_INTERVAL"] = str(
            args.epoch_progress_interval
        )

    install_upstream_runtime_patches()
    status(
        "Resolved training configuration: "
        f"dataset={cfg.dataset.name}, experiment={args.experiment}, "
        f"epochs={cfg.train.n_epochs}, batch_size={cfg.train.batch_size}, "
        f"diffusion_steps={cfg.model.diffusion_steps}, gpus={cfg.general.gpus}."
    )
    datamodule, model_kwargs, molecular_statistics = build_components(cfg)
    model_class = _build_model_class()
    install_discrete_model_runtime_patches(model_class)
    model = model_class(cfg=cfg, **model_kwargs)

    callbacks = [
        _periodic_checkpoint_callback(
            output_dir=output_dir,
            every_n_epochs=args.save_every_n_epochs,
        )
    ]
    if float(getattr(cfg.train, "ema_decay", 0.0)) > 0:
        from src import utils

        callbacks.append(utils.EMA(decay=float(cfg.train.ema_decay)))

    from omegaconf import OmegaConf
    from pytorch_lightning import Trainer
    import torch
    import torch_geometric
    import pytorch_lightning as pl

    resolved_config = output_dir / "resolved_config.yaml"
    OmegaConf.save(config=cfg, f=str(resolved_config), resolve=True)
    if molecular_statistics is not None:
        atomic_json(output_dir / "molecular_statistics.json", molecular_statistics)

    trainer = Trainer(**_trainer_kwargs(cfg, output_dir=output_dir, callbacks=callbacks))
    status(
        "Effective Lightning runtime: "
        f"accelerator={type(trainer.accelerator).__name__}, "
        f"strategy={type(trainer.strategy).__name__}, devices=1."
    )
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=(str(resume_from) if resume_from is not None else None),
    )
    final_checkpoint = checkpoint_dir / "model.ckpt"
    trainer.save_checkpoint(str(final_checkpoint))
    status(f"Final checkpoint saved: {final_checkpoint}.")

    finished_at = datetime.now(timezone.utc)
    manifest = {
        "format": FORMAT,
        "status": "complete",
        "dataset": str(cfg.dataset.name),
        "experiment": str(args.experiment),
        "run_name": str(args.run_name),
        "seed": int(args.seed),
        "configured_n_epochs": int(cfg.train.n_epochs),
        "completed_epochs": int(cfg.train.n_epochs),
        "global_step": int(trainer.global_step),
        "batch_size": int(cfg.train.batch_size),
        "diffusion_steps": int(cfg.model.diffusion_steps),
        "checkpoint": str(final_checkpoint),
        "resolved_config": str(resolved_config),
        "molecular_statistics": (
            str(output_dir / "molecular_statistics.json")
            if molecular_statistics is not None
            else None
        ),
        "resume_from": str(resume_from) if resume_from is not None else None,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": time.monotonic() - started,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_geometric": str(torch_geometric.__version__),
            "pytorch_lightning": str(pl.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "device": (
                str(torch.cuda.get_device_name(0))
                if torch.cuda.is_available() and torch.cuda.device_count() > 0
                else "cpu"
            ),
        },
    }
    atomic_json(args.manifest.expanduser().resolve(), manifest)
    status(
        "Training worker completed: "
        f"epochs={manifest['completed_epochs']}, global_step={manifest['global_step']}."
    )


if __name__ == "__main__":
    main()
