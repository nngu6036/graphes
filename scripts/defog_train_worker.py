#!/usr/bin/env python
"""Launch upstream DeFoG training with GraphER's artifact safeguards."""

from __future__ import annotations

import os


def _install_final_checkpoint_policy() -> None:
    """Save one explicit checkpoint after a successful upstream ``fit``.

    DeFoG ties periodic checkpointing to validation cadence, so an arbitrary
    training horizon can end between saved epochs.  Patching ``Trainer.fit``
    leaves that upstream schedule untouched and writes exactly one additional
    ``grapher_final.ckpt`` after the final optimizer epoch.
    """

    from pathlib import Path

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    if getattr(Trainer, "_grapher_final_checkpoint_patch", False):
        return
    original_fit = Trainer.fit

    def fit_and_save_final(self, *args, **kwargs):
        import json

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
    if dataset in {"qm9", "zinc"}:
        from defog_molecular_runtime import install_dataset_info_patch

        install_dataset_info_patch(dataset)
    _install_final_checkpoint_policy()
    from main import main as upstream_main

    # Hydra consumes the original command line unchanged.
    upstream_main()


if __name__ == "__main__":
    main()
