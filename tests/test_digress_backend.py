from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from grapher.models.digress.backend import generate_digress_graphs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_backend_runs_shell_free_and_decodes_export(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "DiGress"
    (root / "src").mkdir(parents=True)
    dataset = tmp_path / "native_dataset"
    dataset.mkdir()
    config = tmp_path / "resolved_config.yaml"
    config.write_text("dataset:\n  name: comm20\n", encoding="utf-8")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")

    def fake_run(command, **kwargs):
        assert kwargs["shell"] is False
        assert kwargs["cwd"] == str(root / "src")
        output = Path(command[command.index("--output") + 1])
        manifest = Path(command[command.index("--manifest") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            np.savez_compressed(
                handle,
                node_offsets=np.asarray([0, 3], dtype=np.int64),
                node_types=np.asarray([0, 0, 0], dtype=np.int64),
                edge_offsets=np.asarray([0, 2], dtype=np.int64),
                edge_endpoints=np.asarray([(0, 1), (1, 2)], dtype=np.int64),
                edge_types=np.asarray([1, 1], dtype=np.int64),
            )
        manifest.write_text(
            json.dumps(
                {
                    "format": "grapher_digress_export_v1",
                    "num_generated": 1,
                    "output": {"sha256": _sha256(output)},
                }
            ),
            encoding="utf-8",
        )
        kwargs["stdout"].write("worker complete\n")
        kwargs["stdout"].flush()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("grapher.models.digress.backend.subprocess.run", fake_run)
    result = generate_digress_graphs(
        digress_root=root,
        python_executable=Path(__import__("sys").executable),
        dataset="comm20",
        dataset_datadir=dataset,
        resolved_config_path=config,
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "output",
        num_graphs=1,
        generation_seed=42,
        batch_size=1,
    )

    assert len(result.graphs) == 1
    assert set(result.graphs[0].edges()) == {(0, 1), (1, 2)}
    assert result.export_path.is_file()
    assert result.manifest_path.is_file()
    assert result.log_path.is_file()
