from __future__ import annotations

import threading
import time
import sys
from pathlib import Path

import networkx as nx
import pytest
import torch

from grapher.utils.io import load_yaml, save_pickle
from scripts.train_attributed_grapher import _heartbeat_loop, _run_epoch


class _FakeBatch:
    def __init__(self, loss: float) -> None:
        self.loss = float(loss)
        self.graph_size = torch.ones(1)

    def to(self, _device: torch.device) -> "_FakeBatch":
        return self


class _FakeModel:
    def train(self, _training: bool) -> None:
        return None

    def loss(
        self,
        batch: _FakeBatch,
        *,
        loss_weights: dict[str, float],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert loss_weights == {}
        return torch.tensor(batch.loss), {
            "loss": batch.loss,
            "auxiliary": 2.0 * batch.loss,
        }


def test_run_epoch_reports_first_periodic_and_final_batches(capsys) -> None:
    metrics = _run_epoch(
        _FakeModel(),
        [_FakeBatch(1.0), _FakeBatch(3.0), _FakeBatch(5.0)],
        device=torch.device("cpu"),
        optimizer=None,
        loss_weights={},
        phase="val",
        epoch=2,
        total_epochs=4,
        expected_examples=3,
        expected_batches=3,
        batch_progress_interval=2,
    )

    output = capsys.readouterr().out
    assert "epoch=0002/0004 phase=val status=waiting_for_first_batch" in output
    assert "batch=1/3" in output
    assert "batch=2/3" in output
    assert "batch=3/3" in output
    assert "status=complete" in output
    assert "running_loss=3.00000" in output
    assert metrics == pytest.approx({"auxiliary": 6.0, "loss": 3.0})


def test_run_epoch_progress_is_disabled_by_default(capsys) -> None:
    _run_epoch(
        _FakeModel(),
        [_FakeBatch(1.0)],
        device=torch.device("cpu"),
        optimizer=None,
        loss_weights={},
    )

    assert capsys.readouterr().out == ""


def test_heartbeat_reports_active_blocked_batch_deterministically(capsys) -> None:
    class StopAfterOneHeartbeat:
        calls = 0

        def wait(self, _interval_seconds: float) -> bool:
            self.calls += 1
            return self.calls > 1

        def is_set(self) -> bool:
            return False

    _heartbeat_loop(
        StopAfterOneHeartbeat(),  # type: ignore[arg-type]
        threading.Lock(),
        {
            "status": "loading_batch",
            "active_batch": 1,
            "completed_batches": 0,
            "completed_examples": 0,
            "running_loss": None,
        },
        interval_seconds=30.0,
        epoch=1,
        total_epochs=1,
        phase="train",
        expected_batches=1,
        expected_examples=1,
        started=time.perf_counter(),
    )

    output = capsys.readouterr().out
    assert "status=loading_batch" in output
    assert "active_batch=1" in output


def test_full_qm9_config_enables_intra_epoch_progress() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_yaml(
        root / "configs/experiments/grapher/qm9_attributed_spectral_graphlet.yaml"
    )
    predictor = config["attributed_predictor"]

    assert predictor["batch_progress_interval"] == 10
    assert predictor["progress_interval_seconds"] == 30


@pytest.mark.parametrize('flag', ['--max-train-graphs', '--num-train-graphs'])
def test_legacy_cli_samples_prepared_train_before_fitting_vocabulary(tmp_path, monkeypatch, flag):
    from scripts import train_attributed_grapher as cli

    root = tmp_path/'toy'
    root.mkdir()
    graphs = [nx.path_graph(4) for _ in range(8)]
    for index, graph in enumerate(graphs):
        graph.graph['prepared_index'] = index
    for split in ('train', 'val', 'test'):
        save_pickle(graphs, root/f'{split}.pkl')
    config = {'seed': 7, 'dataset': {'name': 'toy', 'root': str(tmp_path),
                                   'max_train_graphs': 2, 'build_if_missing': False}}
    monkeypatch.setattr(cli, 'load_yaml', lambda _: config)
    monkeypatch.setattr(sys, 'argv', ['train', '--config', 'unused.yaml', flag, '3', '--seed', '42'])
    observed = []

    class SelectionReachedVocabulary(Exception):
        pass

    def vocabulary(cls, selected, _config):
        observed.extend(g.graph['prepared_index'] for g in selected)
        raise SelectionReachedVocabulary

    monkeypatch.setattr(cli.GraphCategoryVocabulary, 'from_graphs', classmethod(vocabulary))
    with pytest.raises(SelectionReachedVocabulary):
        cli.main()
    assert len(observed) == 3 and observed != [0, 1, 2]
    assert config['seed'] == 42
    assert config['dataset']['training_subset']['indices'] == observed
    assert config['dataset']['max_train_graphs'] == 3
