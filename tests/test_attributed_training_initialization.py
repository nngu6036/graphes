"""Check initialization requirements before expensive training preparation."""
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml

from grapher.rewiring_mlp.attributed import joint_typed_edge_training as training
from grapher.utils.io import apply_config_overrides


def _zinc_config():
    path = Path(__file__).resolve().parents[1] / 'configs/experiments/grapher/zinc_joint_typed_edge_laplacian_graphlets345_learned.yaml'
    return yaml.safe_load(path.read_text(encoding='utf-8'))


@pytest.mark.parametrize('missing', [True, False])
def test_invalid_initializer_fails_before_dataset_or_output_creation(tmp_path, monkeypatch, missing):
    config = _zinc_config()
    config['joint_typed_degree']['initialize_degree_checkpoint'] = str(tmp_path / 'missing.pt') if missing else None
    output = tmp_path / 'training'
    args = SimpleNamespace(seed=42, output_dir=str(output), epochs=None, batch_size=None)

    def unexpected_dataset_load(*args, **kwargs):
        pytest.fail('Dataset loading must follow initialization validation.')

    monkeypatch.setattr(training, 'load_splits', unexpected_dataset_load)
    error_type = FileNotFoundError if missing else ValueError
    with pytest.raises(error_type) as caught:
        training.train_joint_typed_edge(config, args)
    assert '--set joint_typed_degree.freeze_epochs=0' in str(caught.value)
    if missing:
        assert '--set joint_typed_degree.initialize_degree_checkpoint=null' in str(caught.value)
    assert not output.exists()


def test_documented_scratch_overrides_reach_training_preparation(tmp_path, monkeypatch):
    config = _zinc_config()
    apply_config_overrides(config, [
        'joint_typed_degree.initialize_degree_checkpoint=null',
        'joint_typed_degree.freeze_epochs=0',
    ])
    args = SimpleNamespace(seed=42, output_dir=str(tmp_path / 'training'), epochs=None, batch_size=None)
    reached_dataset = []

    class DatasetReached(Exception):
        pass

    def load(config):
        reached_dataset.append(config)
        raise DatasetReached

    monkeypatch.setattr(training, 'load_splits', load)
    with pytest.raises(DatasetReached):
        training.train_joint_typed_edge(config, args)
    assert len(reached_dataset) == 1
    assert reached_dataset[0]['joint_typed_degree']['initialize_degree_checkpoint'] is None
    assert reached_dataset[0]['joint_typed_degree']['freeze_epochs'] == 0
