from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from grapher.models.digress.workers import common


torch = pytest.importorskip("torch")


class _FakeDistributionNodes:
    def __init__(self, distribution):
        self.distribution = distribution


class _FakePlaceHolder(SimpleNamespace):
    pass


def _install_upstream_stubs(monkeypatch) -> None:
    source_package = ModuleType("src")
    source_package.__path__ = []
    utils_module = ModuleType("src.utils")
    utils_module.PlaceHolder = _FakePlaceHolder
    source_package.utils = utils_module

    diffusion_package = ModuleType("src.diffusion")
    diffusion_package.__path__ = []
    distributions_module = ModuleType("src.diffusion.distributions")
    distributions_module.DistributionNodes = _FakeDistributionNodes
    source_package.diffusion = diffusion_package
    diffusion_package.distributions = distributions_module

    monkeypatch.setitem(sys.modules, "src", source_package)
    monkeypatch.setitem(sys.modules, "src.utils", utils_module)
    monkeypatch.setitem(sys.modules, "src.diffusion", diffusion_package)
    monkeypatch.setitem(
        sys.modules,
        "src.diffusion.distributions",
        distributions_module,
    )


@pytest.mark.parametrize(
    ("dataset", "experiment", "expected"),
    [
        ("zinc", "zinc_no_h", ("qm9", "qm9_no_h")),
        ("ZINC", "custom_zinc", ("qm9", "custom_zinc")),
        ("qm9", "qm9_no_h", ("qm9", "qm9_no_h")),
    ],
)
def test_upstream_config_templates_use_stock_qm9_as_zinc_template(
    dataset: str,
    experiment: str,
    expected: tuple[str, str],
) -> None:
    assert common.upstream_config_templates(dataset, experiment) == expected


def test_compose_config_uses_qm9_hydra_templates_then_restores_zinc(
    monkeypatch,
    tmp_path,
) -> None:
    config_dir = tmp_path / "configs"
    (config_dir / "dataset").mkdir(parents=True)
    (config_dir / "experiment").mkdir()
    (config_dir / "config.yaml").write_text("defaults: []\n", encoding="utf-8")
    (config_dir / "dataset" / "qm9.yaml").write_text("name: qm9\n", encoding="utf-8")
    (config_dir / "experiment" / "qm9_no_h.yaml").write_text(
        "dataset: qm9\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class InitializationContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def initialize_config_dir(*, version_base, config_dir):
        captured["version_base"] = version_base
        captured["config_dir"] = config_dir
        return InitializationContext()

    def compose(*, config_name, overrides):
        captured["config_name"] = config_name
        captured["overrides"] = list(overrides)
        return SimpleNamespace(dataset=SimpleNamespace(name="qm9"))

    hydra_module = ModuleType("hydra")
    hydra_module.initialize_config_dir = initialize_config_dir
    hydra_module.compose = compose
    monkeypatch.setitem(sys.modules, "hydra", hydra_module)

    class FakeOmegaConf:
        @staticmethod
        def resolve(cfg):
            captured["resolved"] = cfg

    omegaconf_module = ModuleType("omegaconf")
    omegaconf_module.OmegaConf = FakeOmegaConf
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf_module)

    cfg = common.compose_config(
        digress_root=tmp_path,
        dataset="zinc",
        experiment="zinc_no_h",
        dataset_datadir=tmp_path / "dataset",
        run_name="zinc_regression",
        seed=42,
        gpus=1,
    )

    assert captured["config_name"] == "config"
    assert captured["config_dir"] == str(config_dir.resolve())
    overrides = captured["overrides"]
    assert isinstance(overrides, list)
    assert "+experiment=qm9_no_h" in overrides
    assert "dataset=qm9" in overrides
    assert "+experiment=zinc_no_h" not in overrides
    assert "dataset=zinc" not in overrides
    assert cfg.dataset.name == "zinc"
    assert captured["resolved"] is cfg


def test_zinc_dataset_infos_declares_exact_model_vocabulary() -> None:
    infos = common.GraphERMolecularDatasetInfos("zinc")

    assert infos.name == "zinc"
    assert infos.atom_decoder == ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    assert infos.atom_encoder == {
        atom: index for index, atom in enumerate(infos.atom_decoder)
    }
    assert infos.num_atom_types == 9
    assert infos.valencies == [4, 3, 2, 1, 5, 6, 1, 1, 1]
    assert infos.atom_weights == {
        0: 12.0,
        1: 14.0,
        2: 16.0,
        3: 19.0,
        4: 30.0,
        5: 32.0,
        6: 35.5,
        7: 78.0,
        8: 127.0,
    }
    assert infos.max_n_nodes == 38
    assert infos.max_weight == 500.0
    assert infos.aromatic is False


def _zinc_statistics() -> dict[str, object]:
    n_nodes = [0.0] * 39
    n_nodes[2] = 3.0
    return {
        "format": "grapher_digress_molecular_statistics_v1",
        "dataset": "zinc",
        "n_nodes": n_nodes,
        "node_types": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "edge_types": [4.0, 3.0, 2.0, 1.0],
        "valency_distribution": [1.0, 2.0, 1.0],
    }


def test_compute_zinc_statistics_scans_training_split_once() -> None:
    nodes = torch.zeros((3, 9), dtype=torch.float32)
    nodes[0, 0] = 1.0
    nodes[1, 2] = 1.0
    nodes[2, 1] = 1.0
    edges = torch.zeros((2, 4), dtype=torch.float32)
    edges[:, 2] = 1.0
    batch = SimpleNamespace(
        x=nodes,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=edges,
        batch=torch.tensor([0, 0, 1], dtype=torch.long),
    )

    class DataModule:
        calls = 0

        def train_dataloader(self):
            self.calls += 1
            return [batch]

    datamodule = DataModule()
    statistics = common.compute_molecular_statistics(
        datamodule, dataset="zinc"
    )

    assert datamodule.calls == 1
    assert statistics["dataset"] == "zinc"
    assert statistics["atom_decoder"] == [
        "C",
        "N",
        "O",
        "F",
        "P",
        "S",
        "Cl",
        "Br",
        "I",
    ]
    assert statistics["n_nodes"] == pytest.approx([0.0, 0.5, 0.5])
    assert statistics["node_types"][:3] == pytest.approx([1 / 3] * 3)
    assert statistics["edge_types"] == pytest.approx([0.0, 0.0, 1.0, 0.0])
    assert statistics["valency_distribution"] == pytest.approx(
        [1 / 3, 0.0, 2 / 3, 0.0]
    )
    assert statistics["node_count_source"] == "train_and_validation"
    assert statistics["categorical_source"] == "train"


def test_compute_zinc_statistics_includes_validation_node_count_support() -> None:
    train_nodes = torch.zeros((2, 9), dtype=torch.float32)
    train_nodes[:, 0] = 1.0
    train_edges = torch.zeros((2, 4), dtype=torch.float32)
    train_edges[:, 1] = 1.0
    train_batch = SimpleNamespace(
        x=train_nodes,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=train_edges,
        batch=torch.zeros(2, dtype=torch.long),
    )
    validation_nodes = torch.zeros((3, 9), dtype=torch.float32)
    validation_nodes[:, 1] = 1.0
    validation_batch = SimpleNamespace(
        x=validation_nodes,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 4), dtype=torch.float32),
        batch=torch.zeros(3, dtype=torch.long),
    )

    class DataModule:
        def train_dataloader(self):
            return [train_batch]

        def val_dataloader(self):
            return [validation_batch]

    statistics = common.compute_molecular_statistics(
        DataModule(), dataset="zinc"
    )

    assert statistics["n_nodes"] == pytest.approx([0.0, 0.0, 0.5, 0.5])
    # Validation molecules expand node-count support without affecting the
    # atom, bond, or valency categorical priors learned from training data.
    assert statistics["node_types"][:2] == pytest.approx([1.0, 0.0])
    assert statistics["edge_types"] == pytest.approx([0.0, 1.0, 0.0, 0.0])


def test_apply_zinc_statistics_accepts_nine_atom_and_four_edge_classes(
    monkeypatch,
) -> None:
    _install_upstream_stubs(monkeypatch)
    infos = common.GraphERMolecularDatasetInfos("zinc")

    common.apply_molecular_statistics(
        infos,
        _zinc_statistics(),
        dataset="zinc",
    )

    assert infos.node_types.shape == (9,)
    assert infos.edge_types.shape == (4,)
    assert torch.isclose(infos.node_types.sum(), torch.tensor(1.0))
    assert torch.isclose(infos.edge_types.sum(), torch.tensor(1.0))
    assert infos.max_n_nodes == 38
    assert isinstance(infos.nodes_dist, _FakeDistributionNodes)
    assert torch.equal(infos.nodes_dist.distribution, infos.n_nodes)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("node_types", [1.0] * 8, "expected 9 classes"),
        ("edge_types", [1.0] * 5, "expected 4 classes"),
    ],
)
def test_apply_zinc_statistics_rejects_categorical_width_mismatch(
    monkeypatch,
    field: str,
    replacement: list[float],
    message: str,
) -> None:
    _install_upstream_stubs(monkeypatch)
    statistics = _zinc_statistics()
    statistics[field] = replacement

    with pytest.raises(ValueError, match=message):
        common.apply_molecular_statistics(
            common.GraphERMolecularDatasetInfos("zinc"),
            statistics,
            dataset="zinc",
        )


def test_apply_zinc_statistics_rejects_statistics_from_another_dataset(
    monkeypatch,
) -> None:
    _install_upstream_stubs(monkeypatch)
    statistics = _zinc_statistics()
    statistics["dataset"] = "qm9"

    with pytest.raises(ValueError, match="expected zinc, got qm9"):
        common.apply_molecular_statistics(
            common.GraphERMolecularDatasetInfos("zinc"),
            statistics,
            dataset="zinc",
        )


def test_zinc_molecular_features_respect_mask_and_kekulized_bond_orders(
    monkeypatch,
) -> None:
    _install_upstream_stubs(monkeypatch)
    infos = common.GraphERMolecularDatasetInfos("zinc")
    features = common.GraphERMolecularFeatures(infos, dataset="zinc")

    # A two-node C=O fragment plus one padded node. Padded categorical state is
    # all-zero, matching the dense masked representation used by DiGress.
    nodes = torch.zeros((1, 3, 9), dtype=torch.float32)
    nodes[0, 0, 0] = 1.0
    nodes[0, 1, 2] = 1.0
    edges = torch.zeros((1, 3, 3, 4), dtype=torch.float32)
    edges[0, 0, 1, 2] = 1.0
    edges[0, 1, 0, 2] = 1.0
    node_mask = torch.tensor([[True, True, False]])

    result = features({"X_t": nodes, "E_t": edges, "node_mask": node_mask})

    # Columns are formal charge proxy and current valency respectively.
    assert torch.equal(
        result.X,
        torch.tensor([[[2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]]),
    )
    assert result.E.shape == (1, 3, 3, 0)
    assert torch.allclose(result.y, torch.tensor([[(12.0 + 16.0) / 500.0]]))
