from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts import defog_molecular_runtime as runtime


class _TopLevelDatasetInfos:
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.original_init_called = True


@pytest.mark.parametrize(
    ("dataset", "module_name", "class_name"),
    [
        ("qm9", "datasets.qm9_dataset", "QM9infos"),
        ("zinc", "datasets.zinc_dataset", "ZINCinfos"),
    ],
)
def test_statistics_patch_preserves_importable_class_identity(
    monkeypatch,
    dataset: str,
    module_name: str,
    class_name: str,
) -> None:
    datasets_package = ModuleType("datasets")
    dataset_module = ModuleType(module_name)
    info_class = type(class_name, (_TopLevelDatasetInfos,), {})
    info_class.__module__ = module_name
    setattr(dataset_module, class_name, info_class)
    setattr(datasets_package, module_name.rsplit(".", 1)[1], dataset_module)
    monkeypatch.setitem(sys.modules, "datasets", datasets_package)
    monkeypatch.setitem(sys.modules, module_name, dataset_module)
    monkeypatch.setattr(
        runtime,
        "apply_empirical_statistics",
        lambda instance, datamodule, *, dataset: {"dataset": dataset},
    )

    runtime.install_dataset_info_patch(dataset)

    # The wrapper must patch the upstream top-level class rather than replace
    # it with a non-pickleable function-local subclass.
    assert getattr(dataset_module, class_name) is info_class
    assert "<locals>" not in info_class.__qualname__
    instance = info_class(SimpleNamespace(), SimpleNamespace())
    assert instance.original_init_called is True
    assert instance.grapher_empirical_statistics == {"dataset": dataset}
    assert info_class._grapher_empirical_patch is True
