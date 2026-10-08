from pathlib import Path
import yaml
from grapher.models.gdsm_simple import hybrid_attributed_loggap as hybrid


def _load(path: str):
    data = yaml.safe_load(Path(path).read_text())
    return data["gdsm_simple"]


def test_hybrid_accepts_topology_and_typed_graphlets_through_6():
    cfg = _load(
        "configs/experiments/gdsm_laplacian_loggap_hybrid_categorical_attributed_g346_explicit/qm9_seed_42.yaml"
    )
    hybrid.validate_options(cfg)


def test_hybrid_accepts_topology_6_typed_5_ablation():
    cfg = _load(
        "configs/experiments/gdsm_laplacian_loggap_hybrid_topology_g346_typed_g345_explicit/qm9_seed_42.yaml"
    )
    hybrid.validate_options(cfg)
