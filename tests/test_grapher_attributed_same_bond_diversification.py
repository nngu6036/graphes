import networkx as nx
import numpy as np

from grapher.properties.summary import SummaryConfig
from grapher.models.grapher import attributed_diversification
from grapher.models.grapher import hybrid_attributed_loggap as hybrid
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary, GraphletBasis
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.graphlets import extract_topology_graphlet_target
from grapher.rewiring_mlp.molecular.typed_invariants import (
    extract_typed_invariant,
    typed_invariant_matches_graph,
)


def _graph():
    g = nx.Graph()
    for i, atom in enumerate([6, 6, 6, 6]):
        g.add_node(i, atomic_num=atom)
    g.add_edge(0, 1, bond_type=1, bond_order=1.0)
    g.add_edge(1, 2, bond_type=2, bond_order=2.0)
    g.add_edge(2, 3, bond_type=1, bond_order=1.0)
    return g


def _summary_cfg(attributed=False):
    return SummaryConfig.from_dict({
        "graphlet_history": True,
        "graphlet_k_min": 3,
        "graphlet_k_max": 4,
        "graphlet_connected_only": True,
        "graphlet_topology_filter": "all",
        "graphlet_backend": "exact",
        "graphlet_num_samples": None,
        **({
            "attributed": True,
            "node_attribute": "atomic_num",
            "edge_attribute": "bond_type",
            "attributed_backend": "python",
        } if attributed else {}),
    })


def test_same_bond_diversification_preserves_typed_degrees():
    g = _graph()
    vocab = GraphCategoryVocabulary.from_graphs(
        [g],
        {
            "node_attribute": "atomic_num",
            "node_categories": [6],
            "edge_attribute": "bond_type",
            "edge_categories": [1, 2],
        },
    )
    top_cfg = _summary_cfg(False)
    typed_cfg = _summary_cfg(True)
    top_basis = TopologyGraphletBasis.from_config(top_cfg)
    typed_basis = GraphletBasis.fit_from_graphs(
        [g], typed_cfg, vocabulary=vocab, attributed=True, seed=1
    )
    top_hist, top_mass = extract_topology_graphlet_target(
        g, graphlet_basis=top_basis, summary_config=top_cfg
    )
    typed_history, typed_mass_dict = typed_basis.statistics_for_graph(g, typed_cfg)
    typed_hist = typed_basis.flatten_history(typed_history)
    typed_mass = typed_basis.flatten_mass(typed_mass_dict)

    cfg = attributed_diversification.resolve_options({
        "enabled": True,
        "random_steps": {"distribution": "uniform_integer", "min_steps": 1, "max_steps": 1},
        "proposal_budget": 32,
        "valid_candidate_budget": 16,
        "rdkit_candidate_filter": False,
        "guidance": {
            "topology_graphlet_weight": 0.0,
            "topology_graphlet_mass_weight": 0.0,
            "topology_orbit_weight": 0.0,
            "typed_graphlet_weight": 1.0,
            "typed_graphlet_mass_weight": 0.0,
        },
        "soft_selection": {
            "temperature_start": 1.0,
            "temperature_end": 1.0,
            "temperature_schedule": "constant",
            "score_normalization": "none",
            "max_target_worsening": 10.0,
            "max_relative_target_worsening": 10.0,
        },
        "target_trust_region": {
            "enabled": True,
            "absolute_worsening": 10.0,
            "relative_worsening": 10.0,
        },
    })
    invariant = extract_typed_invariant(
        g, edge_types=vocab.edge_values, node_attribute="atomic_num", edge_attribute="bond_type"
    )
    final, diag = attributed_diversification.diversify_same_bond_graph(
        g,
        topology_prediction={
            "graphlet_histogram": top_hist,
            "graphlet_mass": top_mass,
        },
        attribute_prediction={
            "typed_graphlet_histogram": typed_hist,
            "typed_graphlet_mass": typed_mass,
        },
        topology_basis=top_basis,
        typed_basis=typed_basis,
        topology_summary_config=top_cfg,
        typed_summary_config=typed_cfg,
        vocabulary=vocab,
        config=cfg,
        rng=np.random.default_rng(3),
    )
    assert diag["accepted_steps"] == 1
    assert diag["changed"] is True
    assert typed_invariant_matches_graph(final, invariant)
    assert nx.is_connected(final)
    assert sorted(d["bond_type"] for _, _, d in final.edges(data=True)) == [1, 1, 2]


def test_random_zero_budget_keeps_graph_unchanged():
    g = _graph()
    vocab = GraphCategoryVocabulary.from_graphs(
        [g], {"node_attribute": "atomic_num", "edge_attribute": "bond_type"}
    )
    top_cfg = _summary_cfg(False)
    typed_cfg = _summary_cfg(True)
    top_basis = TopologyGraphletBasis.from_config(top_cfg)
    typed_basis = GraphletBasis.fit_from_graphs([g], typed_cfg, vocabulary=vocab, attributed=True)
    top_hist, top_mass = extract_topology_graphlet_target(g, graphlet_basis=top_basis, summary_config=top_cfg)
    history, mass = typed_basis.statistics_for_graph(g, typed_cfg)
    final, diag = attributed_diversification.diversify_same_bond_graph(
        g,
        topology_prediction={"graphlet_histogram": top_hist, "graphlet_mass": top_mass},
        attribute_prediction={
            "typed_graphlet_histogram": typed_basis.flatten_history(history),
            "typed_graphlet_mass": typed_basis.flatten_mass(mass),
        },
        topology_basis=top_basis,
        typed_basis=typed_basis,
        topology_summary_config=top_cfg,
        typed_summary_config=typed_cfg,
        vocabulary=vocab,
        config={"enabled": True, "random_steps": {"min_steps": 0, "max_steps": 0}, "rdkit_candidate_filter": False},
        rng=np.random.default_rng(1),
    )
    assert diag["accepted_steps"] == 0
    assert diag["changed"] is False
    assert nx.utils.graphs_equal(final, g)


def test_hybrid_options_accept_new_diversification_block():
    options = hybrid.default_options()
    options["attributed_diversification"]["enabled"] = True
    options["attributed_diversification"]["rdkit_candidate_filter"] = False
    hybrid.validate_options(options)


def test_hybrid_generation_writes_paired_same_bond_outputs(tmp_path):
    import json
    import pickle
    import torch
    from pathlib import Path
    from grapher.models.base import DatasetReference, GenerateRequest, RunSpec, TrainRequest
    from grapher.models.grapher.wrapper import GraphERWrapper
    import importlib.util
    helper_path = Path(__file__).with_name("test_grapher_hybrid_attributed_loggap.py")
    spec = importlib.util.spec_from_file_location("_hybrid_test_helpers", helper_path)
    helper = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(helper)

    torch.set_num_threads(1)
    dataset = helper._write_dataset(tmp_path)
    options = helper._options()
    options["attributed_diversification"] = {
        "enabled": True,
        "mode": "stochastic_same_bond_diversification",
        "random_steps": {"distribution": "uniform_integer", "min_steps": 1, "max_steps": 1},
        "proposal_budget": 16,
        "valid_candidate_budget": 8,
        "preserve_connectivity": True,
        "strict_same_bond": True,
        "rdkit_candidate_filter": False,
        "reject_revisited_states": True,
        "normalization": "initial",
        "epsilon": 1e-6,
        "guidance": {
            "topology_graphlet_weight": 0.5,
            "topology_graphlet_mass_weight": 0.05,
            "topology_orbit_weight": 0.25,
            "typed_graphlet_weight": 1.0,
            "typed_graphlet_mass_weight": 0.1,
        },
        "soft_selection": {
            "temperature_start": 1.0,
            "temperature_end": 1.0,
            "temperature_schedule": "constant",
            "score_normalization": "std",
            "max_target_worsening": 10.0,
            "max_relative_target_worsening": 10.0,
        },
        "target_trust_region": {
            "enabled": True,
            "absolute_worsening": 10.0,
            "relative_worsening": 10.0,
        },
    }
    wrapper = GraphERWrapper()
    run = RunSpec("grapher", "qm9", "hybrid_div", 42, tmp_path / "runs")
    trained = wrapper.train(TrainRequest(run, dataset, options=options))
    generated = wrapper.generate(
        GenerateRequest(run, trained.checkpoint_path, 6, 99, generation_id="sample")
    )
    manifest = json.loads(generated.manifest_path.read_text())
    assert manifest["sampling"]["rewiring"] is True
    assert manifest["sampling"]["same_bond_type_rewiring"] is True
    assert manifest["sampling"]["typed_degree_preserved_by_rewiring"] is True
    assert (generated.generation_dir / "pre_refinement_graphs.pkl").is_file()
    assert (generated.generation_dir / "attributed_diversification_diagnostics.json").is_file()

    with (generated.generation_dir / "pre_refinement_graphs.pkl").open("rb") as handle:
        before = pickle.load(handle)
    with generated.graphs_path.open("rb") as handle:
        after = pickle.load(handle)
    assert len(before) == len(after) == 6
    for left, right in zip(before, after):
        invariant = extract_typed_invariant(
            left, edge_types=(1, 2, 3), node_attribute="atomic_num", edge_attribute="bond_type"
        )
        assert typed_invariant_matches_graph(right, invariant)


def test_explicit_qm9_same_bond_configs_enable_generation_only_diversification(tmp_path):
    from pathlib import Path
    from grapher.models.base import DatasetReference, RunSpec, TrainRequest
    from grapher.models.grapher.wrapper import GraphERWrapper

    root = Path(__file__).resolve().parents[1]
    for seed in (41, 42, 43, 44, 45):
        config = root / (
            "configs/experiments/grapher_final/"
            f"qm9_seed_{seed}.yaml"
        )
        request = TrainRequest(
            RunSpec("grapher", "qm9", "cfg", seed, tmp_path / "runs"),
            DatasetReference("qm9", tmp_path / "datasets", "unused"),
            config_path=config,
        )
        options = GraphERWrapper()._options(request)
        div = options["attributed_diversification"]
        assert div["enabled"] is True
        assert div["strict_same_bond"] is True
        assert div["preserve_neighbor_atom_type_by_bond_type"] is True
        assert div["verify_bond_valence_preservation"] is True
        assert div["preserve_connectivity"] is True
        assert div["random_steps"]["min_steps"] == 0
        assert div["random_steps"]["max_steps"] == 4
        assert options["attribute_summary"]["typed_graphlet"]["orders"] == [3, 4, 5]
