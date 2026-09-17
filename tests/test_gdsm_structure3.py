from __future__ import annotations

import copy
import itertools
import json
import pickle
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.models.base import DatasetReference, RunSpec, TrainRequest, GenerateRequest
from grapher.models.gdsm_simple.degree_initialization import (
    degree_basis_anchor, conditioning_features, realize_degrees, validate_degree_bank,
)
from grapher.models.gdsm_simple.model import make_schedule
from grapher.models.gdsm_simple.refiner import normalized_adjacency_eigenvalues
from grapher.models.gdsm_simple.structure3 import (
    graph_summary3, reconcile_summary3, refine_structure3, validate_structure_options,
)
from grapher.models.gdsm_simple.structured_model import (
    StructuredEigenvalueDenoiser, centered_q_sample, centered_x0, centered_ddim_step,
    summary_losses, summary_probabilities,
)
from grapher.models.gdsm_simple.wrapper import GDSMSimpleWrapper


def test_summary3_known_graphs():
    path, triangle = graph_summary3(nx.path_graph(3), 10), graph_summary3(nx.complete_graph(3), 10)
    np.testing.assert_allclose(path["graphlet_counts"], [1, 0])
    np.testing.assert_allclose(path["orbit_mean"], [4/3, 2/3, 1/3, 0])
    np.testing.assert_allclose(triangle["graphlet_counts"], [0, 1])
    np.testing.assert_allclose(triangle["orbit_mean"], [2, 0, 0, 1])
    assert path["clustering_histogram"][0] == 1
    assert triangle["clustering_histogram"][-1] == 1


def test_summary3_matches_bruteforce_all_atlas_graphs_up_to_five_nodes():
    for g in nx.graph_atlas_g():
        n = len(g)
        if not 1 <= n <= 5:
            continue
        expected = np.zeros((n, 4)); expected[:, 0] = [g.degree(v) for v in range(n)]
        counts = np.zeros(2)
        for nodes in itertools.combinations(range(n), 3):
            sub = g.subgraph(nodes)
            if sub.number_of_edges() == 3:
                counts[1] += 1
                for v in nodes: expected[v, 3] += 1
            elif sub.number_of_edges() == 2:
                counts[0] += 1
                for v in nodes: expected[v, 2 if sub.degree(v) == 2 else 1] += 1
        actual = graph_summary3(g, 10)
        np.testing.assert_array_equal(actual["orbit_node_counts"], expected)
        np.testing.assert_array_equal(actual["graphlet_counts"], counts)
        np.testing.assert_allclose(actual["clustering_histogram"].sum(), 1)


def test_summary3_permutation_invariance_and_empty_connected_histogram():
    g = nx.wheel_graph(8)
    permuted = nx.relabel_nodes(g, {v: (3*v+1) % 8 for v in g})
    a, b = graph_summary3(g), graph_summary3(permuted)
    for key in ("clustering_histogram", "orbit_mean", "graphlet_counts", "graphlet_histogram"):
        np.testing.assert_allclose(a[key], b[key])
    empty = graph_summary3(nx.empty_graph(4))
    np.testing.assert_array_equal(empty["graphlet_histogram"], [0, 0])
    assert np.isfinite(empty["orbit_log_mean"]).all()


def test_reconciled_orbits_match_degree_count_identities():
    d = np.array([3, 2, 2, 1])
    raw = {"clustering_histogram": np.ones(10)/10, "orbit_log_mean": np.log1p([900., 800., 100., 300.]), "graphlet_histogram": np.array([.2, .8])}
    r = reconcile_summary3(raw, d)
    o = np.expm1(r["orbit_log_mean"])
    assert o[0] == pytest.approx(d.mean())
    assert o[1] == pytest.approx(2*o[2])
    assert len(d)*(o[2]+o[3]) == pytest.approx(np.sum(d*(d-1)/2))
    assert r["clustering_histogram"][0] >= .25
    assert r["clustering_histogram"].sum() == pytest.approx(1)


def test_degree_basis_anchor_trace_moment_and_sign_invariance():
    g = nx.path_graph(7)
    _, u = np.linalg.eigh(nx.to_numpy_array(g))
    d = np.array([g.degree(v) for v in g])
    anchor, diag = degree_basis_anchor(u, d)
    signs = np.array([1, -1, 1, -1, -1, 1, 1])
    second, _ = degree_basis_anchor(u*signs[None, :], d)
    np.testing.assert_allclose(anchor, second, atol=1e-8)
    assert np.all(np.diff(anchor) >= -1e-9)
    assert abs(anchor.sum()) < 1e-8
    assert len(d)*float(anchor@anchor) == pytest.approx(float(d.sum()))
    assert diag["solver_success"]
    np.testing.assert_allclose(conditioning_features(u, d, anchor, 10), conditioning_features(u*signs, d, second, 10), atol=1e-7)


def test_incompatible_basis_is_reported_not_treated_as_exact():
    anchor, diag = degree_basis_anchor(np.eye(5), np.full(5, 2))
    assert np.isfinite(anchor).all()
    assert diag["row_sum_rmse"] > .1
    assert diag["diagonal_rmse"] > .1
    partial, pd = degree_basis_anchor(np.eye(5), np.full(5, 2), seed_top_k=2)
    assert np.count_nonzero(partial) <= 2
    assert pd["num_seeded_modes"] == 2
    with pytest.raises(ValueError, match="orthonormal"):
        degree_basis_anchor(np.ones((5, 5)), np.full(5, 2))


def test_centered_forward_and_reverse_algebra():
    schedule = make_schedule(steps=50, beta_start=.001, beta_end=.1)
    torch.manual_seed(21)
    clean, anchor, noise = torch.randn(3, 8), torch.randn(3, 8), torch.randn(3, 8)
    t = torch.tensor([0, 20, 49])
    noisy = centered_q_sample(clean, t, noise, anchor, schedule)
    recovered = centered_x0(noisy, noise, t, anchor, schedule)
    torch.testing.assert_close(recovered, clean, atol=2e-6, rtol=2e-6)
    previous = torch.tensor([0, 10, 30])
    torch.testing.assert_close(centered_ddim_step(clean, noise, previous, anchor, schedule), centered_q_sample(clean, previous, noise, anchor, schedule))


def test_summary_heads_train_and_respect_masks():
    model = StructuredEigenvalueDenoiser(max_nodes=8, hidden_dim=16, num_layers=1, num_heads=4, ff_dim=32, clustering_bins=10)
    mask = torch.tensor([[True]*5+[False]*3, [True]*8])
    eps, pred = model(torch.randn(2, 8)*mask, torch.tensor([1, 8]), mask, mask.sum(1), 10, torch.randn(2, 32))
    assert (eps[~mask] == 0).all()
    prob = summary_probabilities(pred)
    torch.testing.assert_close(prob["clustering_histogram"].sum(1), torch.ones(2))
    losses = summary_losses(pred, torch.ones(2, 10)/10, torch.zeros(2, 4), torch.ones(2, 2)/2)
    (eps.square().mean()+sum(losses.values())).backward()
    for head in (model.clustering_head, model.orbit_head, model.graphlet_head, model.condition):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def test_structure_refiner_accepts_oracle_target_and_preserves_degrees():
    source = nx.Graph()
    source.add_nodes_from(range(8))
    source.add_edges_from([(0,4),(1,2),(1,4),(1,5),(1,7),(2,4),(3,5),(4,7),(5,6),(6,7)])
    target = source.copy(); target.remove_edges_from([(0,4),(5,6)]); target.add_edges_from([(0,5),(4,6)])
    goal = graph_summary3(target, 20)
    out, diag = refine_structure3(source, normalized_adjacency_eigenvalues(target), goal, source=source,
                                 config={"max_steps_per_event":1,"proposal_budget":-1,"valid_candidate_budget":-1,"lambda_weight":0.,"source_weight":0.}, rng=np.random.default_rng(5))
    assert diag["accepted_steps"] == 1
    assert diag["final"]["structure"] < diag["initial"]["structure"]
    assert dict(out.degree()) == dict(source.degree())
    assert nx.is_connected(out)
    assert diag["all_accepted_steps_improve_energy"]


def test_refiner_no_move_is_safe_for_complete_graph():
    g = nx.complete_graph(5)
    out, diag = refine_structure3(g, normalized_adjacency_eigenvalues(g), graph_summary3(g), source=g,
                                 config={"max_steps_per_event":1,"proposal_budget":-1,"valid_candidate_budget":-1}, rng=np.random.default_rng(0))
    assert diag["accepted_steps"] == 0
    assert set(out.edges()) == set(g.edges())


def test_degree_realization_and_stale_prior_rejection():
    d = np.full(8, 2)
    g, diag = realize_degrees(d, np.eye(8), ensure_connected=True,
                             config={"realization_fit_steps":1,"proposal_budget":16,"valid_candidate_budget":8}, rng=np.random.default_rng(2))
    assert [g.degree(v) for v in range(8)] == d.tolist()
    assert nx.is_connected(g) and diag["degree_preserved"]
    validate_degree_bank(SimpleNamespace(empirical_degree_sequences=[[2,1,1]]), [[1,2,1]])
    with pytest.raises(ValueError, match="multiset"):
        validate_degree_bank(SimpleNamespace(empirical_degree_sequences=[[2,1,1],[2,1,1]]), [[2,1,1]])


def small_options():
    return {"train":{"epochs":2,"batch_size":4,"validation_every":1,"log_every":1},
            "model":{"max_nodes":6,"hidden_dim":16,"num_layers":1,"num_heads":4,"ff_dim":32},
            "diffusion":{"steps":30,"beta_start":.01,"beta_end":.3},"sample":{"steps":8},
            "generation_batch_size":2,"runtime":{"device":"cpu"},
            "extensions":{"degree_conditioning":True,"structural_summary":{"enabled":True,"graphlet_size":3,"clustering_bins":10,"basis_pairings_per_graph":2},
                          "degree_preserving_rewiring":True,"initialization":{"mode":"degree_basis","degree_generator":{"type":"empirical"},"ensure_connected":True},
                          "structure_guidance":{"every":2,"start_fraction":.7,"max_steps_per_event":1,"proposal_budget":24,"valid_candidate_budget":12,"realization_fit_steps":1,"save_intermediate_graphs":True}}}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("structure3")
    data = root/"data"/"toy"; data.mkdir(parents=True)
    # Different held-out graphs, all within the training size support.
    train = [nx.cycle_graph(6), nx.complete_bipartite_graph(3,3), nx.wheel_graph(6), nx.path_graph(6)]
    for split, graphs in (("train", train), ("val", [nx.star_graph(5), nx.complete_graph(6)]), ("test", [nx.ladder_graph(3)])):
        with (data/(split+".pkl")).open("wb") as handle: pickle.dump(graphs, handle)
    run = RunSpec("gdsm_simple","community_small","smoke",42,root/"runs")
    wrapper = GDSMSimpleWrapper()
    request = TrainRequest(run,DatasetReference("community_small",root/"data","toy"), options=small_options())
    artifacts = wrapper.train(request)
    return wrapper, request, artifacts, train, root


def test_training_generation_roundtrip_and_checkpoint_reuse(trained):
    wrapper, request, artifacts, train, root = trained
    assert wrapper.train(request).checkpoint_path == artifacts.checkpoint_path
    state = torch.load(artifacts.checkpoint_path, weights_only=False)
    assert len(state["basis_eigenvectors"]) == len(train)
    result = wrapper.generate(GenerateRequest(request.run,artifacts.checkpoint_path,4,42,generation_id="guided"))
    record = json.loads((result.generation_dir/"manifest.json").read_text())
    assert record["structural_rewiring"]["degree_preservation_rate"] == 1.
    assert record["structural_rewiring"]["mean_guidance_events"] > 1
    assert record["empirical_prior"]["test_conditioning"] is False
    with (result.generation_dir/"base_graphs.pkl").open("rb") as h: graphs=pickle.load(h)
    with (result.generation_dir/"sampled_degree_sequences.pkl").open("rb") as h: degrees=pickle.load(h)
    for g, d in zip(graphs, degrees):
        assert [g.degree(v) for v in range(len(g))] == d
        assert nx.is_connected(g) and nx.number_of_selfloops(g) == 0
    repeat = wrapper.generate(GenerateRequest(request.run,artifacts.checkpoint_path,4,42,generation_id="repeated"))
    assert repeat.graphs_sha256 == result.graphs_sha256
    diag = json.loads((result.generation_dir/"rewiring_diagnostics.json").read_text())
    assert all(any(e["timestep"] > 0 for e in row["events"]) for row in diag["per_graph"])


def test_generation_rejects_changed_training_corruption_and_v1_checkpoint(trained):
    wrapper, request, artifacts, _, _ = trained
    with pytest.raises(ValueError, match="initialization.mode"):
        wrapper.generate(GenerateRequest(request.run,artifacts.checkpoint_path,1,42,generation_id="wrong",options={"extensions":{"initialization":{"mode":"gaussian"}}}))
    from grapher.models.gdsm_simple.structured_pipeline import _validate_generation_contract
    with pytest.raises(ValueError, match="Retrain"):
        _validate_generation_contract({"format":"gdsm_simple_checkpoint_v1"}, small_options())


def test_disabled_structure_guidance_has_zero_accepted_steps(trained):
    wrapper, request, artifacts, _, _ = trained
    result = wrapper.generate(GenerateRequest(request.run,artifacts.checkpoint_path,2,51,generation_id="no_rewire",
                              options={"extensions":{"degree_preserving_rewiring":False}}))
    record = json.loads((result.generation_dir/"manifest.json").read_text())
    assert record["structural_rewiring"]["mean_accepted_steps"] == 0
    assert record["structural_rewiring"]["spectrum_feedback"] == 0


def test_gaussian_structure_mode_can_train_and_generate(tmp_path):
    options = small_options(); options["extensions"]["initialization"]["mode"]="gaussian"
    options["train"]["epochs"] = 1
    data=tmp_path/"datasets"/"toy"; data.mkdir(parents=True)
    for split in ("train","val","test"):
        with (data/(split+".pkl")).open("wb") as h: pickle.dump([nx.cycle_graph(6),nx.path_graph(6)],h)
    run=RunSpec("gdsm_simple","community_small","gaussian",42,tmp_path/"runs")
    w=GDSMSimpleWrapper()
    art=w.train(TrainRequest(run,DatasetReference("community_small",tmp_path/"datasets","toy"),options=options))
    gen=w.generate(GenerateRequest(run,art.checkpoint_path,2,42))
    with (gen.generation_dir/"initial_eigenvalue_anchors.pkl").open("rb") as h:
        assert all(np.all(a==0) for a in pickle.load(h))


def test_real_dhvae_checkpoint_integration(trained):
    from grapher.models.dhvae_hh.degree_vae import DegreeVectorizer, build_degree_vae, degree_vae_loss, save_degree_vae_checkpoint
    wrapper, request, artifacts, train, root = trained
    torch.manual_seed(73)
    vectorizer=DegreeVectorizer.fit(train,require_connected=True)
    x, targets=vectorizer.to_training_arrays(train)
    x=torch.from_numpy(x); targets={k:torch.from_numpy(v) for k,v in targets.items()}
    model=build_degree_vae(vectorizer,latent_dim=4,hidden_dim=16,size_condition_dim=4,edge_condition_dim=4,
                           use_edge_count_conditioning=True,prior_condition_on_edges=True,prior_type="conditional_gaussian",prior_components=1,num_layers=1)
    optimizer=torch.optim.Adam(model.parameters(),lr=.015)
    for _ in range(40):
        optimizer.zero_grad(); out,mu,lv=model(x,targets["num_nodes_count"],targets["num_edges_count"])
        loss,_=degree_vae_loss(out,targets,mu,lv,weights={"num_edges":2.,"degree":5.})
        loss.backward();optimizer.step()
    path=root/"degree.pt"
    save_degree_vae_checkpoint(path,model,vectorizer)
    prior={"type":"dhvae","checkpoint_path":str(path),"sample_num_nodes":"empirical","sample_num_edges":"model",
           "fallback":"error","postprocess_policy":"reject_only","max_resample":100,"model_resample_attempts":8,
           "exact_degree_sum_conditioning":True,"parity_conditioned":False}
    result=wrapper.generate(GenerateRequest(request.run,artifacts.checkpoint_path,2,79,generation_id="learned",
                            options={"extensions":{"initialization":{"degree_generator":prior}}}))
    manifest=json.loads((result.generation_dir/"manifest.json").read_text())
    assert manifest["degree_prior"]["learned"] is True
    assert manifest["degree_prior"]["training_degree_multiset_verified"] is True
    assert manifest["structural_rewiring"]["degree_preservation_rate"] == 1


@pytest.mark.parametrize("override", [
    {"structural_summary":{"enabled":True,"graphlet_size":4}},
    {"structural_summary":{"enabled":True,"orbit_scope":"up_to_4"}},
    {"structure_guidance":{"every":0}},
    {"structure_guidance":{"spectrum_feedback":2.}},
])
def test_invalid_new_options_are_rejected(override):
    options=small_options()
    options["extensions"].update(override)
    with pytest.raises(ValueError): validate_structure_options(options)


def test_feedback_actually_changes_the_later_spectral_trajectory(trained):
    wrapper, request, artifacts, _, _ = trained
    common={"feedback_only_after_accept":False}
    paths=[]
    for label, weight in (("feedback_on", .5),("feedback_off",0.)):
        result=wrapper.generate(GenerateRequest(request.run,artifacts.checkpoint_path,2,89,generation_id=label,
                               options={"extensions":{"structure_guidance":{**common,"spectrum_feedback":weight}}}))
        with (result.generation_dir/"target_adjacency_eigenvalues.pkl").open("rb") as h: paths.append(pickle.load(h))
    assert any(not np.allclose(a,b) for a,b in zip(*paths))


def test_primary_and_ablation_configs_load_through_shared_cli():
    from pathlib import Path
    from grapher.models.external_cli import _generation_options, build_parser
    import yaml
    root=Path(__file__).resolve().parents[1]
    for path in (root/"configs"/"baselines").glob("gdsm_simple_community_small_structure3*.yaml"):
        raw=yaml.safe_load(path.read_text())["gdsm_simple"]
        validate_structure_options(raw)
        args=build_parser("gdsm_simple").parse_args(["--dataset","community_small","--stage","generate","--wrapper-config",str(path),"--device","cpu"])
        generated=_generation_options("gdsm_simple",args,wrapper_config=path)
        assert generated["extensions"] == raw["extensions"]
        assert generated["sample"] == raw["sample"]
        assert set(raw)-{"train","model","diffusion","sample","generation_batch_size","extensions"} == set()
