from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.data.io import save_dataset_splits
from grapher.models.dhvae_hh.degree_vae import (
    DegreeVectorizer, build_degree_vae, load_degree_vae_checkpoint, save_degree_vae_checkpoint,
)
from grapher.rewiring_mlp.generic.joint_degree_model import (
    JointDegreeSpectralPredictor, exact_degree_inputs, degree_consistent_orbits,
    orbit_identity_residual, build_embedded_degree_sampler,
)
from grapher.rewiring_mlp.generic.joint_degree_training import build_joint_model, run_joint_epoch, _endpoints
from grapher.rewiring_mlp.generic.spectral_data import TopologySpectralExample, collate_spectral_examples
from grapher.rewiring_mlp.generic.spectral_model import (
    TopologySpectralTransformerPredictor, save_topology_spectral_checkpoint, load_topology_spectral_checkpoint,
)
from grapher.rewiring_mlp.generic.spectral import laplacian_eigenvalues
from grapher.rewiring_mlp.generic.clustering import extract_clustering_histogram
from grapher.rewiring_mlp.generic.orbit import extract_orbit_summary
from grapher.utils.io import load_yaml, save_yaml, load_pickle

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT/'configs/experiments/grapher/community_small_topology_joint_degree.yaml'


@pytest.fixture(autouse=True)
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def graphs():
    g = nx.cycle_graph(6); g.add_edge(0, 3)
    return [nx.path_graph(5), nx.cycle_graph(5), nx.path_graph(6), g]


def configuration():
    cfg = load_yaml(CFG)
    cfg['joint_degree'].update(
        initialize_degree_checkpoint=None, initialize_topology_checkpoint=None,
        freeze_epochs=1, graphs_per_batch=2, conditioning_dim=16, learning_rate=1e-3,
    )
    cfg['joint_degree']['degree_model'].update(
        latent_dim=4, hidden_dim=8, size_condition_dim=4, edge_condition_dim=4,
        prior_hidden_dim=8, num_layers=1,
    )
    cfg['topology_predictor'].update(hidden_dim=8, edge_dim=8, graph_dim=8,
        spectral_dim=16, spectral_layers=1, spectral_heads=4, spectral_ff_dim=32,
        epochs=3, learning_rate=1e-3, progress_interval=1)
    cfg['summary_diffusion'].update(samples_per_graph=2, paths_per_graph=1)
    cfg['topology_refiner'].update(steps=2, proposal_budget=32, valid_candidate_budget=8)
    cfg['generation'].update(num_generate=2, max_attempts_per_graph=8)
    cfg['evaluation'].update(compute_graphlet_history=False, compute_orbit=True)
    return cfg


def example(g):
    spectrum = laplacian_eigenvalues(g)
    return TopologySpectralExample(
        current_graph=g, time=0.3, current_spectrum=spectrum,
        source_spectrum=spectrum, clean_spectrum_target=spectrum,
        clean_clustering_histogram_target=extract_clustering_histogram(g, 100),
        clean_orbit_summary_target=extract_orbit_summary(g),
    )


def joint():
    return build_joint_model(configuration(), graphs())[0]


def test_padding_is_not_counted_and_features_match_canonical_vectorizer():
    model = joint()
    batch = collate_spectral_examples([example(g) for g in graphs()])
    x, target, totals = exact_degree_inputs(batch, model.degree_vectorizer)
    x_expected, y_expected = model.degree_vectorizer.to_training_arrays(graphs())
    np.testing.assert_allclose(x.numpy(), x_expected, atol=1e-7)
    for key, expected in y_expected.items():
        np.testing.assert_allclose(target[key].numpy(), expected, atol=1e-7)
    np.testing.assert_allclose(target['degree'].sum(-1), 1)
    assert torch.all(totals >= 0)


@pytest.mark.parametrize('g', [nx.empty_graph(1), nx.path_graph(2), nx.path_graph(4), nx.star_graph(4), nx.complete_graph(5), nx.cycle_graph(6)])
def test_nine_identities_and_nonnegativity(g):
    v = DegreeVectorizer.fit([g], max_degree=max(g.number_of_nodes()-1, 1))
    b = collate_spectral_examples([example(g)])
    _, _, totals = exact_degree_inputs(b, v)
    raw = (torch.rand((1,15))*7).requires_grad_(True)
    projected = degree_consistent_orbits(raw, totals)
    assert torch.all(projected >= 0)
    assert orbit_identity_residual(projected, totals).max() < 2e-6
    actual = torch.tensor(extract_orbit_summary(g), dtype=torch.float32).unsqueeze(0)
    assert orbit_identity_residual(actual, totals).max() < 2e-6
    projected.sum().backward()
    assert torch.isfinite(raw.grad).all()


def test_joint_structural_gradients_reach_degree_encoder_and_shared_decoder():
    model = joint()
    batch = collate_spectral_examples([example(g) for g in graphs()])
    loss, metrics = model.loss(batch, loss_weights=configuration()['topology_predictor']['loss_weights'])
    loss.backward()
    for prefix in ('degree_model.encoder', 'degree_model.mu', 'degree_model.degree_decoder', 'degree_conditioner', 'spectral_transformer'):
        total = sum(float(p.grad.abs().sum()) for name,p in model.named_parameters() if name.startswith(prefix) and p.grad is not None)
        assert total > 0, prefix
    assert model.degree_model.degree_head.weight.grad is None  # output layer only gets degree loss
    assert metrics['orbit_identity_max_abs'] < 1e-5


def test_degree_objective_updates_decoder_head_edge_head_and_learned_prior():
    model = joint()
    loss, _ = model.degree_loss(collate_spectral_examples([example(g) for g in graphs()]),
         beta=.005, weights=configuration()['joint_degree']['degree_loss_weights'])
    loss.backward()
    for prefix in ('degree_model.degree_head', 'degree_model.num_edges_head', 'degree_model.conditional_prior', 'degree_model.logvar'):
        total = sum(float(p.grad.abs().sum()) for name,p in model.named_parameters() if name.startswith(prefix) and p.grad is not None)
        assert total > 0, prefix


def test_freeze_blocks_degree_updates_but_not_adapter_gradients():
    model = joint(); model.set_degree_trainable(False); model.train()
    assert not model.degree_model.training
    loss, _ = model.loss(collate_spectral_examples([example(g) for g in graphs()]))
    loss.backward()
    assert all(p.grad is None for p in model.degree_model.parameters())
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.degree_conditioner.parameters())
    model.set_degree_trainable(True); model.train()
    assert model.degree_model.training
    assert all(p.requires_grad for p in model.degree_model.parameters())


def test_inference_conditioning_deterministic_and_permutation_invariant():
    model = joint().eval()
    g = graphs()[-1]
    a = example(g); b = deepcopy(a)
    b.current_graph = nx.relabel_nodes(g, {i:(i+3)%len(g) for i in g})
    ba, bb = collate_spectral_examples([a]), collate_spectral_examples([b])
    with torch.no_grad():
        first, second = model(ba), model(bb)
        torch.manual_seed(987)
        third = model(ba)
    for key in ('clean_spectrum','clean_clustering_histogram','clean_orbit_summary'):
        torch.testing.assert_close(first[key], second[key], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(first[key], third[key], atol=0, rtol=0)


def test_model_roundtrip_and_standalone_degree_export(tmp_path):
    model = joint().eval()
    path = tmp_path/'checkpoint.pt'
    save_topology_spectral_checkpoint(model, path, report={'val_spectral_normalized_rmse':.1})
    loaded, _, checkpoint = load_topology_spectral_checkpoint(path, device='cpu')
    assert isinstance(loaded, JointDegreeSpectralPredictor)
    assert checkpoint['joint_degree_enabled']
    batch = collate_spectral_examples([example(graphs()[-1])])
    with torch.no_grad():
        torch.testing.assert_close(model(batch)['clean_orbit_summary'], loaded(batch)['clean_orbit_summary'])
    degree_file = tmp_path/'degree_checkpoint.pt'
    save_degree_vae_checkpoint(degree_file, model.degree_model, model.degree_vectorizer)
    degree, vectorizer, _ = load_degree_vae_checkpoint(degree_file, device='cpu')
    for key,value in degree.state_dict().items():
        torch.testing.assert_close(value, model.degree_model.state_dict()[key])
    assert vectorizer.__dict__ == model.degree_vectorizer.__dict__


def test_old_checkpoint_still_loads(tmp_path):
    model = TopologySpectralTransformerPredictor(spectral_dim=16, spectral_heads=4, spectral_layers=1)
    path = tmp_path/'old.pt'; save_topology_spectral_checkpoint(model,path)
    other,_,_=load_topology_spectral_checkpoint(path,device='cpu')
    assert not getattr(other, 'joint_degree_enabled', False)
    assert other.model_config() == model.model_config()


def test_sampling_embedded_model_never_opens_external_file_and_preserves_sums():
    model = joint().eval()
    cfg = configuration()['degree_generator']
    sampler = build_embedded_degree_sampler(model, cfg, seed=42)
    assert sampler._model is model.degree_model
    rng = np.random.default_rng(43)
    for _ in range(5):
        row = sampler.sample(rng)
        degrees = row['degree_sequence']
        assert nx.is_graphical(degrees)
        assert sum(degrees)==2*row['num_edges']
        assert min(degrees)>0
    cfg['checkpoint_path']='old_degree_model.pt'
    with pytest.raises(ValueError,match='embedded'):
        build_embedded_degree_sampler(model,cfg,seed=42)


def test_support_overflow_fails_not_clips():
    model = joint()
    with pytest.raises(ValueError, match='size outside'):
        model(collate_spectral_examples([example(nx.cycle_graph(7))]))
    cfg = configuration()
    cfg['joint_degree']['max_degree']=3
    restricted,_ = build_joint_model(cfg,graphs())
    with pytest.raises(ValueError, match='Degree exceeds'):
        restricted(collate_spectral_examples([example(nx.star_graph(5))]))


def test_fresh_joint_support_covers_denser_graphs_without_changing_empirical_prior():
    model, report = build_joint_model(configuration(), graphs())
    vectorizer = model.degree_vectorizer
    assert vectorizer.max_edges == 15
    assert report['degree_edge_support']['initial_max_edges'] == 7
    assert vectorizer.empirical_edge_counts == [4, 5, 5, 7]
    assert vectorizer.empirical_node_counts == [5, 5, 6, 6]
    batch = collate_spectral_examples([example(nx.complete_graph(6))])
    _, targets, _ = exact_degree_inputs(batch, vectorizer)
    assert targets['num_edges_count'].item() == 15
    loss, metrics = model.degree_loss(
        batch, beta=.005, weights=configuration()['joint_degree']['degree_loss_weights'],
    )
    assert torch.isfinite(loss)
    assert np.isfinite(metrics['num_edges_loss'])
    # The node and degree bounds still apply, independently of edge support.
    cfg = configuration()
    cfg['joint_degree']['max_degree'] = 3
    restricted, _ = build_joint_model(cfg, graphs())
    assert restricted.degree_vectorizer.max_edges == 9


@pytest.mark.parametrize('edge_conditioning,edge_prior', [(False, False), (True, False), (True, True)])
def test_warm_start_expands_edge_support_preserving_conditioning_and_roundtrips(
    tmp_path, edge_conditioning, edge_prior,
):
    cfg = configuration()
    vectorizer = DegreeVectorizer.fit(graphs(), max_degree=5, require_connected=True)
    prior_cfg = dict(cfg['joint_degree']['degree_model'])
    prior_cfg.update(use_edge_count_conditioning=edge_conditioning, prior_condition_on_edges=edge_prior)
    source = build_degree_vae(vectorizer, **prior_cfg).eval()
    # A nontrivial learned prior exposes errors in edge-feature rescaling.
    with torch.no_grad():
        source.conditional_prior.net[-1].weight.normal_(0, 0.1)
    source_path = tmp_path / 'component.pt'
    save_degree_vae_checkpoint(source_path, source, vectorizer)
    cfg['joint_degree']['initialize_degree_checkpoint'] = str(source_path)
    model, report = build_joint_model(cfg, graphs())
    model.eval()
    expanded = model.degree_model
    assert expanded.max_edges == model.degree_vectorizer.max_edges == 15
    assert report['degree_edge_support']['expanded']
    assert model.degree_vectorizer.empirical_degree_sequences == vectorizer.empirical_degree_sequences
    z, n, m = torch.randn(2, source.latent_dim), torch.tensor([5, 6]), torch.tensor([5, 7])
    with torch.no_grad():
        before = source.decode(z, n, m, return_hidden=True)
        after = expanded.decode(z, n, m, return_hidden=True)
        for key in ('degree_hidden', 'degree_logits', 'num_nodes_logits'):
            torch.testing.assert_close(after[key], before[key])
        before_prior, after_prior = source.prior_parameters(n, m), expanded.prior_parameters(n, m)
        for key in before_prior:
            torch.testing.assert_close(after_prior[key], before_prior[key])
        if edge_conditioning:
            torch.testing.assert_close(after['num_edges_logits'][:, :8], before['num_edges_logits'], atol=0, rtol=0)
            assert torch.all(after['num_edges_logits'].softmax(-1)[:, 8:].sum(-1) <= 1.01e-6)
            torch.testing.assert_close(
                after['num_edges_logits'].softmax(-1)[:, :8],
                before['num_edges_logits'].softmax(-1), atol=1e-6, rtol=1e-6,
            )
    dense = collate_spectral_examples([example(nx.complete_graph(6))])
    loss, _ = model.degree_loss(
        dense, beta=.005, weights=cfg['joint_degree']['degree_loss_weights'],
    )
    assert torch.isfinite(loss)
    loss.backward()
    if edge_conditioning:
        assert torch.isfinite(expanded.num_edges_head.bias.grad).all()
        assert expanded.num_edges_head.bias.grad[15] < 0
    checkpoint = tmp_path / 'joint.pt'
    save_topology_spectral_checkpoint(model, checkpoint)
    loaded, _, _ = load_topology_spectral_checkpoint(checkpoint, device='cpu')
    export = tmp_path / 'export.pt'
    save_degree_vae_checkpoint(export, expanded, model.degree_vectorizer)
    standalone, exported_vectorizer, _ = load_degree_vae_checkpoint(export, device='cpu')
    assert exported_vectorizer.max_edges == 15
    with torch.no_grad():
        torch.testing.assert_close(loaded(dense)['clean_spectrum'], model(dense)['clean_spectrum'])
        torch.testing.assert_close(standalone.decode(z, n, m)['degree_logits'], after['degree_logits'])
    # Migration never rewrites the original component checkpoint.
    original, original_vectorizer, _ = load_degree_vae_checkpoint(source_path, device='cpu')
    assert original.max_edges == original_vectorizer.max_edges == 7
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[key], atol=0, rtol=0)


def test_graph_balanced_degree_loss_evaluations_do_not_multiply_with_bridge_views():
    cfg=configuration();model=joint();eps=_endpoints(graphs(),cfg,42)
    for views in [1,4]:
        cfg['summary_diffusion'].update(samples_per_graph=views,paths_per_graph=2)
        torch.manual_seed(52)
        metrics=run_joint_epoch(model,eps,config=cfg,epoch=0,seed=42,device=torch.device('cpu'))
        assert metrics['degree_loss_evaluations']==len(graphs())
        assert metrics['num_bridge_views']==len(graphs())*views*2
        if views==1:
            first_degree=metrics['degree_loss']
        else:
            assert metrics['degree_loss']==pytest.approx(first_degree,abs=1e-8)


def test_component_warm_start_and_stale_prior_guard(tmp_path):
    cfg=configuration();m=joint()
    d=tmp_path/'degree.pt'; save_degree_vae_checkpoint(d,m.degree_model,m.degree_vectorizer)
    base_cfg=m.model_config();base_cfg.pop('joint_degree_config')
    base=TopologySpectralTransformerPredictor(**base_cfg)
    t=tmp_path/'topology.pt'; save_topology_spectral_checkpoint(base,t)
    cfg['joint_degree'].update(initialize_degree_checkpoint=str(d), initialize_topology_checkpoint=str(t))
    loaded,report=build_joint_model(cfg,graphs())
    torch.testing.assert_close(loaded.gap_head[0].weight,base.gap_head[0].weight)
    torch.testing.assert_close(loaded.degree_model.degree_head.weight,m.degree_model.degree_head.weight)
    assert report['topology_checkpoint']==str(t)
    with pytest.raises(ValueError,match='degree multisets differ'):
        build_joint_model(cfg,[nx.cycle_graph(5)]*4)


@pytest.mark.parametrize('warm_degree', [False, True])
def test_end_to_end_cli_train_generate_diagnose_evaluate(tmp_path,monkeypatch,warm_degree):
    from scripts import train_topology_grapher as train
    from scripts import run_topology_grapher as generate
    from scripts import diagnose_spectral_denoiser as diagnose
    from scripts import evaluate_graph_generation_report as evaluate
    cfg=configuration()
    if warm_degree:
        vectorizer = DegreeVectorizer.fit(graphs(), max_degree=5, require_connected=True)
        degree = build_degree_vae(vectorizer, **cfg['joint_degree']['degree_model'])
        component = tmp_path / 'degree_component.pt'
        save_degree_vae_checkpoint(component, degree, vectorizer)
        cfg['joint_degree']['initialize_degree_checkpoint'] = str(component)
    # Held-out edge count 10 exceeds the old training-only support of 7.
    root=tmp_path/'data';save_dataset_splits('joint_fixture', {'train':graphs(), 'val':[nx.wheel_graph(6)], 'test':[nx.path_graph(5)]}, {}, root)
    cfg['dataset'].update(name='joint_fixture',root=str(root),config_path=None)
    train_dir=tmp_path/'train'; gen_dir=tmp_path/'generated'
    cfg['topology_predictor']['checkpoint_path']=str(train_dir/'checkpoint.pt')
    config_file=tmp_path/'config.yaml';save_yaml(cfg,config_file)
    monkeypatch.setattr(sys,'argv',['train','--config',str(config_file),'--output-dir',str(train_dir),'--seed','42','--device','cpu'])
    train.main()
    report=json.loads((train_dir/'report.json').read_text())
    assert report['best_epoch'] > cfg['joint_degree']['freeze_epochs']
    assert report['warm_start']['degree_edge_support']['initial_max_edges'] == 7
    assert report['warm_start']['degree_edge_support']['max_edges'] == 15
    model,_,cp=load_topology_spectral_checkpoint(train_dir/'checkpoint.pt',device='cpu')
    assert cp['report']['degree_parameter_l2_change_from_initialization']>0
    assert cp['report']['val_orbit_identity_max_abs'] < 1e-5
    # Generation must not need the separately exported degree checkpoint.
    (train_dir/'degree_checkpoint.pt').rename(train_dir/'unused_export.pt')
    monkeypatch.setattr(sys,'argv',['generate','--config',str(config_file),'--output-dir',str(gen_dir),'--num-generate','2','--seed','42','--device','cpu'])
    generate.main()
    r=json.loads((gen_dir/'report.json').read_text())
    assert r['degree_source']=='learned'
    assert r['degree_sampler_source']=='joint_checkpoint_embedded'
    assert r['diagnostics']['degree_preservation_rate']==1.0
    assert r['diagnostics']['connectedness_rate']==1.0
    a=load_pickle(gen_dir/'coarse_graphs.pkl'); b=load_pickle(gen_dir/'topology_refined_graphs.pkl')
    assert len(a)==len(b)==2
    assert all(dict(x.degree())==dict(y.degree()) for x,y in zip(a,b))
    monkeypatch.setattr(sys,'argv',['diagnose','--config',str(config_file),'--checkpoint',str(train_dir/'checkpoint.pt'),'--source-endpoint-only','--samples-per-graph','1','--paths-per-graph','1','--device','cpu','--json-out',str(tmp_path/'diagnostic.json')])
    diagnose.main()
    assert (tmp_path/'diagnostic.json').exists()
    monkeypatch.setattr(sys,'argv',['evaluate','--config',str(config_file),'--generated-dir',str(gen_dir),'--reference-split','val','--output-dir',str(gen_dir/'evaluation_val')])
    evaluate.main()
    assert (gen_dir/'evaluation_val'/'graph_evaluation_report.json').exists()
    # Same checkpoint supports empirical degree conditions too.
    monkeypatch.setattr(sys,'argv',['generate','--config',str(config_file),'--output-dir',str(tmp_path/'empirical'),'--num-generate','2','--device','cpu','--set','generation.degree_source=train_empirical','--set','topology_refiner.steps=0'])
    generate.main()
    assert json.loads((tmp_path/'empirical'/'report.json').read_text())['degree_sampler_source']=='train_empirical'

    # Existing DH-VAE evaluator can read the exact best-checkpoint degree export.
    from scripts import evaluate_degree_generator as eval_degree
    degree_out=tmp_path/'degree_evaluation'
    monkeypatch.setattr(sys,'argv',['degree-evaluate','--config',str(config_file),
        '--checkpoint',str(train_dir/'unused_export.pt'),'--num-samples','8',
        '--batch-size','4','--seed','42','--device','cpu','--output-dir',str(degree_out)])
    eval_degree.main()
    assert (degree_out/'degree_evaluation.json').exists()
    # Read-only structural-shift audit recognizes the new checkpoint subtype.
    from scripts import diagnose_degree_structural_shift as audit
    monkeypatch.setattr(sys,'argv',['audit','--config',str(config_file),
        '--checkpoint',str(train_dir/'checkpoint.pt'),
        '--empirical-dir',str(tmp_path/'empirical'),'--learned-dir',str(gen_dir),
        '--reference-split','val','--histogram-feasibility','--device','cpu',
        '--output-dir',str(tmp_path/'audit')])
    audit.main()
    ar=json.loads((tmp_path/'audit'/'degree_structural_audit.json').read_text())
    assert max(row['orbit_identity_scaled_residual'] for row in ar['graphs']['learned']) < 1e-5
    # Dataset replacement is rejected at generation, before any new samples.
    save_dataset_splits('joint_fixture', {'train':graphs(), 'val':[nx.path_graph(6)], 'test':[nx.path_graph(5)]}, {}, root)
    monkeypatch.setattr(sys,'argv',['generate','--config',str(config_file),'--output-dir',str(tmp_path/'stale'),
                                  '--num-generate','1','--device','cpu'])
    with pytest.raises(ValueError,match='fingerprint mismatch'):
        generate.main()


def test_orbit_identity_parameterization_is_identity_on_actual_atlas_vectors():
    # Graph atlas tests constrain the algebra on many graphlet combinations,
    # not only the handful of hand-written graphs above.
    for g in nx.graph_atlas_g():
        if not g.number_of_nodes():
            continue
        degrees=torch.tensor([d for _,d in g.degree()],dtype=torch.float64)
        n=len(degrees)
        totals=torch.stack([degrees.mean(),(degrees*(degrees-1)/2).sum()/n,
                            (degrees*(degrees-1)*(degrees-2)/6).sum()/n]).unsqueeze(0)
        actual=torch.tensor(extract_orbit_summary(g),dtype=torch.float64).unsqueeze(0)
        constrained=degree_consistent_orbits(actual,totals)
        torch.testing.assert_close(constrained,actual,atol=1e-10,rtol=1e-10)
        assert orbit_identity_residual(constrained,totals).max() < 1e-10


def test_none_consistency_ablation_retains_raw_orbit_output():
    cfg=configuration();cfg['joint_degree']['orbit_consistency']='none'
    model,_=build_joint_model(cfg,graphs())
    out=model(collate_spectral_examples([example(graphs()[-1])]))
    torch.testing.assert_close(out['clean_orbit_summary'],out['unconstrained_orbit_summary'])


def test_dhvae_decode_does_not_change_legacy_outputs_when_hidden_requested():
    model=joint().degree_model.eval()
    z=torch.randn(2,model.latent_dim);n=torch.tensor([5,6]);m=torch.tensor([5,7])
    a=model.decode(z,n,m);b=model.decode(z,n,m,return_hidden=True)
    assert 'degree_hidden' not in a
    for key in a:
        torch.testing.assert_close(a[key],b[key])
    torch.testing.assert_close(model.degree_head(b['degree_hidden']).masked_fill(
        torch.arange(model.max_degree+1)[None,:]>=n[:,None],-1e9),a['degree_logits'])
