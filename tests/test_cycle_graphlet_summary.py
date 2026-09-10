from __future__ import annotations

from dataclasses import replace
from math import comb

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.rewiring_mlp.generic.cycle_graphlets import (
    count_cycle_graphlets, cycle_graphlet_k, cycle_graphlet_histogram_distance,
    cycle_histogram_from_count, extract_cycle_graphlet_histogram,
    validate_cycle_graphlet_histogram, validate_cycle_graphlet_k,
)
from grapher.rewiring_mlp.generic.clustering import extract_clustering_histogram
from grapher.rewiring_mlp.generic.orbit import extract_orbit_summary
from grapher.rewiring_mlp.generic.rewiring import propose_valid_topology_swaps
from grapher.rewiring_mlp.generic.spectral import laplacian_eigenvalues
from grapher.rewiring_mlp.generic.spectral_data import (
    TopologySpectralExample, TopologySpectralDiffusionIterableDataset,
    build_spectral_diffusion_examples, collate_spectral_examples,
)
from grapher.rewiring_mlp.generic.spectral_model import (
    TopologySpectralTransformerPredictor, save_topology_spectral_checkpoint,
    load_topology_spectral_checkpoint,
)
from grapher.rewiring_mlp.generic.spectral_refiner import (
    SpectralPrediction, SpectralRefinerConfig, score_spectral_candidates,
    predict_clean_spectrum, refine_graph_with_spectral_predictions,
)


def graph():
    return nx.Graph([(0,1),(0,2),(0,3),(1,2),(2,4),(3,5),(4,5)])


def example(g):
    spec = laplacian_eigenvalues(g)
    return TopologySpectralExample(
        current_graph=g, time=.4, current_spectrum=spec, source_spectrum=spec,
        clean_spectrum_target=spec,
        clean_cycle_graphlet_histogram_target=extract_cycle_graphlet_histogram(g),
    )


def model(enabled=True):
    return TopologySpectralTransformerPredictor(
        hidden_dim=8, edge_dim=8, graph_dim=8, num_layers=1,
        spectral_dim=16, spectral_layers=1, spectral_heads=4, spectral_ff_dim=32,
        use_graph_context=False, predict_cycle_graphlet_histogram=enabled,
    )


@pytest.mark.parametrize('g,count', [
    (nx.empty_graph(0),0), (nx.empty_graph(2),0), (nx.path_graph(3),0),
    (nx.cycle_graph(3),1), (nx.cycle_graph(4),0), (nx.complete_graph(4),4),
    (nx.complete_bipartite_graph(3,3),0), (graph(),1),
])
def test_exact_triangle_counts_and_nondegenerate_histogram(g,count):
    assert count_cycle_graphlets(g)==count
    hist=extract_cycle_graphlet_histogram(g)
    expected=count/comb(len(g),3) if len(g)>=3 else 0
    np.testing.assert_allclose(hist,[expected,1-expected])
    assert hist.sum()==pytest.approx(1)



def test_c4_c5_count_induced_chordless_cycles():
    for k in (4, 5):
        cycle = nx.cycle_graph(k)
        assert count_cycle_graphlets(cycle, k=k) == 1
        np.testing.assert_allclose(extract_cycle_graphlet_histogram(cycle, k=k), [1.0, 0.0])

        chorded = cycle.copy()
        chorded.add_edge(0, 2)
        assert count_cycle_graphlets(chorded, k=k) == 0
        np.testing.assert_allclose(extract_cycle_graphlet_histogram(chorded, k=k), [0.0, 1.0])

    c5_plus_isolate = nx.cycle_graph(5)
    c5_plus_isolate.add_node(5)
    assert count_cycle_graphlets(c5_plus_isolate, k=5) == 1
    np.testing.assert_allclose(
        extract_cycle_graphlet_histogram(c5_plus_isolate, k=5),
        [1 / comb(6, 5), 1 - 1 / comb(6, 5)],
    )


def test_c5_relabel_invariance():
    g = nx.Graph([(0,1),(0,3),(0,4),(1,5),(2,3),(2,5)])
    renamed = nx.relabel_nodes(g, {v: f"v{v}" for v in g})
    assert count_cycle_graphlets(g, k=5) == 1
    assert count_cycle_graphlets(renamed, k=5) == 1
    np.testing.assert_array_equal(
        extract_cycle_graphlet_histogram(g, k=5),
        extract_cycle_graphlet_histogram(renamed, k=5),
    )

def test_c3_counts_all_triangles_not_cycle_basis_and_residual_includes_paths():
    g=nx.complete_graph(4)
    assert count_cycle_graphlets(g)==4  # basis has only 3 cycles
    g=nx.path_graph(3)
    np.testing.assert_allclose(extract_cycle_graphlet_histogram(g),[0,1])
    one_triangle=nx.cycle_graph(3);one_triangle.add_node(3)
    assert extract_cycle_graphlet_histogram(one_triangle)[0]==.25
    assert extract_cycle_graphlet_histogram(nx.complete_graph(4))[0]==1


def test_relabel_invariance_and_orbit_triangle_identity():
    for g in (graph(), nx.complete_graph(6), nx.cycle_graph(6), nx.gnp_random_graph(10,.4,seed=4)):
        renamed=nx.relabel_nodes(g,{v:f'node_{v}' for v in g})
        np.testing.assert_array_equal(extract_cycle_graphlet_histogram(g),extract_cycle_graphlet_histogram(renamed))
        assert len(g)*extract_orbit_summary(g)[3]/3==pytest.approx(count_cycle_graphlets(g))


@pytest.mark.parametrize('k',[3,4,5])
def test_accept_supported_cycle_sizes(k):
    assert validate_cycle_graphlet_k(k) == k


@pytest.mark.parametrize('k',[2,6,3.0,True,'3',None])
def test_reject_unsupported_sizes(k):
    with pytest.raises(ValueError,match=r'\{3,4,5\}'):
        validate_cycle_graphlet_k(k)


@pytest.mark.parametrize('values',[[1], [.1,.2,.7],[-.1,1.1],[.2,.3],[np.nan,1],[[.5,.5]]])
def test_histogram_validation(values):
    with pytest.raises(ValueError):
        validate_cycle_graphlet_histogram(values)


def test_config_and_distance():
    assert cycle_graphlet_k({}) is None
    assert cycle_graphlet_k({'cycle_graphlet_histogram':True})==3
    assert cycle_graphlet_histogram_distance([.2,.8],[.5,.5])==pytest.approx(.3)
    with pytest.raises(ValueError):cycle_histogram_from_count(5,4)
    with pytest.raises(ValueError):count_cycle_graphlets(nx.DiGraph([(0,1)]))
    with pytest.raises(ValueError):count_cycle_graphlets(nx.Graph([(0,0)]))


@pytest.mark.parametrize('cache',[True,False])
def test_streaming_clean_target_constant_and_not_diffused(cache):
    g=graph()
    dataset=TopologySpectralDiffusionIterableDataset(
        [g], diffusion_config={'samples_per_graph':3,'paths_per_graph':2,'cache_endpoints':cache},
        source_config={'random_relabel_source':False,'ensure_connected_source':True},
        structure_summary_config={'cycle_graphlet_histogram':True,'cycle_graphlet_k':3},
        seed=42,
    )
    for epoch in (0,1):
        dataset.set_epoch(epoch)
        rows=list(dataset);assert len(rows)==6
        for row in rows:
            np.testing.assert_allclose(row.clean_cycle_graphlet_histogram_target,extract_cycle_graphlet_histogram(g))
            assert row.current_graphlet_logits is None
        batch=collate_spectral_examples(rows)
        assert batch.clean_cycle_graphlet_histogram_target.shape==(6,2)
        assert batch.to('cpu').clean_cycle_graphlet_histogram_target is not None


def test_materialized_targets_and_mixed_target_rejection():
    rows,_=build_spectral_diffusion_examples(
        [graph()],diffusion_config={'samples_per_graph':3},
        structure_summary_config={'cycle_graphlet_histogram':True},seed=4,
    )
    assert all(row.clean_cycle_graphlet_histogram_target is not None for row in rows)
    with pytest.raises(ValueError,match='mix'):
        collate_spectral_examples([rows[0], replace(rows[0],clean_cycle_graphlet_histogram_target=None)])


def test_head_gradient_and_no_clean_target_leakage():
    m=model();b=collate_spectral_examples([example(graph()),example(nx.complete_graph(4))])
    output=m(b)
    torch.testing.assert_close(output['clean_cycle_graphlet_histogram'].sum(-1),torch.ones(2))
    loss,metrics=m.loss(b,loss_weights={'spectrum':0,'moment2':0,'cycle_graphlet_histogram':1})
    assert np.isfinite(metrics['cycle_graphlet_histogram_tv'])
    loss.backward()
    for p in (m.cycle_graphlet_histogram_head[-1].weight,m.spectral_token_in[0].weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    m.eval()
    altered=replace(b,clean_cycle_graphlet_histogram_target=1-b.clean_cycle_graphlet_histogram_target)
    torch.testing.assert_close(m(b)['clean_cycle_graphlet_histogram'],m(altered)['clean_cycle_graphlet_histogram'])


def test_small_graph_placeholder_masked_out_of_loss_and_prediction():
    m=model();b=collate_spectral_examples([example(nx.path_graph(2))])
    out=m(b)['clean_cycle_graphlet_histogram'];torch.testing.assert_close(out,torch.tensor([[0.,1.]]))
    loss,metrics=m.loss(b,loss_weights={'spectrum':0,'moment2':0,'cycle_graphlet_histogram':1,'cycle_graphlet_histogram_ce':1})
    assert loss.item()==0 and metrics['cycle_graphlet_valid_fraction']==0
    loss.backward()
    assert m.cycle_graphlet_histogram_head[-1].weight.grad.abs().sum()==0


def test_checkpoint_roundtrip_old_checkpoint_and_predict(tmp_path):
    p=tmp_path/'cycle.pt';m=model().eval()
    save_topology_spectral_checkpoint(m,p)
    loaded,_,_=load_topology_spectral_checkpoint(p,device='cpu')
    assert loaded.predict_cycle_graphlet_histogram and loaded.cycle_graphlet_k==3
    torch.testing.assert_close(m(collate_spectral_examples([example(graph())]))['clean_cycle_graphlet_histogram'],loaded(collate_spectral_examples([example(graph())]))['clean_cycle_graphlet_histogram'])
    assert predict_clean_spectrum(loaded,graph(),time=0,device='cpu').clean_cycle_graphlet_histogram.shape==(2,)
    save_topology_spectral_checkpoint(model(False),p)
    state=torch.load(p,weights_only=False)
    state['model_config'].pop('predict_cycle_graphlet_histogram');state['model_config'].pop('cycle_graphlet_k')
    torch.save(state,p)
    loaded,_,_=load_topology_spectral_checkpoint(p,device='cpu')
    assert not loaded.predict_cycle_graphlet_histogram and loaded.cycle_graphlet_histogram_head is None


def fixture_guidance(mode='cycle'):
    g=graph()
    candidates,candidate_graphs,_=propose_valid_topology_swaps(g,proposal_budget=-1,valid_candidate_budget=-1,preserve_connectivity=True,rng=np.random.default_rng(1))
    target=next(c for c in candidate_graphs.values() if count_cycle_graphlets(c)!=count_cycle_graphlets(g))
    spectrum=laplacian_eigenvalues(target)
    cfg=SpectralRefinerConfig.from_dict({
        'steps':2,'proposal_budget':-1,'valid_candidate_budget':-1,'guidance_mode':mode,
        'spectral_guidance':{'weight':1 if 'spectral' in mode else 0,'min_clean_mix':1,'max_clean_mix':1,'expand_on_plateau':False},
        'clustering_guidance':{'statistic':'histogram','weight':.25,'histogram_bins':100},
        'orbit_guidance':{'weight':1},'cycle_guidance':{'k':3,'distance':'total_variation','weight':.25},
        'candidate_search':{'compute_spectral_diagnostics':False},
        'prediction_horizon':{'mode':'fixed','k':1,'refresh_on_plateau':False},
    })
    kwargs={'clean_spectrum':spectrum,'next_spectrum_target':spectrum,
        'clean_clustering_histogram':extract_clustering_histogram(target,100),
        'clean_orbit_summary':extract_orbit_summary(target),
        'clean_cycle_graphlet_histogram':extract_cycle_graphlet_histogram(target),
        'candidate_graphs':candidate_graphs}
    return g,candidates,cfg,kwargs


@pytest.mark.parametrize('mode',['cycle','clustering_cycle','orbit_cycle','clustering_orbit_cycle','spectral_clustering_orbit_cycle'])
def test_combined_score_exact_and_spectra_optional(mode):
    g,actions,cfg,kwargs=fixture_guidance(mode)
    rows=score_spectral_candidates(g,actions,config=cfg,**kwargs)
    assert rows
    for row in rows:
        components=mode.split('_')
        expected=sum(getattr(cfg,c+'_weight')*row[c+'_relative_improvement'] for c in components)
        assert row['energy_improvement']==pytest.approx(expected)
        assert row['current_cycle_discrepancy']>=0
        if 'spectral' not in mode:assert row['candidate_spectrum'] is None
    eager=score_spectral_candidates(g,actions,config=replace(cfg,compute_candidate_spectral_diagnostics=True),**kwargs)
    np.testing.assert_allclose([r['energy_improvement'] for r in eager],[r['energy_improvement'] for r in rows])


def test_cycle_oracle_rewiring_preserves_degrees_and_connectivity():
    g,actions,cfg,kwargs=fixture_guidance()
    target=kwargs['clean_cycle_graphlet_histogram'];spec=kwargs['clean_spectrum']
    def oracle(_model,current,**kw):
        return SpectralPrediction(spec,laplacian_eigenvalues(current),float(spec.sum()),float(spec@spec),clean_cycle_graphlet_histogram=target)
    final,trace=refine_graph_with_spectral_predictions(g,model=None,refiner_config=cfg,prediction_fn=oracle,return_trace=True)
    assert dict(final.degree())==dict(g.degree()) and nx.is_connected(final)
    assert cycle_graphlet_histogram_distance(extract_cycle_graphlet_histogram(final),target)==pytest.approx(0)
    accepted=[r for r in trace if r.get('accepted')]
    assert accepted and all(r['cycle_gain']>0 for r in accepted)
    assert all(r['scoring_components']==['cycle'] for r in accepted)



def test_c5_cycle_guidance_scores_induced_five_cycles():
    g = nx.Graph([(0,1),(0,3),(0,4),(1,5),(2,3),(2,5)])
    candidates, candidate_graphs, _ = propose_valid_topology_swaps(
        g, proposal_budget=-1, valid_candidate_budget=-1, preserve_connectivity=True,
        rng=np.random.default_rng(1),
    )
    current_count = count_cycle_graphlets(g, k=5)
    target = next(
        candidate for candidate in candidate_graphs.values()
        if count_cycle_graphlets(candidate, k=5) != current_count
    )
    spectrum = laplacian_eigenvalues(target)
    cfg = SpectralRefinerConfig.from_dict({
        'steps': 2, 'proposal_budget': -1, 'valid_candidate_budget': -1,
        'guidance_mode': 'cycle',
        'cycle_guidance': {'k': 5, 'distance': 'total_variation', 'weight': .1},
        'spectral_guidance': {'weight': 0, 'min_clean_mix': 1, 'max_clean_mix': 1, 'expand_on_plateau': False},
        'candidate_search': {'compute_spectral_diagnostics': False},
        'prediction_horizon': {'mode': 'fixed', 'k': 1, 'refresh_on_plateau': False},
    })
    rows = score_spectral_candidates(
        g, candidates, clean_spectrum=spectrum, next_spectrum_target=spectrum,
        clean_cycle_graphlet_histogram=extract_cycle_graphlet_histogram(target, k=5),
        config=cfg, candidate_graphs=candidate_graphs,
    )
    assert rows
    assert any(abs(row['cycle_gain']) > 0 for row in rows)
    assert all(row['cycle_graphlet_k'] == 5 for row in rows)


def test_missing_cycle_checkpoint_head_fails_in_scoring():
    g,actions,cfg,kwargs=fixture_guidance()
    kwargs.pop('clean_cycle_graphlet_histogram')
    with pytest.raises(ValueError,match='checkpoint'):
        score_spectral_candidates(g,actions,config=cfg,**kwargs)


def test_invalid_cycle_config_fails():
    for spec in ({'k':6},{'distance':'rmse'},{'weight':0},{'weight':float('nan')}):
        with pytest.raises(ValueError):SpectralRefinerConfig.from_dict({'guidance_mode':'cycle','cycle_guidance':spec})


def test_cycle3_train_checkpoint_generate_diagnose_evaluate_smoke(tmp_path,monkeypatch):
    import json
    import sys
    from pathlib import Path
    from grapher.data.io import save_dataset_splits
    from grapher.utils.io import load_pickle,load_yaml,save_yaml
    from scripts import train_topology_grapher as train
    from scripts import run_topology_grapher as generate
    from scripts import diagnose_spectral_denoiser as diagnose
    from scripts import evaluate_graph_generation_report as evaluate

    repo=Path(__file__).resolve().parents[1]
    config=load_yaml(repo/'configs/experiments/grapher/community_small_topology_spectral_clustering_orbit_cycle3.yaml')
    dataset=tmp_path/'datasets'
    save_dataset_splits('tinycycle',{'train':[graph(),nx.cycle_graph(6),nx.complete_graph(4)],'val':[graph(),nx.cycle_graph(6)],'test':[graph(),nx.complete_graph(4)]},{},dataset)
    config['benchmark']='tinycycle'
    config['dataset']={'name':'tinycycle','root':str(dataset),'build_if_missing':False}
    config['summary_diffusion'].update(samples_per_graph=2,paths_per_graph=1,storage='streaming')
    config['topology_predictor'].update(epochs=2,batch_size=2,hidden_dim=8,edge_dim=8,graph_dim=8,spectral_dim=16,spectral_layers=1,spectral_heads=4,spectral_ff_dim=32)
    config['topology_refiner'].update(steps=2,proposal_budget=32,valid_candidate_budget=8)
    cfg=tmp_path/'cycle3.yaml';save_yaml(config,cfg)
    train_dir=tmp_path/'train'
    monkeypatch.setattr(sys,'argv',['train','--config',str(cfg),'--output-dir',str(train_dir),'--seed','42','--device','cpu'])
    train.main()
    report=json.loads((train_dir/'training_report.json').read_text())
    assert 'cycle_graphlet_histogram' in report['active_losses']
    assert report['predictor_targets']['cycle_graphlet_k']==3
    assert report['predictor_targets']['cycle_graphlet_histogram_is_diffused'] is False
    assert np.isfinite(report['history'][0]['val_cycle_graphlet_histogram_tv'])
    ckpt=train_dir/'checkpoint.pt'

    gen=tmp_path/'gen'
    monkeypatch.setattr(sys,'argv',['generate','--config',str(cfg),'--checkpoint',str(ckpt),'--output-dir',str(gen),'--num-generate','2','--seed','42','--device','cpu'])
    generate.main()
    generated=json.loads((gen/'report.json').read_text());diag=generated['diagnostics']
    assert diag['guidance_mode']=='clustering_orbit_cycle'
    assert diag['scoring_components']==['clustering','cycle','orbit']
    assert diag['cycle_guidance_weight']==.25
    assert diag['source_enrichment_enabled'] is False
    assert diag['predictor_cycle_graphlet_histogram_enabled'] is True
    source=load_pickle(gen/'coarse_graphs.pkl');final=load_pickle(gen/'topology_refined_graphs.pkl')
    assert len(source)==len(final)==2
    assert all(dict(a.degree())==dict(b.degree()) and nx.is_connected(b) for a,b in zip(source,final))
    if diag['accepted_cycle_measurements']:
        assert diag['mean_accepted_cycle_gain'] is not None

    # Demonstrate all existing objectives can reuse the new checkpoint.
    for mode in ('clustering_orbit','cycle'):
        out=tmp_path/mode
        monkeypatch.setattr(sys,'argv',['generate','--config',str(cfg),'--checkpoint',str(ckpt),'--output-dir',str(out),'--num-generate','1','--device','cpu','--set',f'topology_refiner.guidance_mode={mode}'])
        generate.main()
        diag2=json.loads((out/'report.json').read_text())['diagnostics']
        assert diag2['guidance_mode']==mode
        if mode=='clustering_orbit':
            assert diag2['cycle_guidance_weight']==0 and diag2['mean_accepted_cycle_gain'] is None

    # Explicit source endpoint and noisy-bridge diagnostics both exercise the
    # actual CLI, not manually constructed toy model inputs.
    for endpoint in (False,True):
        out=tmp_path/f'diagnostic_{endpoint}.json'
        argv=['diagnose','--config',str(cfg),'--checkpoint',str(ckpt),'--split','val','--samples-per-graph','1','--paths-per-graph','1','--device','cpu','--json-out',str(out)]
        if endpoint:argv+=['--source-endpoint-only']
        monkeypatch.setattr(sys,'argv',argv);diagnose.main()
        r=json.loads(out.read_text())
        assert r['cycle_graphlet_k']==3
        assert np.isfinite(r['overall']['cycle_graphlet_histogram_tv'])
        assert np.isfinite(r['overall']['cycle_orbit_triangle_count_gap'])

    evaluation=tmp_path/'evaluation_val'
    monkeypatch.setattr(sys,'argv',['evaluate','--config',str(cfg),'--generated-dir',str(gen),'--reference-split','val','--output-dir',str(evaluation),'--dpi','40'])
    evaluate.main()
    csv=(evaluation/'graph_mmd_metrics.csv').read_text()
    assert 'train_to_val' in csv and 'topology_final_to_val' in csv


def test_generation_rejects_old_checkpoint_for_cycle_guidance_before_sampling(tmp_path,monkeypatch):
    import sys
    from pathlib import Path
    from grapher.data.io import save_dataset_splits
    from grapher.utils.io import load_yaml,save_yaml
    from scripts import run_topology_grapher as generate
    cfg=load_yaml(Path(__file__).resolve().parents[1]/'configs/experiments/grapher/community_small_topology_spectral_clustering_orbit_cycle3.yaml')
    data=tmp_path/'data';save_dataset_splits('tiny',{s:[graph()] for s in ('train','val','test')},{},data)
    cfg['benchmark']='tiny';cfg['dataset']={'name':'tiny','root':str(data),'build_if_missing':False}
    cfg['topology_refiner']['guidance_mode']='cycle'
    path=tmp_path/'config.yaml';save_yaml(cfg,path)
    checkpoint=tmp_path/'old.pt';save_topology_spectral_checkpoint(model(False),checkpoint, report={'val_spectral_normalized_rmse': .1})
    monkeypatch.setattr(sys,'argv',['generate','--config',str(path),'--checkpoint',str(checkpoint),'--output-dir',str(tmp_path/'bad_gen'),'--num-generate','1','--device','cpu'])
    with pytest.raises(ValueError,match='no cycle graphlet histogram head'):
        generate.main()


def test_cycle5_train_generate_and_diagnose_smoke(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path

    from grapher.data.io import save_dataset_splits
    from grapher.utils.io import load_pickle, load_yaml, save_yaml
    from scripts import diagnose_spectral_denoiser as diagnose
    from scripts import run_topology_grapher as generate
    from scripts import train_topology_grapher as train

    repo = Path(__file__).resolve().parents[1]
    config = load_yaml(
        repo / "configs/experiments/grapher/community_small_topology_spectral_clustering_orbit_cycle5.yaml"
    )
    # All graphs contain at least five nodes so the C5 target is active.
    g0 = nx.Graph([(0,1),(0,3),(0,4),(1,5),(2,3),(2,5)])
    g1 = nx.cycle_graph(6)
    g2 = nx.complete_bipartite_graph(3, 3)
    dataset = tmp_path / "datasets"
    save_dataset_splits(
        "tinycycle5",
        {"train": [g0, g1, g2], "val": [g0, g1], "test": [g0, g2]},
        {},
        dataset,
    )
    config["benchmark"] = "tinycycle5"
    config["dataset"] = {"name": "tinycycle5", "root": str(dataset), "build_if_missing": False}
    config["summary_diffusion"].update(samples_per_graph=1, paths_per_graph=1, storage="streaming")
    config["topology_predictor"].update(
        epochs=1, batch_size=2, hidden_dim=8, edge_dim=8, graph_dim=8,
        spectral_dim=16, spectral_layers=1, spectral_heads=4, spectral_ff_dim=32,
    )
    config["topology_refiner"].update(steps=1, proposal_budget=16, valid_candidate_budget=4)
    path = tmp_path / "cycle5.yaml"
    save_yaml(config, path)

    train_dir = tmp_path / "train"
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--config", str(path), "--output-dir", str(train_dir), "--seed", "42", "--device", "cpu"],
    )
    train.main()
    report = json.loads((train_dir / "training_report.json").read_text())
    assert report["predictor_targets"]["cycle_graphlet_k"] == 5
    assert report["predictor_targets"]["cycle_graphlet_histogram_normalization"] == "all_node_5_subsets"

    ckpt = train_dir / "checkpoint.pt"
    gen = tmp_path / "gen"
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate", "--config", str(path), "--checkpoint", str(ckpt), "--output-dir", str(gen),
         "--num-generate", "1", "--seed", "42", "--device", "cpu"],
    )
    generate.main()
    generated = json.loads((gen / "report.json").read_text())
    diag = generated["diagnostics"]
    assert diag["guidance_mode"] == "clustering_orbit_cycle"
    assert diag["cycle_guidance_k"] == 5
    assert diag["cycle_guidance_weight"] == pytest.approx(.1)
    assert "cycle" in diag["scoring_components"]
    source = load_pickle(gen / "coarse_graphs.pkl")
    final = load_pickle(gen / "topology_refined_graphs.pkl")
    assert dict(source[0].degree()) == dict(final[0].degree())

    out = tmp_path / "diagnose.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["diagnose", "--config", str(path), "--checkpoint", str(ckpt), "--split", "val",
         "--source-endpoint-only", "--samples-per-graph", "1", "--paths-per-graph", "1",
         "--device", "cpu", "--json-out", str(out)],
    )
    diagnose.main()
    diagnostic = json.loads(out.read_text())
    assert diagnostic["cycle_graphlet_k"] == 5
    assert diagnostic["cycle_graphlet_representation"] == "[C5, other] / choose(n,5)"
    assert np.isfinite(diagnostic["overall"]["cycle_graphlet_histogram_tv"])
    # 15-D ORCA only covers graphlets up to four nodes, so there is no C5/orbit
    # triangle-equivalence diagnostic.
    assert "cycle_orbit_triangle_count_gap" not in diagnostic["overall"]
