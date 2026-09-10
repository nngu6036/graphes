#!/usr/bin/env python
"""Audit empirical-degree vs learned-degree GraphER runs without generating graphs.

Run from the GraphES repository root with PYTHONPATH=src. Reads only saved graphs,
reports, a trusted checkpoint and the existing dataset. No sampling/refinement or
model update is performed. A missing paired clean target is NEVER replaced with a
nearest training graph and called ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any

import networkx as nx
import numpy as np
from scipy.optimize import linprog
import torch

from grapher.data.io import load_dataset_splits
from grapher.utils.io import load_yaml
from grapher.utils.device import resolve_torch_device
from grapher.rewiring_mlp.generic.clustering import (
    extract_clustering_histogram, clustering_histogram_wasserstein,
)
from grapher.rewiring_mlp.generic.orbit import extract_orbit_summary, orbit_summary_distance
from grapher.rewiring_mlp.generic.spectral import laplacian_eigenvalues, spectral_scale
from grapher.rewiring_mlp.generic.spectral_data import _resolve_spectral_diffusion_endpoints
from grapher.rewiring_mlp.generic.spectral_model import load_topology_spectral_checkpoint
from grapher.rewiring_mlp.generic.spectral_refiner import predict_clean_spectrum
from grapher.rewiring_mlp.evaluation.metrics import mmd_gaussian_emd, mmd_rbf


ORBIT_IDENTITIES = (
    'mean_degree', 'path3_endpoint_balance', 'centered_wedges',
    'path4_endpoint_balance', 'star3_leaf_balance', 'paw_degree2_balance',
    'paw_degree3_balance', 'diamond_balance', 'three_neighbor_subsets',
)


def degree_tuple(g: nx.Graph) -> tuple[int, ...]:
    return tuple(sorted((int(v) for _, v in g.degree()), reverse=True))


def degree_features(d: tuple[int, ...]) -> dict[str, float | int]:
    n = len(d)
    if n == 0:
        raise ValueError('Empty graphs are not supported in this audit.')
    a = np.asarray(d, dtype=float)
    return dict(n=n, m=sum(d)//2, mean_degree=float(a.mean()),
                degree_variance=float(a.var()), degree_second_moment=float(np.mean(a*a)),
                max_degree=max(d), leaf_fraction=float(np.mean(a == 1)),
                low_degree_fraction=float(np.mean(a < 2)),
                wedge_count=sum(math.comb(v, 2) for v in d if v >= 2),
                three_star_count=sum(math.comb(v, 3) for v in d if v >= 3))


def degree_hist(d: tuple[int, ...], width: int) -> np.ndarray:
    return np.bincount(d, minlength=width).astype(float)/len(d)


def orbit_constraints(d: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Necessary equalities for the ORCA 0..14 mean-per-node convention.

    The last identity partitions triples of neighbors by the number of edges
    among those neighbors (0,1,2,3 -> orbits 7,11,13,14 respectively).
    """
    f = degree_features(d)
    a = np.zeros((9, 15), dtype=float)
    b = np.zeros(9, dtype=float)
    a[0, 0] = 1; b[0] = f['mean_degree']
    a[1, 1] = 1; a[1, 2] = -2
    a[2, [2, 3]] = 1; b[2] = f['wedge_count']/f['n']
    a[3, 4] = 1; a[3, 5] = -1
    a[4, 6] = 1; a[4, 7] = -3
    a[5, 10] = 1; a[5, 9] = -2
    a[6, 11] = 1; a[6, 9] = -1
    a[7, 12] = 1; a[7, 13] = -1
    a[8, [7, 11, 13, 14]] = 1
    b[8] = f['three_star_count']/f['n']
    return a, b


def orbit_feasibility(o: np.ndarray, d: tuple[int, ...]) -> dict[str, float]:
    a, b = orbit_constraints(d)
    r = a @ np.asarray(o) - b
    # Orthogonal distance to the AFFINE relaxation, not a graph projection.
    # Every graph in this degree fibre satisfies these equalities, so this is
    # a lower bound on raw orbit RMSE to a realized graph. Not a log-RMSE bound.
    correction = a.T @ np.linalg.solve(a @ a.T, r)
    result = {f'orbit_residual_{name}': float(abs(value))
              for name, value in zip(ORBIT_IDENTITIES, r)}
    result['orbit_affine_raw_rmse_lower_bound'] = float(np.linalg.norm(correction)/np.sqrt(15))
    result['orbit_identity_scaled_residual'] = float(np.mean(abs(r)/np.maximum(abs(b), 1.0)))
    result['orbit_log_rmse_lower_bound_from_degree0'] = float(
        abs(np.log1p(o[0])-np.log1p(b[0]))/np.sqrt(15))
    return result


def histogram_feasibility_lower_bound(hist: np.ndarray, d: tuple[int, ...]) -> float:
    """W1 lower bound using a fractional degree-conditioned histogram relaxation.

    Each node of degree q can choose C=t/choose(q,2), t an integer, with C=0
    for q<2. This LP allows arbitrary fractional mixtures per degree, ignoring
    adjacency/triangle consistency and integer node allocations. Therefore it
    can UNDERESTIMATE, never intentionally overestimate, attainable W1. Zero
    is not a realizability certificate. Uses the evaluator's bin boundaries.
    """
    h = np.asarray(hist, dtype=float)
    bins = len(h)
    if bins < 2 or np.any(h < 0) or not np.isclose(h.sum(), 1.0, atol=1e-5):
        raise ValueError('Expected a normalized histogram.')
    h = h/h.sum()
    counts = Counter(d)
    levels = sorted(counts)
    variables = []
    edges = np.linspace(0.0, 1.0, bins+1)
    for group, q in enumerate(levels):
        denom = math.comb(q, 2) if q >= 2 else 0
        coeffs = np.arange(denom+1, dtype=float)/denom if denom else np.array([0.0])
        allowed = np.clip(np.searchsorted(edges, coeffs, side='right')-1, 0, bins-1)
        variables.extend((group, int(b)) for b in np.unique(allowed))
    nv, ne = len(variables), bins-1
    ae = np.zeros((len(levels), nv+ne))
    cdf = np.zeros((ne, nv))
    for j, (group, b) in enumerate(variables):
        ae[group, j] = 1
        if b < ne:
            cdf[b:, j] = 1
    be = np.asarray([counts[q]/len(d) for q in levels])
    target = np.cumsum(h)[:-1]
    au = np.vstack((np.c_[cdf, -np.eye(ne)], np.c_[-cdf, -np.eye(ne)]))
    bu = np.r_[target, -target]
    cost = np.r_[np.zeros(nv), np.full(ne, 1.0/bins)]
    sol = linprog(cost, A_ub=au, b_ub=bu, A_eq=ae, b_eq=be,
                  bounds=(0, None), method='highs')
    if not sol.success:
        raise RuntimeError(f'Histogram relaxation failed: {sol.message}')
    return max(0.0, float(sol.fun))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def graph_fingerprint(graphs: list[nx.Graph]) -> str:
    # Includes graph order and indexed edges; not an isomorphism hash.
    h = hashlib.sha256()
    for g in graphs:
        h.update(repr(list(g.nodes())).encode())
        h.update(repr(sorted(tuple(sorted(e)) for e in g.edges())).encode())
    return h.hexdigest()


def load_graphs(path: Path) -> list[nx.Graph]:
    # Pickle is appropriate only for the user's own trusted saved artifacts.
    with path.open('rb') as f:
        objects = pickle.load(f)
    if not isinstance(objects, (list, tuple)):
        raise TypeError(f'{path}: expected a sequence of NetworkX graphs')
    result = []
    for i, g in enumerate(objects):
        if not isinstance(g, nx.Graph) or g.is_directed() or g.is_multigraph():
            raise TypeError(f'{path}, graph {i}: expected simple undirected nx.Graph')
        if not g.number_of_nodes() or nx.number_of_selfloops(g):
            raise ValueError(f'{path}, graph {i}: empty graph or self-loop')
        result.append(g.copy())
    if not result:
        raise ValueError(f'{path}: no graphs')
    return result


def describe(values: list[float]) -> dict[str, Any]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return dict(count=0, mean=None, median=None, q90=None)
    return dict(count=len(v), mean=float(v.mean()), median=float(np.median(v)),
                q90=float(np.quantile(v, .9)), min=float(v.min()), max=float(v.max()))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({k for row in rows for k, v in row.items()
                   if isinstance(v, (float, int, bool, np.number)) and k != 'index'})
    return {key: describe([row.get(key) for row in rows if row.get(key) is not None])
            for key in keys}


def flatten(value: Any, prefix: str = '') -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    return {k: v for name, child in value.items()
            for k, v in flatten(child, f'{prefix}.{name}' if prefix else name).items()}


def load_run(directory: Path, max_graphs: int | None) -> dict[str, Any]:
    report_path = directory/'report.json'
    report = json.loads(report_path.read_text())
    source_path, final_path = directory/'coarse_graphs.pkl', directory/'topology_refined_graphs.pkl'
    sources, finals = load_graphs(source_path), load_graphs(final_path)
    if len(sources) != len(finals):
        raise ValueError(f'{directory}: source/final lengths differ')
    if max_graphs:
        sources, finals = sources[:max_graphs], finals[:max_graphs]
    return dict(directory=str(directory), report=report, sources=sources, finals=finals,
                metadata=dict(report_sha256=sha256(report_path), source_file_sha256=sha256(source_path),
                    final_file_sha256=sha256(final_path), num_loaded=len(sources),
                    indexed_source_batch_sha256=graph_fingerprint(sources),
                    degree_batch_sha256=hashlib.sha256(repr([degree_tuple(g) for g in sources]).encode()).hexdigest(),
                    degree_source=report.get('degree_source'), seed=report.get('seed')))


def descriptor_metrics(reference: list[nx.Graph], candidates: list[nx.Graph], bins: int,
                       orbit_cache: dict) -> dict[str, float]:
    def orbit(g):
        key = graph_fingerprint([g])
        if key not in orbit_cache:
            orbit_cache[key] = extract_orbit_summary(g)
        return orbit_cache[key]
    width = max(max(degree_tuple(g)) for g in reference+candidates)+1
    rh = np.stack([degree_hist(degree_tuple(g), width) for g in reference])
    ch = np.stack([degree_hist(degree_tuple(g), width) for g in candidates])
    rc = np.stack([extract_clustering_histogram(g, bins) for g in reference])
    cc = np.stack([extract_clustering_histogram(g, bins) for g in candidates])
    ro = np.stack([orbit(g) for g in reference]); co = np.stack([orbit(g) for g in candidates])
    return dict(degree_mmd=float(mmd_gaussian_emd(rh, ch, sigma=1.0)),
                clustering_mmd=float(mmd_gaussian_emd(rc, cc, sigma=.1, distance_scaling=bins)),
                orbit_mmd=float(mmd_rbf(ro, co, sigma=30.0)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True, help='Trusted joint spectral-summary checkpoint used for BOTH runs')
    p.add_argument('--empirical-dir', type=Path, required=True)
    p.add_argument('--learned-dir', type=Path, required=True)
    p.add_argument('--reference-split', choices=['val', 'test'], default='val')
    p.add_argument('--max-graphs', type=int, default=None)
    p.add_argument('--histogram-feasibility', action='store_true', help='Compute optional fractional histogram W1 lower bounds')
    p.add_argument('--training-source-draws', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.training_source_draws < 1 or (args.max_graphs is not None and args.max_graphs < 1):
        p.error('Counts must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir/'degree_structural_audit.json').exists():
        p.error('Output report already exists; choose a fresh output directory')
    torch.set_num_threads(args.threads)
    cfg = load_yaml(args.config); dc = cfg.get('dataset', {})
    splits = load_dataset_splits(dc.get('name', 'sbm'), root=dc.get('root', 'outputs/datasets'),
                                config_path=dc.get('config_path'), build_if_missing=False)
    train = list(splits['train']); val = list(splits['val']); reference = list(splits[args.reference_split])
    if not train or not val or not reference:
        raise ValueError('Train, validation and reference splits must be nonempty')
    device = resolve_torch_device(args.device)
    model, _, checkpoint = load_topology_spectral_checkpoint(args.checkpoint, device=device)
    if not model.predict_clustering_histogram or not model.predict_orbit_summary:
        raise ValueError('This audit needs both histogram and orbit heads')
    model.eval()
    bins = model.clustering_histogram_bins
    if bins != 100:
        raise ValueError('This comparison targets the 100-bin GraphRNN clustering protocol')
    runs = {label: load_run(directory, args.max_graphs) for label, directory in
            [('empirical', args.empirical_dir), ('learned', args.learned_dir)]}
    warnings = []
    for label, run in runs.items():
        expected = {'empirical': {'empirical', 'train_empirical'}, 'learned': {'learned', 'degree_vae'}}[label]
        if run['metadata']['degree_source'] not in expected:
            warnings.append(f"{label}: recorded degree_source={run['metadata']['degree_source']!r}, not {expected}")
        stored = run['report'].get('diagnostics', {}).get('checkpoint_validation_metrics')
        if stored and stored != checkpoint.get('report'):
            warnings.append(f'{label}: stored validation metrics differ from supplied checkpoint report; verify checkpoint identity')
    a, b = [flatten(runs[label]['report'].get('config', {}).get('topology_refiner', {})) for label in ['empirical', 'learned']]
    differences = {key: [a.get(key), b.get(key)] for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
    if differences:
        warnings.append('Refiner configurations differ; not a degree-source-only intervention')
    if len(runs['empirical']['sources']) != len(runs['learned']['sources']):
        warnings.append('Generated graph counts differ')
    # Stored config, not today's edited YAML, defines the trained source process.
    tc = checkpoint.get('config') or cfg
    if not checkpoint.get('config'):
        warnings.append('Checkpoint lacks config: reconstructed training HH conditions use supplied config')
    diffusion = tc.get('summary_diffusion', {})
    ctor = tc.get('constructor', {})
    source_cfg = dict(ensure_connected_source=diffusion.get('ensure_connected_source', ctor.get('ensure_connected', True)),
                      max_repair_trials=diffusion.get('max_repair_trials', ctor.get('max_repair_trials', 10000)),
                      random_relabel_source=diffusion.get('random_relabel_source', ctor.get('random_relabel', True)),
                      source_randomization_steps=diffusion.get('source_randomization_steps', 0))
    if (tc.get('training_sources', {}) or {}).get('mode', 'target_degree_havel_hakimi') not in {'target_degree_havel_hakimi', 'spectral_havel_hakimi'}:
        raise ValueError('This audit assumes target-degree HH spectral training')
    rng = np.random.default_rng(args.seed)
    training_hh = []
    for draw in range(args.training_source_draws):
        for g in train:
            s, _, _ = _resolve_spectral_diffusion_endpoints(g, source_config=source_cfg,
                    require_same_degree_sequence=True, rng=rng)
            training_hh.append(s)
    val_hh = [_resolve_spectral_diffusion_endpoints(g, source_config=source_cfg,
              require_same_degree_sequence=True, rng=rng)[0] for g in val]
    train_d = [degree_tuple(g) for g in train]
    degree_set = set(train_d); nm_set = {(len(d), sum(d)//2) for d in train_d}
    exact_moment_set = {(f['n'], f['m'], f['wedge_count'], f['three_star_count']) for f in map(degree_features, train_d)}
    all_g = train+val+[g for r in runs.values() for g in r['sources']]
    width = max(max(degree_tuple(g)) for g in all_g)+1
    train_cdf = np.cumsum(np.stack([degree_hist(d, width) for d in train_d]), axis=1)
    spectrum_pool = [(g.number_of_nodes(), g.number_of_edges(), laplacian_eigenvalues(g)/spectral_scale(g)) for g in training_hh]
    orbit_cache = {}
    def orbit(g):
        key = graph_fingerprint([g])
        if key not in orbit_cache:
            orbit_cache[key] = extract_orbit_summary(g)
        return orbit_cache[key]
    def features(g):
        d = degree_tuple(g); f = degree_features(d)
        f.update(degree_seen_in_train=d in degree_set, nm_seen_in_train=(f['n'], f['m']) in nm_set,
                 degree_moments_seen_in_train=(f['n'], f['m'], f['wedge_count'], f['three_star_count']) in exact_moment_set)
        dist = abs(train_cdf-np.cumsum(degree_hist(d, width))).sum(axis=1)
        f['nearest_training_degree_hist_emd'] = float(dist.min())
        mask_n = np.array([len(x) == len(d) for x in train_d])
        mask_nm = np.array([(len(x), sum(x)//2) == (len(d), sum(d)//2) for x in train_d])
        for name, mask in [('same_n', mask_n), ('same_nm', mask_nm)]:
            f[f'nearest_training_degree_hist_emd_{name}'] = float(dist[mask].min()) if mask.any() else None
        spectrum = laplacian_eigenvalues(g)/spectral_scale(g)
        for name, predicate in [('same_n', lambda n,m: n == f['n']), ('same_nm', lambda n,m: (n,m)==(f['n'], f['m']))]:
            distances = [float(np.sqrt(np.mean((spectrum-sp)**2))) for n,m,sp in spectrum_pool if predicate(n,m)]
            f[f'nearest_reconstructed_training_HH_spectrum_rmse_{name}'] = min(distances) if distances else None
        return f
    def prediction_data(g, source, time):
        with torch.inference_mode():
            return predict_clean_spectrum(model, g, time=time, device=device, conditioning_graph=source)
    def prediction_features(g, pred):
        d = degree_tuple(g)
        f = orbit_feasibility(pred.clean_orbit_summary, d)
        f['predicted_orbit_summary'] = pred.clean_orbit_summary.tolist()
        f['predicted_clustering_histogram'] = pred.clean_clustering_histogram.tolist()
        if args.histogram_feasibility:
            f['histogram_relaxed_w1_lower_bound'] = histogram_feasibility_lower_bound(pred.clean_clustering_histogram, d)
        if pred.clean_cycle_graphlet_histogram is not None and model.cycle_graphlet_k == 3:
            n = len(d)
            f['cycle_orbit_triangle_count_gap'] = float(abs(pred.clean_cycle_graphlet_histogram[0]*math.comb(n,3)-pred.clean_orbit_summary[3]*n/3)) if n>=3 else 0.0
        return f
    rows_by_group = {}
    for label, sources, targets in [('train_endpoint', training_hh[:len(train)], train), ('validation_endpoint', val_hh, val)]:
        print(f'Auditing {label}: {len(sources)} graphs', flush=True)
        rows = []
        for i, (g, target) in enumerate(zip(sources, targets)):
            pred = prediction_data(g, g, 0.0)
            row = dict(index=i, degree_sequence=list(degree_tuple(g)), **features(g), **prediction_features(g, pred))
            row['pred_to_paired_clean_hist_w1'] = clustering_histogram_wasserstein(pred.clean_clustering_histogram, extract_clustering_histogram(target,bins))
            row['pred_to_paired_clean_orbit_log_rmse'] = orbit_summary_distance(pred.clean_orbit_summary, orbit(target))
            rows.append(row)
        rows_by_group[label] = rows
    metric_rows = {'train_to_reference': descriptor_metrics(reference, train, bins, orbit_cache)}
    for label, run in runs.items():
        print(f'Auditing {label}: {len(run["sources"])} saved source/final pairs', flush=True)
        traces = run['report'].get('traces', [])
        records = run['report'].get('pipeline_records', [])
        rows = []
        for i,(s,g) in enumerate(zip(run['sources'], run['finals'])):
            if degree_tuple(s) != degree_tuple(g) or dict(s.degree()) != dict(g.degree()):
                raise AssertionError(f'{label} graph {i}: degree changed between saved source/final')
            d = degree_tuple(s); a,b = orbit_constraints(d)
            for actual in (s,g):
                if np.max(abs(a @ orbit(actual)-b)) > 1e-8:
                    raise AssertionError('Orbit backend violates the necessary degree identities')
            trace = traces[i] if i < len(traces) else []
            record = records[i] if i < len(records) else {}
            accepted = [x for x in trace if x.get('accepted')]
            times = [x.get('prediction_time') for x in trace if x.get('prediction_time') is not None]
            # Last logged prediction time is exact if present. Otherwise there
            # is no justified inference-time target to report at the final state.
            final_time = float(times[-1]) if times else None
            p0 = prediction_data(s,s,0.0)
            row = dict(index=i, degree_sequence=list(d), **features(s), **prediction_features(s,p0))
            h0,hf = extract_clustering_histogram(s,bins),extract_clustering_histogram(g,bins)
            o0,of = orbit(s),orbit(g)
            row.update(source_hist_w1_to_initial_prediction=clustering_histogram_wasserstein(h0,p0.clean_clustering_histogram),
                       final_hist_w1_to_initial_prediction=clustering_histogram_wasserstein(hf,p0.clean_clustering_histogram),
                       source_orbit_log_rmse_to_initial_prediction=orbit_summary_distance(o0,p0.clean_orbit_summary),
                       final_orbit_log_rmse_to_initial_prediction=orbit_summary_distance(of,p0.clean_orbit_summary),
                       indexed_degree_preserved=True, source_connected=nx.is_connected(s), final_connected=nx.is_connected(g),
                       accepted_swaps=record.get('accepted_swaps',len(accepted)),
                       generation_attempts=record.get('generation_attempts'),
                       candidate_pass_rate=record.get('candidate_pass_rate'), stop_reason=record.get('stop_reason'),
                       last_prediction_time=final_time)
            for stat in ('clustering','orbit','cycle'):
                changes = [float(x[f'{stat}_gain']) for x in accepted if x.get(f'{stat}_gain') is not None]
                row[f'accepted_{stat}_worsening_fraction'] = float(np.mean(np.asarray(changes)<-1e-10)) if changes else None
            if final_time is not None:
                pf = prediction_data(g,s,final_time)
                row['hist_target_drift_w1'] = clustering_histogram_wasserstein(p0.clean_clustering_histogram,pf.clean_clustering_histogram)
                row['orbit_target_drift_log_rmse'] = orbit_summary_distance(p0.clean_orbit_summary,pf.clean_orbit_summary)
                row['final_hist_w1_to_last_prediction'] = clustering_histogram_wasserstein(hf,pf.clean_clustering_histogram)
                row['final_orbit_log_rmse_to_last_prediction'] = orbit_summary_distance(of,pf.clean_orbit_summary)
                row.update({f'last_prediction_{k}':v for k,v in orbit_feasibility(pf.clean_orbit_summary,d).items()})
            rows.append(row)
        rows_by_group[label] = rows
        metric_rows[f'{label}_hh_to_reference'] = descriptor_metrics(reference,run['sources'],bins,orbit_cache)
        metric_rows[f'{label}_final_to_reference'] = descriptor_metrics(reference,run['finals'],bins,orbit_cache)
    summaries = {k:summarize_rows(v) for k,v in rows_by_group.items()}
    for group in ['empirical','learned','validation_endpoint']:
        for seen in (True,False):
            sub = [r for r in rows_by_group[group] if r['degree_seen_in_train'] == seen]
            summaries[f'{group}_degree_{"seen" if seen else "novel"}'] = summarize_rows(sub)
    report = dict(format='degree_structural_shift_audit_v1', checkpoint=str(Path(args.checkpoint).resolve()),
                  checkpoint_sha256=sha256(Path(args.checkpoint)), use_graph_context=model.use_graph_context,
                  reference_split=args.reference_split, reference_graphs=len(reference), train_graphs=len(train), val_graphs=len(val),
                  dataset_indexed_fingerprints={name:graph_fingerprint(list(splits[name])) for name in ['train','val','test']},
                  orbit_backend='repository pure-Python 15-D mean-per-node descriptor; no ORCA subprocess',
                  settings_differences=differences, warnings=warnings,
                  source_metadata={label:run['metadata'] for label,run in runs.items()},
                  mmd=metric_rows, summary=summaries, graphs=rows_by_group,
                  caveats=['No clean paired target exists for arbitrary learned-degree draws; their distances are to predictions, not supervised errors.',
                           'Novel degree sequences are not necessarily bad: compare held-out validation novelty as a calibration.',
                           'The orbit affine and histogram fractional bounds are relaxations, not realizability certificates.',
                           'Training HH spectra are reconstructed with declared settings; not guaranteed identical to cached training sources.',
                           'Current checkpoint hash documents this audit, not historical generation checkpoint identity.',
                           'Metrics average graphs equally and use the selected reference split; no cross-split comparison.',
                           'Missing final prediction time is reported as null, not guessed.'])
    out = args.output_dir/'degree_structural_audit.json'
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    all_rows = [dict(group=group,**row) for group,rows in rows_by_group.items() for row in rows]
    fields = ['group','index']+sorted(set().union(*(row.keys() for row in all_rows))-{'group','index'})
    with (args.output_dir/'per_graph.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for row in all_rows:
            w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    print('\nMMD against',args.reference_split)
    for name,v in metric_rows.items():
        print(f'{name:32s} degree={v["degree_mmd"]:.6f} clustering={v["clustering_mmd"]:.6f} orbit={v["orbit_mmd"]:.6f}')
    keys=['degree_seen_in_train','nm_seen_in_train','degree_variance','wedge_count','three_star_count',
          'nearest_training_degree_hist_emd_same_nm','nearest_reconstructed_training_HH_spectrum_rmse_same_nm',
          'orbit_identity_scaled_residual','orbit_affine_raw_rmse_lower_bound',
          'histogram_relaxed_w1_lower_bound','final_hist_w1_to_initial_prediction',
          'final_orbit_log_rmse_to_initial_prediction','hist_target_drift_w1','orbit_target_drift_log_rmse','generation_attempts']
    for group in ['validation_endpoint','empirical','learned']:
        print(f'\n{group} (means; prediction compatibility is NOT ground-truth error)')
        for key in keys:
            val_ = summaries[group].get(key,{}).get('mean')
            if val_ is not None:
                print(f'  {key}: {val_:.6g}')
    for message in warnings:
        print('WARNING:',message)
    print('Saved',out)


if __name__ == '__main__':
    main()
