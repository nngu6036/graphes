"""Audits and separately named attributed-feature MMDs; no repairs or ORCA replacement."""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from grapher.models.gdsm_simple.wrapper import _sha256
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary
from .data import TypedGraphlets3,encode_graph,typed_counts


def load_pickle(path):
    # Only use project-generated or otherwise trusted pickle files.
    with Path(path).open('rb') as handle:
        return pickle.load(handle)


def audit(generated_dir):
    root=Path(generated_dir);manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('format')!='grapher_gdsm_spectral_categorical_generation_v1':
        raise ValueError('Expected a managed spectral-categorical generation directory')
    for name,digest in manifest['artifact_sha256'].items():
        if _sha256(root/name)!=digest:raise ValueError(f'Artifact checksum mismatch: {name}')
    schema=json.loads((root/'categorical_schema.json').read_text())
    vocab=GraphCategoryVocabulary.from_dict(schema['category_vocabulary'])
    final=load_pickle(root/'base_graphs.pkl');initial=load_pickle(root/'initial_graphs.pkl')
    pre=load_pickle(root/'final_pre_rewire_graphs.pkl');degrees=load_pickle(root/'sampled_degree_sequences.pkl')
    vals=load_pickle(root/'final_adjacency_eigenvalues.pkl');vectors=load_pickle(root/'final_eigenvectors.pkl')
    n=len(final)
    if not n or any(len(a)!=n for a in (initial,pre,degrees,vals,vectors)):
        raise ValueError('Empty or mismatched output batches')
    prior_match=initial_match=connected=0;max_error=0.;max_orthogonal_error=0.
    for i,(g,g0,gpre,d,z,u) in enumerate(zip(final,initial,pre,degrees,vals,vectors)):
        if len(g)!=len(g0) or len(g)!=len(gpre) or len(g)!=len(d):raise AssertionError('Node count changed')
        x,e=encode_graph(g,vocab,len(g));xp,ep=encode_graph(gpre,vocab,len(g))
        x0,e0=encode_graph(g0,vocab,len(g))
        if list(g.nodes())!=list(range(len(g))):raise AssertionError('Final node indices must be 0..n-1')
        np.testing.assert_array_equal(x,xp)
        np.testing.assert_array_equal((e>0).sum(1),(ep>0).sum(1))
        for k in range(1,vocab.num_edge_categories):
            np.testing.assert_array_equal((e==k).sum(1),(ep==k).sum(1))
        error=float(np.max(np.abs((u*(np.asarray(z)*np.sqrt(len(g)))[None])@u.T-(e>0))))
        orth=float(np.max(np.abs(u.T@u-np.eye(len(g)))))
        max_error=max(max_error,error);max_orthogonal_error=max(max_orthogonal_error,orth)
        if error>2e-4 or orth>2e-4:raise AssertionError(f'Final eigenpairs do not represent retained adjacency #{i}')
        prior_match+=int(np.array_equal(np.sort((e>0).sum(1)),np.sort(d)))
        initial_match+=int(np.array_equal((e>0).sum(1),(e0>0).sum(1)))
        connected+=int(nx.is_connected(g))
    diag=json.loads((root/'rewiring_diagnostics.json').read_text())['aggregate']
    return {'status':'passed','num_graphs':n,'artifact_hashes_verified':True,
            'node_counts_preserved':True,'final_local_node_types_preserved':True,
            'final_local_indexed_degrees_and_typed_degrees_preserved':True,
            'prior_degree_preservation_rate':prior_match/n,'initial_degree_preservation_rate':initial_match/n,
            'connectedness_rate':connected/n,'max_eigenpair_reconstruction_error':max_error,
            'max_eigenvector_orthogonality_error':max_orthogonal_error,
            'recorded_basis_updates_per_graph':diag['basis_updates_per_graph'],
            'recorded_categorical_degree_change_steps_mean':diag['categorical_degree_change_steps_mean'],
            'degree_changes_are_allowed_not_required':True}


def rbf_mmd2(a,b,sigma=1.,block=256):
    """Biased nonnegative RBF MMD^2, unit-Euclidean histograms, blocked pair memory."""
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    if not len(a) or not len(b) or not np.isfinite(sigma) or sigma<=0:raise ValueError('Empty samples or invalid sigma')
    def average(x,y):
        total=0.
        for i in range(0,len(x),block):
            for j in range(0,len(y),block):
                total+=np.exp(-cdist(x[i:i+block],y[j:j+block],metric='sqeuclidean')/(2*sigma*sigma)).sum()
        return total/(len(x)*len(y))
    return float(max(0.,average(a,a)+average(b,b)-2*average(a,b)))


def evaluate(generated_dir,reference_graphs,*,sigma=1.,max_reference=None,seed=42):
    root=Path(generated_dir);schema=json.loads((root/'categorical_schema.json').read_text())
    if schema.get('graphlet_schema_version')==2:
        from .multiscale_evaluation import evaluate_multi
        return evaluate_multi(generated_dir,reference_graphs,sigma=sigma,max_reference=max_reference,seed=seed)
    vocab=GraphCategoryVocabulary.from_dict(schema['category_vocabulary'])
    generated=load_pickle(root/'base_graphs.pkl');reference=load_pickle(reference_graphs)
    if not isinstance(reference,(list,tuple)) or not reference:raise ValueError('Reference must be a nonempty graph list')
    if max_reference is not None:
        if max_reference<1:raise ValueError('max_reference must be positive')
        if max_reference<len(reference):
            idx=np.random.default_rng(seed).choice(len(reference),max_reference,replace=False)
            reference=[reference[i] for i in idx]
    all_rows=[];keys=set()
    for graphs in (generated,reference):
        rows=[]
        for g in graphs:
            x,e=encode_graph(g,vocab,max(1,len(g)));counts=typed_counts(x,e);keys.update(counts)
            rows.append((x,e,counts))
        all_rows.append(rows)
    # External evaluation distinguishes all observed typed classes; it does NOT
    # modify the checkpoint's training-only vocabulary or collapse unseen types.
    basis=TypedGraphlets3(sorted(keys));train_basis=TypedGraphlets3(schema['graphlet_keys'])
    features=[];overflow=[]
    for rows in all_rows:
        f={k:[] for k in ('node_category_histogram','edge_category_histogram','typed_graphlet3_joint_mass','connected_triple_mass')}
        ov=[]
        for x,e,counts in rows:
            nh=np.bincount(x,minlength=vocab.num_node_categories).astype(float)/len(x)
            ij=np.triu_indices(len(x),1);eh=np.bincount(e[ij],minlength=vocab.num_edge_categories).astype(float)/max(len(ij[0]),1)
            h,m=basis.encode_counts(counts,len(x))
            f['node_category_histogram'].append(nh);f['edge_category_histogram'].append(eh)
            # Unconditional mass per triple plus disconnected mass. Avoids
            # arbitrary normalized histograms when no connected triple exists.
            f['typed_graphlet3_joint_mass'].append(np.r_[h[:-1]*m,1-m])
            f['connected_triple_mass'].append([m])
            ov.append(float(train_basis.encode_counts(counts,len(x))[0][-1]))
        features.append(f);overflow.append(float(np.mean(ov)))
    return {'num_generated':len(generated),'num_reference':len(reference),'reference_path':str(Path(reference_graphs).resolve()),
            'reference_sha256':_sha256(Path(reference_graphs)),'reference_subsample_seed':seed if max_reference is not None else None,
            'protocol':'biased_RBF_MMD_squared_Euclidean','sigma':sigma,
            'not_standard_graphRNN_ORCA_or_NSPDK':True,'raw_graphs_no_sanitization_or_filter':True,
            'external_typed_graphlet3_class_count':len(keys),
            'mmd2':{k:rbf_mmd2(features[0][k],features[1][k],sigma) for k in features[0]},
            'mean_conditional_overflow_against_training_vocabulary':{'generated':overflow[0],'reference':overflow[1]}}
