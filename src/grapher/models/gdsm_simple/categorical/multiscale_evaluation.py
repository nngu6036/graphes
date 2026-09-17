"""Exact per-order graphlet diagnostics with sparse feature storage.

The evaluation vocabulary is the union of generated/reference patterns; it
never changes the model vocabulary. These are explicitly RBF MMD^2 metrics,
not GraphRNN/ORCA, NSPDK, FCD, or repair-based molecular metrics.
"""
from __future__ import annotations
import json
from pathlib import Path
from math import comb

import numpy as np
from scipy import sparse

from .data import encode_graph
from .multiscale import TypedGraphletsMulti, count_multi
from .evaluation import load_pickle, rbf_mmd2
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary
from grapher.models.gdsm_simple.wrapper import _sha256


def sparse_rbf_mmd2(a,b,sigma=1.,block=256):
    a=sparse.csr_matrix(a,dtype=np.float64);b=sparse.csr_matrix(b,dtype=np.float64)
    if not a.shape[0] or not b.shape[0] or not np.isfinite(sigma) or sigma<=0:
        raise ValueError('Nonempty features and a positive finite bandwidth required')
    def average(x,y):
        xn=np.asarray(x.multiply(x).sum(axis=1)).ravel()
        yn=np.asarray(y.multiply(y).sum(axis=1)).ravel()
        total=0.
        for i in range(0,x.shape[0],block):
            for j in range(0,y.shape[0],block):
                distance=xn[i:i+block,None]+yn[None,j:j+block]-2*(x[i:i+block]@y[j:j+block].T).toarray()
                total+=np.exp(-np.maximum(distance,0)/(2*sigma*sigma)).sum()
        return total/(x.shape[0]*y.shape[0])
    return float(max(0.,average(a,a)+average(b,b)-2*average(a,b)))


def evaluate_multi(generated_dir,reference_graphs,*,sigma=1.,max_reference=None,seed=42):
    root=Path(generated_dir);schema=json.loads((root/'categorical_schema.json').read_text())
    vocab=GraphCategoryVocabulary.from_dict(schema['category_vocabulary'])
    train_basis=TypedGraphletsMulti.from_schema(schema)
    generated=load_pickle(root/'base_graphs.pkl');reference=load_pickle(reference_graphs)
    if not generated or not isinstance(reference,(list,tuple)) or not reference:
        raise ValueError('Nonempty generated and reference graph lists required')
    if max_reference is not None:
        if max_reference<1:raise ValueError('max_reference must be positive')
        if max_reference<len(reference):
            ids=np.random.default_rng(seed).choice(len(reference),max_reference,replace=False)
            reference=[reference[i] for i in ids]
    # Compact local IDs avoid a huge dense [graphs x typed-vocabulary] matrix.
    keys={k:{} for k in train_basis.orders};groups=[];sizes=[];node=[];edge=[];overflow=[]
    for graphs in (generated,reference):
        rows=[];ns=[];nh=[];eh=[];ov=[]
        for g in graphs:
            x,e=encode_graph(g,vocab,max(1,len(g)));n=len(x)
            counts=count_multi(x,e,train_basis.orders,limit=train_basis.limit)
            row={}
            for k in train_basis.orders:
                values={}
                for key,value in counts[k].items():
                    if key not in keys[k]:keys[k][key]=len(keys[k])
                    values[keys[k][key]]=value
                row[k]=values
            rows.append(row);ns.append(n)
            nh.append(np.bincount(x,minlength=vocab.num_node_categories)/n)
            ij=np.triu_indices(n,1)
            eh.append(np.bincount(e[ij],minlength=vocab.num_edge_categories)/max(len(ij[0]),1))
            h,_=train_basis.encode_counts(counts,n);ov.append(train_basis.overflow_by_order(h))
        groups.append(rows);sizes.append(ns);node.append(nh);edge.append(eh);overflow.append(ov)
    metrics={'node_category_histogram':rbf_mmd2(node[0],node[1],sigma),
             'edge_category_histogram':rbf_mmd2(edge[0],edge[1],sigma)}
    by_order={}
    for k in train_basis.orders:
        dim=len(keys[k]);joint=[];conditional=[];masses=[]
        for rows,ns in zip(groups,sizes):
            rr=[];cc=[];vv=[];cr=[];cv=[];ci=[];condition_rows=0;ms=[]
            for i,(row,n) in enumerate(zip(rows,ns)):
                total=sum(row[k].values());den=max(comb(n,k),1) if n>=k else 1
                mass=total/den;ms.append([mass])
                for key,value in row[k].items():
                    rr.append(i);cc.append(key);vv.append(value/den)
                    if total:
                        cr.append(condition_rows);ci.append(key);cv.append(value/total)
                rr.append(i);cc.append(dim);vv.append(1-mass)
                condition_rows+=int(total>0)
            joint.append(sparse.csr_matrix((vv,(rr,cc)),shape=(len(rows),dim+1)))
            conditional.append(sparse.csr_matrix((cv,(cr,ci)),shape=(condition_rows,dim)))
            masses.append(ms)
        mj=sparse_rbf_mmd2(joint[0],joint[1],sigma)
        mm=rbf_mmd2(masses[0],masses[1],sigma)
        hc=(sparse_rbf_mmd2(conditional[0],conditional[1],sigma)
            if all(c.shape[0] for c in conditional) else None)
        by_order[str(k)]={'joint_mass_mmd2':mj,'selected_mass_mmd2':mm,
                          'conditional_histogram_mmd2':hc,'external_class_count':dim,
                          'generated_with_connected_subsets':conditional[0].shape[0],
                          'reference_with_connected_subsets':conditional[1].shape[0],
                          'generated_order_available':sum(n>=k for n in sizes[0]),
                          'reference_order_available':sum(n>=k for n in sizes[1]),
                          'mean_overflow_vs_train':{'generated':float(np.mean([r[str(k)] for r in overflow[0]])),
                                                    'reference':float(np.mean([r[str(k)] for r in overflow[1]]))}}
        metrics[f'typed_graphlet{k}_joint_mass']=mj
        metrics[f'connected_subset_mass_{k}']=mm
    return {'num_generated':len(generated),'num_reference':len(reference),
            'reference_path':str(Path(reference_graphs).resolve()),'reference_sha256':_sha256(Path(reference_graphs)),
            'reference_subsample_seed':seed if max_reference is not None else None,
            'graphlet_orders':list(train_basis.orders),'graphlet_counting':'exact_connected_induced',
            'protocol':'biased_RBF_MMD_squared_Euclidean','sigma':sigma,
            'not_standard_graphRNN_ORCA_or_NSPDK':True,'raw_graphs_no_sanitization_or_filter':True,
            'external_vocabulary':'union_generated_reference_no_overflow_collapse',
            'conditional_histogram_policy':'only_graphs_with_positive_connected_count; denominators_reported_per_order',
            'mmd2':metrics,'graphlet_metrics_by_order':by_order}
