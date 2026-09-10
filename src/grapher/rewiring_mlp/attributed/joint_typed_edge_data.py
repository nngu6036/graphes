"""Aligned molecular endpoints and graph-balanced batches for joint typed GraphER."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import networkx as nx
import numpy as np
import torch

from grapher.models.dhvae_hh.typed_constructor import construct_typed_graph
from grapher.models.dhvae_hh.typed_degree_vae import TypedSignatureVectorizer
from grapher.rewiring_mlp.attributed.spectral import normalize_attributed_graph, attributed_laplacian_spectra
from grapher.rewiring_mlp.generic.clustering import extract_clustering_histogram
from grapher.rewiring_mlp.generic.orbit import extract_orbit_summary
from grapher.rewiring_mlp.molecular.typed_invariants import extract_typed_invariant, typed_invariant_matches_graph
from grapher.utils.io import load_pickle


ENDPOINT_VALENCE_POLICY = 'preserve_prepared_target_bond_types'


def endpoint_constructor_config(config: dict) -> dict:
    """Reconstruct observed typed graphs without applying generation valence caps.

    Canonical QM9 is loaded with sanitize=False and may contain stored bond-order
    sums beyond the generation envelope (for example, raw pentavalent nitrogen).
    Sanitizing or repairing that target would change its indexed signatures.
    The target itself witnesses a simple connected realization, so retain its
    labels exactly and leave chemical acceptance to unconditional generation.
    """
    constructor = deepcopy(config.get('constructor', {}))
    constructor['randomize_assignment'] = False
    constructor['max_weighted_valence'] = None
    return constructor


def graph_record(graph: nx.Graph, node_attribute='atomic_num', edge_attribute='bond_type') -> dict:
    g = normalize_attributed_graph(graph)
    return {'nodes': [int(g.nodes[i][node_attribute]) for i in range(len(g))],
            'edges': sorted([min(int(u),int(v)), max(int(u),int(v)), int(d[edge_attribute])]
                            for u,v,d in g.edges(data=True))}


def graph_from_record(record: dict, node_attribute='atomic_num', edge_attribute='bond_type') -> nx.Graph:
    g = nx.Graph()
    for i, x in enumerate(record['nodes']):
        g.add_node(i, **{node_attribute: x, 'atomic_num': x, 'atom_type': x})
    for u,v,r in record['edges']:
        g.add_edge(u,v, **{edge_attribute: r, 'bond_type': r,
                          'bond_order': 1.5 if r == 4 else float(r)})
    return g


def record_hash(record: Any) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_splits(config: dict) -> tuple[dict, dict]:
    """Read existing splits only. Never rebuild or fall back to training validation."""
    from grapher.rewiring_mlp.generic.joint_checkpointing import file_sha256
    ds = config['dataset']; root = Path(ds.get('root','outputs/datasets')) / ds['name']
    paths = {name: root/f'{name}.pkl' for name in ('train','val','test')}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(f"Required processed split missing: {p}. Build the dataset explicitly first.")
    metadata = {'name': ds['name'], 'benchmark': config.get('benchmark'),
                'split_sha256': {k: file_sha256(p) for k,p in paths.items()}}
    metadata['fingerprint'] = record_hash(metadata)
    splits = {k: list(load_pickle(p)) for k,p in paths.items()}
    if not splits['train'] or not splits['val']:
        raise ValueError('Nonempty separate train and validation splits are required.')
    return splits, metadata


def validate_graph(graph: nx.Graph, vectorizer: TypedSignatureVectorizer,
                   atom_types: tuple[int,...]) -> nx.Graph:
    v = vectorizer; vocab = v.vocabulary
    g = normalize_attributed_graph(graph)
    if not g or not nx.is_connected(g):
        raise ValueError('Only nonempty connected input graphs are supported.')
    if not v.min_nodes <= len(g) <= v.max_nodes:
        raise ValueError(f'Graph size {len(g)} outside typed checkpoint support [{v.min_nodes},{v.max_nodes}].')
    for _, d in g.nodes(data=True):
        if d.get(vocab.node_attribute) not in atom_types:
            raise ValueError(f'Unknown atom category {d.get(vocab.node_attribute)}.')
        # The supplied dataset codec is atomic_num/bond_type with implicit H.
        # Do not silently throw away user-supplied charge/stereo semantics.
        if d.get('formal_charge', 0) != 0:
            raise ValueError('Explicit nonzero formal_charge must be part of the typed signature before training this model.')
    invariant = extract_typed_invariant(g, edge_types=vocab.edge_types,
                                       node_attribute=vocab.node_attribute, edge_attribute=vocab.edge_attribute)
    for sig in invariant.signatures:
        try: vocab.index(sig)
        except ValueError as e:
            raise ValueError(f'Unseen typed signature {sig}. Vocabulary is training-only; do not expand it from validation/test.') from e
    return g


class EndpointStore:
    """Lazy, bounded RAM cache plus optional content-addressed SQLite endpoint cache.

    Cache entries are JSON, namespaced by graph contents, constructor settings,
    seed, and summary definition. Validation never fits a vocabulary. Data
    workers are intentionally not used by the graph-balanced trainer.
    """
    def __init__(self, graphs, vectorizer, atom_types, config, *, seed, cache_path=None):
        self.graphs = graphs; self.vectorizer = vectorizer; self.atom_types = tuple(atom_types)
        self.config = deepcopy(config); self.seed = int(seed)
        self.memory = OrderedDict()
        self.capacity = int(config.get('training_sources',{}).get('memory_cache_graphs',256))
        self.db = None
        if cache_path:
            path=Path(cache_path); path.parent.mkdir(parents=True,exist_ok=True)
            self.db=sqlite3.connect(path)
            self.db.execute('CREATE TABLE IF NOT EXISTS endpoints (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            self.db.commit()

    def __len__(self): return len(self.graphs)

    def close(self):
        if self.db is not None: self.db.close(); self.db=None

    def __getitem__(self, index):
        if index in self.memory:
            self.memory.move_to_end(index); return self.memory[index]
        v = self.vectorizer; vocab=v.vocabulary
        target=validate_graph(self.graphs[index],v,self.atom_types)
        constructor=endpoint_constructor_config(self.config)
        ss=self.config.get('structure_summary_prediction',{})
        rec=graph_record(target,vocab.node_attribute,vocab.edge_attribute)
        key=record_hash({'version':2,'graph':rec,'seed':self.seed+index*1009,
                         'valence_policy':ENDPOINT_VALENCE_POLICY,
                         'constructor':constructor,'summaries':ss,'edges':list(vocab.edge_types)})
        stored=self.db.execute('SELECT value FROM endpoints WHERE key=?',(key,)).fetchone() if self.db else None
        if stored:
            raw=json.loads(stored[0]); source=graph_from_record(raw['source'],vocab.node_attribute,vocab.edge_attribute)
        else:
            invariant=extract_typed_invariant(target,edge_types=vocab.edge_types,
                    node_attribute=vocab.node_attribute,edge_attribute=vocab.edge_attribute)
            # No arbitrary rematching: retain indexed target signatures throughout.
            source,diag=construct_typed_graph(invariant,constructor,np.random.default_rng(self.seed+index*1009))
            if not typed_invariant_matches_graph(source,invariant): raise AssertionError('Training endpoint misalignment.')
            diag['valence_policy']=ENDPOINT_VALENCE_POLICY
            raw={'source':graph_record(source,vocab.node_attribute,vocab.edge_attribute),'constructor':diag,
                 'source_spectra':attributed_laplacian_spectra(source,edge_attribute=vocab.edge_attribute).tolist(),
                 'target_spectra':attributed_laplacian_spectra(target,edge_attribute=vocab.edge_attribute).tolist()}
            if ss.get('clustering_histogram',True):
                raw['histogram']=extract_clustering_histogram(target,int(ss.get('clustering_bins',100))).tolist()
            if ss.get('orbit_summary',True): raw['orbit']=extract_orbit_summary(target).tolist()
            if self.db:
                self.db.execute('INSERT OR REPLACE INTO endpoints VALUES (?,?)',(key,json.dumps(raw)))
                self.db.commit()
        item={'source':source,'target':target, **{k:val for k,val in raw.items() if k!='source'}}
        self.memory[index]=item
        while len(self.memory)>max(self.capacity,0): self.memory.popitem(last=False)
        return item


def inference_item(source: nx.Graph, vectorizer, atom_types) -> dict:
    source=validate_graph(source,vectorizer,tuple(atom_types))
    return {'source':source,'source_spectra':attributed_laplacian_spectra(source,
             edge_attribute=vectorizer.vocabulary.edge_attribute).tolist()}


def collate(items: list[dict], vectorizer, atom_types, *, device='cpu', rng=None) -> dict[str,torch.Tensor]:
    """Node indices are arbitrary; only JOINT source/target permutations are used."""
    v=vectorizer; vocab=v.vocabulary; atom_types=tuple(atom_types)
    B=len(items); N=max(len(x['source']) for x in items); R=len(vocab.edge_types)
    out={'mask':torch.zeros(B,N,dtype=torch.bool),
         'atom':torch.zeros(B,N,dtype=torch.long),
         'typed_degrees':torch.zeros(B,N,R),
         'source_labels':torch.zeros(B,N,N,dtype=torch.long),
         'source_spectra':torch.zeros(B,2,N), 'n':torch.zeros(B,dtype=torch.long)}
    has_targets=all('target' in x for x in items)
    if any('target' in x for x in items) and not has_targets: raise ValueError('Mixed train/inference batch.')
    if has_targets:
        out['target_labels']=torch.zeros(B,N,N,dtype=torch.long)
        out['target_spectra']=torch.zeros(B,2,N)
        for key in ('histogram','orbit'):
            if key in items[0]: out[key]=torch.tensor(np.asarray([x[key] for x in items]),dtype=torch.float32)
    degree_graphs=[]
    for b,item in enumerate(items):
        source=normalize_attributed_graph(item['source']); n=len(source)
        target=normalize_attributed_graph(item['target']) if has_targets else None
        inv=extract_typed_invariant(source,edge_types=vocab.edge_types,
                                    node_attribute=vocab.node_attribute,edge_attribute=vocab.edge_attribute)
        if target is not None and not typed_invariant_matches_graph(target,inv):
            raise ValueError('Source and target indexed typed signatures differ.')
        # Canonical index convention is restored before arrays are populated.
        p=np.arange(n) if rng is None else rng.permutation(n)
        mapping={i:int(p[i]) for i in range(n)}
        source=nx.relabel_nodes(source,mapping,copy=True)
        if target is not None: target=nx.relabel_nodes(target,mapping,copy=True)
        degree_graphs.append(source)
        out['mask'][b,:n]=True; out['n'][b]=n
        inv=extract_typed_invariant(source,edge_types=vocab.edge_types,
                                    node_attribute=vocab.node_attribute,edge_attribute=vocab.edge_attribute)
        out['atom'][b,:n]=torch.tensor([atom_types.index(sig.node_type) for sig in inv.signatures])
        out['typed_degrees'][b,:n]=torch.tensor([sig.edge_degrees for sig in inv.signatures],dtype=torch.float32)
        for graph, key in ((source,'source_labels'),(target,'target_labels')):
            if graph is not None:
                for u,w,data in graph.edges(data=True):
                    r=vocab.edge_types.index(data[vocab.edge_attribute])+1
                    out[key][b,u,w]=out[key][b,w,u]=r
        out['source_spectra'][b,:,:n]=torch.as_tensor(np.asarray(item['source_spectra']))
        if target is not None: out['target_spectra'][b,:,:n]=torch.as_tensor(np.asarray(item['target_spectra']))
    features,targets=v.to_training_arrays(degree_graphs)
    out['typed_features']=torch.from_numpy(features)
    for key,val in targets.items(): out['degree_'+key]=torch.from_numpy(val)
    degrees=out['typed_degrees'].sum(-1); n=out['n'].float()
    out['orbit_totals']=torch.stack([(degrees).sum(-1)/n,
           (degrees*(degrees-1)/2).sum(-1)/n,(degrees*(degrees-1)*(degrees-2)/6).sum(-1)/n],-1)
    # Two exact Laplacian traces: ordinary and bond-order weighted.
    orders=torch.tensor([1.5 if int(r)==4 else float(r) for r in vocab.edge_types])
    out['spectral_trace']=torch.stack([degrees.sum(-1),(out['typed_degrees']*orders).sum((1,2))],-1)
    out['spectral_scale']=(out['spectral_trace']/n[:,None]).clamp_min(1)
    return {k:t.to(device) for k,t in out.items()}
