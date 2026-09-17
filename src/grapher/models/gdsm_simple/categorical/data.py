"""Aligned categorical states and exact connected, induced typed size-3 targets."""
from __future__ import annotations
from collections import Counter
from functools import lru_cache
from itertools import combinations, permutations
from math import comb
import networkx as nx
import numpy as np
import torch
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary


@lru_cache(maxsize=200000)
def canonical_triple(raw: tuple[int, ...]) -> tuple[int, ...]:
    x, uv = raw[:3], {(0,1): raw[3], (0,2): raw[4], (1,2): raw[5]}
    return min(tuple(x[i] for i in p) + tuple(uv[tuple(sorted((p[i],p[j])))] for i,j in ((0,1),(0,2),(1,2)))
               for p in permutations(range(3)))


def typed_counts(x: np.ndarray, e: np.ndarray) -> Counter:
    # Enumerate wedges; a triangle is visited at each of its three centres.
    # Deduplicate triples before canonicalizing; disconnected triples are omitted.
    seen, result = set(), Counter()
    for centre in range(len(x)):
        for left, right in combinations(np.flatnonzero(e[centre]).tolist(), 2):
            ijk = tuple(sorted((left, centre, right)))
            if ijk in seen:
                continue
            seen.add(ijk)
            i,j,k = ijk
            raw = (int(x[i]),int(x[j]),int(x[k]),int(e[i,j]),int(e[i,k]),int(e[j,k]))
            result[canonical_triple(raw)] += 1
    return result


def topology_summary(e: np.ndarray, bins: int):
    a = (e > 0).astype(np.float64)
    d = a.sum(1)
    triangles = np.rint(((a@a)*a).sum(1)/2)
    wedges = d*(d-1)/2
    centre = np.maximum(wedges-triangles, 0)
    endpoints = np.maximum(a@(d-1)-2*triangles, 0)
    orbit = np.log1p(np.stack((d,endpoints,centre,triangles), -1).mean(0)).astype(np.float32)
    clustering = np.divide(triangles, wedges, out=np.zeros_like(d), where=wedges>0)
    hist = np.histogram(clustering, bins=bins, range=(0.,1.))[0].astype(np.float32)/len(d)
    return hist, orbit


class TypedGraphlets3:
    def __init__(self, keys):
        self.keys = tuple(tuple(int(i) for i in k) for k in keys)
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("Duplicate typed graphlet vocabulary entries")
        self.index = {k:i for i,k in enumerate(self.keys)}
        self.overflow = len(self.keys)
        self.dimension = len(self.keys)+1

    def encode_counts(self, counts, n):
        out = np.zeros(self.dimension, np.float32)
        for key, count in counts.items():
            out[self.index.get(key, self.overflow)] += count
        total = float(out.sum())
        return out/max(total,1.), total/max(comb(n,3),1)

    def summary(self, x, e):
        return self.encode_counts(typed_counts(x,e),len(x))


def encode_graph(graph, vocab: GraphCategoryVocabulary, max_nodes):
    if not isinstance(graph,nx.Graph) or graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Expected a simple undirected loop-free NetworkX graph")
    nodes = list(graph.nodes())
    if not 1 <= len(nodes) <= max_nodes:
        raise ValueError(f"Graph has {len(nodes)} nodes, outside 1..{max_nodes}")
    lookup = {v:i for i,v in enumerate(nodes)}
    x = np.array([vocab.node_index(graph.nodes[v]) for v in nodes],np.int16)
    e = np.zeros((len(nodes),len(nodes)),np.int16)
    for u,v,attrs in graph.edges(data=True):
        e[lookup[u],lookup[v]]=e[lookup[v],lookup[u]]=vocab.edge_index(attrs)
    return x,e


def decode_graph(x, e, vocab: GraphCategoryVocabulary):
    graph = nx.Graph()
    for i,value in enumerate(x):
        attrs = {vocab.node_attribute: vocab.node_value(int(value))} if vocab.node_attribute else {}
        graph.add_node(i, **attrs)
    for i,j in zip(*np.nonzero(np.triu(e,1))):
        attrs = {vocab.edge_attribute: vocab.edge_value(int(e[i,j]))} if vocab.edge_attribute else {}
        graph.add_edge(int(i),int(j),**attrs)
    return graph


def make_record(graph, vocab, max_nodes, bins, graphlet_config=None):
    x,e = encode_graph(graph,vocab,max_nodes)
    adjacency=(e>0).astype(np.float64)
    vals,u = np.linalg.eigh(adjacency)
    clustering,orbit=topology_summary(e,bins)
    if graphlet_config is not None and graphlet_config.get('sizes') is not None:
        from .multiscale import count_multi
        counts=count_multi(x,e,graphlet_config['sizes'],limit=graphlet_config['max_connected_subsets'])
    else:
        counts=typed_counts(x,e)
    return {"x":x,"e":e,"z":(vals/len(x)**.5).astype(np.float32),"clustering":clustering,
            "orbit":orbit,"counts":counts,"degrees":adjacency.sum(1).astype(np.int64)}, u.astype(np.float32)


def collate(records, basis: TypedGraphlets3, *, device):
    b=len(records); width=max(len(row["x"]) for row in records)
    x=np.zeros((b,width),np.int64); e=np.zeros((b,width,width),np.int64)
    z=np.zeros((b,width),np.float32); anchor=np.zeros_like(z); mask=np.zeros((b,width),bool)
    hist,mass=[],[]
    for i,row in enumerate(records):
        n=len(row["x"]); x[i,:n]=row["x"]; e[i,:n,:n]=row["e"]; z[i,:n]=row["z"]
        anchor[i,:n]=row["anchor"]; mask[i,:n]=True
        h,m=basis.encode_counts(row["counts"],n); hist.append(h); mass.append(m)
    result={"x":x,"e":e,"z":z,"anchor":anchor,"mask":mask,"histogram":np.stack(hist),
            "mass":np.asarray(mass,np.float32),"clustering":np.stack([r["clustering"] for r in records]),
            "orbit":np.stack([r["orbit"] for r in records])}
    if getattr(basis,'multiscale',False):
        result['graphlet_order_mask']=np.array([[len(r['x'])>=k for k in basis.orders] for r in records],bool)
    return {k:torch.as_tensor(v,device=device) for k,v in result.items()}


def permute_aligned(batch, generator):
    # Graph-index permutations do NOT permute rank-indexed spectra or anchors.
    x,e=batch["x"].clone(),batch["e"].clone()
    for i,n in enumerate(batch["mask"].sum(1).tolist()):
        p=torch.randperm(n,device=x.device,generator=generator)
        x[i,:n]=x[i,:n][p]; e[i,:n,:n]=e[i,:n,:n][p][:,p]
    return {**batch,"x":x,"e":e}
