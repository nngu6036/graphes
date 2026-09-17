#!/usr/bin/env python3
"""Dataset-free eigensolver + joint forward/backward preflight for GDSM.

Run from the project root with PYTHONPATH=src. GDSM_EIGH_BACKEND=cpu bypasses
CUDA eigh only; --device still selects where the neural network is trained.
This synthetic check is not a QM9 quality or throughput benchmark.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from grapher.models.gdsm_simple.categorical.config import DEFAULTS
from grapher.models.gdsm_simple.categorical.data import collate
from grapher.models.gdsm_simple.categorical.model import SpectralCategoricalDenoiser
from grapher.models.gdsm_simple.categorical.noise import MarginalNoise, cosine_alpha_bar
from grapher.models.gdsm_simple.categorical.pipeline import prepare_data, _loss_batch
from grapher.models.gdsm_simple.categorical.spectral import eigenpairs, eigh_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--graphs', type=int, default=256)
    parser.add_argument('--max-nodes', type=int, default=9)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-json', type=Path)
    args = parser.parse_args()
    if min(args.graphs, args.max_nodes, args.batch_size) < 1:
        parser.error('Graph count, maximum nodes and batch size must be positive')
    device = torch.device('cuda:0' if args.device == 'gpu' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable in this Python environment')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    graphs = []
    for i in range(args.graphs):
        n = args.max_nodes if i < 5 else int(rng.integers(1, args.max_nodes+1))
        kind = i % 5
        if kind == 0:
            graph = nx.empty_graph(n)
        elif kind == 1:
            graph = nx.complete_graph(n)
        elif kind == 2:
            graph = nx.path_graph(n)
        elif kind == 3:
            graph = nx.cycle_graph(n) if n > 2 else nx.path_graph(n)
        else:
            graph = nx.gnp_random_graph(n, .25, seed=int(rng.integers(2**31)))
        for v in graph:
            graph.nodes[v]['node_type'] = str(int(rng.integers(4)))
        for u, v in graph.edges:
            graph.edges[u, v]['edge_type'] = int(rng.integers(1, 4))
        graphs.append(graph)
    cfg = copy.deepcopy(DEFAULTS)
    cfg['categories'] = {'node_attribute': 'node_type', 'edge_attribute': 'edge_type'}
    cfg['graphlets']['clustering_bins'] = 10
    rows, _, vocab, basis, _, metadata = prepare_data(graphs, graphs[:1], args.max_nodes, cfg, args.seed)
    model = SpectralCategoricalDenoiser(
        node_classes=vocab.num_node_categories, edge_classes=vocab.num_edge_categories,
        graphlet_classes=basis.dimension, hidden_dim=16, num_layers=1, num_heads=2,
        ff_dim=32, clustering_bins=10, max_nodes=args.max_nodes,
    ).to(device)
    schedule = cosine_alpha_bar(20, device=device)
    xn, en = MarginalNoise(metadata['node_marginal'], schedule), MarginalNoise(metadata['edge_marginal'], schedule)
    generator = torch.Generator(device=device).manual_seed(args.seed+1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    max_residual = 0.
    max_orthogonality = 0.
    losses = []
    checked = 0
    for start in range(0, len(rows), args.batch_size):
        part = rows[start:start+args.batch_size]
        batch = collate(part, basis, device=device)
        values, vectors = eigenpairs(batch['e'], batch['mask'])
        for i, n in enumerate(batch['mask'].sum(1).tolist()):
            u = vectors[i, :n, :n].double().cpu()
            v = values[i, :n].double().cpu() * n**.5
            a = (batch['e'][i, :n, :n] > 0).double().cpu()
            max_residual = max(max_residual, float(((u*v[None]) @ u.T-a).abs().max()))
            max_orthogonality = max(max_orthogonality, float((u.T @ u-torch.eye(n, dtype=u.dtype)).abs().max()))
            checked += 1
        loss, _ = _loss_batch(model, part, basis, cfg, schedule, xn, en, generator, device, permutations=True)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite synthetic joint loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                raise RuntimeError('Nonfinite synthetic gradient')
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if max_residual > 2e-5 or max_orthogonality > 2e-6:
        raise RuntimeError('Returned eigenpairs do not reconstruct the original binary graphs accurately')
    result = {
        'status': 'passed', 'test_kind': 'synthetic_numerical_preflight_not_quality_benchmark',
        'python': platform.python_version(), 'torch': torch.__version__,
        'cuda_build': torch.version.cuda, 'device': str(device),
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'seed': args.seed, 'max_nodes': args.max_nodes, 'checked_graphs': checked,
        'optimizer_steps': len(losses), 'finite_losses_and_gradients': True,
        'max_eigenpair_reconstruction_error': max_residual,
        'max_orthogonality_error': max_orthogonality,
        'solver': eigh_diagnostics(),
    }
    text = json.dumps(result, indent=2)+'\n'
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text)
    print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
