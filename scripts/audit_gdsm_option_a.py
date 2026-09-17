#!/usr/bin/env python
"""Audit trusted locally generated Option-A artifacts without running MMD."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np

from grapher.utils.networkx_pickle import load_trusted_networkx_pickle


def audit(directory: Path) -> dict:
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('generation_mode') != 'spectral_decode':
        raise ValueError('This is not an Option-A spectral_decode generation.')
    names = ('initial_graphs.pkl', 'base_graphs.pkl', 'final_pre_rewire_graphs.pkl',
             'sampled_degree_sequences.pkl', 'final_degree_sequences.pkl')
    loaded = {name: load_trusted_networkx_pickle(directory/name) for name in names}
    n = len(loaded['base_graphs.pkl'])
    if n == 0 or any(len(batch) != n for batch in loaded.values()):
        raise ValueError('Artifact batches are empty or have inconsistent lengths.')
    initial_same, prior_same, local_same, connected, simple = [], [], [], [], []
    for i in range(n):
        source, final, pre = (loaded[k][i] for k in names[:3])
        d = [int(x) for x in loaded['sampled_degree_sequences.pkl'][i]]
        if any(set(g) != set(range(len(d))) for g in (source, final, pre)):
            raise AssertionError(f'Graph {i} lost or relabelled nodes.')
        df = [final.degree(v) for v in range(len(final))]
        ds = [source.degree(v) for v in range(len(source))]
        dp = [pre.degree(v) for v in range(len(pre))]
        if df != loaded['final_degree_sequences.pkl'][i]:
            raise AssertionError(f'Graph {i}: saved final degrees do not match the graph.')
        local_same.append(dp == df)
        initial_same.append(sorted(ds) == sorted(df))
        prior_same.append(sorted(d) == sorted(df))
        connected.append(nx.is_connected(final))
        simple.append(not final.is_directed() and not final.is_multigraph() and not nx.number_of_selfloops(final))
    if not all(local_same) or not all(simple):
        raise AssertionError('Final local rewiring violated degree preservation or graph simplicity.')
    diagnostics = json.loads((directory/'rewiring_diagnostics.json').read_text())
    records = diagnostics['per_graph']
    if len(records) != n:
        raise ValueError('Diagnostic batch length differs from graph batch length.')
    all_events = [e for row in records for e in row['events']]
    if any(e['degrees_before_rewiring'] != e['degrees_after_rewiring'] for e in all_events):
        raise AssertionError('An intermediate local swap changed degrees.')
    return {
        'num_graphs': n, 'global_degree_constraint': False,
        'final_graphs_simple_rate': float(np.mean(simple)),
        'final_connected_rate': float(np.mean(connected)),
        'initial_to_final_degree_multiset_preservation_rate': float(np.mean(initial_same)),
        'prior_to_final_degree_multiset_preservation_rate': float(np.mean(prior_same)),
        'final_event_degree_preservation_rate': float(np.mean(local_same)),
        'all_local_rewiring_events_preserve_degrees': True,
        'mean_spectral_degree_change_steps': float(np.mean([r['spectral_degree_change_steps'] for r in records])),
        'mean_feedback_events': float(np.mean([sum(e['spectrum_feedback_applied'] for e in r['events']) for r in records])),
        'note': 'Initial/prior degrees may change; final_pre_rewire and final degrees must match. Equality in some samples is allowed.',
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.generated_dir), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
