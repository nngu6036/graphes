#!/usr/bin/env python3
"""Verify complete topology-state hashes AND serialized graph topologies.

Only load trusted generation artifacts: NetworkX pickle is executable data.
No model is trained or sampled and no experimental parameters are inferred.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx

from grapher.utils.networkx_pickle import load_trusted_networkx_pickle


def verify_pairing(directories: list[Path]) -> dict[str, Any]:
    if len(directories) < 2:
        raise ValueError('Provide at least two generation directories')
    records = []
    for directory in directories:
        meta = json.loads((directory / 'topology_pairing.json').read_text())
        manifest = json.loads((directory / 'manifest.json').read_text())
        graphs = load_trusted_networkx_pickle(directory / 'base_graphs.pkl')
        if meta.get('format') != 'gdsm_attributed_topology_pairing_v1':
            raise ValueError(f'Unsupported pairing record in {directory}')
        if len(graphs) != len(meta['per_graph_sha256']) or len(graphs) != manifest['num_requested']:
            raise ValueError(f'Incomplete generation denominator in {directory}')
        records.append((meta, graphs))
    base, base_graphs = records[0]
    comparisons = []
    for directory, (other, graphs) in zip(directories[1:], records[1:]):
        if len(graphs) != len(base_graphs):
            raise ValueError('Compared runs have different sample counts')
        if other['checkpoint_sha256'] != base['checkpoint_sha256']:
            raise ValueError('Compared runs use different trained checkpoints')
        state_mismatches = [i for i, (a, b) in enumerate(zip(base['per_graph_sha256'], other['per_graph_sha256'])) if a != b]
        graph_mismatches = []
        for i, (a, b) in enumerate(zip(base_graphs, graphs)):
            # Compare indexed topology, ignoring attributes and graph metadata.
            if (set(a.nodes) != set(b.nodes)
                or {frozenset(e) for e in a.edges} != {frozenset(e) for e in b.edges}):
                graph_mismatches.append(i)
        comparisons.append({'directory': str(directory), 'state_mismatch_indices': state_mismatches,
                            'topology_mismatch_indices': graph_mismatches})
    return {'format': 'gdsm_decoder_pairing_verification_v1', 'reference': str(directories[0]),
            'num_graphs': len(base_graphs), 'comparisons': comparisons,
            'paired': all(not x['state_mismatch_indices'] and not x['topology_mismatch_indices'] for x in comparisons)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated-dirs', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = verify_pairing(args.generated_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['paired'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
