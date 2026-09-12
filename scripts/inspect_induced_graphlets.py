#!/usr/bin/env python
"""Print/export the exact versioned, untyped induced graphlet catalogue."""
import argparse
from pathlib import Path
from grapher.rewiring_mlp.generic.induced_graphlets import InducedGraphletSpec, catalogue
from grapher.rewiring_mlp.generic.joint_checkpointing import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--scope', choices=('all', 'connected'), default='all')
    p.add_argument('--json-out')
    args = p.parse_args()
    spec = InducedGraphletSpec(args.k, args.scope)
    cat = catalogue(spec.k, spec.scope)
    print(f'Induced topology graphlets: k={spec.k} scope={spec.scope} bins={spec.width}')
    print('Normalization:', spec.metadata()['normalization'])
    print('Fingerprint:', cat['fingerprint'])
    print('Atom and bond labels: ignored (topology-only summary)')
    for index, row in enumerate(cat['bins']):
        print(f"{index:3d} {row['id']:18s} connected={str(row['connected']):5s} edges={row['num_edges']} degrees={row['degrees']}")
    if args.json_out:
        atomic_json(cat, Path(args.json_out))

if __name__ == '__main__':
    main()
