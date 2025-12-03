import networkx as nx
import numpy as np
import random
import os
import argparse
import toml
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# bring in the helpers we use
from utils import (
    load_graph_from_directory,
    deterministic_connected_havel_hakimi_from_graph,
    make_lambda_dist,
    transform_to_hh_via_guided_rewiring,
    draw_graphs_grid
)

def _sample_indices(n, max_samples=12):
    """
    Pick up to max_samples indices from [0..n-1], covering the whole range.
    Always include first and last.
    """
    if n <= max_samples:
        return list(range(n))
    # spread roughly evenly
    idx = np.linspace(0, n - 1, num=max_samples, dtype=int)
    # ensure unique and sorted
    idx = sorted(set(idx.tolist() + [0, n - 1]))
    return idx

def _compute_distance_series(traj, lambda_dist):
    """
    Returns a list of distances d(G_t, H) for each graph in traj.
    `traj` is a list of (G_after_rewire, added_edge_pair)
    """
    vals = []
    for Gt, _ in traj:
        vals.append(lambda_dist(Gt))
    return vals

# Example Usage:
def main(args):
    config_dir = Path("configs")
    dataset_dir = Path("datasets") / args.dataset_dir
    config = toml.load(config_dir / args.config)
    graphs, max_node = load_graph_from_directory(dataset_dir)

    max_steps = config.get('rewiring', {}).get('max_steps', 100)

    # pick an initial graph and its HH target
    G_init = graphs[0]
    G_hh = deterministic_connected_havel_hakimi_from_graph(G_init)

    # fixed positions for consistent visualization across trajectory
    # we use the same node set (rewiring preserves node set)
    pos = nx.spring_layout(G_init, seed=42)

    names = [
            'symmetric_edit',
            'spectral',
            'bures_wasserstein',
            'effective_resistance_fro',
            'deltacon',
            'netlsd'
        ]
    name = 'netlsd'
    for _ in range(6):

        print("Computing trajectory & plots for:", name)
        frames = []

        # Build callable d(G, H)
        lambda_dist = make_lambda_dist(name, G_hh)

        # Build trajectory (list of (G_after_rewire, added_pair))
        trajectory = transform_to_hh_via_guided_rewiring(
            G_init, G_hh, lambda_dist, max_steps
        )
        frames = [graph for graph, _, _ in trajectory]
        print("Trajectory length", len(frames))
        frames = [G_init] + frames + [G_hh]

        draw_graphs_grid(frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot graph evolution along guided-rewiring trajectory')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Path to the directory containing graph files (under datasets/)')
    parser.add_argument('--config', type=str, required=True, help='TOML config filename under configs/')
    args = parser.parse_args()
    main(args)
