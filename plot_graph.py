import networkx as nx
import numpy as np
import random
import os
import argparse
import toml
from pathlib import Path
import matplotlib.pyplot as plt
from utils import load_graph_from_directory, havel_hakimi_construction
import matplotlib.patheffects as pe

from create_dataset import generate_community_graph

def plot_graph_evolution(snapshots):
    """
    Plot a sequence of graph snapshots in one horizontal figure.

    Args:
        snapshots (list[tuple[nx.Graph, int]]): list of (graph, step) pairs, for ONE graph.
        idx (int): 0-based index of this graph (used in filename).
        out_dir (str): output folder.
    """

    if not snapshots:
        return

    # Use a FIXED layout across all panels for comparability.
    # Compute on the first snapshot and reuse positions.
    G0, _ = snapshots[0]
    # Seeded layout for reproducibility; tweak seed if you like.
    pos = nx.spring_layout(G0, seed=42)

    fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4))
    if len(snapshots) == 1:
        axes = [axes]

    for ax, (G, label) in zip(axes, snapshots):
        # Draw with fixed positions; nodes that don't exist will be ignored (same N here).
        nx.draw(G, pos=pos, node_size=40, with_labels=False, ax=ax)
        ax.set_title(label)
        ax.axis("off")

    plt.show()


def rewire_edges(
    G: nx.Graph,
    max_retry_step: int = 64,
):
    for _ in range(max_retry_step):
        all_edges = list(G.edges())
        e1, e2 = random.sample(all_edges, 2)
        u, v = e1
        x, y = e2
        # Disjoint endpoints for a valid 2-edge swap
        if len({u, v, x, y}) != 4:
            continue
        if not G.has_edge(u, x) and not G.has_edge(v, y):
            G.remove_edges_from([(u, v), (x, y)])
            G.add_edges_from([(u, x), (v, y)])
            return G
        if not G.has_edge(u, y) and not G.has_edge(v, x):
            G.remove_edges_from([(u, v), (x, y)])
            G.add_edges_from([(u, y), (v, x)])
            return G
    return G


# Example Usage:
def main():
    G = generate_community_graph(25,30)
    seq = [deg for _, deg in G.degree()]
    G_hh = havel_hakimi_construction(seq)
    num_steps = 2000
    snapshots = []
    step_size = max(1, num_steps // 8)   # ~8 panels
    plot_index = num_steps
    snapshots.append((G.copy(), "G"))
    for t in reversed(range(num_steps + 1)):
        G = rewire_edges(G.copy())
        if t == plot_index:
            snapshots.append((G.copy(), f"Step {t}"))  # store a copy of the graph and the step
            plot_index -= step_size
    
    snapshots.append((G_hh, f"G-HH"))
    plot_graph_evolution(snapshots)


if __name__ == "__main__":
    main()
