import networkx as nx
import numpy as np
import random
import os
import argparse
import toml
from pathlib import Path
import matplotlib.pyplot as plt
from utils import load_graph_from_directory
import matplotlib.patheffects as pe

def random_edge_swap_with_record(G, rng=None, max_tries=200):
    """
    Force one valid edge swap on an undirected simple graph G.
    Returns: G_new, removed_edges, added_edges
    """
    assert not G.is_directed(), "This helper assumes an undirected simple graph."
    rng = rng or random
    edges = list(G.edges())
    if len(edges) < 2:
        raise ValueError("Need at least 2 edges to swap.")

    for _ in range(max_tries):
        (u, v), (x, y) = rng.sample(edges, 2)

        # Two possible rewirings
        if rng.random() < 0.5:
            a, b = u, y
            c, d = x, v
        else:
            a, b = u, x
            c, d = v, y

        # prevent self-loops, duplicates
        if len({a, b}) < 2 or len({c, d}) < 2:
            continue
        if len({tuple(sorted((a, b))), tuple(sorted((c, d)))}) < 2:
            continue
        if G.has_edge(a, b) or G.has_edge(c, d):
            continue

        # valid: do the swap on a copy
        G_new = G.copy()
        # normalize old endpoints for undirected graphs
        u, v = (u, v) if u <= v else (v, u)
        x, y = (x, y) if x <= y else (y, x)
        G_new.remove_edge(u, v)
        G_new.remove_edge(x, y)
        G_new.add_edge(a, b)
        G_new.add_edge(c, d)
        removed = [(u, v), (x, y)]
        added = [(a, b), (c, d)]
        return G_new, removed, added

    raise RuntimeError("Could not find a valid swap; try a different seed/graph.")

def draw_swap_before_after(G, G_swapped, removed, added, pos=None, fname=None, with_labels=False):
    """
    Before: removed edges in RED.
    After : added edges in GREEN.
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    node_kw = dict(
        node_size=2400,
        node_color="skyblue",
        edgecolors="black",
        linewidths=10,
    )
    base_edge_kw = dict(width=10, alpha=0.7, edge_color="black")

    hi_removed_kw = dict(width=10, alpha=1.0, edge_color="red", connectionstyle="arc3,rad=0.15")
    hi_added_kw   = dict(width=10, alpha=1.0, edge_color="green", connectionstyle="arc3,rad=0.15")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax1, ax2 = axes

    # BEFORE
    ax1.set_title("Before")
    nx.draw_networkx_nodes(G, pos, ax=ax1, **node_kw)
    nx.draw_networkx_edges(G, pos, ax=ax1, **base_edge_kw)
    if removed:
        nx.draw_networkx_edges(G, pos, edgelist=removed, ax=ax1, **hi_removed_kw)
    if with_labels:
        nx.draw_networkx_labels(G, pos, ax=ax1, font_size=12)
    ax1.set_axis_off()

    # AFTER
    ax2.set_title("After")
    nx.draw_networkx_nodes(G_swapped, pos, ax=ax2, **node_kw)
    nx.draw_networkx_edges(G_swapped, pos, ax=ax2, **base_edge_kw)
    if added:
        nx.draw_networkx_edges(G_swapped, pos, edgelist=added, ax=ax2, **hi_added_kw)
    if with_labels:
        nx.draw_networkx_labels(G_swapped, pos, ax=ax2, font_size=12)
    ax2.set_axis_off()

    plt.tight_layout()

    plt.show()
# ---------------- Example ----------------

# Example Usage:
def main(args):
    config_dir = Path("configs")
    dataset_dir = Path("datasets") / args.dataset_dir
    graphs, _,min_node = load_graph_from_directory(dataset_dir)
    G = random.choice(graphs)
    G2, removed_edges, added_edges = random_edge_swap_with_record(G, rng=random)

    draw_swap_before_after(
        G, G2,
        removed=removed_edges,
        added=added_edges,
        fname="edge_swap_highlight.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Mixing Time for Graph Generation')
    parser.add_argument('--dataset-dir', type=str, help='Path to the directory containing graph files')
    args = parser.parse_args()
    main(args)
