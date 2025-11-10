import random
import math
import matplotlib.pyplot as plt
from torch_geometric.data import Batch
import toml
import torch
import torch.nn as nn
from utils import *
from model_graph_sed import  GraphSEDModel
import os
import argparse
from pathlib import Path
import networkx as nx
from torch_geometric.data import Data


def to_pyg_data(G: nx.Graph) -> Data:
    """Convert NetworkX graph to PyG Data with simple node features (degree as scalar)."""
    Gc = nx.convert_node_labels_to_integers(G, ordering="sorted")
    degs = torch.tensor([Gc.degree(i) for i in range(Gc.number_of_nodes())], dtype=torch.float32).unsqueeze(-1)
    data = from_networkx(Gc)
    data.x = degs  # [N,1]
    return data

@torch.no_grad()
def greedy_hh_rewiring_test(
    model: GraphSEDModel,
    G: nx.Graph,
    steps: int = 100,
    seed: int | None = 0,
    draw_each_step: bool = True,
    figsize_init=(10, 4),
    figsize_step=(4, 4),
):
    """
    Given a trained GraphSEDModel and a graph G:
      1) Plot G and its Havel–Hakimi graph H (same degree sequence).
      2) Compute and print ||h(G) - h(H)||_2 using model embeddings.
      3) Repeat for `steps` iterations:
         - Pick a first edge e1 uniformly at random from current G.
         - For every possible second edge e2 (disjoint from e1), consider the two valid rewiring
           orientations; choose the candidate G_t that minimizes ||h(G_t) - h(H)||_2.
         - Print the chosen distance, draw G_t, then perform the swap (set G := G_t).

    Notes:
      • Rewiring preserves the degree sequence, so H stays fixed throughout.
      • Uses degree scalar as the node feature via to_pyg_data().
    """
    rng = random.Random(seed)
    np_rng = nx.utils.create_random_state(seed)
    model.eval()

    # -- Helpers ----------------------------------------------------------------
    def _embed_graph(graph: nx.Graph) -> torch.Tensor:
        data = to_pyg_data(graph)
        batch = Batch.from_data_list([data])
        h = model.embed(batch)         # [1, D]
        return h.squeeze(0)            # [D]

    def _euclid(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.norm(a - b, p=2).item())

    def _valid_simple_edges(graph: nx.Graph, a, b, c, d) -> bool:
        # simple graph: no self-loop, distinct endpoints, and no parallel edges
        if len({a, b, c, d}) < 4:
            return False
        if a == b or c == d:
            return False
        if graph.has_edge(a, b) or graph.has_edge(c, d):
            return False
        return True

    def _rewired_graph(graph: nx.Graph, e1, e2, orientation: int) -> nx.Graph | None:
        """
        orientation=0 -> (u,x),(v,y)
        orientation=1 -> (u,y),(v,x)
        Returns a new graph if the rewiring is valid; otherwise None.
        """
        (u, v) = e1
        (x, y) = e2
        if len({u, v, x, y}) < 4:
            return None
        if orientation == 0:
            a, b, c, d = u, x, v, y
        else:
            a, b, c, d = u, y, v, x
        if not _valid_simple_edges(graph, a, b, c, d):
            return None
        Htmp = graph.copy()
        # Remove original edges first (ensure they exist)
        if not Htmp.has_edge(u, v) or not Htmp.has_edge(x, y):
            return None
        Htmp.remove_edge(u, v)
        Htmp.remove_edge(x, y)
        # Double-check we didn't accidentally create duplicates
        if Htmp.has_edge(a, b) or Htmp.has_edge(c, d):
            return None
        Htmp.add_edge(a, b)
        Htmp.add_edge(c, d)
        return Htmp

    # -- Build Havel–Hakimi reference (fixed through the trajectory) ------------
    H = havel_hakimi_construction(G)
    # share the same node set ordering; spring layout on the original node set for stable visuals
    nodes_sorted = sorted(set(G.nodes()) | set(H.nodes()))
    pos = nx.spring_layout(nx.relabel_nodes(G, {u: i for i, u in enumerate(nodes_sorted)}), seed=seed)
    pos = {u: pos[nodes_sorted.index(u)] if isinstance(pos, dict) else pos[i] for i, u in enumerate(nodes_sorted)}

    # -- Initial plot: G vs H ---------------------------------------------------
    plt.figure(figsize=figsize_init)
    plt.subplot(1, 2, 1)
    nx.draw(G, pos=pos, with_labels=True, node_size=300)
    plt.title("Initial G")

    plt.subplot(1, 2, 2)
    nx.draw(H, pos=pos, with_labels=True, node_size=300)
    plt.title("Havel–Hakimi H")
    plt.tight_layout()
    plt.show()

    # -- Initial distances ------------------------------------------------------
    h_G = _embed_graph(G)
    h_H = _embed_graph(H)
    d0 = _euclid(h_G, h_H)
    print(f"[Init] ||h(G) - h(H)||_2 = {d0:.6f}")

    # -- Greedy loop towards HH in embedding space ------------------------------
    for t in range(1, steps + 1):
        edges = list(G.edges())
        if len(edges) < 2:
            print(f"[Step {t:03d}] Not enough edges to rewire. Stopping.")
            break

        # pick first edge randomly
        e1 = edges[rng.randrange(len(edges))]

        best_dist = math.inf
        best_graph = None
        best_choice = None  # (e1, e2, orientation)

        # enumerate all candidate second edges (disjoint from e1)
        (u, v) = e1
        for e2 in edges:
            if e2 == e1:
                continue
            (x, y) = e2
            if len({u, v, x, y}) < 4:
                continue

            # Two orientations
            for orient in (0, 1):
                G_cand = _rewired_graph(G, e1, e2, orient)
                if G_cand is None:
                    continue
                # compute embedding distance to H
                h_cand = _embed_graph(G_cand)
                dist = _euclid(h_cand, h_H)
                if dist < best_dist:
                    best_dist = dist
                    best_graph = G_cand
                    best_choice = (e1, e2, orient)

        if best_graph is None:
            print(f"[Step {t:03d}] No valid rewiring found from chosen e1={e1}. Retrying with a new e1.")
            # Try a different e1 in the same step (fallback)
            # If none works after a few tries, we break to avoid infinite loops.
            retries = 0
            max_retries = 5
            success = False
            while retries < max_retries and not success:
                e1 = edges[rng.randrange(len(edges))]
                (u, v) = e1
                for e2 in edges:
                    if e2 == e1:
                        continue
                    (x, y) = e2
                    if len({u, v, x, y}) < 4:
                        continue
                    for orient in (0, 1):
                        G_cand = _rewired_graph(G, e1, e2, orient)
                        if G_cand is None:
                            continue
                        h_cand = _embed_graph(G_cand)
                        dist = _euclid(h_cand, h_H)
                        if dist < math.inf:
                            best_dist = dist
                            best_graph = G_cand
                            best_choice = (e1, e2, orient)
                            success = True
                            break
                    if success:
                        break
                retries += 1

            if best_graph is None:
                print(f"[Step {t:03d}] Failed to find any valid rewiring after retries. Stopping.")
                break

        # print distance and draw chosen G_t
        (e1_chosen, e2_chosen, orient_chosen) = best_choice
        print(f"[Step {t:03d}] e1={e1_chosen}, e2={e2_chosen}, orient={orient_chosen} -> dist={best_dist:.6f}")

        if draw_each_step:
            plt.figure(figsize=figsize_step)
            nx.draw(best_graph, pos=pos, with_labels=True, node_size=300)
            plt.title(f"G_t (step {t})\n||h(G_t)-h(H)||={best_dist:.4f}")
            plt.tight_layout()
            plt.show()

        # apply the swap: advance G
        G = best_graph
        h_G = _embed_graph(G)  # (not strictly needed further, but kept for clarity)

    print("[Done]")

    # return final graph and the HH reference for downstream use if needed
    return G, H


def main(args):
    config_dir = Path("configs")
    dataset_dir = Path("datasets") / args.dataset_dir
    model_dir = Path("models")
    config = toml.load(config_dir / args.config)
    model = GraphSEDModel(
        in_channels=config['training']['in_channels'],
        hidden_dim=config['training']['hidden_dim'],
        num_layers=config['training']['num_layers'],
        proj_dim=config['training']['proj_dim'],
        enc_dropout=0.1,
        l2_normalize=True,
        dec_hidden=config['training']['dec_hidden'],
        dec_dropout=0.1,
    )
    model.load_model(model_dir / args.input_model)
    print(f"Model GraphSEDModel loaded from {args.input_model}")
    graphs, max_node = load_graph_from_directory(dataset_dir)
    G = random.choice(graphs)
    _ = greedy_hh_rewiring_test(model, G, steps=100, seed=0, draw_each_step=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standard-VAE for Graph Generation')
    parser.add_argument('--dataset-dir', type=str, help='Path to the directory containing graph files')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file in TOML format of Graph-ER')
    parser.add_argument('--input-model', type=str, help='Path to load a pre-trained model')
    args = parser.parse_args()
    main(args)