import networkx as nx
import numpy as np
import random
import os
import argparse
import toml
from pathlib import Path
import matplotlib.pyplot as plt
from utils import load_graph_from_directory

# Function for a single edge rewiring step
def edge_rewire(G):
    edges = list(G.edges())
    np.random.shuffle(edges)
    for (u, v) in edges:
        for (x, y) in edges:
            # Check conditions to avoid self-loop and multiple edges
            if len({u, v, x, y}) == 4 and not G.has_edge(u, x) and not G.has_edge(v, y):
                G.remove_edges_from([(u, v), (x, y)])
                G.add_edges_from([(u, x), (v, y)])
                return

# Compute statistical features of the graph
def graph_features(G):
    clustering = np.mean(list(nx.clustering(G).values()))
    degree_seq = np.array(sorted([d for _, d in G.degree()]))
    return clustering, degree_seq

# Measure distance between feature distributions
def features_distance(f1, f2):
    clustering_diff = np.abs(f1[0] - f2[0])
    degree_diff = np.linalg.norm(f1[1] - f2[1])
    return clustering_diff + degree_diff

# Empirical estimation of mixing time
def estimate_mixing_time(G_init, config):
    G = G_init.copy()
    prev_features = graph_features(G)
    distances = []

    for step in range(1, config['mixing_time']['max_steps'] + 1):
        edge_rewire(G)

        if step % config['mixing_time']['check_interval'] == 0:
            current_features = graph_features(G)
            distance = features_distance(prev_features, current_features)
            distances.append(distance)

            if distance < config['mixing_time']['threshold']:
                return step, distances

            prev_features = current_features

    return config['mixing_time']['max_steps'], distances

# Example Usage:
def main(args):
    config_dir = Path("configs")
    dataset_dir = Path("datasets") / args.dataset_dir
    config = toml.load(config_dir / args.config)
    graphs, max_node = load_graph_from_directory(dataset_dir)

    sample_count = config['mixing_time'].get('sample_count', 1)
    total_steps = 0
    all_distances = []

    for _ in range(sample_count):
        G_init = random.choice(graphs)
        mixing_step, distances = estimate_mixing_time(G_init, config)
        total_steps += mixing_step
        all_distances.append(distances)

    avg_mixing_time = total_steps / sample_count
    print(f"Average Mixing Time over {sample_count} samples: {avg_mixing_time:.2f} steps")

    # Compute average distance per time step
    max_len = max(len(d) for d in all_distances)
    padded = [np.pad(d, (0, max_len - len(d)), constant_values=np.nan) for d in all_distances]
    avg_distances = np.nanmean(padded, axis=0)

    # Plot average distance trajectory
    plt.plot(range(len(avg_distances)), avg_distances)
    plt.xlabel('Interval Steps (in hundreds)')
    plt.ylabel('Average Feature Distance')
    plt.title('Average Empirical Mixing Time Trajectory')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Mixing Time for Graph Generation')
    parser.add_argument('--dataset-dir', type=str, help='Path to the directory containing graph files')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file in TOML format')
    args = parser.parse_args()
    main(args)
