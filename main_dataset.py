import json
import networkx as nx
import random
from pathlib import Path
import toml
import argparse
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

def check_sequence_validity(seq):
    """Checks if a degree sequence is valid after removing all zeros."""
    if len(seq) == 0:
        return False,1
    # Degree sequence sum must be even
    if sum(seq) % 2 != 0:
        return False,2
    # Sort in descending order
    sorted_seq = sorted(seq, reverse=True)
    # Apply Erdős–Gallai theorem
    for k in range(1, len(sorted_seq) + 1):
        lhs = sum(sorted_seq[:k])
        rhs = k * (k - 1) + sum(min(d, k) for d in sorted_seq[k:])
        if lhs > rhs:
            return False,3
    return True, 0

def generate_planar_graph(node_count, edge_count):
    """Generate a planar graph using a Delaunay triangulation approach."""
    points = [(random.random(), random.random()) for _ in range(node_count)]
    G = nx.random_geometric_graph(node_count, radius=0.5)
    if len(G.edges) > edge_count:
        edges_to_remove = sorted(G.edges, key=lambda e: random.random())[:len(G.edges) - edge_count]
        G.remove_edges_from(edges_to_remove)
    return G

def generate_lobster_graph(node_count, edge_count):
    """Generate a lobster graph."""
    return nx.random_lobster(node_count // 2, 0.5, 0.5)

def generate_ego_graph(node_count):
    """Generate an ego graph."""
    G = nx.erdos_renyi_graph(node_count, 0.05)
    ego_node = random.choice(list(G.nodes))
    return nx.ego_graph(G, ego_node)

def generate_protein_graph(node_count):
    """Generate a protein-like random graph."""
    return nx.powerlaw_cluster_graph(node_count, 3, 0.05)

def generate_point_cloud_graph(node_count):
    """Generate a 3D point cloud graph."""
    return nx.random_geometric_graph(node_count, radius=0.25)

def generate_sbm_graph(block_sizes, probabilities):
    """Generate a Stochastic Block Model graph."""
    return nx.stochastic_block_model(block_sizes, probabilities)

def generate_grid_graph(dim_x, dim_y):
    """Generate a grid graph."""
    return nx.grid_2d_graph(dim_x, dim_y)

def save_graphs(graphs, edge_list_dir):
    """Save all graphs to edge list files."""
    if edge_list_dir.exists():
        for file in edge_list_dir.iterdir():
            file.unlink()
    else:
        edge_list_dir.mkdir(parents=True, exist_ok=True)

    for i, graph in enumerate(graphs):
        edge_list_path = edge_list_dir / f"graph_{i}.edgelist"

        # For grid graphs, save the conventional edge-list format
        if isinstance(graph, nx.Graph) and all(isinstance(n, tuple) for n in graph.nodes()):
            # Convert tuple-based nodes to simple integers and map edges
            mapping = {node: idx for idx, node in enumerate(graph.nodes())}
            relabeled_graph = nx.relabel_nodes(graph, mapping)
            nx.write_edgelist(relabeled_graph, edge_list_path, data=False)
        else:
            nx.write_edgelist(graph, edge_list_path, data=False)


def main(args):
    """Main function to generate datasets based on a configuration file."""
    config_dir = Path("configs")
    dataset_dir = Path("datasets")
    config_file = args.config_file
    dataset_prefix = args.dataset_prefix

    config_path = config_dir / config_file
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = toml.load(f)

    dataset_dir.mkdir(parents=True, exist_ok=True)

    for dataset in config["datasets"]:
        graph_type = dataset["type"]
        graph_count = dataset["count"]
        node_count = dataset.get("nodes", None)
        edge_count = dataset.get("edges", None)

        graphs = []
        print(f"Generate datasets for graph type {graph_type}")
        for _ in range(graph_count):
            if graph_type == "planar":
                G = generate_planar_graph(node_count, edge_count)
            elif graph_type == "lobster":
                G = generate_lobster_graph(node_count, edge_count)
            elif graph_type == "sbm":
                G = generate_sbm_graph(dataset["block_sizes"], dataset["probabilities"])
            elif graph_type == "ego":
                G = generate_ego_graph(node_count)
            elif graph_type == "grid":
                if "dim_x" in dataset and "dim_y" in dataset:
                    G = generate_grid_graph(dataset["dim_x"], dataset["dim_y"])
                else:
                    raise ValueError("Grid graph requires 'dim_x' and 'dim_y' to be specified in the configuration.")
            elif graph_type == "protein":
                G = generate_protein_graph(node_count)
            elif graph_type == "point_cloud":
                G = generate_point_cloud_graph(node_count)
            else:
                raise ValueError(f"Unsupported graph type: {graph_type}")
            graphs.append(G)

        edge_list_dir = dataset_dir / f"{dataset_prefix}_{graph_type}_edgelists"
        print(f"Saving datasets for graph type {graph_type}")
        save_graphs(graphs, edge_list_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate graphs based on configuration.")
    parser.add_argument("--config-file", type=str, required=True, help="Name of the TOML configuration file in the input folder.")
    parser.add_argument("--dataset-prefix", type=str, required=True, help="Prefix for naming the dataset files.")
    args = parser.parse_args()

    main(args)
