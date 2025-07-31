import json
import networkx as nx
import random
from pathlib import Path
import toml
import argparse
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from scipy.spatial import Delaunay
from torch_geometric.datasets import QM9, ZINC
import numpy as np

def generate_grid_graph(min_node, max_node):
    """Generate a grid graph."""
    r = random.randint(min_node, max_node)
    c = random.randint(min_node, max_node)
    G = nx.grid_2d_graph(r, c, periodic=False)
    return G

def generate_community_graph(min_node, max_node, p_intra=0.3, p_inter = 0.05):
    def generate_connected_er_graph(n, p):
        """Retry ER graph generation until a connected graph is produced."""
        while True:
            G = nx.erdos_renyi_graph(n, p)
            if nx.is_connected(G):
                return G

    V = random.randint(min_node, max_node)
    n = V // 2  # nodes per community
    # Generate connected intra-community graphs
    G1 = generate_connected_er_graph(n, p_intra)
    G2 = generate_connected_er_graph(V - n, p_intra)
    # Relabel nodes in G2 to avoid overlap
    G2 = nx.relabel_nodes(G2, lambda x: x + n)
    G = nx.union(G1, G2)
    # Add inter-community edges and ensure connectivity
    inter_edges = set()
    while not nx.is_connected(G) or len(inter_edges) < int(p_inter * V):
        u = random.randint(0, n - 1)
        v = random.randint(n, V - 1)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            inter_edges.add((u, v))
    return G

def generate_planar_graph(node_count, edge_count):
    """Generate a planar graph using a Delaunay triangulation approach."""
    # Step 1: Generate random 2D points
    points = np.random.rand(node_count, 2)
    # Step 2: Use Delaunay triangulation to ensure a planar graph
    tri = Delaunay(points)
    G = nx.Graph()
    G.add_nodes_from(range(node_count))
    # Step 3: Add edges from triangles
    for triangle in tri.simplices:
        edges = [(triangle[i], triangle[j]) for i in range(3) for j in range(i+1, 3)]
        G.add_edges_from(edges)

    # Step 4: Check planarity (should always be True)
    is_planar, _ = nx.check_planarity(G)
    assert is_planar, "Generated graph is not planar!"
    
    # Step 5: Prune edges if needed
    if G.number_of_edges() > edge_count:
        edges = list(G.edges)
        random.shuffle(edges)
        G.remove_edges_from(edges[:G.number_of_edges() - edge_count])
    return G

def load_ego_graph(min_node, max_node, count):
    """Load ego graph dataset."""
    def can_be_rewired(G):
        """Return True if the graph G can be rewired."""
        edges = list(G.edges())
        if len(edges) < 2:
            return False
        # Try to find two edges with 4 distinct nodes
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                e1, e2 = edges[i], edges[j]
                u, v = e1
                x, y = e2
                if len({u, v, x, y}) == 4:
                    if not G.has_edge(u, x) and not G.has_edge(v, y):
                        return True
                    elif not G.has_edge(u, y) and not G.has_edge(v, x):
                        return True
        return False
        
    dataset = Planetoid(root='./datasets', name='Citeseer')
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)
    ego_graphs = []
    for node in G.nodes:
        ego = nx.ego_graph(G, node, radius=1)
        if ego.number_of_nodes() >= min_node and ego.number_of_nodes() <= max_node and can_be_rewired(ego):  
            ego_graphs.append(ego)
            count-= 1
            if count == 0 :
                break
    return ego_graphs

def load_qm9_graph():
    """Load QM9 graph dataset."""
    qm9_dataset = QM9(root='datasets/QM9')
    qm9_graphs = []
    for qm9_graph in qm9_dataset:
        G = to_networkx(qm9_graph, to_undirected=True)
        qm9_graphs.append(G)
    return qm9_graphs

def load_zinc_graph():
    """Load ZINC graph dataset."""
    zinc_dataset = ZINC(root='datasets/ZINC', subset=True)
    zinc_graphs = []
    for zinc_graph in zinc_dataset:
        G = to_networkx(zinc_graph, to_undirected=True)
        zinc_graphs.append(G)
    return zinc_graphs

def generate_sbm_graph(block_sizes, probabilities):
    """Generate a Stochastic Block Model graph."""
    import networkx as nx
    return nx.stochastic_block_model(block_sizes, probabilities)

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
    dataset_prefix = args.dataset_prefix

    config_path = config_dir / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = toml.load(f)

    dataset_dir.mkdir(parents=True, exist_ok=True)

    for dataset in config["datasets"]:
        graph_type = dataset["type"]
        node_count = dataset.get("nodes", None)
        edge_count = dataset.get("edges", None)

        graphs = []
        if graph_type == "ego":
            graph_count = dataset["count"]
            graphs = load_ego_graph(dataset["min_node"],dataset["max_node"],dataset["hop"],graph_count)
        if graph_type == "qm9":
            graphs = load_qm9_graph()
        if graph_type == "zinc":
            graphs = load_zinc_graph()
        if graph_type == "planar":
            graph_count = dataset["count"]
            for _ in range(graph_count):
                G = generate_planar_graph(node_count, edge_count)
                graphs.append(G)
        if graph_type == "sbm":
            graph_count = dataset["count"]
            for _ in range(graph_count):
                G = generate_sbm_graph(dataset["block_sizes"], dataset["probabilities"])
                graphs.append(G)
        if graph_type == "grid":
            graph_count = dataset["count"]
            for _ in range(graph_count):
                G = generate_grid_graph(dataset["min_node"],dataset["max_node"])
                graphs.append(G)
        if graph_type == "community":
            graph_count = dataset["count"]
            for _ in range(graph_count):
                G = generate_community_graph(dataset["min_node"],dataset["max_node"],dataset["p_intra"],dataset["p_inter"])
                graphs.append(G)

        print(f"Generate synthetic datasets for graph type {graph_type}")
        edge_list_dir = dataset_dir / f"{dataset_prefix}_{graph_type}_edgelists"
        print(f"Saving datasets for graph type {graph_type}")
        save_graphs(graphs, edge_list_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate graphs based on configuration.")
    parser.add_argument("--config", type=str, required=True, help="Name of the TOML configuration file in the input folder.")
    parser.add_argument("--dataset-prefix", type=str, required=True, help="Prefix for naming the dataset files.")
    args = parser.parse_args()

    main(args)
