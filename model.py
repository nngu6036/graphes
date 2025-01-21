import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx, from_networkx
import random
from scipy.stats import wasserstein_distance
import numpy as np


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, degree_set_size):
        self.degree_set_size = degree_set_size
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels + degree_set_size, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index, degree_set):
        # Convert degree set to one-hot encoding
        degree_set_one_hot = torch.nn.functional.one_hot(degree_set.long(), num_classes=self.degree_set_size).float()
        degree_set_one_hot = degree_set_one_hot.to(x.device)

        # Concatenate one-hot degree set to the node features
        x_with_degree = torch.cat([x, degree_set_one_hot], dim=1)
        x = self.conv1(x_with_degree, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

class GNNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, degree_set_size):
        self.degree_set_size = degree_set_size
        super(GNNDecoder, self).__init__()
        self.conv1 = GCNConv(in_channels + degree_set_size, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index, degree_set):
        # Convert degree set to one-hot encoding
        degree_set_one_hot = torch.nn.functional.one_hot(degree_set.long(), num_classes=self.degree_set_size).float()
        degree_set_one_hot = degree_set_one_hot.to(x.device)

        # Concatenate one-hot degree set to the node features
        x_with_degree = torch.cat([x, degree_set_one_hot], dim=1)
        x = self.conv1(x_with_degree, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

class GraphES(nn.Module):
    def __init__(self, node_features, degree_set_size):
        super(GraphES, self).__init__()
        self.encoder = GNNEncoder(node_features, 128, degree_set_size)
        self.decoder = GNNDecoder(128, node_features, degree_set_size)
        self.edge_predictor_1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.edge_predictor_2 = nn.Sequential(
            nn.Linear(128 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def edge_swap(self, edge_index):
        edge_list = edge_index.t().tolist()
        num_edges = len(edge_list)
        # Randomly choose two edges to swap
        e1, e2 = random.sample(range(num_edges), 2)
        (a, b), (c, d) = edge_list[e1], edge_list[e2]
        # Swap edges to form new pairs
        if random.random() > 0.5:
            edge_list[e1], edge_list[e2] = (a, d), (c, b)
        else:
            edge_list[e1], edge_list[e2] = (a, c), (b, d)
        return torch.tensor(edge_list, dtype=torch.long).t()

    def forward_diffusion(self, x, edge_index, degree_set, timesteps):
        z = self.encoder(x, edge_index, degree_set)
        for t in range(timesteps):
            edge_index = self.edge_swap(edge_index)
        return z, edge_index

    def reverse_process(self, z, edge_index, degree_set, timesteps):
        for t in range(timesteps):
            # Predict the first edge to swap
            edge_scores_1 = self.edge_predictor_1(z).squeeze()
            edge_indices_1 = edge_scores_1.argmax().item()
            e1 = edge_index[:, edge_indices_1]

            # Predict the second edge to swap conditioned on the first edge
            e1_features = z[e1]  # Extract node embeddings for the first edge
            e1_features_flat = e1_features.flatten()  # Flatten the features
            conditioned_input = torch.cat([z, e1_features_flat.unsqueeze(0).expand(z.size(0), -1)], dim=1)
            edge_scores_2 = self.edge_predictor_2(conditioned_input).squeeze()
            edge_indices_2 = edge_scores_2.argmax().item()
            e2 = edge_index[:, edge_indices_2]

            # Perform the edge swap
            (a, b), (c, d) = e1.tolist(), e2.tolist()
            if random.random() > 0.5:
                edge_index[:, edge_indices_1], edge_index[:, edge_indices_2] = torch.tensor([a, d]), torch.tensor([c, b])
            else:
                edge_index[:, edge_indices_1], edge_index[:, edge_indices_2] = torch.tensor([a, c]), torch.tensor([b, d])

        decoded_x = self.decoder(z, edge_index, degree_set)
        return decoded_x, edge_index

    def forward(self, x, edge_index, degree_set, timesteps):
        z, noisy_edge_index = self.forward_diffusion(x, edge_index, degree_set, timesteps)
        reconstructed_x, reconstructed_edge_index = self.reverse_process(z, noisy_edge_index, degree_set, timesteps)
        return reconstructed_x, reconstructed_edge_index

    def generate_samples(self, initial_graph, node_features, timesteps=5, num_nodes=None):
        self.eval()
        import pdb
        pdb.set_trace()
        with torch.no_grad():
            if num_nodes is None:
                num_nodes = data.num_nodes  # Default to the number of nodes in the new graph
            data = from_networkx(initial_graph)
            x = torch.eye(num_nodes)
            edge_index = data.edge_index
            degree_set = torch.tensor([d for _, d in initial_graph.degree()], dtype=torch.float32)
            z, noisy_edge_index = self.forward_diffusion(x, edge_index, degree_set, timesteps)
            generated_x, generated_edge_index = self.reverse_process(z, noisy_edge_index, degree_set, timesteps)
            return generated_x, generated_edge_index

def calculate_mmd(graph_list_1, graph_list_2):
    def degree_distribution(graph):
        return [d for _, d in graph.degree()]

    def clustering_coefficient(graph):
        return list(nx.clustering(graph).values())

    def orbit_counts(graph):
        # For simplicity, we consider the counts of 4-node orbits using nx.cycle_basis for a heuristic approximation
        return [len(cycle) for cycle in nx.cycle_basis(graph) if len(cycle) == 4]

    mmd_degree = 0
    mmd_clustering = 0
    mmd_orbit = 0

    for g1, g2 in zip(graph_list_1, graph_list_2):
        degree_g1, degree_g2 = degree_distribution(g1), degree_distribution(g2)
        clustering_g1, clustering_g2 = clustering_coefficient(g1), clustering_coefficient(g2)
        orbit_g1, orbit_g2 = orbit_counts(g1), orbit_counts(g2)

        mmd_degree += wasserstein_distance(degree_g1, degree_g2)
        mmd_clustering += wasserstein_distance(clustering_g1, clustering_g2)
        mmd_orbit += wasserstein_distance(orbit_g1, orbit_g2)

    mmd_degree /= len(graph_list_1)
    mmd_clustering /= len(graph_list_1)
    mmd_orbit /= len(graph_list_1)

    return mmd_degree, mmd_clustering, mmd_orbit

if __name__ == "__main__":
    # Define the number of nodes and features for the graph
    num_nodes = 10
    node_features = 3

    # Generate a random graph and convert it to PyTorch Geometric format
    graph = nx.erdos_renyi_graph(num_nodes, 0.3)
    data = from_networkx(graph)
    x = torch.eye(num_nodes)
    edge_index = data.edge_index

    # Initialize the model and optimizer
    degree_set_size = num_nodes  # Assuming max number of nodes in the dataset is num_nodes
    model = GraphES(node_features, degree_set_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        degree_set = torch.tensor([d for _, d in graph.degree()], dtype=torch.float32)
        reconstructed_x, reconstructed_edge_index = model(x, edge_index, degree_set, timesteps=5)
        loss = nn.MSELoss()(reconstructed_x, x)  # Simplified loss
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Generate new samples after training
    new_graph = nx.erdos_renyi_graph(num_nodes, 0.3)
    generated_x, generated_edge_index = model.generate_samples(new_graph, node_features, timesteps=5, num_nodes=num_nodes)
    print("Generated node features:", generated_x)
    print("Generated edge index:", generated_edge_index)

    # Calculate Maximum Mean Discrepancy (MMD)
    original_graphs = [graph]
    generated_graphs = [to_networkx(from_networkx(new_graph, edge_index=generated_edge_index))]
    mmd_degree, mmd_clustering, mmd_orbit = calculate_mmd(original_graphs, generated_graphs)
    print(f"MMD Degree: {mmd_degree}")
    print(f"MMD Clustering Coefficient: {mmd_clustering}")
    print(f"MMD Orbit Counts: {mmd_orbit}")
