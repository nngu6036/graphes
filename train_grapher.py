import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import os
import argparse
import toml
import math
import random
from pathlib import Path
from torch_geometric.utils import from_networkx
import networkx as nx
from scipy.optimize import linear_sum_assignment
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

from model_msvae import MSVAE
from model_grapher import GraphER
from eval import DegreeSequenceEvaluator, GraphsEvaluator
from utils import *


def rewire_edges(G, num_rewirings):
    edges = list(G.edges())
    last_rewired_pair = None
    while num_rewirings > 0:
        e1, e2 = random.sample(edges, 2)
        u, v = e1
        x, y = e2
        if len({u, v, x, y}) == 4:
            if not G.has_edge(u, x) and not G.has_edge(v, y):
                G.remove_edges_from([e1, e2])
                G.add_edges_from([(u, x), (v, y)])
                last_rewired_pair = (e1, e2)
                num_rewirings-= 1
            elif not G.has_edge(u, y) and not G.has_edge(v, x):
                G.remove_edges_from([e1, e2])
                G.add_edges_from([(u, y), (v, x)])
                last_rewired_pair = (e1, e2)
                num_rewirings-= 1
        edges = list(G.edges())
    return G, last_rewired_pair

def train_grapher(model, graphs, num_epochs, learning_rate, max_node, T, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for G in graphs:
            num_rewirings = random.randint(1, T)
            G_corrupted, last_rewired_pair = rewire_edges(G.copy(), num_rewirings)
            if not last_rewired_pair:
                print("Revire pair null")
                continue
            (u,v), (x,y) = last_rewired_pair
            data = graph_to_data(G_corrupted).to(device)
            candidate_edges = [e for e in G_corrupted.edges()]
            positive_edges = {frozenset((u, x)), frozenset((u, y))}
            labels = torch.tensor([1.0 if frozenset((s, t)) in positive_edges else 0.0 for (s, t) in candidate_edges])
            scores = model(data.x, data.edge_index, (u,v), candidate_edges, t=num_rewirings)
            loss = criterion(scores.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


def load_msvae_from_file(max_node,config, model_path):
    hidden_dim = config['training']['hidden_dim']
    latent_dim = config['training']['latent_dim']
    model = MSVAE(max_input_dim=max_node, hidden_dim=hidden_dim, latent_dim=latent_dim, max_frequency = max_node)
    print(f"MS-VAE Model loaded from {model_path}")
    model.load_model(model_path)
    return model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

def main(args):
    config_dir = Path("configs")
    dataset_dir = Path("datasets") / args.dataset_dir
    model_dir = Path("models")
    config = toml.load(config_dir / args.config)
    msvae_config = toml.load(config_dir / args.msvae_config)
    graphs, max_node = load_graph_from_directory(dataset_dir)
    print(f"Loading graphs dataset {len(graphs)}")
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    msvae_model  = load_msvae_from_file(max_node, msvae_config, model_dir /args.msvae_model)
    hidden_dim = config['training']['hidden_dim']
    latent_dim = config['training']['latent_dim']
    num_layer = config['training']['num_layer']
    model = GraphER(in_channels=1, hidden_dim=hidden_dim,num_layer=num_layer)
    if args.input_model:
        model.load_model(model_dir / args.input_model)
        print(f"Model Graph-ER loaded from {args.input_model}")
    else:
        num_epochs = config['training']['num_epochs']
        learning_rate = config['training']['learning_rate']
        T = config['training']['T']
        train_grapher(model, train_graphs, num_epochs, learning_rate, max_node,T, 'cpu')
    if args.output_model:
        model.save_model(model_dir / args.output_model)
        print(f"Model saved to {args.output_model}")
    if args.evaluate:
        graph_eval = GraphsEvaluator()
        T = config['training']['T']
        if args.ablation:
            sample_graphs = random.sample(train_graphs,min(len(train_graphs),config['inference']['generate_samples']))
            degree_sequences = [[deg for _, deg in graph.degree()] for graph in sample_graphs]
            generated_graphs = model.generate(config['inference']['generate_samples'],T,degree_sequences = degree_sequences, msvae_model = None)
        else:
            generated_graphs = model.generate(config['inference']['generate_samples'],T,degree_sequences = None, msvae_model = msvae_model)
        print(f"Evaluate generated graphs")
        print(f"MMD Degree: {graph_eval.compute_mmd_degree_emd(test_graphs,generated_graphs)}")
        print(f"MMD Clustering Coefficient: {graph_eval.compute_mmd_cluster(test_graphs,generated_graphs)}")
        print(f"MMD Orbit count: {graph_eval.compute_mmd_orbit(test_graphs,generated_graphs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GRAPH-ER Model')
    parser.add_argument('--dataset-dir', type=str, required=True,help='Path to the directory containing graph files')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file in TOML format of Graph-ER')
    parser.add_argument('--msvae-config', type=str, required=True, help='Path to the configuration file in TOML format of MS-VAE')
    parser.add_argument('--msvae-model', type=str, required=True,help='Path to load a pre-trained MS-VAE model')
    parser.add_argument('--output-model', type=str, help='Path to save the trained model')
    parser.add_argument('--input-model', type=str, help='Path to load a pre-trained model')
    parser.add_argument('--evaluate', action='store_true', help='Whether we evaluate the model')
    parser.add_argument('--ablation', action='store_true', help='Whether to run ablation study')
    args = parser.parse_args()
    main(args)