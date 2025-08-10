import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import os
import argparse
import toml
import math
from pathlib import Path
import networkx as nx
from scipy.optimize import linear_sum_assignment
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from model_setvae import SetVAE,decode_degree_sequence, encode_degree_sequence
from utils import load_degree_sequence_from_directory
from eval import DegreeSequenceEvaluator


def train_setvae(model, dataloader, num_epochs, learning_rate,max_degree):
    optimizer = Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)
    model.train()
    for epoch in range(num_epochs):
        print("Traininig Set-VAE iteration ", epoch)
        total_loss = 0
        for (x,) in dataloader:
            m = encode_degree_sequence(x,max_degree) 
            N = m.sum(dim=1).long()
            out = model(m, N)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch Set-VAE [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")


def main(args):
    config_dir = Path("configs")
    dataset_dir = Path("datasets") / args.dataset_dir
    model_dir = Path("models")
    config = toml.load(config_dir / args.config)
    batch_size = config['training']['batch_size']
    input_data, max_node = load_degree_sequence_from_directory(dataset_dir)
    train_data, test_data = train_test_split(input_data, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(torch.stack([encode_degree_sequence(seq,max_node) for seq in train_data]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    hidden_dim = config['training']['hidden_dim']
    latent_dim = config['training']['latent_dim']
    model = SetVAE(hidden_dim=hidden_dim, latent_dim=latent_dim, max_degree = max_node)
    if args.input_model:
        model.load_model(model_dir / args.input_model)
        print(f"Model loaded from {model_dir / args.input_model}")
    else:
        num_epochs = config['training']['num_epochs']
        learning_rate = config['training']['learning_rate']
        train_setvae(model, train_dataloader, num_epochs, learning_rate, max_node)
    if args.output_model:
        model.save_model(model_dir / args.output_model)
        print(f"Model saved to {args.output_model}")
    if args.evaluate:
        deg_eval = DegreeSequenceEvaluator()
        generated_data = model.generate(config['inference']['generate_samples'])
        print(f"Generated degree sequence validity: {deg_eval.evaluate_sequences(generated_data)}")
        print(f"Evaluate baseline: train <-> test")
        #print(f"Chamfer Distance: {deg_eval.evaluate_multisets_chamfer_distance(train_data,test_data)}")
        #print(f"Earth Mover Distance: {deg_eval.evaluate_multisets_earth_mover_distance(train_data,test_data)}")
        print(f"KL Distance: {deg_eval.evaluate_multisets_kl_distance(train_data,test_data,max_node)}")
        print(f"MMD Distance: {deg_eval.evaluate_multisets_mmd_distance(train_data,test_data,max_node)}")
        print(f"Evaluate fit: train <-> generated")
        #print(f"Chamfer Distance: {deg_eval.evaluate_multisets_chamfer_distance(train_data,generated_data)}")
        #print(f"Earth Mover Distance: {deg_eval.evaluate_multisets_earth_mover_distance(train_data,generated_data)}")
        print(f"KL Distance: {deg_eval.evaluate_multisets_kl_distance(train_data,generated_data,max_node)}")
        print(f"MMD Distance: {deg_eval.evaluate_multisets_mmd_distance(train_data,generated_data,max_node)}")
        print(f"Evaluate fit: test <-> generated")
        #print(f"Chamfer Distance: {deg_eval.evaluate_multisets_chamfer_distance(test_data,generated_data)}")
        #print(f"Earth Mover Distance: {deg_eval.evaluate_multisets_earth_mover_distance(test_data,generated_data)}")
        print(f"KL Distance: {deg_eval.evaluate_multisets_kl_distance(test_data,generated_data,max_node)}")
        print(f"MMD Distance: {deg_eval.evaluate_multisets_mmd_distance(test_data,generated_data,max_node)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set-VAE for Graph Generation')
    parser.add_argument('--dataset-dir', type=str, help='Path to the directory containing graph files')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file in TOML format')
    parser.add_argument('--output-model', type=str, help='Path to save the trained model')
    parser.add_argument('--input-model', type=str, help='Path to load a pre-trained model')
    parser.add_argument('--evaluate', action='store_true', help='Whether we evaluate the model')
    args = parser.parse_args()
    main(args)