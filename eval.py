from typing import List
import numpy as np
import tempfile
import subprocess
import os
import networkx as nx
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment
from collections import Counter
import torch
#from networkx.algorithms import graphlet_degree_vectors

from utils import check_sequence_validity
ORCA_EXEC = os.environ.get('ORCA_EXEC', None)

def gaussian_emd_kernel(X, Y, sigma=1.0):
    K = np.zeros((len(X), len(Y)))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            emd = wasserstein_distance(np.arange(len(x)), np.arange(len(y)), x, y)
            K[i, j] = np.exp(-emd**2 / (2 * sigma**2))
    return K

def compute_degree_histograms(sets, max_degree):
    histograms = []
    for seq in sets:
        hist = torch.zeros(max_degree + 1)
        for deg in seq:
            if 0 <= deg <= max_degree:
                hist[deg] += 1
        hist /= hist.sum() + 1e-8  # normalize + stability
        histograms.append(hist)
    return torch.stack(histograms)


class GraphsEvaluator():

	def compute_mmd_degree_emd(self, graphs_1, graphs_2, max_degree, sigma=1.0):
	    def degree_histogram(graphs, max_degree):
	        histograms = []
	        for G in graphs:
	            degree_sequence = [deg for _, deg in G.degree()]
	            hist = np.zeros(max_degree + 1)
	            for deg in degree_sequence:
	                if deg <= max_degree:
	                    hist[deg] += 1
	            if hist.sum() == 0:
	                hist[0] = 1.0
	            hist /= hist.sum()
	            histograms.append(hist)
	        return np.array(histograms)
	    H1 = degree_histogram(graphs_1, max_degree)
	    H2 = degree_histogram(graphs_2, max_degree)
	    K_xx = gaussian_emd_kernel(H1, H1, sigma)
	    K_yy = gaussian_emd_kernel(H2, H2, sigma)
	    K_xy = gaussian_emd_kernel(H1, H2, sigma)
	    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
	    return float(mmd)

	def compute_mmd_cluster(self, graphs_1, graphs_2, bins=10, sigma=1.0):
	    def clustering_histogram(graphs, bins=10):
	        histograms = []
	        for G in graphs:
	            clustering = list(nx.clustering(G).values())
	            hist, _ = np.histogram(clustering, bins=bins, range=(0, 1), density=True)
	            if hist.sum() == 0:
	                hist[0] = 1.0
	            hist /= hist.sum()
	            histograms.append(hist)
	        return np.array(histograms)
	    H1 = clustering_histogram(graphs_1, bins)
	    H2 = clustering_histogram(graphs_2, bins)
	    K_xx = gaussian_emd_kernel(H1, H1, sigma)
	    K_yy = gaussian_emd_kernel(H2, H2, sigma)
	    K_xy = gaussian_emd_kernel(H1, H2, sigma)
	    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
	    return float(mmd)

	def compute_mmd_orbit(self, graphs_1, graphs_2, sigma=1.0):
		if not ORCA_EXEC:
			raise Exception("ORCA module is not found")
		def count_graphlets_orbit(graph, orbit_size: int = 4):
			"""
			Count graphlet orbits occurence using ORCA executable
			Args:
			G (nx.Graph): NetworkX undirected graph.
			orbit_size (int): Size of the graphlets to count (typically 3, 4, or 5). Default is 4.
			"""
			with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp1,tempfile.NamedTemporaryFile(mode='r', delete=False) as temp2:
				temp1_path = temp1.name
				temp2_path = temp2.name
				# Write graph in required format: first line n e, then e lines of edges
				nodes = sorted(graph.nodes())
				node_map = {node: idx for idx, node in enumerate(nodes)}
				n = graph.number_of_nodes()
				e = graph.number_of_edges()
				temp1.write(f"{n} {e}\n")
				for u, v in graph.edges():
					temp1.write(f"{node_map[u]} {node_map[v]}\n")
				temp1.flush()
				# Run ORCA
				try:
					subprocess.run([ORCA_EXEC, "node", str(orbit_size), temp1_path, temp2_path],check=True, capture_output=True)
				except subprocess.CalledProcessError as e:
				    raise RuntimeError(f"ORCA execution failed: {e.stderr.decode()}")

			# Read orbit counts
			with open(temp2_path, 'r') as f:
				orbit_counts = [list(map(int, line.strip().split())) for line in f.readlines()]
			# Clean up
			os.remove(temp1_path)
			os.remove(temp2_path)
			# Sum counts across all nodes
			total_counts = [0] * len(orbit_counts[0])
			for node_orbit in orbit_counts:
				for i, val in enumerate(node_orbit):
					total_counts[i] += val
			# Divide by multiplicity to get true occurrence count
				orbit_multiplicity = [2, 2, 1, 6, 2, 1, 2, 4, 2, 1, 1, 2, 2, 4, 1]  # 15 orbits
			true_occurrences = [
			    total_counts[i] // orbit_multiplicity[i] for i in range(len(total_counts))
			]
			return true_occurrences

		def orbit_histogram(graphs):
			histograms = []
			for G in graphs:
				counts = torch.tensor(count_graphlets_orbit(G))
				counts = counts.float()
				counts /= counts.sum() + 1e-8  # normalize to make it a histogram
				histograms.append(counts)
			max_len = max(len(h) for h in histograms)
			padded = [np.pad(h, (0, max_len - len(h))) for h in histograms]
			return np.array(padded)
		H1 = orbit_histogram(graphs_1)
		H2 = orbit_histogram(graphs_2)
		K_xx = gaussian_emd_kernel(H1, H1, sigma)
		K_yy = gaussian_emd_kernel(H2, H2, sigma)
		K_xy = gaussian_emd_kernel(H1, H2, sigma)
		mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
		return float(mmd)


class DegreeSequenceEvaluator():

	def evaluate_multisets_chamfer_distance(self, sets1, sets2):
		def compute_chamfer_distance(set1, set2):
		    if len(set1) == 0 or len(set2) == 0:
		        return float('inf')  # If one of the sets is empty, the distance is undefined.
		    set1 = torch.tensor(set1, dtype=torch.float).unsqueeze(1)
		    set2 = torch.tensor(set2, dtype=torch.float).unsqueeze(1)
		    dists_1_to_2 = torch.min(torch.cdist(set1, set2, p=2), dim=1).values
		    dists_2_to_1 = torch.min(torch.cdist(set2, set1, p=2), dim=1).values
		    chamfer_distance = torch.sum(dists_1_to_2) + torch.sum(dists_2_to_1)
		    return chamfer_distance.item()
		chamfer_distances = [compute_chamfer_distance(s, t) for s in sets1 for t in sets2]
		avg_chamfer_distance = sum(chamfer_distances) / len(chamfer_distances) if chamfer_distances else float('inf')
		return avg_chamfer_distance

	def evaluate_multisets_earth_mover_distance(self, sets1, sets2):
		def compute_earth_mover_distance(set1, set2):
		    if len(set1) == 0 or len(set2) == 0:
		        return float('inf')  # Undefined if one of the sets is empty.
		    set1 = torch.tensor(set1, dtype=torch.float).unsqueeze(1)
		    set2 = torch.tensor(set2, dtype=torch.float).unsqueeze(1)
		    cost_matrix = torch.cdist(set1, set2, p=2).cpu().numpy()
		    row_ind, col_ind = linear_sum_assignment(cost_matrix)
		    emd = cost_matrix[row_ind, col_ind].sum()
		    return emd
		emd_distances = [compute_earth_mover_distance(s, t) for s in sets1 for t in sets2]
		avg_emd_distance = sum(emd_distances) / len(emd_distances) if emd_distances else float('inf')
		return avg_emd_distance

	def evaluate_multisets_kl_distance(self, sets1, sets2,max_node):
		def compute_kl_divergence(p_hist, q_hist, eps=1e-8):
				p_ = torch.tensor(p_hist) + eps
				q_ = torch.tensor(q_hist) + eps
				p_ = p_ / p_.sum()
				q_ = q_ / q_.sum()
				return torch.sum(p_ * torch.log(p_ / q_)).item()
		source_hist = compute_degree_histograms(sets1,max_node)
		target_hist = compute_degree_histograms(sets2,max_node)
		kl = [compute_kl_divergence(s, t) for s in source_hist for t in target_hist]
		avg_kl = sum(kl) / len(kl) if kl else float('inf')
		return avg_kl

	def evaluate_multisets_mmd_distance(self, sets1, sets2,max_node):
		def compute_mmd(X, Y, sigma=1.0):
		    X_np = X.detach().cpu().numpy()
		    Y_np = Y.detach().cpu().numpy()
		    K_xx = gaussian_emd_kernel(X_np, X_np, sigma=sigma)
		    K_yy = gaussian_emd_kernel(Y_np, Y_np, sigma=sigma)
		    K_xy = gaussian_emd_kernel(X_np, Y_np, sigma=sigma)
		    return float(K_xx.mean() + K_yy.mean() - 2 * K_xy.mean())
		source_hist = compute_degree_histograms(sets1,max_node)
		target_hist = compute_degree_histograms(sets2,max_node)
		mmd = compute_mmd(source_hist, target_hist)
		return mmd

	def evaluate_sequences(self, sets:List):
	    validity_checks = [check_sequence_validity(seq) for seq in sets]
	    degree_validities = [result for result, code in validity_checks if result]
	    error_codes = [code for result, code in validity_checks if not result]
	    error_count = Counter(error_codes)
	    empty_degree = [1] * error_count.get(1, 0)
	    odd_sum_degree = [1] * error_count.get(2, 0)
	    invalid_degree = [1] * error_count.get(3, 0)
	    validity_percentage = (sum(degree_validities) / len(validity_checks)) * 100 if len(validity_checks) > 0 else 0
	    empty_percentage = (sum(empty_degree) / len(validity_checks)) * 100 if len(validity_checks) > 0 else 0
	    odd_percentage = (sum(odd_sum_degree) / len(validity_checks)) * 100 if len(validity_checks) > 0 else 0
	    invalidity_percentage = (sum(invalid_degree) / len(validity_checks)) * 100 if len(validity_checks) > 0 else 0
	    return validity_percentage
