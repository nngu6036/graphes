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
import networkx as nx
from torch_geometric.utils import from_networkx
from scipy.optimize import linear_sum_assignment
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

from model_msvae import MSVAE
from model_grapher import GraphERFlow 
from eval import DegreeSequenceEvaluator, GraphsEvaluator
from utils import *

# ---------- Symmetric difference & alternating cycles ----------
def canonical_edges(G):
    return {(min(u,v), max(u,v)) for (u,v) in G.edges()}

def difference_colors(H, Gt):
    E1, E2 = canonical_edges(H), canonical_edges(Gt)
    red  = E1 - E2  # in H only
    blue = E2 - E1  # in G* only
    return red, blue

def alternating_components(H, Gt):
    """Return list of components; each component as a 2m-cycle ordered (v0,v1,...,v{2m-1}) with e0 red."""
    red, blue = difference_colors(H, Gt)
    D = nx.Graph()
    for (u,v) in red:  D.add_edge(u, v, color='red')
    for (u,v) in blue: D.add_edge(u, v, color='blue')
    cycles = []
    for comp_nodes in nx.connected_components(D):
        S = D.subgraph(comp_nodes).copy()
        # recover an ordered alternating cycle
        # pick a start (u) and its red neighbor v to ensure e0 is red
        u = next(iter(S.nodes()))
        nbrs_red = [w for w in S.neighbors(u) if S[u][w]['color']=='red']
        if not nbrs_red:
            # flip colors if started on blue
            nbrs_blue = [w for w in S.neighbors(u)]
            v = nbrs_blue[0]
        else:
            v = nbrs_red[0]
        order = [u, v]
        prev = u
        curr = v
        prev_color = S[u][v]['color']
        # walk until we return to start
        while True:
            # pick the next neighbor of curr with opposite color, not equal to prev
            next_nodes = []
            for w in S.neighbors(curr):
                if w == prev: 
                    continue
                if S[curr][w]['color'] != prev_color:
                    next_nodes.append(w)
            if not next_nodes:
                break
            w = next_nodes[0]
            order.append(w)
            prev, curr = curr, w
            prev_color = S[prev][curr]['color']
            if curr == order[0]:
                break
        # ensure even length and alternating
        if len(order) >= 4 and len(order)%2==0:
            cycles.append(order)
    return cycles

def cycle_to_swaps(node_cycle, H, Gt):
    """
    Given an alternating node cycle [v0,v1,...,v{2m-1}, v0] with e0 red,
    produce m-1 double-edge swaps:
      remove red e_{2j} and e_{2(j+1)} ; add blue e_{2j+1} and e_{2j+3}
    """
    L = len(node_cycle)
    if node_cycle[0] != node_cycle[-1]:
        node_cycle = node_cycle + [node_cycle[0]]
        L += 1
    m = (L-1)//2
    swaps = []
    # edges along cycle: e_i = (v_i, v_{i+1})
    v = node_cycle
    for j in range(m-1):
        # indices mod (2m)
        i0 = (2*j) % (2*m)
        i2 = (2*j+2) % (2*m)
        i1 = (2*j+1) % (2*m)
        i3 = (2*j+3) % (2*m)
        e_red1  = tuple(sorted((v[i0], v[i0+1])))
        e_red2  = tuple(sorted((v[i2], v[i2+1])))
        e_blue1 = tuple(sorted((v[i1], v[i1+1])))
        e_blue2 = tuple(sorted((v[i3], v[i3+1])))
        swaps.append((e_red1, e_red2, e_blue1, e_blue2))
    return swaps


def transform_to_hh_via_stochastic_rewiring(
    G,
    H,
    max_steps=10000,
    beta=3.0,           # bias toward HH edges
    T0=1.0,             # initial temperature
    cooling=0.995,      # simulated annealing cooling per accepted step
    ensure_connected=True,
    k_hop=2,         # e.g., 2 or 3 to preserve locality; None disables
    locality_reference="initial",  # "initial" (default) or "current"
    seed=None,
):
    """
    Stochastically transform G to its Havel–Hakimi realization using biased 2-edge swaps
    while (a) preserving degree sequence, (b) enforcing k-hop locality for *new* edges,
    and (c) rejecting swaps that break connectivity (if ensure_connected=True).
    """
    rng = random.Random(seed)
    Gc = G.copy()

    # Target HH graph and scoring
    
    H_set = {_ek(u, v) for u, v in H.edges()}
    def matches_in_H(edges): return sum(1 for e in edges if _ek(*e) in H_set)
    cur_matches = matches_in_H(Gc.edges())

    # Precompute k-hop neighborhoods on the chosen reference graph
    neighborhoods = None
    if k_hop is not None:
        ref_graph = G if locality_reference == "initial" else Gc
        neighborhoods = _khop_neighborhoods(ref_graph, k_hop)

    T = T0
    traj = []
    m = Gc.number_of_edges()

    for _ in range(max_steps):
        prop = _propose_swap_with_locality(
            Gc, rng, k_hop, neighborhoods, locality_reference, max_tries=256
        )
        if prop is None:
            # no valid locality-respecting swap found under the budget
            break

        (e1, e2, f1, f2) = prop
        before = int(_ek(*e1) in H_set) + int(_ek(*e2) in H_set)
        after  = int(_ek(*f1) in H_set) + int(_ek(*f2) in H_set)
        dmatches = after - before

        # Metropolis acceptance (symmetric proposals)
        accept = (dmatches >= 0) or (rng.random() < math.exp(beta * dmatches / max(T, 1e-8)))
        if not accept:
            continue

        # Tentatively apply and enforce connectivity
        Gb = Gc.copy()  
        Gc.remove_edges_from([e1, e2])
        Gc.add_edges_from([f1, f2])

        if ensure_connected and not nx.is_connected(Gc):
            # revert if it breaks connectivity
            Gc.remove_edges_from([f1, f2])
            Gc.add_edges_from([e1, e2])
            continue
        else:
            traj.append((Gb,(e1, e2), (f1, f2)))

        # Commit
        cur_matches += dmatches
        T *= cooling

        if cur_matches == m:  # reached HH exactly
            break

        # If using dynamic locality, refresh neighborhoods occasionally (cheap heuristic)
        if k_hop is not None and locality_reference == "current":
            # Only recompute for touched nodes to keep it light
            for u in {e1[0], e1[1], e2[0], e2[1], f1[0], f1[1], f2[0], f2[1]}:
                dists = nx.single_source_shortest_path_length(Gc, u, cutoff=k_hop)
                neighborhoods[u] = {x for x, dist in dists.items() if 0 < dist <= k_hop}
    return traj

    
def build_teacher_sequence(
    G_star,
    H,
    k_local: int = 2,
    enforce_biconnected: bool = False,
    triangle_weight_alpha: float = 0.0,
    allow_non_blue_fallback: bool = True,
):
    """
    Build a connectivity-safe, k-hop-local, triangle-aware teacher trajectory
    from H=HH(deg(G*)) to G* by resolving each alternating component.
    Returns: list of (G_t, G_{t-1}, anchor_e1, partner_e2)
    """

    red, blue = difference_colors(H, G_star)
    steps = []

    # caches
    khop = precompute_khop(H, k_local)     # node -> k-hop set
    clustering = nx.clustering(H)          # local clustering coeffs

    for cyc in alternating_components(H, G_star):
        # cycle_to_swaps yields (e_r1, e_r2, e_b1, e_b2) for each move on this component
        for (e_r1, e_r2, e_b1, e_b2) in cycle_to_swaps(cyc, H, G_star):
            e_r1 = tuple(sorted(e_r1)); e_r2 = tuple(sorted(e_r2))
            e_b1 = tuple(sorted(e_b1)); e_b2 = tuple(sorted(e_b2))
            G_pre = H.copy()

            # k-hop locality (strong: all four endpoints must be pairwise within k hops)
            if not are_four_endpoints_k_local(e_r1, e_r2, khop):
                continue

            # connectivity-safe (no split; optionally no bridges)
            if not swap_is_connectivity_safe(H, e_r1, e_r2, enforce_biconnected=enforce_biconnected):
                continue

            # Two orientations to consider
            u,v = e_r1; x,y = e_r2
            add_parallel = (tuple(sorted((u,x))), tuple(sorted((v,y))))
            add_cross    = (tuple(sorted((u,y))), tuple(sorted((v,x))))

            # We prefer the orientation that matches exactly the blue edges in this cycle
            match_parallel = set(add_parallel) == {e_b1, e_b2}
            match_cross    = set(add_cross)    == {e_b1, e_b2}

            legal_parallel = all((a!=b) and (not H.has_edge(*e)) for e in add_parallel for a,b in [e])
            legal_cross    = all((a!=b) and (not H.has_edge(*e)) for e in add_cross    for a,b in [e])

            candidates = []
            if legal_parallel:
                s_par = triangle_aware_score(H, e_r1, e_r2, list(add_parallel), clustering, triangle_weight_alpha)
                candidates.append(("parallel", add_parallel, match_parallel, s_par))
            if legal_cross:
                s_crs = triangle_aware_score(H, e_r1, e_r2, list(add_cross), clustering, triangle_weight_alpha)
                candidates.append(("cross", add_cross, match_cross, s_crs))
            if not candidates:
                continue

            # Rank: (1) must match blue; (2) higher triangle score
            candidates.sort(key=lambda x: ((0 if x[2] else 1), -x[3]))
            orient, add_edges, matches_blue, _ = candidates[0]

            # if top candidate doesn't match blue and fallback is disallowed, try the other that matches blue
            if not matches_blue and not allow_non_blue_fallback:
                # find any candidate that matches blue
                for c in candidates:
                    if c[2]:
                        orient, add_edges, matches_blue, _ = c
                        break
                else:
                    # no blue-matching orientation under masks; skip this swap
                    continue

            # apply
            H.remove_edges_from([e_r1, e_r2])
            H.add_edges_from(add_edges)

            # record teacher step (anchor=e_r1; partner=e_r2)
            steps.append((G_pre, H.copy(), e_r1, e_r2))

            # refresh cheap caches (global recompute is simplest & safe)
            khop = precompute_khop(H, k_local)
            clustering = nx.clustering(H)

    return steps  # from G_T->...->G_0

# ---------- Mask builder (legality + disjoint endpoints) ----------
def build_candidates_masked(
    G, e1, raw_cands,
    k_local: int = 2,
    enforce_biconnected: bool = True
):
    """
    Return candidate partner edges e2 that have at least one legal, connectivity-safe,
    k-local orientation with e1 (we don't pick orientation here; we only filter).
    """
    u, v = e1
    # precompute once per call (graph changes across steps)
    khop = precompute_khop(G, k_local)

    def legal_adds(e2):
        x, y = e2
        add_parallel = ( (u,x), (v,y) )
        add_cross    = ( (u,y), (v,x) )
        def legal_pair(pair):
            (a,b),(c,d) = pair
            # disjoint endpoints
            if len({u,v,x,y}) != 4:
                return False
            # no self loops / multiedges
            if a==b or c==d: return False
            if G.has_edge(*pair[0]) or G.has_edge(*pair[1]): return False
            return True
        return legal_pair(add_parallel) or legal_pair(add_cross)

    filtered = []
    for e2 in raw_cands:
        e2 = tuple(sorted(e2))
        # locality
        if not are_four_endpoints_k_local(tuple(sorted(e1)), e2, khop):
            continue
        # connectivity safe on removal of the two existing edges
        if not swap_is_connectivity_safe(G, tuple(sorted(e1)), e2, enforce_biconnected=enforce_biconnected):
            continue
        # at least one orientation must be legal
        if not legal_adds(e2):
            continue
        filtered.append(e2)
    return filtered

# ---------- Training (Flow Matching) ----------
def train_fm(model, graphs, num_epochs, lr,k_lap,heat_times, device='cpu'):
    model.to(device)
    opt = Adam(model.parameters(), lr=lr)    # <-- was AdamW but not imported
    graphs = [(G,havel_hakimi_construction(G)) for G in graphs]
    for ep in range(num_epochs):
        model.train()
        total_loss = 0.0
        for G_star, G_hh in graphs:
            teacher = build_teacher_sequence(G_star,G_hh)
            import pdb
            pdb.set_trace()
            T = max(1, len(teacher))
            for t_idx, (G_t, G_tm1, e1, e2) in enumerate(teacher, start=1):
                pyg_t   = to_pyg_with_features(G_t).to(device)
                pyg_tm1 = to_pyg_with_features(G_tm1).to(device)
                z_t,  h_t  = model.encode(pyg_t.x,   pyg_t.edge_index)
                z_tm, h_tm = model.encode(pyg_tm1.x, pyg_tm1.edge_index)
                t_bin = torch.tensor([T - t_idx + 1], dtype=torch.long, device=device)
                v_pred, t_emb = model.velocity(h_t, t_bin)
                delta_tau = 1.0 / float(T)
                v_tgt  = (h_tm - h_t) / delta_tau
                L_cfm  = F.mse_loss(v_pred, v_tgt)
                h_next = h_t + delta_tau * v_pred
                L_next = F.mse_loss(h_next, h_tm)

                edges = list(G_t.edges())
                raw_cands = [(x, y) for (x, y) in edges if (x, y) != e1]
                cand = build_candidates_masked(G_t, e1, raw_cands)
                if len(cand) == 0:
                    L_act = torch.tensor(0.0, device=device)
                else:
                    logits = model.action_logits(z_t, h_t, t_emb, e1, cand)
                    try:
                        y_idx = cand.index(e2) if e2 in cand else cand.index((e2[1], e2[0]))
                    except ValueError:
                        y_idx = None
                    L_act = torch.tensor(0.0, device=device) if y_idx is None \
                        else F.cross_entropy(logits.unsqueeze(0), torch.tensor([y_idx], device=device))

                loss = 1.0*L_cfm + 0.5*L_next + 1.0*L_act
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += float(loss.item())
        print(f"Epoch {ep+1}/{num_epochs} | loss={total_loss:.4f}")

# ---------- Main ----------

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
    num_layer = config['training']['num_layer']
    k_lap = config['training']['k_lap']
    heat_times = config['training']['heat_times']
    n_locals = 3
    in_channels = 2 + k_lap + k_lap * len(heat_times) + n_locals
    model = GraphERFlow(in_channels = in_channels, hidden_dim=hidden_dim, num_layers=num_layer, time_emb_dim=64)
    if args.input_model:
        model.load_model(model_dir / args.input_model)
        print(f"Model Graph-ER loaded from {args.input_model}")
    else:
        num_epochs = config['training']['num_epochs']
        learning_rate = config['training']['learning_rate']
        train_fm(model, train_graphs, num_epochs, learning_rate, k_lap, heat_times, device='cpu')
    if args.output_model:
        model.save_model(model_dir / args.output_model)
        print(f"Model saved to {args.output_model}")
    if args.evaluate:
        graph_eval = GraphsEvaluator()
        deg_eval = DegreeSequenceEvaluator()
        test_seqs = [[deg for _, deg in graph.degree()] for graph in test_graphs ]
    
        generated_graphs, generated_seqs = model.fm_generate_from_degree(degree_sequence, num_steps=32, graph_init='havel_hakimi', mask_builder=None, first_edge_sampler='uniform')
        print(f"Evaluate generated graphs using Havei Hamimi Model and MS-VAE")
        print(f"MMD Degree: {graph_eval.compute_mmd_degree_emd(test_graphs,generated_graphs,max_node)}")
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
    args = parser.parse_args()
    main(args)
