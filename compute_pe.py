#!/usr/bin/env python3
"""
Precompute positional encodings (PE) for PyG QM9 / ZINC and save to disk.

Default:
  - Laplacian eigenvector PE (normalized Laplacian)
  - Skip the first trivial eigenvector
  - Save as a list of [num_nodes, k] tensors to <root>/pe_cache/<dataset>_lap_pe_k{k}.pt

Example:
  python precompute_pe.py --dataset QM9 --root datasets/QM9 --k 8
  python precompute_pe.py --dataset ZINC --root datasets/ZINC --k 16 --subset
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch_geometric.datasets import QM9, ZINC
from torch_geometric.utils import to_undirected

# -------------------------
# PE computation utilities
# -------------------------

def dense_adj_from_edge_index(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Build dense adjacency (0/1) for an undirected simple graph."""
    A = torch.zeros((n, n), dtype=torch.float32)
    if edge_index.numel() == 0:
        return A
    A[edge_index[0], edge_index[1]] = 1.0
    # ensure no self-loops contribute
    A.fill_diagonal_(0.0)
    return A

def normalized_laplacian(A: torch.Tensor) -> torch.Tensor:
    """
    L = I - D^{-1/2} A D^{-1/2}
    where D is the degree diagonal matrix.
    """
    n = A.size(0)
    deg = A.sum(dim=1)  # [n]
    inv_sqrt = torch.zeros_like(deg)
    inv_sqrt[deg > 0] = deg[deg > 0].pow(-0.5)
    D_inv_sqrt = torch.diag(inv_sqrt)
    I = torch.eye(n, dtype=A.dtype)
    L = I - D_inv_sqrt @ A @ D_inv_sqrt
    return L

def combinatorial_laplacian(A: torch.Tensor) -> torch.Tensor:
    """L = D - A."""
    deg = A.sum(dim=1)
    return torch.diag(deg) - A

@torch.no_grad()
def lap_pe(edge_index: torch.Tensor, num_nodes: int, k: int,
           normalized: bool = True,
           skip_first: bool = True,
           deterministic_sign: bool = True) -> torch.Tensor:
    """
    Compute Laplacian eigenvector PE:
      - returns [num_nodes, k] (or fewer if graph too small)
      - optionally skips the first trivial eigenvector
      - uses deterministic sign fixing to make results stable across runs
    """
    if num_nodes <= 0:
        return torch.zeros((0, k), dtype=torch.float32)

    # undirected adjacency
    ei = to_undirected(edge_index, num_nodes=num_nodes)
    A = dense_adj_from_edge_index(ei, num_nodes)

    L = normalized_laplacian(A) if normalized else combinatorial_laplacian(A)

    # eigen-decomposition (symmetric)
    evals, evecs = torch.linalg.eigh(L)  # evals asc

    # choose columns
    start = 1 if skip_first else 0
    avail = max(0, evecs.size(1) - start)
    kk = min(k, avail)
    if kk == 0:
        return torch.zeros((num_nodes, k), dtype=torch.float32)

    pe = evecs[:, start:start + kk].to(torch.float32)  # [n, kk]

    # pad if kk < k (rare: tiny graphs)
    if kk < k:
        pad = torch.zeros((num_nodes, k - kk), dtype=torch.float32)
        pe = torch.cat([pe, pad], dim=1)

    # deterministic sign flip (common trick):
    # make the largest-magnitude entry in each eigenvector positive
    if deterministic_sign:
        for j in range(pe.size(1)):
            col = pe[:, j]
            idx = torch.argmax(col.abs())
            if col[idx] < 0:
                pe[:, j] = -col

    return pe  # [n, k]

# -------------------------
# Main
# -------------------------

def load_dataset(name: str, root: str, subset: bool) -> Tuple[torch.utils.data.Dataset, str]:
    name = name.upper()
    root = str(root)

    if name == "QM9":
        ds = QM9(root=root)
        tag = "QM9"
    elif name == "ZINC":
        # subset=True -> ZINC subset (common in papers); False -> full ZINC
        ds = ZINC(root=root, subset=subset)
        tag = "ZINCsubset" if subset else "ZINCfull"
    else:
        raise ValueError("dataset must be one of: QM9, ZINC")

    return ds, tag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["QM9", "ZINC"])
    parser.add_argument("--root", type=str, required=True, help="PyG dataset root directory")
    parser.add_argument("--k", type=int, default=8, help="PE dimension")
    parser.add_argument("--normalized", action="store_true", help="Use normalized Laplacian (default)")
    parser.add_argument("--combinatorial", action="store_true", help="Use combinatorial Laplacian instead")
    parser.add_argument("--no-skip-first", action="store_true", help="Do NOT skip the first trivial eigenvector")
    parser.add_argument("--no-sign-fix", action="store_true", help="Disable deterministic eigenvector sign fix")
    parser.add_argument("--subset", action="store_true", help="For ZINC: use subset=True")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda (dense eig is small here)")
    args = parser.parse_args()

    if args.combinatorial and args.normalized:
        raise ValueError("Choose at most one: --normalized or --combinatorial")
    use_normalized = True
    if args.combinatorial:
        use_normalized = False

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ds, tag = load_dataset(args.dataset, args.root, args.subset)

    out_dir = Path(args.root) / "pe_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    lap_kind = "norm" if use_normalized else "comb"
    skip_first = not args.no_skip_first
    sign_fix = not args.no_sign_fix

    out_path = out_dir / f"{tag}_lap_pe_{lap_kind}_k{args.k}_skip{int(skip_first)}.pt"

    pe_list: List[torch.Tensor] = []
    num_nodes_list: List[int] = []

    print(f"Dataset: {tag}  | graphs: {len(ds)}")
    print(f"Computing LapPE: k={args.k}, lap={lap_kind}, skip_first={skip_first}, sign_fix={sign_fix}")
    print(f"Saving to: {out_path}")

    for i in range(len(ds)):
        data = ds[i]
        n = int(data.num_nodes)
        ei = data.edge_index

        # compute PE on CPU unless you explicitly want CUDA;
        # these graphs are tiny (QM9/ZINC), so CPU is typically fine.
        ei_ = ei.to(device)
        pe = lap_pe(
            ei_,
            num_nodes=n,
            k=args.k,
            normalized=use_normalized,
            skip_first=skip_first,
            deterministic_sign=sign_fix,
        ).cpu()

        pe_list.append(pe)
        num_nodes_list.append(n)

        if (i + 1) % 10000 == 0:
            print(f"  processed {i+1}/{len(ds)}")

    payload: Dict[str, object] = {
        "dataset": tag,
        "k": args.k,
        "laplacian": lap_kind,
        "skip_first": skip_first,
        "sign_fix": sign_fix,
        "num_graphs": len(ds),
        "num_nodes": num_nodes_list,
        "pe": pe_list,  # list of [n_i, k] tensors
    }
    torch.save(payload, out_path)
    print("Done.")

if __name__ == "__main__":
    main()
