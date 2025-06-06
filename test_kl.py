import torch

def compute_kl_divergence(p_hist, q_hist, eps=1e-8, normalize=True):
    p = p_hist + eps
    q = q_hist + eps
    if normalize:
        p = p / p.sum()
        q = q / q.sum()
    return torch.sum(p * torch.log(p / q)).item()

# Tensors
p = torch.tensor([5.7441, 3.2092, 7.6390, 5.6720, 7.1429, 5.8428, 8.8499, 0.5180])
q = torch.tensor([8.7063, 9.0063, 3.9466, 2.1569, 9.6507, 3.7233, 1.5313, 9.1919])

# Compute KL(P || Q)
kl = compute_kl_divergence(p, q)
print(f"KL Divergence (P || Q): {kl:.4f}")