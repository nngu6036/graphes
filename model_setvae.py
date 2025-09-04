import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
# one-hot enconding 

def encode_degree_sequence(degree_sequence, max_degree, normalize=True):
    """
    Encode a degree sequence into a histogram vector of multiplicities.

    Args:
        degree_sequence (list[int]): List of node degrees.
        max_degree (int): Maximum possible degree value.
        normalize (bool): If True, divide counts by total number of nodes.

    Returns:
        torch.Tensor: Multiplicity vector h of shape (max_degree+1,).
                      Normalized if normalize=True.
    """
    # Clamp degrees into [0, max_degree]
    clamped = [max(0, min(d, max_degree)) for d in degree_sequence]
    # Compute multiplicity counts
    h = torch.zeros(max_degree + 1, dtype=torch.float)
    for deg in clamped:
        h[deg] += 1
    # Normalize so sum(h) == 1 (or keep raw counts)
    if normalize and h.sum() > 0:
        h /= h.sum()
    return h

def decode_degree_sequence(one_hot_tensor):
    # one_hot_tensor is actually a multiplicity/count vector m[0..Dmax]
    degree_sequence = []
    for i, count in enumerate(one_hot_tensor.squeeze()):
        deg = i                    # <-- fix off-by-one (degree == index)
        c = int(count.item())
        degree_sequence.extend([deg] * c)
    return degree_sequence

def _masked_multinomial_nll(logits, m_counts, N_nodes):
    B, Dp1 = logits.shape
    device = logits.device
    losses = []
    for b in range(B):
        n = int(N_nodes[b].item())
        n = max(n, 1)
        mask = torch.arange(Dp1, device=device) <= (n - 1)
        logp = F.log_softmax(logits[b][mask], dim=-1)
        counts = m_counts[b][mask]
        nll = -(counts * logp).sum() / (n + 1e-8)
        losses.append(nll)
    return torch.stack(losses).mean()

class SetVAE(torch.nn.Module):
    def __init__(self, hidden_dim, latent_dim, max_degree):
        super(SetVAE, self).__init__()
        self.max_degree = max_degree
        self.latent_dim = latent_dim
        self.size_embed = nn.Linear(1, hidden_dim)  # scalar N -> hidden
        Dp1 = max_degree + 1

        # phi: learnable embedding for each degree value
        self.phi = nn.Embedding(Dp1, hidden_dim)
        nn.init.xavier_uniform_(self.phi.weight)

        # ρ ∘ sum
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Encoder heads -> q(z|D)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder p(D|z): logits over multiplicities 0..max_degree
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, Dp1)
        )
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, m_counts, N_nodes):
        B, Dp1 = m_counts.shape
        idx = torch.arange(Dp1, device=m_counts.device)
        phi_d = self.phi(idx)
        h = (m_counts.unsqueeze(-1) * phi_d.unsqueeze(0)).sum(dim=1)  # [B, hidden]
        h = h + self.size_embed(N_nodes.view(-1,1).float())           # inject N
        h = self.rho(h)
        mu, logvar = self.enc_mu(h), self.enc_logvar(h)
        return mu, logvar

    def decode(self, z):
        return self.dec(z)
    
    def forward(self, m_counts, N_nodes):
        """
        m_counts: [B, Dp1] integer multiplicities (targets)
        N_nodes:  [B]      sum(m_counts)
        """
        mu, logvar = self.encode(m_counts,N_nodes)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)

        recon = _masked_multinomial_nll(logits, m_counts, N_nodes)  # size-aware
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        loss = recon + kld
        return {"logits": logits, "mu": mu, "logvar": logvar, "loss": loss}

    @torch.no_grad()
    def generate(self, N_nodes):
        self.eval()
        device = next(self.parameters()).device
        B = len(N_nodes)
        N_tensor = torch.tensor(N_nodes, device=device, dtype=torch.long)

        z = torch.randn(B, self.latent_dim, device=device)
        logits = self.decode(z)
        probs_full = F.softmax(logits, dim=-1)

        m_list = []
        Dp1 = probs_full.size(1)
        degs = torch.arange(Dp1, device=device)

        for b in range(B):
            n = int(N_tensor[b].item())
            n = max(n, 1)
            mask = (degs <= (n - 1))
            probs = probs_full[b] * mask
            probs = probs / (probs.sum() + 1e-8)  # renormalize on valid support

            sampled = torch.multinomial(probs, num_samples=n, replacement=True)
            m_counts = torch.bincount(sampled, minlength=self.max_degree + 1)

            m_list.append(m_counts)

        samples = torch.stack(m_list, dim=0).to(device)

        # Parity repair (even sum of degrees)
        degs = torch.arange(self.max_degree + 1, device=device)
        sum_deg = (samples * degs).sum(dim=1)
        odd_idx = torch.where(sum_deg % 2 == 1)[0]

        for b in odd_idx.tolist():
            n = int(N_tensor[b].item())
            # prefer 0 -> 1 if we have any zeros and n >= 2
            if samples[b, 0] > 0 and n >= 2:
                samples[b, 0] -= 1
                samples[b, 1] += 1
            else:
                # otherwise, find any d>=1 with mass and move one to d-1
                d = int((samples[b, 1:] > 0).nonzero(as_tuple=False).min().item()) + 1  # first d>=1 that exists
                samples[b, d] -= 1
                samples[b, d - 1] += 1

        # Optional: EG projection/repair (cheap heuristic)
        degree_sequences = []
        for m in samples:
            dseq = decode_degree_sequence(m)
            n = len(dseq)
            dseq = [min(d, n - 1) for d in dseq]  # hard clip as a safety net
            dseq.sort(reverse=True)
            # Simple downtick projection if not EG-graphical
            while not nx.is_graphical(dseq, method='eg'):
                for i in range(len(dseq)):
                    if dseq[i] > 0:
                        dseq[i] -= 1
                        break
                dseq.sort(reverse=True)
            degree_sequences.append(dseq)

        return degree_sequences

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path, map_location=None):
        if map_location is None:
            map_location = next(self.parameters()).device
        state = torch.load(file_path, map_location=map_location)
        self.load_state_dict(state)
        self.eval()