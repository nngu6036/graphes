import torch
import torch.nn as nn
import torch.nn.functional as F

# one-hot enconding 

def encode_degree_sequence(degree_sequence , max_degree):
    h = torch.bincount(degree_sequence.clamp_min(0).clamp_max(max_degree),
                       minlength=max_degree+1).float()
    return h  # counts (m_0,...,m_max)

def decode_degree_sequence(one_hot_tensor):
    # one_hot_tensor is actually a multiplicity/count vector m[0..Dmax]
    degree_sequence = []
    for i, count in enumerate(one_hot_tensor.squeeze()):
        deg = i                    # <-- fix off-by-one (degree == index)
        c = int(count.item())
        degree_sequence.extend([deg] * c)
    return degree_sequence

class SetVAE(torch.nn.Module):
    def __init__(self, hidden_dim, latent_dim, max_degree):
        super(SetVAE, self).__init__()
        self.max_degree = max_degree
        self.latent_dim = latent_dim

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

    def encode(self, m_counts):
        B, Dp1 = m_counts.shape
        idx = torch.arange(Dp1, device=m_counts.device)  # [0..Dmax]
        phi_d = self.phi(idx)                            # [Dp1, hidden]
        h = (m_counts.unsqueeze(-1) * phi_d.unsqueeze(0)).sum(dim=1)  # [B, hidden]
        h = self.rho(h)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def decode(self, z):
        return self.dec(z)
    
    def forward(self, m_counts, N_nodes):
        """
        m_counts: [B, Dp1] integer multiplicities (targets)
        N_nodes:  [B]      sum(m_counts)
        """
        mu, logvar = self.encode(m_counts)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)  # [B, Dp1]

        # Reconstruction: KL(target || softmax(logits)) == CE(target, logits) - H(target)
        target_prob = (m_counts + 1e-8) / (N_nodes.unsqueeze(1) + 1e-8)
        recon = F.kl_div(F.log_softmax(logits, dim=-1), target_prob, reduction='batchmean')

        # Latent KL
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        # EG soft validity

        total = recon + kld 
        return {
            "logits": logits,
            "mu": mu, "logvar": logvar,
            "loss": total,
            "loss_terms": {"recon": recon.detach(), "kld": kld.detach()()}
        }

    @torch.no_grad()
    def generate(self, batch_size, N_nodes):
        self.eval()
        device = next(self.parameters()).device
        z = torch.randn(batch_size, self.latent_dim, device=device)
        logits = self.decode(z)
        probs = F.softmax(logits, dim=-1)
        m_list = []
        for b in range(batch_size):
            m_b = torch.multinomial(probs[b], num_samples=int(N_nodes[b].item()), replacement=True)
            m_counts = torch.bincount(m_b, minlength=self.max_degree+1)
            m_list.append(m_counts)
        samples = torch.stack(m_list, dim=0).to(device)

        # Parity repair (heuristic)
        degs = torch.arange(self.max_degree+1, device=device)
        sum_deg = (samples * degs).sum(dim=1)
        odd = (sum_deg % 2 == 1)
        if odd.any():
            for b in torch.where(odd)[0].tolist():
                if samples[b, 0] > 0:
                    samples[b, 0] -= 1
                    samples[b, 1] += 1
                else:
                    samples[b, 1] = samples[b, 1].clamp_min(1) - 1
                    samples[b, 0] += 1

        degree_sequences = [decode_degree_sequence(m) for m in samples]
        return degree_sequences

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()