import torch
import torch.nn as nn
import torch.nn.functional as F

# padding 0 to the sequence
def encode_degree_sequence(degree_sequence , max_class):
    sequence = degree_sequence + [0] * (max_class - len(degree_sequence))
    return torch.tensor(sequence).float()


def decode_degree_sequence(tensor):
    tensor = tensor.long()
    non_zero_tensor = tensor[tensor != 0]
    return non_zero_tensor.tolist()


class StdVAEEncoder(torch.nn.Module):
    def __init__(self,  input_dim, hidden_dim, latent_dim):
        super(StdVAEEncoder, self).__init__()
        self.hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.mean_layer = torch.nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = torch.nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.hidden_layer(x))
        mean, logvar = self.mean_layer(h), self.logvar_layer(h)
        return mean, logvar

class StdVAEDecoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, max_output_dim,max_degree):
        super(StdVAEDecoder, self).__init__()
        self.max_output_dim = max_output_dim
        self.max_degree = max_degree
        self.hidden_layer = torch.nn.Linear(latent_dim, hidden_dim)
        
        # Predicts a distribution over {0, 1, ..., N-1} for each degree
        self.logits_layer = torch.nn.Linear(hidden_dim, max_output_dim * max_degree)

    def forward(self, z, batch):
        h = F.relu(self.hidden_layer(z))                         # (B, H)
        logits = self.logits_layer(h)                            # (B, D*N)
        logits = logits.view(-1, self.max_output_dim, self.max_degree)  # (B, D, N)
        return logits

class StdVAE(torch.nn.Module):
    def __init__(self, max_input_dim, hidden_dim, latent_dim):
        super(StdVAE, self).__init__()
        self.max_input_dim = max_input_dim
        self.encoder = StdVAEEncoder( max_input_dim, hidden_dim, latent_dim)
        self.decoder = StdVAEDecoder(latent_dim, hidden_dim,max_input_dim,max_input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, batch):
        mean, logvar = self.encoder(batch)
        z = self.reparameterize(mean, logvar)
        logits = self.decoder(z,batch)
        return logits, mean, logvar

    def fix_degree_sum_even(self, sequence: torch.Tensor) -> torch.Tensor:
        total = sequence.sum().item()
        if total % 2 == 0:
            return sequence
        # Find the largest odd value in the tensor
        odd_mask = (sequence % 2 == 1)
        odd_indices = torch.nonzero(odd_mask, as_tuple=True)[0]
        sequence[odd_indices] += 1
        return sequence

    def generate(self, num_samples):
        self.eval()
        with torch.no_grad():
            z = torch.randn((num_samples, self.latent_dim))
            
            # Get (B, D, N) probabilities from decoder
            logits = self.decoder(z, None)  
            logits = logits.view(-1, self.max_input_dim, self.max_input_dim)
            probs = F.softmax(logits, dim=-1)
            B, D, N = probs.shape
            samples = torch.multinomial(probs.view(-1, N), 1).view(B, D)
            fixed_sequences = []
            for seq in samples:
                seq_fixed = self.fix_degree_sum_even(seq)
                fixed_sequences.append(decode_degree_sequence(seq_fixed))
            return fixed_sequences

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()
