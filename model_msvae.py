import torch
import torch.nn as nn
import torch.nn.functional as F

# one-hot enconding 

def encode_degree_sequence(degree_sequence , max_class):
    sorted_degree_sequence = sorted(degree_sequence, reverse=True)
    one_hot_tensor = torch.zeros(max_class, dtype=torch.float32)
    for deg in sorted_degree_sequence:
        if 1 <= deg <= max_class:
            one_hot_tensor[deg - 1] += 1
        else:
            raise ValueError(f'Invalid degree sequence {degree_sequence}')
    return one_hot_tensor

def decode_degree_sequence(one_hot_tensor):
    degree_sequence = []
    for i, count in enumerate(one_hot_tensor.squeeze()):
        degree = i + 1  # Degree is index + 1
        count = int(count.item())  # Convert float to int
        degree_sequence.extend([degree] * count)  # Append 'count' times
    return degree_sequence


class MSVAEEncoder(torch.nn.Module):
    def __init__(self,  input_dim, hidden_dim, latent_dim):
        super(MSVAEEncoder, self).__init__()
        self.hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.mean_layer = torch.nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = torch.nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.hidden_layer(x))
        mean, logvar = self.mean_layer(h), self.logvar_layer(h)
        return mean, logvar

class MSVAEDecoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, max_output_dim,max_frequency):
        super(MSVAEDecoder, self).__init__()
        self.max_output_dim = max_output_dim
        self.max_frequency = max_frequency
        self.hidden_layer = torch.nn.Linear(latent_dim, hidden_dim)
        # Predicts a distribution over {0, 1, ..., N-1} for each degree frequency
        self.logits_layer = torch.nn.Linear(hidden_dim, max_output_dim * max_frequency)

    def forward(self, z, batch):
        h = F.relu(self.hidden_layer(z))                         # (B, H)
        logits = self.logits_layer(h)                            # (B, D*N)
        logits = logits.view(-1, self.max_output_dim, self.max_frequency)  # (B, D, N)
        return logits

class MSVAE(torch.nn.Module):
    def __init__(self, max_input_dim, hidden_dim, latent_dim, max_frequency):
        super(MSVAE, self).__init__()
        self.max_input_dim = max_input_dim
        self.latent_dim = latent_dim
        self.max_frequency = max_frequency
        self.encoder = MSVAEEncoder( max_input_dim, hidden_dim, latent_dim)
        self.decoder = MSVAEDecoder(latent_dim, hidden_dim,max_input_dim,max_frequency)
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, batch):
        mean, logvar = self.encoder(batch)
        z = self.reparameterize(mean, logvar)
        logits = self.decoder(z,batch)
        return logits, mean, logvar

    def fix_degree_sum_even(self, freq: torch.Tensor) -> torch.Tensor:
        """
        Ensures the sum of degrees in `freq` is even.
        Subtracts 1 from the highest non-zero entry if sum is odd.
        """
        indices = torch.arange(1, self.max_input_dim+1).float().to(freq.device)
        total = torch.sum(freq * indices).item()
        if total % 2 != 0:
            even_indices = torch.arange(0, freq.size(0), 2)  # get even indices: 0, 2, 4, ...
            even_values = freq[even_indices]                # values at even indices
            relative_max_idx = torch.argmax(even_values)    # index within the even subset
            max_idx = even_indices[relative_max_idx]        # map back to original index
            if freq[max_idx] > 0:
                freq[max_idx] += 1
        return freq

    def generate(self, num_samples ):
        self.eval()
        with torch.no_grad():
            z = torch.randn((num_samples, self.latent_dim))
            
            # Get (B, D, N) probabilities from decoder
            logits = self.decoder(z, None) 
            probs = F.softmax(logits, dim=-1)
            B, D, N = probs.shape
            samples = torch.multinomial(probs.view(-1, N), 1).view(B, D)

            fixed_sequences = []
            for freq in samples:
                freq_fixed = self.fix_degree_sum_even(freq.float())
                fixed_sequences.append(decode_degree_sequence(freq_fixed))
            return fixed_sequences
            
    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()