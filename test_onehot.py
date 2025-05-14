import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class OneHotDecoder(nn.Module):
    def __init__(self, input_dim):
        super(OneHotDecoder, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, 64),  # First hidden layer
            nn.ReLU(),
            nn.Linear(64, 32),  # Second hidden layer
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.ReLU()
        )

    def forward(self, one_hot_tensor):
        seq = self.hidden(one_hot_tensor)  # Apply the neural network layers
        return seq

# Generate synthetic one-hot data
input_dim = 10
num_samples = 50000

# Generate random one-hot vectors
def generate_random_vector(N):
    # Generate N random values from uniform distribution
    random_values = torch.rand(N)

    # Scale the values so that their sum is at most N
    scaled_values = (random_values / random_values.sum()) * N

    # Convert to integers while keeping sum ≤ N
    vector = torch.floor(scaled_values).to(torch.int)

    # Adjust for rounding errors to ensure sum ≤ N
    while vector.sum() > N:
        idx = torch.randint(0, N, (1,)).item()  # Pick a random index
        if vector[idx] > 0:
            vector[idx] -= 1  # Reduce element to satisfy constraint

    return vector

def construct_vector(x, N):
    # Initialize an empty list to collect values
    y_list = []

    # Iterate over each element in x
    for i, val in enumerate(x):
        y_list.extend([i + 1] * val)  # Create a list of length x[i] with value i+1

    # Convert to a PyTorch tensor
    y = torch.tensor(y_list, dtype=torch.int)

    # Pad with zeros if necessary
    if len(y) < N:
        padding = torch.zeros(N - len(y), dtype=torch.int)
        y = torch.cat([y, padding])  # Append zeros to make length N

    return y

# Generate random one-hot vectors
x_train = torch.stack([generate_random_vector(input_dim) for _ in range(num_samples)])
y_train = torch.stack([construct_vector(x,input_dim) for x in x_train])


# Initialize model, loss, and optimizer
model = OneHotDecoder(input_dim)
criterion = nn.MSELoss()  # Regression-style loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train.float())
    loss = criterion(y_pred, y_train.float())
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Test the model
x_test = generate_random_vector(input_dim)
y_test = construct_vector(x_test, input_dim)
y_pred = model(y_test.float())

print("Test data:", x_test.tolist())
print("Test output:", y_test.tolist())
print("Predicted output:", y_pred.tolist())
