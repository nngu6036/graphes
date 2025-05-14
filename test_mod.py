import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Mod2SinPredictor(nn.Module):
    def __init__(self, input_dim):
        super(Mod2SinPredictor, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)  # Linear sum

    def forward(self, one_hot_tensor):
        sum_pred = self.linear(one_hot_tensor)  # Compute sum
        return (1 - torch.cos(np.pi * sum_pred)) / 2  # Cosine approximation of mod 2

# Generate synthetic one-hot data
input_dim = 10
num_samples = 1000

# Generate random one-hot vectors
x_train = torch.eye(input_dim)[torch.randint(0, input_dim, (num_samples,))]
sum_values = torch.sum(x_train * torch.arange(1, input_dim+1, dtype=torch.float32), dim=1, keepdim=True)
y_train = (sum_values % 2)  # Modulo 2 labels (0 for even, 1 for odd)

# Initialize model, loss, and optimizer
model = Mod2SinPredictor(input_dim)
criterion = nn.MSELoss()  # Regression-style loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 2000
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Test the model
x_test = torch.eye(input_dim)[torch.tensor([0, 2, 4, 7, 9])]  # Test cases
y_pred = model(x_test).detach().numpy()
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert predictions to 0 or 1

print("Predicted mod 2 values:", y_pred_binary.flatten())
