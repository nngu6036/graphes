import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Simulating swap intensity over time for a large graph (50 nodes, 20 edge pairs tracked)
num_time_steps = 256
num_edge_pairs = 16

data = np.random.rand(num_edge_pairs, num_time_steps)  # Random transition intensities

# Create heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(data, cmap="coolwarm", annot=False, ax=ax)

# Label axes
ax.set_xlabel("Time Step")
ax.set_ylabel("Edge Pairs")
ax.set_title("Edge Swapping Transition Heatmap (Large Graph)")

# Show plot
plt.show()