import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

# Define a set of alpha and beta parameter pairs
params = [
    (0.5, 0.5),
    (2, 5),
    (1e-3, 10)
]

x = np.linspace(0, 1, 1000)

# Plotting
plt.figure(figsize=(12, 8))

for a, b in params:
    y = beta.pdf(x, a, b)
    plt.plot(x, y, label=f'α={a}, β={b}')

plt.title('Beta Distribution for Various α and β')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend(title='Parameters')
plt.grid(True)
plt.tight_layout()
plt.show()
