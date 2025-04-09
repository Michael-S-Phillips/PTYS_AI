# Topographic Diffusion of Lunar Impact Craters

This repository contains code and resources for simulating and analyzing the topographic diffusion of lunar impact craters using Physics-Informed Neural Networks (PINNs). The goal is to recreate and extend the results from Fasset and Thomson (2014) and other studies.

## Directory Structure

- `crater_diffusion/`
  - `crater_diffusion_pinn.py`: Contains the `CraterDiffusionPINN` class for simulating crater diffusion using PINNs.
  - `crater_diffusion_utils.py`: Utility functions for crater topography calculations and data preparation.
  - `crater_diffusion.ipynb`: Jupyter notebook for running simulations and visualizations.
  - `files/`: Directory containing images and other resources.

## Usage

### Crater Diffusion Simulation

1. **Initial Crater Topography**: The initial topography of craters is calculated using empirical equations from various studies.

2. **1D Diffusion Simulation**: Simulate the topographic diffusion in one spatial dimension using an implicit solution.

3. **2D Diffusion Simulation**: Scale up to two spatial dimensions and simulate the diffusion using both explicit and implicit solutions.

4. **Physics-Informed Neural Network (PINN)**: Use PINNs to simulate the diffusion process and compare the results with traditional numerical methods.

### Parameter Discovery

The `CraterDiffusionPINN` class can be used to discover the optimal `kappa`, radius, and time values that fit a given height array.

### Example Usage

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from crater_diffusion_pinn import CraterDiffusionPINN
from crater_diffusion_utils import plot_crater_times

# Parameters
kappa = 5.5  # diffusion coefficient (m²/Myr)
D = 100 * np.pi / 2 #np.sqrt(kappa)  # crater diameter (m), sqrt kappa makes the time scaling 1:1 at 1 million years.
radius = D / 2.0
r_max = tf.cast(D, dtype=tf.float32)  # half grid extent in meters
t_max = np.ceil(D**2 / kappa) # appx 4.5 Gyr

# Create and train the PINN
pinn = CraterDiffusionPINN(kappa, radius, r_max, t_max)
pinn.model = pinn._build_network()
# pinn.model.load_weights("pinn_pure_10000_epoch_varD.keras")
losses = pinn.train(n_epochs=50000, n_points = 1000)

# Predict at different times
times = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
# times = [t_max * ts / (0.5 * times[-1])  for ts in times]
predictions = []

# Parameters for visualization
kappa = 5.5  # diffusion coefficient (m²/Myr)
D = 300
radius = D / 2.0
r_max = tf.cast(D, dtype=tf.float32)  # half grid extent in meters
t_max = 4500

# Create prediction grid
x = np.linspace(-r_max, r_max, 100)  # Grid in meters
y = np.linspace(-r_max, r_max, 100)  # Grid in meters
X, Y = np.meshgrid(x, y)

pinn.__init__(kappa, radius, r_max, t_max)

for t in times:
    h_pred = pinn.predict(X.flatten(), Y.flatten(), t * np.ones_like(X.flatten()))
    predictions.append(D * h_pred.reshape(X.shape))

fig = plot_crater_times(predictions, x, y, kappa, times, "Crater Diffusion Over Time")
fig.show()

# Plot loss history
plt.figure(figsize=(8, 4))
plt.semilogy(losses[:, 0], label='Total Loss')
plt.semilogy(losses[:, 1], label='PDE Loss')
plt.semilogy(losses[:, 2], label='Initial Condition Loss')
plt.semilogy(losses[:, 3], label='Boundary Condition Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
```
![Crater Diffusion Simulation](crater_diffusion_300m_pinn.gif)
## Visualization

The repository includes functions for visualizing the evolution of crater topography over time and comparing the results with empirical data.

## References

- Fasset and Thomson (2014): [Link to paper](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2014JE004698)
- Yang et al. (2021): [Link to paper](http://dx.doi.org/10.1029/2021GL095537)
- Fasset et al. (2022): [Link to paper](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022JE007510)

## License

This project is licensed under the MIT License.