import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.PINN import PINN  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = os.path.join(os.path.dirname(__file__), "../results/heat1d.pth")
model_path = os.path.abspath(model_path) 

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")



# Model architecture (should match training)
input_dim = 2
hidden_dims = [32, 64, 64, 32]
output_dim = 1

model = PINN(input_dim, hidden_dims, output_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()

# Define the plotting domain
x0, x1 = -1, 1  # Spatial domain
t0, t1 = 0, 1  # Time domain

n_x, n_t = 1000, 1000  # Grid resolution
x_vals = torch.linspace(x0, x1, n_x).view(-1, 1).to(device)
t_vals = torch.linspace(t0, t1, n_t).view(-1, 1).to(device)

# Create a meshgrid for evaluation
X, T = torch.meshgrid(x_vals.squeeze(), t_vals.squeeze(), indexing='ij')
X_flat = X.reshape(-1, 1)
T_flat = T.reshape(-1, 1)

# Make predictions
with torch.no_grad():
    U_pred = model(X_flat.to(device), T_flat.to(device))
    U_pred = U_pred.cpu().numpy().reshape(n_x, n_t)

# Convert tensors to numpy for plotting
X = X.cpu().numpy()
T = T.cpu().numpy()

# Plot the solution
plt.figure(figsize=(8, 6))
plt.contourf(X, T, U_pred, levels=50, cmap="coolwarm")
plt.colorbar(label="Solution u(x, t)")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Heat Equation Solution (PINN)")
plt.show()
