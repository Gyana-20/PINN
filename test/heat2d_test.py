import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.PINN import PINN  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = os.path.join(os.path.dirname(__file__), "../results/heat2d.pth")
model_path = os.path.abspath(model_path) 

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

input_dim = 3
hidden_dims = [32, 64, 64, 32]
output_dim = 1

model = PINN(input_dim, hidden_dims, output_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()

x0, x1 = 0, 1
y0, y1 = 0, 1
t0, t1 = 0, 1

x_vals = torch.linspace(x0, x1, 100)
y_vals = torch.linspace(y0, y1, 100)
X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')

# Flatten grid for model input
X_flat, Y_flat = X.flatten().unsqueeze(1), Y.flatten().unsqueeze(1)

# Define time steps
num_frames = 50
t_vals = torch.linspace(t0, t1, num_frames)

# Prepare figure
fig, ax = plt.subplots(figsize=(8, 6))
fig.set_size_inches(8, 6)  

# Initial temperature field
T_fixed = torch.ones_like(X_flat) * t_vals[0]
with torch.no_grad():
    U_pred = model(X_flat.to(device), Y_flat.to(device), T_fixed.to(device))
U_pred = U_pred.cpu().numpy().reshape(100, 100)

# Initial plot
contour = ax.contourf(X.numpy(), Y.numpy(), U_pred, cmap="coolwarm", levels=50)
ax.set_xlim([x0, x1])  # Fix x-axis limits
ax.set_ylim([y0, y1])  # Fix y-axis limits
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Heat Distribution at t = {t_vals[0]:.2f}")
cbar = fig.colorbar(contour, ax=ax)

def update(frame):
    global contour

    T_fixed = torch.ones_like(X_flat) * t_vals[frame]
    with torch.no_grad():
        U_pred = model(X_flat.to(device), Y_flat.to(device), T_fixed.to(device))
    U_pred = U_pred.cpu().numpy().reshape(100, 100)

    # Update contour plot
    for coll in contour.collections:
        coll.remove()
    contour = ax.contourf(X.numpy(), Y.numpy(), U_pred, cmap="coolwarm", levels=50)

    # Keep axis fixed
    ax.set_xlim([x0, x1])
    ax.set_ylim([y0, y1])
    ax.set_title(f"Heat Distribution at t = {t_vals[frame]:.2f}")

    # Update colorbar
    cbar.update_normal(contour)

# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100)
plt.tight_layout()
plt.show()

ani.save("heat2d.mp4", writer="ffmpeg", fps=10)