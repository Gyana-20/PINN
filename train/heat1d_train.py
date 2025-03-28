import torch
import torch.optim as optim
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.PINN import PINN
from pde_residue.heat1d import pde_residual

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# fix the domains and alpha
alpha = 0.01

x0, x1 = -1, 1
t0, t1 = 0, 1

# create the model
input_dim = 2
hidden_dims = [32, 64, 64, 32]
output_dim = 1

model = PINN(input_dim, hidden_dims, output_dim).to(device)

# create the collocation data points
n_collocation = 50000
x_collocation = (x0 + (x1-x0)*torch.rand(n_collocation, 1)).to(device)
t_collocation = (t0 + (t1-t0)*torch.rand(n_collocation, 1).to(device))

x_collocation.requires_grad = True
t_collocation.requires_grad = True

# initial condition
n_ic = 500
x_ic = torch.linspace(x0, x1, n_ic).view(-1, 1).to(device)
t_ic = torch.zeros_like(x_ic).to(device)
u_ic = torch.sin(torch.pi * x_ic).to(device)

# boundary condition (Neumann)
n_bc = 500
t_bc = (t0 + (t1 - t0) * torch.rand(n_bc, 1)).to(device)
x_bc_left = torch.full_like(t_bc, x0).to(device)
x_bc_right = torch.full_like(t_bc, x1).to(device)

x_bc_left.requires_grad = True
x_bc_right.requires_grad = True

f_t = torch.sin(torch.pi*t_bc).to(device)
g_t = torch.cos(torch.pi*t_bc).to(device)

optimizer = optim.LBFGS(model.parameters(), max_iter=500, tolerance_grad=1e-7, tolerance_change=1e-9)


def closure():
    optimizer.zero_grad()
    u_pred = model(x_collocation, t_collocation)
    u_pred_ic = model(x_ic, t_ic)
    u_bc_left_pred = model(x_bc_left, t_bc)
    u_bc_right_pred = model(x_bc_right, t_bc)
    
    u_x_left = torch.autograd.grad(u_bc_left_pred, x_bc_left, torch.ones_like(u_bc_left_pred), 
                               create_graph=True, retain_graph=True)[0]
    u_x_right = torch.autograd.grad(u_bc_right_pred, x_bc_right, torch.ones_like(u_bc_right_pred),
                                create_graph= True, retain_graph=True)[0] 

    pde_res = pde_residual(u_pred, x_collocation, t_collocation, alpha)
    pde_loss = torch.mean(pde_res**2)
    ic_loss = torch.mean((u_pred_ic - u_ic)**2)
    bc_loss = torch.mean((u_x_left-f_t)**2 + (u_x_right-g_t)**2)
    
    loss = pde_loss + ic_loss + bc_loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    loss.backward()
    
    return loss

num_epochs = 10

for epoch in range (num_epochs):
    loss = optimizer.step(closure)
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_dir, exist_ok=True)

model_save_path = os.path.join(results_dir, "heat1d.pth")
torch.save(model.state_dict(), model_save_path)

print(f"Training complete. Model saved at: {model_save_path}")
          
    