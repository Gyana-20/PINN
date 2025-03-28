import torch
import torch.optim as optim
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.PINN import PINN
from pde_residue.burger2d import residual_burger_2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name(0)}")

# Define the domain and nu

nu = 1/3000
x0 , x1 = 0, 1
y0 , y1 = 0, 1
t0, t1 = 0, 1

# interior points
x_int = (x0 + (x1-x0)*(torch.rand(5000, 1))).to(device)
y_int = (y0 + (y1-y0)*(torch.rand(5000, 1))).to(device)
t_int = (t0 + (t1-t0)*(torch.rand(5000, 1))).to(device)

x_int.requires_grad = True
y_int.requires_grad = True
t_int.requires_grad = True

# initial points
x_ic = (torch.rand(100, 1)*(x1-x0)+x0).to(device)
y_ic = (torch.rand(100, 1)*(y1-y0)+y0).to(device)
t_ic = torch.zeros_like(x_ic).to(device)

u_0 = torch.sin(torch.pi * x_ic) * torch.cos(torch.pi * y_ic)
v_0 = torch.cos(torch.pi * x_ic) * torch.sin(torch.pi * y_ic)

#boundary points
t_bc = (t0 + (t1-t0)*(torch.rand(100, 1))).to(device)
y_bc = (y0 + (y1 - y0) * torch.rand(100, 1)).to(device)
x_bc = (x0 + (x1 - x0) * torch.rand(100, 1)).to(device)

x0_bc = torch.zeros_like (t_bc).to(device)
x1_bc = torch.ones_like (t_bc).to(device)
y0_bc = torch.zeros_like (t_bc).to(device)
y1_bc = torch.ones_like (t_bc).to(device)

u_bc_x0 = torch.zeros_like(t_bc).to(device)
u_bc_x1 = torch.zeros_like(t_bc).to(device)
v_bc_y0 = torch.zeros_like(t_bc).to(device)
v_bc_y1 = torch.zeros_like(t_bc).to(device)

# define the model

input_dim = 3
hidden_dims = [128, 256, 256,128]
output_dim = 2

model = PINN(input_dim, hidden_dims, output_dim).to(device)

# optimization

optimizer = optim.LBFGS (model.parameters(), lr = 0.01,
                         max_iter = 500, 
                         tolerance_grad= 1e-7,
                         tolerance_change= 1e-9)

def closure():
    optimizer.zero_grad()
    
    u_int, v_int = model(x_int, y_int, t_int).T

    u0_hat, v0_hat = model (x_ic, y_ic, t_ic).T
    
    u_bc_x0_hat, v_bc_x0_hat = model(x0_bc, y_bc, t_bc).T  # u at x=0
    u_bc_x1_hat, v_bc_x1_hat = model(x1_bc, y_bc, t_bc).T  # u at x=1
    u_bc_y0_hat, v_bc_y0_hat = model(x_bc, y0_bc, t_bc).T  # v at y=0
    u_bc_y1_hat, v_bc_y1_hat = model(x_bc, y1_bc, t_bc).T  # v at y=1
    
    res1, res2 = residual_burger_2d(u_int, v_int, x_int, y_int, t_int, nu)
    
    pde_loss = torch.mean(res1**2 + res2**2)
    
    ic_loss = torch.mean((u0_hat - u_0)**2 + (v0_hat - v_0)**2)
    
    bc_loss = torch.mean((u_bc_x0 - u_bc_x0_hat)**2 +
                         (u_bc_x1 - u_bc_x1_hat)**2 +
                         (v_bc_y0 - v_bc_y0_hat)**2 +
                         (v_bc_y1 - v_bc_y1_hat)**2)
    
    loss = 0.9*pde_loss + 0.01*ic_loss + 0.09*bc_loss
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    loss.backward()
    
    return loss
num_epochs = 10
for epoch in range (num_epochs):
    loss = optimizer.step(closure)
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_dir, exist_ok=True)

model_save_path = os.path.join(results_dir, "burger2d.pth")
torch.save(model.state_dict(), model_save_path)

print(f"Training complete. Model saved at: {model_save_path}")
    
    
    
    
