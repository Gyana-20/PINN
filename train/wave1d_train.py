import torch
import torch.optim as optim
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.PINN import PINN
from pde_residue.wave1d import pde_residue_wave

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    
# constant and domains
x0, x1 = 0, 1
t0, t1 = 0, 1
c = 0.01

# collocation points/ interior
x_int = (torch.rand(10000, 1)*(x1-x0)+x0).to(device)
t_int = (torch.rand(10000, 1)*(t1-t0)+t0).to(device)

x_int.requires_grad = True
t_int.requires_grad = True

# initial displacement points
x_ic = (torch.rand(100, 1)*(x1-x0)+x0).to(device)
t_ic = torch.zeros_like(x_ic).to(device)
t_ic.requires_grad = True

initial_dis = torch.exp(-10*(x_ic - 0.5)**2).to(device)
initial_vel = torch.cos(2*torch.pi*x_ic)

# boundary points
t_bc = (torch.rand(100,1)*(t1-t0)+t0).to(device)
x_left_bc = torch.zeros_like(t_bc).to(device)
x_right_bc = torch.ones_like(t_bc).to(device)
x_left_bc.requires_grad = True
x_right_bc.requires_grad = True

f1 = torch.zeros_like(t_bc).to(device)
f2 = torch.zeros_like(t_bc).to(device)

# define the model
input_dim = 2
output_dim = 1
hidden_dims = [32, 64, 64, 32]
model = PINN(input_dim, hidden_dims, output_dim)
model.to(device)

#optimizer
optimizer = optim.LBFGS(model.parameters(), lr=0.01,
                        max_iter=500, tolerance_change=1e-9, 
                        tolerance_grad= 1e-7)

def closure():
    optimizer.zero_grad()
    
    u_pred = model (x_int, t_int)
    
    initial_dis_pred = model(x_ic, t_ic)
    initial_vel_pred = torch.autograd.grad(initial_dis_pred, t_ic,
                                           grad_outputs = torch.ones_like(initial_dis_pred),
                                           create_graph= True, retain_graph= True)[0]
    
    u_left = model (x_left_bc, t_bc)
    u_right = model (x_right_bc, t_bc)
    
    u_left_x = torch.autograd.grad(u_left, x_left_bc,
                                   grad_outputs = torch.ones_like(u_left),
                                   create_graph= True, retain_graph= True)[0]

    u_right_x = torch.autograd.grad(u_right, x_right_bc,
                                   grad_outputs = torch.ones_like(u_right),
                                   create_graph= True, retain_graph= True)[0]
        
    pde_res = pde_residue_wave(u_pred, x_int, t_int, c)
    pde_loss = torch.mean(pde_res**2)
    
    displacement_loss = torch.mean((initial_dis - initial_dis_pred)**2)
    velocity_loss = torch.mean((initial_vel - initial_vel_pred)**2)
    
    boundary_loss = torch.mean((u_left_x-f1)**2 + (u_right_x - f2)**2)
    
    loss = pde_loss + displacement_loss + velocity_loss + boundary_loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    loss.backward()
    
    return loss

num_epochs = 10
for epoch in range (num_epochs):
    loss = optimizer.step(closure)
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_dir, exist_ok=True)

model_save_path = os.path.join(results_dir, "wave1d.pth")
torch.save(model.state_dict(), model_save_path)

print(f"Training complete. Model saved at: {model_save_path}")
    
    
    