import torch
import torch.optim as optim
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.PINN import PINN
from pde_residue.heat2d import residual_heat2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    print(f"Using device: {torch.cuda.get_device_name(0)}")

# alpha and domain
alpha = 0.01
x0, x1 = 0, 1
y0, y1 = 0, 1
t0, t1 = 0, 1

# interior points
x_int = (x0 + (x1-x0)*(torch.rand(10000, 1))).to(device)
y_int = (y0 + (y1-y0)*(torch.rand(10000, 1))).to(device)
t_int = (t0 + (t1-t0)*(torch.rand(10000, 1))).to(device)

x_int.requires_grad = True
y_int.requires_grad = True
t_int.requires_grad = True

# initial points
x_ic = (torch.rand(100, 1)*(x1-x0)+x0).to(device)
y_ic = (torch.rand(100, 1)*(y1-y0)+y0).to(device)
t_ic = torch.zeros_like(x_ic).to(device)

u_initial = torch.exp(-10*((x_ic - 0.5)**2 + (y_ic-0.5)**2))

# boundary points
t_bc = (t0 + (t1-t0)*(torch.rand(10000, 1))).to(device)
x_bc_l = torch.zeros_like (t_bc).to(device)
x_bc_r = torch.ones_like (t_bc).to(device)
y_bc_l = torch.zeros_like (t_bc).to(device)
y_bc_r = torch.ones_like (t_bc).to(device)

x_bc_l.requires_grad = True
x_bc_r.requires_grad = True
y_bc_l.requires_grad = True
y_bc_r.requires_grad = True


f1 = torch.zeros_like(t_bc)
f2 = torch.zeros_like(t_bc)
g1 = torch.zeros_like(t_bc)
g2 = torch.zeros_like(t_bc)

# model
input_dim = 3
hidden_dims = [32, 64, 64, 32]
output_dim = 1

model = PINN(input_dim, hidden_dims, output_dim).to(device)

# optimization

optimizer = optim.LBFGS(model.parameters(), lr = 0.01,
                        max_iter = 500,tolerance_grad=1e-7,
                        tolerance_change= 1e-9)
def closure():
    optimizer.zero_grad()
    
    u_int = model(x_int, y_int, t_int)
    
    u_ic = model(x_ic, y_ic, t_ic)
    
    u_bc_l = model (x_bc_l, y_bc_l, t_bc)
    u_bc_r = model (x_bc_r, y_bc_r, t_bc)
    
    u_bc_l_x = torch.autograd.grad(u_bc_l, x_bc_l, torch.ones_like(u_bc_l),
                                   create_graph= True, retain_graph=True)[0]
    u_bc_r_x = torch.autograd.grad(u_bc_r, x_bc_r, torch.ones_like(u_bc_r),
                                   create_graph= True, retain_graph=True)[0]
    
    u_bc_l_y = torch.autograd.grad(u_bc_l, y_bc_l, torch.ones_like(u_bc_l),
                                   create_graph= True, retain_graph=True)[0]
    u_bc_r_y = torch.autograd.grad(u_bc_r, y_bc_r, torch.ones_like(u_bc_r),
                                   create_graph= True, retain_graph=True)[0]
    
    res = residual_heat2d(u_int, x_int, y_int, t_int, alpha)
    loss1 = torch.mean(res**2)
    
    loss2 = torch.mean((u_initial - u_ic)**2)
    loss3 = torch.mean ((u_bc_l_x - f1)**2 + (u_bc_r_x - f2)**2+
                         (u_bc_l_y - g1 )**2 + (u_bc_r_y - g2)**2)
    
    loss = loss1 + loss2 + loss3
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    loss.backward()
    
    return loss

num_epochs = 10
for epoch in range (num_epochs):
    loss = optimizer.step(closure)
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_dir, exist_ok=True)

model_save_path = os.path.join(results_dir, "heat2d.pth")
torch.save(model.state_dict(), model_save_path)

print(f"Training complete. Model saved at: {model_save_path}")
    
    
    
    




  