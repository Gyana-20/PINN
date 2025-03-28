import torch

def pde_residual (u_pred, x, t, alpha):
    
    u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), 
                              create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True, retain_graph=True)[0]
    u_t = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred),
                              create_graph=True, retain_graph=True)[0]
    
    residue = u_t - alpha * u_xx

    return residue