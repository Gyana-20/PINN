import torch

def residual_heat2d (u_pred, x, y, t, alpha):
    
    u_x = torch.autograd.grad(u_pred, x,
                              grad_outputs = torch.ones_like(u_pred),
                              create_graph= True,
                              retain_graph = True)[0]
    u_xx = torch.autograd.grad(u_x, x,
                              grad_outputs = torch.ones_like(u_x),
                              create_graph= True,
                              retain_graph = True)[0]
    u_y = torch.autograd.grad(u_pred, y,
                              grad_outputs = torch.ones_like(u_pred),
                              create_graph= True,
                              retain_graph = True)[0]
    u_yy = torch.autograd.grad(u_y, y,
                              grad_outputs = torch.ones_like(u_y),
                              create_graph= True,
                              retain_graph = True)[0]
    u_t = torch.autograd.grad(u_pred, t,
                              grad_outputs = torch.ones_like(u_pred),
                              create_graph= True,
                              retain_graph = True)[0]
    
    return u_t - alpha * (u_xx + u_yy)