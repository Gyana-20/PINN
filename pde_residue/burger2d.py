import torch

def residual_burger_2d (u_pred, v_pred, 
                        x, y, t,
                        nu = 1/3000):
    u_pred_t = torch.autograd.grad(
        u_pred, t, grad_outputs = torch.ones_like(u_pred),
        create_graph = True, 
        retain_graph = True
    )[0]
    
    
    u_pred_x = torch.autograd.grad(
        u_pred, x, grad_outputs = torch.ones_like(u_pred),
        create_graph = True, 
        retain_graph = True
    )[0]
    
    u_pred_xx = torch.autograd.grad(
        u_pred_x, x, grad_outputs = torch.ones_like(u_pred_x),
        create_graph = True, 
        retain_graph = True
    )[0]
    
    u_pred_y = torch.autograd.grad(
        u_pred, y, grad_outputs = torch.ones_like(u_pred),
        create_graph = True, 
        retain_graph = True
    )[0]
    
    u_pred_yy = torch.autograd.grad(
        u_pred_y, y, grad_outputs = torch.ones_like(u_pred_y),
        create_graph = True, 
        retain_graph = True
    )[0]
    
    v_pred_t = torch.autograd.grad(
        v_pred, t, grad_outputs = torch.ones_like(v_pred),
        create_graph = True, 
        retain_graph = True
    )[0]
    
    v_pred_x = torch.autograd.grad(
        v_pred, x, grad_outputs = torch.ones_like(v_pred),
        create_graph = True, 
        retain_graph = True
    )[0]
    
    v_pred_xx = torch.autograd.grad(
        v_pred_x, x, grad_outputs = torch.ones_like(v_pred_x),
        create_graph = True, 
        retain_graph = True
    )[0]
    
    v_pred_y = torch.autograd.grad(
        v_pred, y, grad_outputs = torch.ones_like(v_pred),
        create_graph = True, 
        retain_graph = True
    )[0]
    
    v_pred_yy = torch.autograd.grad(
        v_pred_y, y, grad_outputs = torch.ones_like(v_pred_y),
        create_graph = True, 
        retain_graph = True
    )[0]
    
    residue1 =  (u_pred_t + u_pred * u_pred_x + v_pred * u_pred_y) - nu*(u_pred_xx + u_pred_yy)
    residue2 =  (v_pred_t + u_pred * v_pred_x + v_pred * v_pred_y) - nu*(v_pred_xx + v_pred_yy)
    
    return residue1 , residue2
    
    