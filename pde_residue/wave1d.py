import torch 

def pde_residue_wave (u_pred, x, t, c):
    
    u_t = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), 
                              create_graph=True, retain_graph=True)[0]
    
    u_tt = torch.autograd.grad(u_t, t, grad_outputs = torch.ones_like(u_t),
                               create_graph=True, retain_graph=True)[0]
    
    u_x = torch.autograd.grad (u_pred, x, grad_outputs = torch.ones_like(u_pred),
                               create_graph = True, retain_graph = True)[0]
    
    u_xx = torch.autograd.grad (u_x, x, grad_outputs= torch.ones_like(u_x), 
                                create_graph = True, retain_graph = True)[0]
    
    res = u_tt - c* u_xx
    
    return res