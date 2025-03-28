import torch
import torch.nn as nn
import torch.nn.functional as F

class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(PINN, self).__init__()
        
        layers = []
        
        # add the input -> hidden[0] layer and activation function
        
        layers.append (nn.Linear(input_dim, hidden_dims[0]))
        nn.init.xavier_uniform_(layers[-1].weight)
        nn.init.constant_(layers[-1].bias, 0.1)
        layers.append(nn.Tanh())        
        
        # add rest of the hidden layers and activation functions
        
        for i in range (len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            nn.init.xavier_uniform_(layers[-1].weight)
            nn.init.constant_(layers[-1].bias, 0.1)
            layers.append(nn.Tanh())
        
        # add the output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        nn.init.xavier_uniform_(layers[-1].weight)
        nn.init.constant_(layers[-1].bias, 0.1)
    
        self.model = nn.Sequential(*layers)
    
    def forward (self, *args):
        args = [arg.view(-1, 1) if arg.dim() == 1 else arg for arg in args]
        inputs = torch.cat(args, dim=1)
        return self.model(inputs)
