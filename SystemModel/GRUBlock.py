# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:15:41 2024

@author: Hamza
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GraphNorm, global_add_pool, global_mean_pool
from torch.nn import Embedding, GRU, Parameter

class GRUBlock(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=5):
        super(GRUBlock, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.gru_block = GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        #self.initial_hs = Parameter(torch.zeros(hidden_dim), requires_grad=True)

    def forward(self, x, window_size):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        outputs = []
        for i in range(0, x.size(1)-window_size+1):
            # Extract the window of size 'window_size'
            window = x[:, i:i+window_size, :]
            
            # Forward pass through GRU
            out, _ = self.gru_block(window, h0)
            
            '''
            if i==0:
                # Append the outputs of GRU to the list
                outputs.append(out)
            else:
            '''
            outputs.append(out[:,-1,:].unsqueeze(dim=1))
        
        # Stack the outputs along the num_timesteps dimension
        stacked_outputs = torch.cat(outputs, dim=1)
        
        return stacked_outputs
        
    