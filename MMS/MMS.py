# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 18:49:03 2024

@author: Hamza
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GraphNorm, global_add_pool, global_mean_pool
from torch.nn import Embedding, GRU, Parameter
from GNN import SystemGNN
from GRUBlock import GRUBlock

class MMS(torch.nn.Module):
    def __init__(self, input_dim, gnn_hidden, gru_hidden):
        super(MMS, self).__init__()
        self.gnn = SystemGNN(input_dim, gnn_hidden)
        self.gru_block = GRUBlock(gnn_hidden, gru_hidden)

    def forward(self, x):
        # Initialize hidden states
        predictions, hidden_state = self.gru_block(x, self.initial_hs)
        return predictions