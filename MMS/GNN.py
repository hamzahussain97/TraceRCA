# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:13:45 2024

@author: Hamza
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GraphNorm, GATConv, global_add_pool, global_mean_pool
from torch.nn import Embedding, GRU, Parameter

class SystemGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SystemGNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.norm = GraphNorm(hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.norm1 = GraphNorm(hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim)
        self.norm2 = GraphNorm(hidden_dim)

    def forward(self, data, batch):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # Compute the norms of each row
        norms = torch.norm(x, dim=1, keepdim=True)
        norms[norms == 0] = 1e-8
        # Normalize each row
        x = x.div(norms)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.gelu(x)
        #x = self.norm(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.gelu(x)
        #x = self.norm1(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.gelu(x)
        return x

class TraceGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, predict_graph=False, pool='add'):
        super(TraceGNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.predict_graph = predict_graph
        if self.predict_graph:
            self.fc = torch.nn.Linear(hidden_dim, output_dim)
        else:
            self.fc = torch.nn.Linear(2 * hidden_dim, output_dim)
        self.pool = pool

    def forward(self, data, batch, node_embeddings):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = node_embeddings[x]
        x = self.conv1(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.gelu(x)
        if self.predict_graph == False:
            #Start decoding
            row, col = edge_index
            x = torch.cat([x[row], x[col]], dim = -1)
            # Fully connected layer
        else:
            if self.pool == 'mean':
                x = global_mean_pool(x, batch)
            else:
                x = global_add_pool(x, batch)
        # Fully connected layer
        x = self.fc(x)
        x = F.gelu(x)
        
        return x.squeeze(dim=1)
    
class EdgeGNNGRU(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, predict_graph=True):
        super(EdgeGNNGRU, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        #self.fc = torch.nn.Linear(hidden_dim)
        self.gru_cell = GRU(hidden_dim, output_dim, batch_first=True)
        self.initial_hs = Parameter(torch.zeros(1, 1), requires_grad=True)
        self.predict_graph = predict_graph

    def forward(self, data, batch, node_embeddings):
        x, edge_index = data.x, data.edge_index
        x = node_embeddings[x]
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.conv2(x, edge_index)
        x = F.gelu(x)
        #x = self.fc(x)
        #x = F.gelu(x)
        
        # Start decoding
        batch_edge = batch[edge_index[0]]
        row, col = edge_index
        x = (x[row] + x[col])
        #Construct GRU Input
        gru_input, mask, max_edges = construct_tensor(x, batch_edge)
        # Initialize hidden states
        hidden_state = self.initial_hs.expand(1, batch_edge.max().item() + 1, 1)
        predictions, hidden_state = self.gru_cell(gru_input, hidden_state)
        # Apply mask to predictions
        masked_predictions = predictions.squeeze(dim=2) * mask
        
        if self.predict_graph:
            # Find the index of the last non-zero value in the max_nodes dimension for each sample
            last_non_zero_indices = torch.tensor([torch.nonzero(row).tolist()[-1][-1] if torch.sum(row) > 0 else 0 for row in masked_predictions])
    
            # Extract the corresponding prediction for each sample
            selected_predictions = masked_predictions[torch.arange(masked_predictions.size(0)), last_non_zero_indices]
        else:
            selected_predictions = masked_predictions[mask.bool()]
        return selected_predictions
    
def construct_tensor(x, batch):
    # Find unique values (graphs) and their counts
    unique_values, counts = torch.unique(batch, return_counts=True)
    batch_size = counts.size(0)
    max_nodes = counts.max().item()

    # Initialize the output tensor with zeros
    output = torch.zeros(batch_size, max_nodes, x.size(1))
    mask = torch.zeros(batch_size, max_nodes, dtype=torch.int)

    # Iterate over unique values (graphs)
    for i, graph_id in enumerate(unique_values):
        # Get the indices of nodes belonging to the current graph
        graph_indices = (batch == graph_id).nonzero(as_tuple=True)[0]

        # Pad the graph's nodes with zeros to match the maximum count
        padded_nodes = x[graph_indices]
        padding = max_nodes - len(graph_indices)
        if padding != 0:
            padded_nodes = torch.cat([x[graph_indices], torch.zeros(max_nodes - len(graph_indices), x.size(1))])
        
        # Compute the mask tensor based on the condition
        mask[i, :len(graph_indices)] = 1
        # Assign the padded nodes to the output tensor
        output[i] = padded_nodes

    return output, mask, max_nodes