# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:32:12 2024

@author: Hamza
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm, global_add_pool, global_mean_pool
from torch.nn import Embedding, GRU, Parameter


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

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim_two, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim_two)
        self.fc = torch.nn.Linear(2 * hidden_dim_two, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope = 0.1)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, negative_slope = 0.1)
        # Start decoding
        row, col = edge_index
        x = torch.cat([x[row], x[col]], dim = -1)
        # Fully connected layer
        x = self.fc(x)
        x = F.relu(x, negative_slope = 0.1)
        return x


class EmbeddingGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, embedding_dim, output_dim):
        super(EmbeddingGNN, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.conv1 = GCNConv(input_dim + embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(2 * hidden_dim, input_dim)

    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
        node_index = x[:,-1].long()
        emb_vecs = self.embedding(node_index)
        assert x.shape[0] == emb_vecs.shape[0]
        x = torch.cat((x[:,:-1], emb_vecs), axis=1)
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.conv2(x, edge_index)
        x = F.gelu(x)
        x = self.conv3(x, edge_index)
        x = F.gelu(x)
        #Start decoding
        row, col = edge_index
        x = torch.cat([x[row], x[col]], dim = -1)
        # Fully connected layer
        x = self.fc(x)
        x = F.gelu(x)
        return x

class EmbNodeGNNGRU(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, embedding_dim, output_dim):
        super(EmbNodeGNNGRU, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.conv1 = GCNConv(input_dim + embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, input_dim)
        self.gru_cell = GRU(input_dim, output_dim, batch_first=True)
        self.initial_hs = Parameter(torch.zeros(1, 1), requires_grad=True)

    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
        node_index = x[:,-1].long()
        emb_vecs = self.embedding(node_index)
        assert x.shape[0] == emb_vecs.shape[0]
        x = torch.cat((x[:,:-1], emb_vecs), axis=1)
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.conv2(x, edge_index)
        x = F.gelu(x)
        x = self.conv3(x, edge_index)
        x = F.gelu(x)
        x = self.fc(x)
        x = F.gelu(x)
        
        # Construct input tensor for GRU
        gru_input, mask, max_nodes = construct_tensor(x, batch)
        
        # Initialize hidden states
        hidden_state = self.initial_hs.expand(1, batch.max().item() + 1, 1)
        predictions, hidden_state = self.gru_cell(gru_input, hidden_state)
        
        # Apply mask to predictions
        masked_predictions = predictions.squeeze() * mask

        # Find the index of the last non-zero value in the max_nodes dimension for each sample
        last_non_zero_indices = torch.tensor([torch.nonzero(row).tolist()[-1][-1] if torch.sum(row) > 0 else 0 for row in masked_predictions])

        # Extract the corresponding prediction for each sample
        selected_predictions = masked_predictions[torch.arange(masked_predictions.size(0)), last_non_zero_indices]
        '''
        for node in range(batch.max() + 1):
            graph_indices = (batch == graph_id).nonzero(as_tuple=True)[0]
            node_embeddings = x[graph_indices]
            hs = hidden_state[graph_id]
            for step in range(node_embeddings.size(0)):
                hs = self.gru_cell(node_embeddings[step], hs)
            outputs[graph_id] = hs
        '''
        return selected_predictions

class EmbEdgeGNNGRU(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, embedding_dim, output_dim):
        super(EmbEdgeGNNGRU, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.conv1 = GCNConv(input_dim + embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, input_dim)
        self.gru_cell = GRU(input_dim, output_dim, batch_first=True)
        self.initial_hs = Parameter(torch.zeros(1, 1), requires_grad=True)

    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
        node_index = x[:,-1].long()
        emb_vecs = self.embedding(node_index)
        assert x.shape[0] == emb_vecs.shape[0]
        x = torch.cat((x[:,:-1], emb_vecs), axis=1)
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.conv2(x, edge_index)
        x = F.gelu(x)
        x = self.conv3(x, edge_index)
        x = F.gelu(x)
        x = self.fc(x)
        x = F.gelu(x)
        
        # Start decoding
        batch_edge = batch[edge_index[0]]
        row, col = edge_index
        #x = torch.cat([x[row], x[col]], dim = -1)
        x = (x[row] + x[col]) / 2
        #Construct GRU Input
        gru_input, mask, max_edges = construct_tensor(x, batch_edge)
        # Initialize hidden states
        hidden_state = self.initial_hs.expand(1, batch_edge.max().item() + 1, 1)
        predictions, hidden_state = self.gru_cell(gru_input, hidden_state)
        
        # Apply mask to predictions
        masked_predictions = predictions.squeeze() * mask

        # Find the index of the last non-zero value in the max_nodes dimension for each sample
        last_non_zero_indices = torch.tensor([torch.nonzero(row).tolist()[-1][-1] if torch.sum(row) > 0 else 0 for row in masked_predictions])

        # Extract the corresponding prediction for each sample
        selected_predictions = masked_predictions[torch.arange(masked_predictions.size(0)), last_non_zero_indices]
        masked_predictions = masked_predictions[mask.bool()]
        return selected_predictions.squeeze(), masked_predictions.squeeze()

class GRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Update gate weights
        self.weight_ih = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        # Reset gate weights
        self.weight_ir = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hr = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        # New gate weights
        self.weight_in = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hn = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights
        torch.nn.init.xavier_uniform_(self.weight_ih)
        torch.nn.init.xavier_uniform_(self.weight_hh)
        torch.nn.init.xavier_uniform_(self.weight_ir)
        torch.nn.init.xavier_uniform_(self.weight_hr)
        torch.nn.init.xavier_uniform_(self.weight_in)
        torch.nn.init.xavier_uniform_(self.weight_hn)

    def forward(self, input, hx):
        # Transpose input if batch_first=True
        if input.dim() == 3:
            input = input.transpose(0, 1)

        # Calculate gates for each time step
        outputs = []
        for i in range(input.size(0)):
            input_gate = torch.sigmoid(torch.mm(self.weight_ih, input[i]) + torch.mm(self.weight_hh, hx))
            reset_gate = torch.sigmoid(torch.mm(self.weight_ir, input[i]) + torch.mm(self.weight_hr, hx))
            new_gate = torch.tanh(torch.mm(self.weight_in, input[i]) + torch.mm(self.weight_hn, reset_gate * hx))
            hx = (1 - input_gate) * hx + input_gate * new_gate
            outputs.append(hx)

        # Concatenate outputs along time dimension
        output = torch.stack(outputs)

        # Transpose output if batch_first=True
        if input.dim() == 3:
            output = output.transpose(0, 1)

        return output, hx
