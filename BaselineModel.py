# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:32:12 2024

@author: Hamza
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GATConv, GraphNorm, global_add_pool, global_mean_pool
from torch.nn import Embedding, GRU, Parameter, RNN

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, predict_graph=True, pool='add'):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.predict_graph = predict_graph
        if self.predict_graph:
            self.fc = torch.nn.Linear(hidden_dim, output_dim)
        else:
            self.fc = torch.nn.Linear(2 * hidden_dim, output_dim)
        self.pool = pool

    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
        # Compute the norms of each row
        norms = torch.norm(x, dim=1, keepdim=True)
        norms[norms == 0] = 1e-8
        # Normalize each row
        x = x.div(norms)
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.conv2(x, edge_index)
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


class EmbGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, total_traces, embedding_dim, output_dim, predict_graph=True, pool='add'):
        super(EmbGNN, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.conv1 = GATConv(input_dim+embedding_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim)
        self.predict_graph = predict_graph
        if self.predict_graph:
            self.fc = torch.nn.Linear(hidden_dim, output_dim)
        else:
            self.fc = torch.nn.Linear(2 * hidden_dim, output_dim)
        self.pool = pool

    def forward(self, data, batch):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        node_index = x[:,-1].long()
        emb_vecs = self.embedding(node_index)
        assert x.shape[0] == emb_vecs.shape[0]
        x = x[:,:-1]
        norms = torch.norm(x, dim=1, keepdim=True)
        norms[norms == 0] = 1e-8
        # Normalize each row
        x = x.div(norms)
        x = torch.cat((x, emb_vecs), axis=1)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.gelu(x)
        if self.predict_graph == False:
            #Start decoding
            row, col = edge_index
            x = torch.cat([x[row], x[col]], dim = -1)
        else:
            if self.pool == 'mean':
                x = global_mean_pool(x, batch)
            else:
                x = global_add_pool(x, batch)
        # Fully connected layer
        x = self.fc(x)
        x = F.leaky_relu(x, 0.01)
        return x

class EmbEdgeGNNGRU(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, total_traces, embedding_dim, output_dim, predict_graph=True):
        super(EmbEdgeGNNGRU, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.conv1 = GATConv(input_dim+embedding_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.gru_cell = GRUModel(hidden_dim, output_dim, output_dim)
        self.initial_hs = Parameter(torch.zeros(6, 1, output_dim), requires_grad=True)
        self.predict_graph = predict_graph
        self.output_dim = output_dim
        
    def forward(self, data, batch):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        node_index = x[:,-1].long()
        emb_vecs = self.embedding(node_index)
        assert x.shape[0] == emb_vecs.shape[0]
        x = torch.cat((x[:,:-1], emb_vecs), axis=1)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.gelu(x)
        
        batch_edge = batch[edge_index[0]]
        row, col = edge_index
        x = torch.cat([x[row], x[col]], dim = -1)
        
        # Fully connected layer
        x = self.fc(x)
        x = F.gelu(x)
        
        #Construct GRU Input
        gru_input, mask, max_edges = construct_tensor(x, batch_edge)
        # Initialize hidden states
        # hidden_state = self.initial_hs.expand(6, batch_edge.max().item() + 1, self.output_dim)
        predictions, hidden_state = self.gru_cell(gru_input)
        #predictions = F.gelu(predictions).div(predictions)

        # Apply mask to predictions
        masked_predictions = predictions.view(predictions.size(0), -1, self.output_dim) * mask.unsqueeze(2)
        
        if self.predict_graph:
            # Find the index of the last non-zero value in the max_nodes dimension for each sample
            last_non_zero_indices = torch.tensor([torch.nonzero(row).tolist()[-1][-1] if torch.sum(row) > 0 else 0 for row in mask])
    
            # Extract the corresponding prediction for each sample
            selected_predictions = masked_predictions[torch.arange(masked_predictions.size(0)), last_non_zero_indices]
        else:
            selected_predictions = masked_predictions[mask.bool()]
        return selected_predictions

class RNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=6, batch_first=True):
        super(RNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        k = 1 / hidden_size  # Calculate k

        # Define learnable parameters
        self.weight_ih_l = Parameter(torch.FloatTensor(num_layers, hidden_size, input_size).uniform_(-k, k))
        self.weight_hh_l = Parameter(torch.FloatTensor(num_layers, hidden_size, hidden_size).uniform_(-k, k))
        self.bias_ih_l = Parameter(torch.FloatTensor(num_layers, hidden_size).uniform_(-k, k))
        self.bias_hh_l = Parameter(torch.FloatTensor(num_layers, hidden_size).uniform_(-k, k))
        
    def forward(self, x, h_0=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        h_t_minus_1 = h_0
        h_t = h_0.clone()
        output = []
        for t in range(seq_len):
            for layer in range(self.num_layers):
                h_t[layer] = F.leaky_relu(
                x[t] @ self.weight_ih_l[layer].T
                + self.bias_ih_l[layer]
                + h_t_minus_1[layer] @ self.weight_hh_l[layer].T
                + self.bias_hh_l[layer], 0.01
            )
            output.append(h_t[-1])
            h_t_minus_1 = h_t.clone()
        output = torch.stack(output)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h_t

# Define the GRU cell
class VanillaGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VanillaGRU, self).__init__()
        self.hidden_size = hidden_size

        # Update gate
        self.Wz = torch.nn.Linear(input_size + hidden_size, hidden_size)
        
        # Reset gate
        self.Wr = torch.nn.Linear(input_size + hidden_size, hidden_size)
        
        # Candidate activation
        self.Wh = torch.nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 1)
        
        z = torch.sigmoid(self.Wz(combined))
        r = torch.sigmoid(self.Wr(combined))
        
        combined_reset = torch.cat((x, r * hidden), 1)
        h_tilde = F.leaky_relu(self.Wh(combined_reset), 0.01)
        
        hidden = (1 - z) * hidden + z * h_tilde
        return hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

# Define the complete GRU model
class GRUModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = VanillaGRU(input_size, hidden_size)
        #self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        hidden = self.gru.init_hidden(batch_size)
        outputs = torch.zeros(batch_size, seq_length, self.hidden_size)
        for t in range(seq_length):
            hidden = self.gru(x[:, t, :], hidden)
            outputs[:, t, :] = hidden
        #output = self.fc(hidden)
        return outputs, hidden

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


class EmbNodeGNNGRU(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, embedding_dim, output_dim, predict_graph = True):
        super(EmbNodeGNNGRU, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.conv1 = GraphConv(input_dim + embedding_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, input_dim)
        self.gru_cell = GRU(input_dim, output_dim, batch_first=True)
        self.initial_hs = Parameter(torch.zeros(1, 1), requires_grad=True)
        self.predict_graph = predict_graph
        if self.predict_graph == False:
            self.prediction_layer = torch.nn.Linear(2 * output_dim, output_dim)

    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
        node_index = x[:,-1].long()
        emb_vecs = self.embedding(node_index)
        assert x.shape[0] == emb_vecs.shape[0]
        x = torch.cat((x[:,:-1], emb_vecs), axis=1)
        # Compute the norms of each row
        norms = torch.norm(x, dim=1, keepdim=True)
        # Normalize each row
        x = x.div(norms)
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
        if self.predict_graph == False:
            x = masked_predictions[mask.bool()]
            #Start decoding
            row, col = edge_index
            x = torch.stack([x[row], x[col]], dim = 1)
            x = self.prediction_layer(x)
            x = F.gelu(x).squeeze(dim=1)
        else:
            x = masked_predictions[torch.arange(masked_predictions.size(0)), last_non_zero_indices]
        return x


'''
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
'''