# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 14:17:06 2024

@author: Hamza
"""

import torch
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from CustomDataset import CustomDataset, CustomDataLoader
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torchmetrics.functional.regression import explained_variance
from SystemGraphProcessor import system_graph_processor, recover_values
from Preprocess import recover_by_trace
from ModelTrainer import MAPE, percentile_mape, percentile_mae, plot
import sys
sys.path.append('./MMS')
from GNN import SystemGNN, TraceGNN, EdgeGNNGRU
from GRUBlock import GRUBlock
import numpy as np


class SystemModelTrainer():
    def __init__(self, data_dir, batch_size, features, predict_graph=True,\
                 normalize_features=[], normalize_by_node_features=[], \
                 scale_features=[], validate_on_trace=False):
        
        self.batch_size=batch_size
        self.features=features
        self.num_features=len(features)
        self.predict_graph=predict_graph
        self.normalize_features=normalize_features 
        self.normalize_by_node_features=normalize_by_node_features
        self.scale_features=scale_features
        self.validate_on_trace=validate_on_trace
        
        assert not(self.predict_graph and self.validate_on_trace)
        
        path = './A/microservice/test/'
        
        #Pass the directory that contains data as pickle files to the preprocessing function
        data, graphs, num_nodes, measures = system_graph_processor(path, features)

        dataset = CustomDataset(graphs)
    
        # Split the dataset
        train_size = int(0.8 * len(dataset))
        
        # Split the dataset into training and validation sets
        self.train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        self.val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
        #train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create DataLoaders for training and validation
        self.train_loader = CustomDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        self.val_loader = CustomDataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.data = data
        self.graphs = graphs
        self.num_nodes = num_nodes
        self.measures = measures
    
    def initialize_models(self, input_dim, gnn_hidden, gru_hidden, trace_gnn_hidden):
        self.gnn = SystemGNN(input_dim, gnn_hidden)
        self.gru_block = GRUBlock(gnn_hidden, gru_hidden)
        self.trace_gnn = EdgeGNNGRU(gru_hidden, trace_gnn_hidden, 1)
        
    
    def train(self, epochs, loss_fn, criterion, optimizer):
        train_loader = self.train_loader
        val_loader = self.val_loader
        # Training loop
        for epoch in range(1, epochs+1):
            self.set_to_train()
            total_loss = 0
            total_crit = 0
            for batch in train_loader:
                #optimizer.zero_grad()
                outputs, c_targets, trace_ints, loss, crit = self.step(batch, loss_fn, criterion, optimizer)
                #loss.backward()
                #optimizer.step()
                total_crit += crit.item()
                total_loss += loss.item()
                train_crit = total_crit/len(train_loader)
                train_loss = total_loss/len(train_loader)
                
            self.set_to_eval()
            total_val_loss = 0
            total_val_crit = 0
            with torch.no_grad():
                targets = torch.tensor([])
                predictions = torch.tensor([])
                trace_integers = torch.tensor([])
                for batch in val_loader:
                    outputs, c_targets, trace_ints, loss, crit = self.step(batch, loss_fn, criterion, optimizer)
                    total_val_loss += loss.item()
                    total_val_crit += crit.item()
                    val_crit = total_val_crit/len(val_loader)
                    val_loss = total_val_loss/len(val_loader)
                    targets = torch.cat([targets, c_targets], axis=0)
                    trace_integers = torch.cat([trace_integers, trace_ints], axis=0)
                    predictions = torch.cat([predictions, outputs], axis=0)
            
            targets = recover_by_trace(targets, trace_integers, self.measures)
            predictions = recover_by_trace(predictions, trace_integers, self.measures)
            #targets = torch.pow(10, targets)
            #predictions = torch.pow(10, predictions)
            mape = MAPE(predictions, targets)
            e_var = explained_variance(predictions, targets)
            print(f"Epoch: {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Train criterion: {train_crit:.4f}, Val Loss: {val_loss:.4f}, Val criterion: {val_crit:.4f}, Val MAPE: {mape:.4f}, Exp Var: {e_var:.4f}")
            p_mape = percentile_mape(targets, predictions)
            print(f"MAPE by percentiles: {', '.join(f'{tensor.item():.4f}' for tensor in p_mape.values())}")
            p_mae = percentile_mae(targets, predictions)
            print(f"MAE by percentiles: {', '.join(f'{tensor.item():.4f}' for tensor in p_mae.values())}")
            print("\n")
            if epoch == epochs: 
                plot(targets, predictions)
    
    def set_to_train(self):
        self.gnn.train(True)
        self.gru_block.train(True)
        self.trace_gnn.train(True)
        self.validate = False
    
    def set_to_eval(self):
        self.gnn.eval()
        self.gru_block.eval()
        self.trace_gnn.eval()
        self.validate = True
    
    def step(self, batch, loss_fn, criterion, optimizer):
        outputs = torch.tensor([])
        targets = torch.tensor([])
        trace_integers = torch.tensor([])
        '''
        for timestep in range(timesteps):
            graph_indices = (batch.batch == timestep).nonzero(as_tuple=True)[0]
            node_embeddings = system_embeddings[graph_indices]
        '''
        trace_graphs = batch.trace_graphs[self.batch_size-1]
        trace_data = CustomDataset(trace_graphs)
        trace_loader = DataLoader(trace_data, batch_size=128, shuffle=False)
        
        for trace_batch in trace_loader:
            if not self.validate: 
                optimizer.zero_grad()
            num_timesteps = batch.batch.max().item() + 1
            system_embeddings = self.gnn(batch, batch.batch)
            num_features = system_embeddings.size(1)
            system_embeddings = system_embeddings.view(num_timesteps, self.num_nodes, num_features)
            system_embeddings = system_embeddings.permute(1,0,2)
            
            system_embeddings = self.gru_block(system_embeddings, self.batch_size)
            system_embeddings = system_embeddings.permute(1, 0, 2).contiguous().view(-1, num_features)
            out = self.trace_gnn(trace_batch, trace_batch.batch, system_embeddings)
            outputs = torch.cat([outputs, out], axis=0)
            targets = torch.cat([targets, trace_batch.targets], axis=0)
            trace_integers = torch.cat([trace_integers, trace_batch.trace_integer], axis=0)
            loss = torch.sqrt(loss_fn(out, trace_batch.targets))
            if not self.validate:
                loss.backward()
                optimizer.step()
        batch_edge = batch.batch[batch.edge_index[0]]
        target_indices = (batch_edge == self.batch_size-1).nonzero(as_tuple=True)[0]
        loss = torch.sqrt(loss_fn(outputs, targets))
        crit = criterion(outputs, targets)
                
        return outputs, targets, trace_integers, loss, crit
    
if __name__ == "__main__":
    path = './A/microservice/test/'
    features = ['cpu_use', 'mem_use_percent', 'mem_use_amount', 'net_send_rate', 'net_receive_rate', 'file_read_rate']
    model_trainer = SystemModelTrainer(path, 10, features)
    model_trainer.initialize_models(6, 64, 64, 32)
    loss = torch.nn.MSELoss(reduction='mean')
    criterion = torch.nn.L1Loss(reduction='mean')
    model_parameters = list(model_trainer.gnn.parameters()) + list(model_trainer.gru_block.parameters()) + list(model_trainer.trace_gnn.parameters())
    optimizer = torch.optim.Adam(model_parameters, lr=0.0001)
    model_trainer.train(50, loss, criterion, optimizer)
    