# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:21:41 2024

@author: Hamza
"""

import torch
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys
sys.path.append('./Alibaba')
sys.path.append('./MicroSS')
from CustomDataset import CustomDataset
from Preprocess import preprocess, recover, recover_by_node, recover_value, recover_by_trace
from process_alibaba import process_alibaba
from GraphConstructor import process_micross
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from BaselineModel import GNN
from torchmetrics.functional.regression import explained_variance
from SystemGraphProcessor import system_graph_processor
import numpy as np
from torchmetrics import MeanAbsolutePercentageError


class ModelTrainer():
    def __init__(self, path, batch_size, quantiles=[], predict_graph=True, one_hot_enc=False, \
                 normalize_features=[], normalize_by_node_features=[], \
                 scale_features=[], validate_on_trace=False):
        
        self.batch_size=batch_size
        self.quantiles=quantiles
        self.predict_graph=predict_graph
        self.one_hot_enc=one_hot_enc
        self.normalize_features=normalize_features 
        self.normalize_by_node_features=normalize_by_node_features
        self.scale_features=scale_features
        self.validate_on_trace=validate_on_trace
        
        assert not(self.predict_graph and self.validate_on_trace)
        
        self.path = path
        if 'TrainTicket' in  path:
            #Pass the directory that contains data as pickle files to the preprocessing function
            data, graphs, global_map, measures = preprocess(path,\
                                                            self.one_hot_enc,\
                                                            self.normalize_features,\
                                                            self.normalize_by_node_features,\
                                                            self.scale_features)
        elif 'Alibaba' in path:
            #Pass the directory containing the data folder that has pt files
            data, graphs, global_map, measures = process_alibaba(path)
        else:
            data, graphs, global_map, measures = process_micross(path)
        
        if 'latency' in measures: measures = measures['latency']
        dataset = CustomDataset(graphs)
    
        # Split the dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create DataLoaders for training and validation
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        self.data = data
        self.measures = measures
        self.global_map  = global_map
        self.graphs = graphs
        
    def set_model(self, model):
        self.model = model
    
    def train(self, epochs, loss_fn, criterion, optimizer):
        train_loader = self.train_loader
        val_loader = self.val_loader
        # Training loop
        for epoch in range(1, epochs+1):
            self.model.train(True)
            total_loss = 0
            total_crit = 0
            for batch in train_loader:
                optimizer.zero_grad()
                recovered, recov_pred, loss, crit = self.step(batch, loss_fn, criterion)
                loss.backward()
                optimizer.step()
                total_crit += crit.item() 
                total_loss += loss.item()
                train_crit = total_crit/len(train_loader)
                train_loss = total_loss/len(train_loader)
                
            self.model.eval()
            total_val_loss = 0
            total_val_crit = 0
            with torch.no_grad():
                target = torch.tensor([])
                predictions = torch.tensor([])
                for batch in val_loader:
                    recovered, recov_pred, loss, crit = self.step(batch, loss_fn, criterion)
                    total_val_loss += loss.item()
                    total_val_crit += crit.item()
                    val_crit = total_val_crit/len(val_loader)
                    val_loss = total_val_loss/len(val_loader)
                    target = torch.cat([target, recovered], axis=0)
                    predictions = torch.cat([predictions, recov_pred], axis=0)
            cov_prob = coverage_probability(target, predictions)
            print(f"Epoch: {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Train criterion: {train_crit:.4f}, Val Loss: {val_loss:.4f}, Val criterion: {val_crit:.4f}, Val Cov Prob: {cov_prob:.4f}")
            calculate_metrics(self.quantiles, target, predictions, epoch, epochs)
            if epoch == epochs:
                plot_percentiles(target, predictions, self.quantiles)
            print("\n")
        return self.model
    
    def validate(self, loss_fn, criterion):
        self.model.eval()
        total_val_loss = 0
        total_val_crit = 0
        with torch.no_grad():
            target = torch.tensor([])
            predictions = torch.tensor([])
            for batch in self.val_loader:
                recovered, recov_pred, loss, crit = self.step(batch, loss_fn, criterion)
                total_val_loss += loss.item()
                total_val_crit += crit.item()
                val_crit = total_val_crit/len(self.val_loader)
                val_loss = total_val_loss/len(self.val_loader)
                target = torch.cat([target, recovered], axis=0)
                predictions = torch.cat([predictions, recov_pred], axis=0)
        return target, predictions
        
    
    def step(self, batch, loss_fn, criterion):
        recov_pred = self.model(batch, batch.batch)
        if self.predict_graph:
            recovered = batch.trace_lat
        else:
            recovered = batch.y
        #target = torch.stack([1 - recovered, recovered], dim=1)
        #prediction = torch.stack([1 - recov_pred, recov_pred], dim=1)
        target = torch.stack([recovered for _ in range(len(self.quantiles))], dim=1)
        loss = loss_fn(recov_pred, target, self.quantiles)
        if self.validate_on_trace:
            edge_index = batch.edge_index
            batch_nodes = batch.batch
            batch_edge = batch_nodes[edge_index[0]]
            recovered, recov_pred = self.extract_trace_lat(recovered, recov_pred, batch_edge)
        #recovered, recov_pred = self.recover_predictions(recovered, recov_pred, node_names, trace_integers)
        #recovered = batch.original
        index = self.quantiles.index(0.5)
        crit = criterion(recov_pred[:,index], recovered)
        return recovered, recov_pred, loss, crit
    
    def extract_trace_lat(self, recovered, recov_pred, batch):
        last_indices = torch.bincount(batch)
        last_indices = torch.cumsum(last_indices, dim=0) - 1
        recovered = recovered[last_indices]
        recov_pred = recov_pred[last_indices]
        return recovered, recov_pred
    
    def recover_predictions(self, recovered, recov_pred, node_names, trace_integers):
        if 'latency' in self.normalize_by_node_features:
            recovered = recover_by_node(recovered, node_names, self.measures['norm_by_node'])
            recov_pred = recover_by_node(recov_pred, node_names, self.measures['norm_by_node'])
        if 'latency' in self.normalize_features:
            recovered = recover(recovered, self.measures['norm'][0], self.measures['norm'][1])
            recov_pred = recover(recov_pred, self.measures['norm'][0], self.measures['norm'][1])
        if 'latency' in self.scale_features:
            recovered = recover_value(recovered, self.measures['scale'][0], self.measures['scale'][1])
            recov_pred = recover_value(recov_pred, self.measures['scale'][0], self.measures['scale'][1])
        else:
            recovered = recover_by_trace(recovered, trace_integers, node_names, self.measures['norm_by_trace'], True)
            recov_pred = recover_by_trace(recov_pred, trace_integers, node_names, self.measures['norm_by_trace'])
        return recovered, recov_pred
    
    def predict(self, graph_idx):
        graph = self.graphs[graph_idx]
        if self.predict_graph:
            recovered = [graph.trace_lat]
        else:
            recovered = graph.y
            
        with torch.no_grad():
            recov_pred = self.model(graph, torch.zeros(graph.x.size(0), dtype=torch.int64))
            
        #recovered, recov_pred = self.recover_predictions(recovered, recov_pred, node_names, trace_integers)
        
        print("*************Prediction*************")
        print(recov_pred)
        print("***************Actual***************")
        print(recovered)
        
        return recov_pred
    
def calculate_metrics(quantiles, target, predictions, epoch, epochs):
    # Store the original values
    o_predictions = predictions
    o_target = target
    
    # Update predictions and target with 10 ** predictions and 10 ** target
    #predictions = 10 ** predictions
    #target = 10 ** target
    for i, quantile in enumerate(quantiles):
        mape = MAPE(predictions[:,i], target)
        e_var = explained_variance(predictions[:,i], target)
        qloss = quantile_loss(predictions[:,i], target, quantile)
        print('***************************************************************')
        print(f"Quantile: {quantile}, Quantile Loss: {qloss}")
        print(f"Val MAPE: {mape:.4f}, Exp Var: {e_var:.4f}")
        p_qloss = percentile_quantile_loss(predictions[:,i], target, quantile)
        print(f"Quantile Loss by percentiles: {', '.join(f'{tensor.item():.4f}' for tensor in p_qloss.values())}")
        p_mape = percentile_mape(target, predictions[:,i])
        print(f"MAPE by percentiles: {', '.join(f'{tensor.item():.4f}' for tensor in p_mape.values())}")
        p_mae = percentile_mae(target, predictions[:,i])
        print(f"MAE by percentiles: {', '.join(f'{tensor.item():.4f}' for tensor in p_mae.values())}")
        if epoch == epochs: 
            plot(target, predictions[:,i], quantile)
    if epoch == epochs:
        p = percentiles(target, predictions)
        p_values = p[100]['p']
        print("\n")
        print("Percentile Values In Validation Data")
        print(p_values)

def percentile_mape(target, predictions):
    p = percentiles(target,predictions, log_scale=True)
    
    m_25 = MAPE(torch.tensor(p[25]['y']),torch.tensor(p[25]['x']))
    m_50 = MAPE(torch.tensor(p[50]['y']),torch.tensor(p[50]['x']))
    m_75 = MAPE(torch.tensor(p[75]['y']),torch.tensor(p[75]['x']))
    m_90 = MAPE(torch.tensor(p[90]['y']),torch.tensor(p[90]['x']))
    m_100 = MAPE(torch.tensor(p[100]['y']),torch.tensor(p[100]['x']))
    
    return {25: m_25, 50: m_50, 75: m_75, 90: m_90, 100:m_100}

def percentile_mae(target, predictions):
    p = percentiles(target,predictions)
    
    m_25 = MAE(torch.tensor(p[25]['y']),torch.tensor(p[25]['x']))
    m_50 = MAE(torch.tensor(p[50]['y']),torch.tensor(p[50]['x']))
    m_75 = MAE(torch.tensor(p[75]['y']),torch.tensor(p[75]['x']))
    m_90 = MAE(torch.tensor(p[90]['y']),torch.tensor(p[90]['x']))
    m_100 = MAE(torch.tensor(p[100]['y']),torch.tensor(p[100]['x']))
    
    return {25: m_25, 50: m_50, 75: m_75, 90: m_90, 100:m_100}

def percentile_quantile_loss(predictions, target, t_value):
    p = percentiles(target, predictions, log_scale=True)
    m_25 = quantile_loss(torch.tensor(p[25]['y']),torch.tensor(p[25]['x']),t_value)
    m_50 = quantile_loss(torch.tensor(p[50]['y']),torch.tensor(p[50]['x']),t_value)
    m_75 = quantile_loss(torch.tensor(p[75]['y']),torch.tensor(p[75]['x']),t_value)
    m_90 = quantile_loss(torch.tensor(p[90]['y']),torch.tensor(p[90]['x']),t_value)
    m_100 = quantile_loss(torch.tensor(p[100]['y']),torch.tensor(p[100]['x']),t_value)
    
    return {25: m_25, 50: m_50, 75: m_75, 90: m_90, 100:m_100}

def coverage_probability(targets, predictions):
    # Check if the target falls within the predicted range for each percentile
    covered = [predictions[j, 0] <= targets[j] <= predictions[j, -1] for j in range(len(targets))]
    
    # Calculate coverage probability
    coverage_prob = sum(covered) / len(covered)
    
    # Return coverage probabilities
    return coverage_prob

def percentiles(x,y, log_scale=False):
    # Store the original values
    o_x = x
    o_y = y
    
    # Update predictions and target with 10 ** predictions and 10 ** target
    x = 10 ** x
    y = 10 ** y
    
    x = x.numpy()
    y = y.numpy()

    percentile_10 = np.percentile(x, 10)
    percentile_25 = np.percentile(x, 25)
    percentile_50 = np.percentile(x, 50)
    percentile_75 = np.percentile(x, 75)
    percentile_90 = np.percentile(x, 90)
    percentile_95 = np.percentile(x, 95)
    
    p_values = [percentile_10, percentile_25, percentile_50, percentile_75, percentile_90, percentile_95]
    '''
    index_25 = np.where((x <= percentile_25))[0]
    index_50 = np.where((x > percentile_25) & (x <= percentile_50))[0]
    index_75 = np.where((x > percentile_50) & (x <= percentile_75))[0]
    index_90 = np.where((x > percentile_75))[0]
    #index_100 = np.where((x <= 200000))[0]
    '''
    
    index_25 = np.where((x <= 50))[0]
    index_50 = np.where((x > 50) & (x <= 100))[0]
    index_75 = np.where((x > 100) & (x <= 1000))[0]
    index_90 = np.where((x > 10000))[0]
    #index_100 = np.where((x <= 200000))[0]
    
    '''
    index_25 = np.where((x <= 50))[0]
    index_50 = np.where((x > 50) & (x <= 100))[0]
    index_75 = np.where((x <= 200))[0]
    index_90 = np.where((x <= 1000))[0]
    index_100 = np.where((x > 1000))[0]
    '''
    
    if log_scale:
        x_25 = o_x[index_25]
        y_25 = o_y[index_25]
        
        x_50 = o_x[index_50].flatten()
        y_50 = o_y[index_50].flatten()
        
        x_75 = o_x[index_75].flatten()
        y_75 = o_y[index_75].flatten()
        
        x_90 = o_x[index_90].flatten()
        y_90 = o_y[index_90].flatten()
        
        p_100 = {'x': o_x, 'y': o_y, 'p': p_values}
    else:
        x_25 = x[index_25]
        y_25 = y[index_25]
        
        x_50 = x[index_50].flatten()
        y_50 = y[index_50].flatten()
        
        x_75 = x[index_75].flatten()
        y_75 = y[index_75].flatten()
        
        x_90 = x[index_90].flatten()
        y_90 = y[index_90].flatten()
        
        p_100 = {'x': x, 'y': y, 'p': p_values}
        
    percentiles = {}
    # Slice values based on percentiles
    p_25 = {'x': x_25, 'y': y_25, 'p': p_values}
    percentiles[25] = p_25
    
    p_50 = {'x': x_50, 'y': y_50, 'p': p_values}
    percentiles[50] = p_50
    
    p_75 = {'x': x_75, 'y': y_75, 'p': p_values}
    percentiles[75] = p_75
    
    p_90 = {'x': x_90, 'y': y_90, 'p': p_values}
    percentiles[90] = p_90
    
    percentiles[100] = p_100

    return percentiles

def plot(x, y, quantile):
    p = percentiles(x, y, log_scale=True)
    plot_figure(1, p, 25, quantile)
    plot_figure(2, p, 50, quantile)
    plot_figure(3, p, 75, quantile)
    plot_figure(4, p, 90, quantile)
    plot_figure(5, p, 100, quantile)
    
def plot_figure(i, p, u_l, quantile):
    quantile = quantile * 100
    plt.figure(i)
    plt.scatter(p[u_l]['x'],p[u_l]['y'])
    
    if len(p[u_l]['x']) > 0:
        max_val = max(max(p[u_l]['x']), max(p[u_l]['y']))
        plt.grid(True)
        plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='y=x')
        plt.title(f'Predicted vs Target values for {quantile}th quantile')
        plt.show()
        if u_l == 100:
            plt.figure(i+1, figsize=(10, 6))  # Adjust the figure size as needed
            plt.hist(p[u_l]['x'], bins=30, color='skyblue', edgecolor='black')
            plt.title('Distribution of Latencies')
            plt.xlabel('Latency')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()
            
            plt.figure(i+2, figsize=(10, 6))  # Adjust the figure size as needed
            plt.hist(p[u_l]['y'], bins=30, color='skyblue', edgecolor='black')
            plt.title('Distribution of Predicted {quantile}th quantile')
            plt.xlabel('Latency')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()
        
def plot_percentiles(targets, predictions, quantiles):
    # Generate random colors using a color map
    colors = cm.viridis(np.linspace(0, 1, len(quantiles)))
    #colors = ['r', 'g', 'b']  # Colors for percentiles (10th, 50th, 90th)
    
    percentiles = list(zip(*predictions))
    #min_val = min(min(targets), min(predictions[:,2]))
    #max_val = max(max(targets), max(predictions[:,2]))
    min_val = min(targets)
    max_val = max(targets)
    plt.figure(figsize=(10, 6))
    # Scatter plot for each percentile
    sorted_indexes = np.argsort(targets)
    targets = np.array(targets)[sorted_indexes]
    for i, percentile_values in enumerate(percentiles):
        percentile_values = np.array(percentile_values)[sorted_indexes]
        plt.plot(targets, percentile_values, c=colors[i], label=f'{quantiles[i]}th Percentile')
    
    # Add labels and legend
    plt.grid(True)
    plt.xlabel('Target Latency')
    plt.ylabel('Prediction')
    plt.title('Plot of Predicted Percentiles against Target Latency')
    plt.legend(fontsize='xx-small', loc='upper right')
    
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y=x')
    # Show plot
    plt.show()
    
    plt.figure(figsize=(10, 6))
    index = quantiles.index(0.5)
    mean_pred = predictions[:,index]
    
    residuals = targets - np.array(mean_pred)[sorted_indexes]
    # Plot residual vs. fitted
    plt.scatter(mean_pred, residuals, color='blue')
    plt.axhline(y=0, color='black', linestyle='--')  # Add a horizontal line at y=0 for reference
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Fitted Values')
    plt.grid(True)
    plt.show()
    

def MAPE(output, target):
    #error = target - output
    #abs_error = error.abs()
    #p_error = abs_error / target
    #mape = torch.mean(p_error)
    mean_abs_percentage_error = MeanAbsolutePercentageError()
    mape = mean_abs_percentage_error(output, target)
    return mape

def multi_quantile_loss(preds, target, quantiles):
    assert isinstance(preds, torch.Tensor), "Predictions must be a torch.Tensor"
    assert isinstance(target, torch.Tensor), "Target must be a torch.Tensor"
    assert isinstance(quantiles, (list, torch.Tensor)), "Quantiles must be a list or torch.Tensor"
    assert len(preds.shape) == 2, "Predictions must have 2 dimensions (batch_size, num_quantiles)"
    assert preds.shape[1] == len(quantiles), f"Number of predictions ({preds.shape[1]}) must match the number of quantiles ({len(quantiles)})"
    assert preds.shape == target.shape, "Shape of predictions must match shape of target"

    if isinstance(quantiles, list):
        assert all(0 < q < 1 for q in quantiles), "Quantiles should be in (0, 1) range"
    else:
        assert torch.all((0 < quantiles) & (quantiles < 1)), "Quantiles should be in (0, 1) range"

    # Convert quantiles to a tensor if it's a list
    if isinstance(quantiles, list):
        quantiles_tensor = torch.tensor(quantiles, device=preds.device).view(1, -1)
    else:
        quantiles_tensor = quantiles.view(1, -1)

    # Calculate errors
    errors = (target - preds)
    squared_errors = errors ** 2
    
    #losses = torch.where(errors < 0, squared_errors * (1 - quantiles_tensor), squared_errors * quantiles_tensor)
    # Calculate losses for each quantile
    losses = torch.max((quantiles_tensor - 1) * errors, quantiles_tensor * errors)

    # Sum the losses and take the mean
    loss = torch.mean(torch.sum(losses, dim=1))

    return loss

def MAE(output, target):
    criterion = torch.nn.L1Loss(reduction='mean')
    MAE = criterion(output, target)
    #MAE = torch.mean(output - target)
    return MAE

def quantile_loss(preds, target, quantile=0.5):
    assert 0 < quantile < 1, "Quantile should be in (0, 1) range"
    errors = target - preds
    loss = torch.max((quantile - 1) * errors, quantile * errors)
    return torch.abs(loss).mean()
'''
if __name__ == "__main__":
    data, measures, global_map, loaders = prepare_data(batch_size=128, one_hot_enc=True)
    # Initialize the model
    input_dim = 2 + len(global_map)
    hidden_dim = 128
    hidden_dim_two = 128
    output_dim = 1  # Assuming binary classification
    model = GNN(input_dim, hidden_dim, hidden_dim_two, output_dim)
    
    # Loss and optimizer
    MSE = torch.nn.MSELoss(reduction='mean')
    MAE = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = train(model, MSE, MAE, optimizer, measures, 5, loaders)
    predict(model, data[0], measures)
'''