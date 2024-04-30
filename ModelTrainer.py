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

from CustomDataset import CustomDataset
from Preprocess import preprocess, recover, recover_by_node, recover_value, recover_by_trace
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from BaselineModel import GNN
from torchmetrics.functional.regression import explained_variance
from SystemGraphProcessor import system_graph_processor
import numpy as np
from torchmetrics import MeanAbsolutePercentageError


class ModelTrainer():
    def __init__(self, data_dir, batch_size, quantiles=[], predict_graph=True, one_hot_enc=False, \
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
        
        path = './A/microservice/test/'
        
        #Pass the directory that contains data as pickle files to the preprocessing function
        data, graphs, global_map, measures = preprocess(path,\
                                                        self.one_hot_enc,\
                                                        self.normalize_features,\
                                                        self.normalize_by_node_features,\
                                                        self.scale_features)

            
        if 'latency' in measures: measures = measures['latency']
        dataset = CustomDataset(graphs)
    
        # Split the dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        # Split the dataset into training and validation sets
        #train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        #val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
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
            for i, quantile in enumerate(self.quantiles):
                mape = MAPE(predictions[:,i], target)
                e_var = explained_variance(predictions[:,i], target)
                print(f"Quantile: {quantile}")
                print(f"Val MAPE: {mape:.4f}, Exp Var: {e_var:.4f}")
                p_mape = percentile_mape(target, predictions[:,i])
                print(f"MAPE by percentiles: {', '.join(f'{tensor.item():.4f}' for tensor in p_mape.values())}")
                p_mae = percentile_mae(target, predictions[:,i])
                print(f"MAE by percentiles: {', '.join(f'{tensor.item():.4f}' for tensor in p_mae.values())}")
                if epoch == epochs: 
                    plot(target, predictions[:,i])
            if epoch == epochs:
                plot_percentiles(target, predictions, self.quantiles)
            print("\n")
        return self.model
    
    def step(self, batch, loss_fn, criterion):
        recov_pred = self.model(batch, batch.batch)
        if self.predict_graph:
            recovered = batch.trace_lat
            node_names = batch.first_node
            trace_integers = batch.trace_integer
        else:
            recovered = batch.y
            node_names = batch.node_names
            trace_integers = batch.trace_integers
        #target = torch.stack([1 - recovered, recovered], dim=1)
        #prediction = torch.stack([1 - recov_pred, recov_pred], dim=1)
        target = torch.stack([recovered for _ in range(len(self.quantiles))], dim=1)
        loss = loss_fn(recov_pred, target, self.quantiles)
        if self.validate_on_trace:
            edge_index = batch.edge_index
            batch_nodes = batch.batch
            batch_edge = batch_nodes[edge_index[0]]
            recovered, recov_pred = self.extract_trace_lat(recovered, recov_pred, batch_edge)
            node_names = batch.first_node
            trace_integers = batch.trace_integer
        #recovered, recov_pred = self.recover_predictions(recovered, recov_pred, node_names, trace_integers)
        #recovered = batch.original
        crit = criterion(recov_pred[:,1], recovered)
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
            node_names = [graph.first_node]
            trace_integers = [graph.trace_integer]
        else:
            recovered = graph.y
            node_names = [graph.node_names]
            trace_integers = graph.trace_integers
            
        with torch.no_grad():
            recov_pred = self.model(graph, torch.zeros(graph.x.size(0), dtype=torch.int64))
            
        #recovered, recov_pred = self.recover_predictions(recovered, recov_pred, node_names, trace_integers)
        
        print("*************Prediction*************")
        print(recov_pred)
        print("***************Actual***************")
        print(recovered)
        
        return recov_pred
    
    

'''
def prepare_data(batch_size, one_hot_enc=False, normalize_features=[], normalize_by_node_features=[], scale_features=[]):
    #Pass the directory that contains data as pickle files to the preprocessing function
    data, graphs, global_map, measures = preprocess('./A/microservice/test/', one_hot_enc, normalize_features, normalize_by_node_features, scale_features)
    if 'latency' in measures: measures = measures['latency']
    dataset = CustomDataset(graphs)

    
    # Split the dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return data, graphs, measures, global_map, [train_loader, val_loader]

def train(model, loss_fn, criterion, optimizer, measures, epochs, loaders, recov = False, recov_by_node = False, recov_scaling = False):
    train_loader = loaders[0]
    val_loader = loaders[1]
    # Training loop
    for epoch in range(1, epochs+1):
        model.train(True)
        total_loss = 0
        total_crit = 0
        for batch in train_loader:
            optimizer.zero_grad()
            recovered, recov_pred, loss, crit = step(batch, model, loss_fn, criterion, measures, recov, recov_by_node, recov_scaling)
            loss.backward()
            optimizer.step()
            total_crit += crit.item() 
            total_loss += loss.item()
            train_crit = total_crit/len(train_loader)
            train_loss = total_loss/len(train_loader)
            
        model.eval()
        total_val_loss = 0
        total_val_crit = 0
        with torch.no_grad():
            target = torch.tensor([])
            predictions = torch.tensor([])
            for batch in val_loader:
                recovered, recov_pred, loss, crit = step(batch, model, loss_fn, criterion, measures, recov, recov_by_node, recov_scaling)
                total_val_loss += loss.item()
                total_val_crit += crit.item()
                val_crit = total_val_crit/len(val_loader)
                val_loss = total_val_loss/len(val_loader)
                target = torch.cat([target, recovered], axis=0)
                predictions = torch.cat([predictions, recov_pred], axis=0)
        
        #print(outputs)
        #print(predictions)
        print(f"Epoch: {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Train criterion: {train_crit:.4f}, Val Loss: {val_loss:.4f}, Val criterion: {val_crit:.4f}")
        mape = percentile_mape(target, predictions)
        print(f"MAPE by percentiles: {', '.join(str(tensor.item()) for tensor in mape.values())}")
        if epoch == epochs: 
            plot(target, predictions)
    return model

def step(batch, model, loss_fn, criterion, measures, recov = False, recov_by_node = False, recov_scaling = False):
    recov_pred = model(batch, batch.batch)
    recovered = batch.trace_lat
    loss = torch.sqrt(loss_fn(recov_pred, recovered))
    recovered, recov_pred = recover_predictions(recovered, recov_pred, batch, measures, recov, recov_by_node, recov_scaling)
    crit = criterion(recov_pred, recovered)
    return recovered, recov_pred, loss, crit

def recover_predictions(recovered, recov_pred, batch, measures, recov=False, recov_by_node=False, recov_scaling=False):
    if recov == True:
        recovered = recover(recovered, measures[0], measures[1])
        recov_pred = recover(recov_pred, measures[0], measures[1])
    elif recov_by_node == True:
        recovered = recover_by_node(recovered, batch.node_names, measures)
        recov_pred = recover_by_node(recov_pred, batch.node_names, measures)
    elif recov_scaling == True:
        recovered = recover_value(recovered, measures[0], measures[1])
        recov_pred = recover_value(recov_pred, measures[0], measures[1])
    return recovered, recov_pred

'''

def predict(model, graph, measures, recov = False, recov_by_node = False, recov_scaling = False):
    with torch.no_grad():
        recov_pred = model(graph, torch.zeros(graph.x.size(0), dtype=torch.int64))
    if recov == True:
        recovered = recover(graph.trace_lat, measures[0], measures[1])
        recov_pred = recover(recov_pred, measures[0], measures[1])
        print("*************Prediction*************")
        print(recov_pred)
        print("***************Actual***************")
        print(recovered)
    elif recov_by_node == True:
        recovered = recover_by_node(graph.trace_lat, [graph.node_names], measures)
        recov_pred = recover_by_node(recov_pred, [graph.node_names], measures)
        print("*************Prediction*************")
        print(recov_pred)
        print("***************Actual***************")
        print(recovered)
    elif recov_scaling == True:
        recovered = recover_value(graph.trace_lat, measures[0], measures[1])
        recov_pred = recover_value(recov_pred, measures[0], measures[1])
        print("*************Prediction*************")
        print(recov_pred)
        print("***************Actual***************")
        print(recovered)
    else:
        print("*************Prediction*************")
        print(recov_pred)
        print("***************Actual***************")
        print(graph.y)
    return recov_pred

def percentile_mape(target, predictions):
    p = percentiles(target,predictions)
    
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

def threshold_breach_acc(target, predictions):
    # Convert tensors to numpy arrays
    target_np = target.numpy()
    predictions_np = predictions.numpy()
    
    # Calculate percentiles of the target tensor using numpy
    percentiles = np.array([75, 90, 95])
    target_percentiles = np.percentile(target_np, percentiles)
    
    # Initialize lists to store metrics for each percentile
    accuracy_list = []
    precision_list = []
    recall_list = []

    for percentile in target_percentiles:
        # Create tensors indicating whether values in the target tensor exceed percentile
        target_exceed_percentiles = (target_np > percentile).astype(float)
    
        # Create tensors indicating whether values in the predictions tensor exceed percentile
        prediction_exceed_percentiles = (predictions_np > percentile).astype(float)
    
        # Calculate accuracy
        accuracy = np.mean(target_exceed_percentiles == prediction_exceed_percentiles)
        accuracy_list.append(accuracy)
    
        # Precision: true positives / (true positives + false positives)
        precision = (prediction_exceed_percentiles * target_exceed_percentiles).sum() / prediction_exceed_percentiles.sum() if prediction_exceed_percentiles.sum() != 0 else 0  # handle division by zero
        precision_list.append(precision)
    
        # Recall: true positives / (true positives + false negatives)
        recall = (prediction_exceed_percentiles * target_exceed_percentiles).sum() / target_exceed_percentiles.sum() if target_exceed_percentiles.sum() != 0 else 0  # handle division by zero
        recall_list.append(recall)
        
    for i, percentile in enumerate([75, 90, 95]):
        print(f"Percentile: {percentile}, Accuracy: {accuracy_list[i]:.3f}, Precision: {precision_list[i]:.3f}, Recall: {recall_list[i]:.3f}")

def coverage_probability(targets, predictions):
    # Check if the target falls within the predicted range for each percentile
    covered = [predictions[j, 0] <= targets[j] <= predictions[j, -1] for j in range(len(targets))]
    
    # Calculate coverage probability
    coverage_prob = sum(covered) / len(covered)
    
    # Return coverage probabilities
    return coverage_prob

def percentiles(x,y):
    x = x.numpy()
    y = y.numpy()

    percentile_10 = np.percentile(x, 10)
    percentile_25 = np.percentile(x, 25)
    percentile_50 = np.percentile(x, 50)
    percentile_75 = np.percentile(x, 75)
    percentile_90 = np.percentile(x, 90)
    percentile_95 = np.percentile(x, 95)
    
    index_25 = np.where((x <= 1))[0]
    index_50 = np.where((x > 1) & (x <= 2))[0]
    index_75 = np.where((x > 2) & (x <= 3))[0]
    index_90 = np.where((x > 3) & (x <= 4))[0]
    index_100 = np.where((x > 4) & (x <= 5))[0]
    
    percentiles = {}
    # Slice values based on percentiles
    x_25 = x[index_25]
    y_25 = y[index_25]
    p_25 = {'x': x_25, 'y': y_25}
    percentiles[25] = p_25
    
    
    x_50 = x[index_50].flatten()
    y_50 = y[index_50].flatten()
    p_50 = {'x': x_50, 'y': y_50}
    percentiles[50] = p_50
    
    x_75 = x[index_75].flatten()
    y_75 = y[index_75].flatten()
    p_75 = {'x': x_75, 'y': y_75}
    percentiles[75] = p_75
    
    x_90 = x[index_90].flatten()
    y_90 = y[index_90].flatten()
    p_90 = {'x': x_90, 'y': y_90}
    percentiles[90] = p_90
    
    x_100 = x[index_100].flatten()
    y_100 = y[index_100].flatten()
    p_100 = {'x': x_100, 'y': y_100}
    percentiles[100] = p_100
    
    return percentiles


def plot(x, y):
    p = percentiles(x, y)
    plot_figure(1, p, 25)
    plot_figure(2, p, 50)
    plot_figure(3, p, 75)
    plot_figure(4, p, 90)
    plot_figure(5, p, 100)
    
def plot_figure(i, p, u_l):
    plt.figure(i)
    plt.scatter(p[u_l]['x'],p[u_l]['y'])
    
    if len(p[u_l]['x']) > 0:
        max_val = max(max(p[u_l]['x']), max(p[u_l]['y']))
        plt.grid(True)
        plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='y=x')
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
            plt.title('Distribution of Latencies')
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
    plt.legend()
    
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y=x')
    # Show plot
    plt.show()
    
    plt.figure(figsize=(10, 6))
    mean_pred = predictions[:,3]
    
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
    errors = target - preds

    # Calculate losses for each quantile
    losses = torch.max((quantiles_tensor - 1) * errors, quantiles_tensor * errors)

    # Sum the losses and take the mean
    loss = torch.mean(torch.sum(losses, dim=1))

    return loss

def RRMSE(output, target):
    target[target==0] = 1
    SE = torch.mean(((output - target)**2) / ((target**2)))
    return SE

def MBE(output, target):
    SE = torch.mean(output - target)
    return SE

def MAE(output, target):
    #criterion = torch.nn.L1Loss(reduction='mean')
    #MAE = criterion(output, target)
    MAE = torch.mean(output - target)
    return MAE

def quantile_loss(preds, target, quantile):
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