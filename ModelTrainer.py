# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:21:41 2024

@author: Hamza
"""

import torch
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from CustomDataset import CustomDataset
from Preprocess import preprocess, recover, recover_by_node, recover_value
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from BaselineModel import GNN
import numpy as np


def prepare_data(batch_size, one_hot_enc=False, normalize_features=[], normalize_by_node_features=[], scale_features=[]):
    #Pass the directory that contains data as pickle files to the preprocessing function
    data, graphs, global_map, measures = preprocess('./A/microservice/test/', one_hot_enc, normalize_features, normalize_by_node_features, scale_features)
    if 'latency' in measures: measures = measures['latency']
    dataset = CustomDataset(graphs)
    '''
    for i in range(10):
        graph = graphs[100*i]
        inv_map = inv_maps[100*i]
        g = to_networkx(graph, to_undirected=False)
        plt.figure(i)
        nx.draw(g, labels = inv_map, with_labels = True)
        plt.show
    
    '''
    # Split the dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return data, graphs, measures, global_map, [train_loader, val_loader]

def MAPE(target, output):
    return torch.mean((target - output).abs() / target.abs())

def RRMSE(output, target):
    target[target==0] = 1
    SE = torch.mean(((output - target)**2) / ((target**2)))
    return SE

def MBE(output, target):
    SE = torch.mean(output - target)
    return SE

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
            max_pred, recov_pred = model(batch, batch.batch)
            recovered = batch.y
            trace_lat = batch.trace_lat
            loss = torch.sqrt(loss_fn(recov_pred, recovered))
            if recov == True:
                recovered = recover(recovered, measures[0], measures[1])
                recov_pred = recover(recov_pred, measures[0], measures[1])
            elif recov_by_node == True:
                recovered = recover_by_node(recovered, batch.node_names, measures)
                recov_pred = recover_by_node(recov_pred, batch.node_names, measures)
            elif recov_scaling == True:
                recovered = recover_value(recovered, measures[0], measures[1])
                recov_pred = recover_value(recov_pred, measures[0], measures[1])
                trace_lat = recover_value(trace_lat, measures[0], measures[1])
                max_pred = recover_value(max_pred, measures[0], measures[1])
            crit = criterion(max_pred, trace_lat)
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
                max_pred, recov_pred = model(batch, batch.batch)
                recovered = batch.y
                trace_lat = batch.trace_lat
                loss = torch.sqrt(loss_fn(recov_pred, recovered))
                if recov == True:
                    recovered = recover(recovered, measures[0], measures[1])
                    recov_pred = recover(recov_pred, measures[0], measures[1])
                elif recov_by_node == True:
                    recovered = recover_by_node(recovered, batch.node_names, measures)
                    recov_pred = recover_by_node(recov_pred, batch.node_names, measures)
                elif recov_scaling == True:
                    recovered = recover_value(recovered, measures[0], measures[1])
                    recov_pred = recover_value(recov_pred, measures[0], measures[1])
                    trace_lat = recover_value(trace_lat, measures[0], measures[1])
                    max_pred = recover_value(max_pred, measures[0], measures[1])
                    
                crit = criterion(max_pred, trace_lat)
                total_val_loss += loss.item()
                total_val_crit += crit.item()
                val_crit = total_val_crit/len(val_loader)
                val_loss = total_val_loss/len(val_loader)
                target = torch.cat([target, trace_lat], axis=0)
                predictions = torch.cat([predictions, max_pred], axis=0)
        
        #print(outputs)
        #print(predictions)
        print(f"Epoch: {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Train criterion: {train_crit:.4f}, Val Loss: {val_loss:.4f}, Val criterion: {val_crit:.4f}")
        mape = percentile_mape(target, predictions)
        print(f"MAPE by percentiles: {', '.join(str(tensor.item()) for tensor in mape.values())}")
        if epoch == epochs: 
            plot(target, predictions)
    return model

def predict(model, graph, measures, recov = False, recov_by_node = False, recov_scaling = False):
    with torch.no_grad():
        recov_pred, out = model(graph, torch.zeros(graph.x.size(0), dtype=torch.int))
    if recov == True:
        recovered = recover(graph.y, measures[0], measures[1])
        recov_pred = recover(out, measures[0], measures[1])
        print("*************Prediction*************")
        print(recov_pred)
        print("***************Actual***************")
        print(recovered)
    elif recov_by_node == True:
        recovered = recover_by_node(graph.y, [graph.node_names], measures)
        recov_pred = recover_by_node(out, [graph.node_names], measures)
        print("*************Prediction*************")
        print(recov_pred)
        print("***************Actual***************")
        print(recovered)
    elif recov_scaling == True:
        recovered = recover_value(graph.y, measures[0], measures[1])
        recov_pred = recover_value(out, measures[0], measures[1])
        print("*************Prediction*************")
        print(recov_pred)
        print("***************Actual***************")
        print(recovered)
    else:
        print("*************Prediction*************")
        print(out)
        print("***************Actual***************")
        print(graph.y)
    return out

def percentile_mape(target, predictions):
    p = percentiles(target,predictions)
    
    m_25 = MAPE(torch.tensor(p[25]['x']),torch.tensor(p[25]['y']))
    m_50 = MAPE(torch.tensor(p[50]['x']),torch.tensor(p[50]['y']))
    m_90 = MAPE(torch.tensor(p[90]['x']),torch.tensor(p[90]['y']))
    m_100 = MAPE(torch.tensor(p[100]['x']),torch.tensor(p[100]['y']))
    
    return {25: m_25, 50: m_50, 90: m_90, 100:m_100}
    
def percentiles(x,y):
    x = x.numpy()
    y = y.numpy()
    percentile_25 = np.percentile(x, 25)
    percentile_50 = np.percentile(x, 50)
    percentile_90 = np.percentile(x, 90)
    
    index_25 = np.where(x <= percentile_25)[0]
    index_50 = np.where((x > percentile_25) & (x <= percentile_50))[0]
    index_90 = np.where((x > percentile_50) & (x <= percentile_90))[0]
    index_100 = np.where((x > percentile_90))[0]
    
    percentiles = {}
    # Slice values based on percentiles
    x_25 = x[index_25]
    y_25 = y[index_25]
    p_25 = {'x': x_25, 'y': y_25}
    percentiles[25] = p_25
    
    x_50 = x[index_50]
    y_50 = y[index_50]
    p_50 = {'x': x_50, 'y': y_50}
    percentiles[50] = p_50
    
    x_90 = x[index_90]
    y_90 = y[index_90]
    p_90 = {'x': x_90, 'y': y_90}
    percentiles[90] = p_90
    
    x_100 = x[index_100]
    y_100 = y[index_100]
    p_100 = {'x': x_100, 'y': y_100}
    percentiles[100] = p_100
    
    return percentiles


def plot(x, y):
    p = percentiles(x, y)
    
    plt.figure(1)
    plt.scatter(p[25]['x'],p[25]['y'])
    max_val = max(max(p[25]['x']), max(p[25]['y']))
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='y=x')
    plt.show()
    
    plt.figure(2)
    plt.scatter(p[50]['x'],p[50]['y'])
    max_val = max(max(p[50]['x']), max(p[50]['y']))
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='y=x')
    plt.show()
    
    
    plt.figure(3)
    plt.scatter(p[90]['x'],p[90]['y'])
    max_val = max(max(p[90]['x']), max(p[90]['y']))
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='y=x')
    plt.show()
    
    plt.figure(4)
    plt.scatter(p[100]['x'],p[100]['y'])
    max_val = max(max(p[100]['x']), max(p[100]['y']))
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='y=x')
    plt.show()
    return 0

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