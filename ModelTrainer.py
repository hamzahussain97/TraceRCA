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
            outputs = torch.tensor([])
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
                if epoch == epochs:
                    outputs = torch.cat([outputs, trace_lat], axis=0)
                    predictions = torch.cat([predictions, max_pred], axis=0)
        
        #print(outputs)
        #print(predictions)
        if epoch == epochs: plot(outputs, predictions)
        print(f"Epoch: {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Train criterion: {train_crit:.4f}, Val Loss: {val_loss:.4f}, Val criterion: {val_crit:.4f}")
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

def plot(x, y):
    x = x.numpy()
    y = y.numpy()
    percentile_25 = np.percentile(x, 25)
    percentile_50 = np.percentile(x, 50)
    percentile_75 = np.percentile(x, 90)
    percentile_100 = np.percentile(x,90)
    
    print(percentile_25)
    print(percentile_50)
    print(percentile_75)
    print(percentile_100)
    
    index_25 = np.where(x < percentile_25)[0]
    index_50 = np.where(x < percentile_50)[0]
    index_75 = np.where(x < percentile_75)[0]
    
    # Slice values based on percentiles
    x_25 = x[index_25]
    y_25 = y[index_25]
    
    x_50 = x[index_50]
    y_50 = y[index_50]
    
    x_75 = x[index_75]
    y_75 = y[index_75]
    
    plt.figure(1)
    plt.scatter(x_25,y_25)
    max_val = max(max(x_25), max(y_25))
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='y=x')
    plt.show()
    
    plt.figure(2)
    plt.scatter(x_50,y_50)
    max_val = max(max(x_50), max(y_50))
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='y=x')
    plt.show()
    
    
    plt.figure(3)
    plt.scatter(x_75,y_75)
    max_val = max(max(x_75), max(y_75))
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='y=x')
    plt.show()
    
    plt.figure(4)
    plt.scatter(x,y)
    max_val = max(max(x), max(y))
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