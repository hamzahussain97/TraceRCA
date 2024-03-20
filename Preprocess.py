# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:23:12 2024

@author: Hamza
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import math
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from torch_geometric.utils import to_networkx



def prepare_data(path, normalize_features= [], normalize_by_node_features = [], scale_features = []):
    data = pd.DataFrame()
    data_dir = Path(path)
    file_list = list(map(str, data_dir.glob("*admin-order_abort_1011.pkl")))
    '''
    ##################################################
    print("\n***********File List************")
    print(*file_list, sep='\n')
    print("********************************\n")
    ##################################################
    '''
    print("\n********************************")
    print("*********Loading Files**********")
    print("********************************\n")
    for data_file in tqdm(file_list):
        with open(data_file, 'rb') as file:
            file_data = pickle.load(file)
        df = pd.DataFrame(file_data)
        df = df.apply(lambda row: order_data(row), axis=1)
        #df['starttime'] = df.apply(lambda row: get_start_times(row['timestamp'], row['latency']), axis=1)
        df['timestamp'] = df['timestamp'].apply(lambda stamps: stamps_to_time(stamps))
        df['latency'] = df['latency'].apply(lambda latencies: micro_to_mili(latencies))
        data = pd.concat([data,df])
    
    counts = data['label'].value_counts()
    ##################################################
    print("\n***********Fault Distribution************")
    print(counts)
    print("*****************************************\n")
    ##################################################
    data = data[data['label'] != 1]
    
    counts = data['label'].value_counts()
    ##################################################
    print("\n***********Fault Distribution************")
    print(counts)
    print("*****************************************\n")
    ##################################################
    
    outliers = ['d3fdfb558dfb754de55b9e8d80eeb7a3', \
                'f8398b6b1ad61f915ff275141eb345e7', \
                'd50503eb258fcf371b719b716555f55d', \
                '9f6fb14ccb19fc48668c1898c4835905', \
                '6b479c5de1a70eb50b1ea151c93b6181']
    data = data[~data['trace_id'].isin(outliers)]
    measures = {}
    for feature in normalize_by_node_features:
        data, feature_measures = normalize_by_node(data, feature)
        measures[feature] = feature_measures
    for feature in normalize_features:
        data, feature_mean, feature_std = normalize(data, feature)
        measures[feature] = [feature_mean, feature_std]
    for feature in scale_features:
        data, feature_max, feature_min = scale(data, feature)
        measures[feature] = [feature_max, feature_min]
    
    global_map = prepare_global_map(data)
    return data, global_map, measures

def order_data(data_row):
    latencies = data_row['latency']
    sorted_indices = sorted(range(len(latencies)), key=lambda i: latencies[i])
    data_row['latency'] = [data_row['latency'][i] for i in sorted_indices]
    data_row['cpu_use'] = [data_row['cpu_use'][i] for i in sorted_indices]
    data_row['mem_use_percent'] = [data_row['mem_use_percent'][i] for i in sorted_indices]
    data_row['net_send_rate'] = [data_row['net_send_rate'][i] for i in sorted_indices]
    data_row['net_receive_rate'] = [data_row['net_receive_rate'][i] for i in sorted_indices]
    return data_row

def stamps_to_time(stamps):
    time = []
    for stamp in stamps:
        time = time + [datetime.fromtimestamp(stamp/1000000)\
                       .strftime("%d/%m/%Y %H:%M:%S.%f")]
    return time

def micro_to_mili(latencies):
    return [latency / 1000 for latency in latencies]

def scale(data, column):
    #exploded_data = data.explode('latency')
    #transformed_values, lambda_parameter = boxcox(exploded_data['latency'].astype(float))
    #exploded_data['new_latency'] = transformed_values
    #df_reversed_latency = exploded_data.groupby(exploded_data.index).agg({'latency': list, 'new_latency': list})
    #df_other_columns = exploded_data.groupby(exploded_data.index).first().drop(columns=['latency', 'new_latency'])
    #reversed_data = pd.merge(df_reversed_latency, df_other_columns, left_index=True, right_index=True)

    values = pd.DataFrame()
    values = data.explode(column)
    
    maximum = values[column].max()
    minimum = values[column].min()
    data['original_latency'] = data['latency']
    data[column] = data[column].apply(lambda row: scale_values(row, maximum, minimum))
    return data, maximum, minimum

def scale_values(values, maximum, minimum):
    scaled_values = []
    maximum = np.log10(maximum)
    minimum = np.log10(minimum)
    for value in values:
        value = np.log10(value)
        scaled_value = (value - minimum) / (maximum-minimum)
        scaled_values = scaled_values + [scaled_value]
    return scaled_values

def recover_value(values, maximum, minimum):
    recovered_values = []
    maximum = np.log10(maximum)
    minimum = np.log10(minimum)
    #recovered_values = inv_boxcox(values, maximum)
    for value in values:
        value = ((maximum - minimum) * value) + minimum
        recovered_value = 10 ** value
        recovered_values = recovered_values + [recovered_value]
    
    return torch.tensor(recovered_values, dtype=torch.float32)

def normalize_by_node(data, column):
    values = pd.DataFrame()
    filtered_data = data[[column, 's_t']].copy()
    
    values[column] = data[column].explode(column)
    df_edges = data['s_t'].explode('s_t')
    nodes = df_edges.apply(lambda x: x[1])
    
    values['node_name'] = nodes
    
    print("\n********************************")
    print("***Normalizing " + column + " by node***")
    print("********************************\n")
   
    # Group by 'node_name' and calculate mean and std of column values
    result = values.groupby('node_name')[column].agg(['mean', 'std'])

    # Rename the columns for clarity
    result.columns = ['average', 'std_dev']

    # If you want to reset the index and have 'node_name' as a regular column:
    #result.reset_index(inplace=True)
    result.fillna(1, inplace=True)
    result['std_dev'].replace(0,1, inplace=True)
    if column == 'latency':
        data[column] = filtered_data.progress_apply(lambda row: centre_by_node(row[column], row['s_t'], result), axis=1)
    else:
        data[column+'_normalized'] = filtered_data.progress_apply(lambda row: centre_by_node(row[column], row['s_t'], result), axis=1)
    return data, result

def centre_by_node(values, nodes, measures):
    centred_values = []
    for (value, node) in zip(values, nodes):
        mean = measures.loc[node[1], 'average']
        std = measures.loc[node[1], 'std_dev']
        centred_value = [(value - mean) / std]
        centred_values = centred_values + centred_value
    return centred_values

def recover_by_node(values, node_names, measures):
    recovered_values = []
    concatenated_names = pd.concat(node_names).reset_index(drop=True)
    for (value, node) in zip(values, concatenated_names):
        mean = measures.loc[node, 'average']
        std = measures.loc[node, 'std_dev']
        recovered_value = [(value * std) + mean]
        recovered_values = recovered_values + [recovered_value]
    return torch.tensor(recovered_values, dtype=torch.float32)

def normalize(data, column):
    total = 0
    count = 0
    squared_sum = 0
    
    for values in data[column]:
        # Update total by summing up all values in each list
        total += sum(values)
        # Update count by adding the length of each list
        count += len(values)
        # Update squared_sum by summing up the squares of all values in each list
        squared_sum += sum(map(lambda x: x**2, values))
    
    # Calculate the mean
    mean = total / count
    
    # Calculate the mean of squares
    squared_mean = squared_sum / count
    
    # Calculate the standard deviation
    std = math.sqrt(squared_mean - mean**2)
    data[column] = data[column].apply(lambda values: centre(values, mean, std))
    return data, mean, std

def centre(values, mean, std):
    centred_values = []
    for value in values:
        centred_values = centred_values + [(value - mean) / std]
    return centred_values

def recover(values, mean, std):
    recovered_values = []
    for value in values:
        recovered_values = recovered_values + [(value * std) + mean]
    return torch.tensor(recovered_values, dtype=torch.float32)
    

def prepare_global_map(data):
    global_map ={}
    for s_t in data['s_t']:
        edges = pd.DataFrame(s_t, columns = ['source', 'target'])
        unique_nodes = pd.concat([edges['source'], edges['target']]).unique()
        for node in unique_nodes:
            if global_map == {}:
                global_map[node] = 0
            elif node not in global_map.keys():
                global_map[node] = max(global_map.values()) + 1
    return global_map

def prepare_graph(trace, global_map, one_hot_enc, normalize_by_node_features = []):
    nodes = {'cpu_use': trace['cpu_use'], \
             'mem_use_percent': trace['mem_use_percent'],
             'net_send_rate': trace['net_send_rate'],
             'net_receive_rate': trace['net_receive_rate']}
        
    for feature in normalize_by_node_features:
        if feature != 'latency':
            nodes[feature+'_normalized'] = trace[feature+'_normalized']
    
    nodes = pd.DataFrame(nodes)

    #Create dataframe of edges
    edges = pd.DataFrame(trace['s_t'], columns = ['source', 'target'])

    #Assume that the metrics belong to the target node in the edge. Store the 
    #node name of the target with the metrics
    nodes['node_name'] = edges['target']
    node_names = nodes['node_name']

    y_edge_features = {'latency': trace['latency']}
    y_edge_features = pd.DataFrame(y_edge_features)
    trace_lat = y_edge_features.max()
    
    #Find all unique node names
    unique_nodes = pd.concat([edges['source'], edges['target']]).unique()
    
    #Check nodes that only occur as source in edges, therefore will have no metric
    #values in the node feature matrix.
    not_in_target = edges[~edges['source'].isin(edges['target'])]

    if len(not_in_target) > 0:
    # Create a DataFrame filled with zeros with the same number of rows as 'not_in_target'
    # and one less column than 'nodes'
        zero_df = pd.DataFrame(0, index=range(len(not_in_target)), columns=nodes.columns[:-1])
        source_df = pd.DataFrame(not_in_target['source'].values, columns=['node_name'])
        
        # Reset the index of zero_df and source_df to align them properly
        zero_df.reset_index(drop=True, inplace=True)
        source_df.reset_index(drop=True, inplace=True)
        # Add 'node_name' column to zero_df and populate it with values from 'source' column of not_in_target
        zero_df = pd.concat([zero_df, source_df], axis = 1)
        # Append zero_df to the bottom of nodes DataFrame
        nodes = pd.concat([nodes, zero_df])

    nodes = nodes.drop_duplicates(subset='node_name')
    nodes = nodes.reset_index(drop=True)
    
    #Map node names to integers
    node_to_int = {node: i for i, node in enumerate(unique_nodes)}
    nodes['node_name'] = nodes['node_name'].map(global_map)
    edges['source'] = edges['source'].map(node_to_int)
    edges['target'] = edges['target'].map(node_to_int)
    
    if one_hot_enc:
        # Convert 'node_name' column to string to ensure proper encoding
        nodes['node_name'] = nodes['node_name'].astype(str)
    
        # Perform one-hot encoding
        one_hot_encoded = pd.get_dummies(nodes['node_name'], prefix='node')
        
        # Reindex the one-hot encoded DataFrame to ensure the desired number of columns
        one_hot_encoded = one_hot_encoded.reindex(columns=[str(i) for i in range(len(global_map))], fill_value=0)
    
        # Concatenate the one-hot encoded columns with the original DataFrame
        nodes = pd.concat([nodes.drop(columns=['node_name']), one_hot_encoded], axis=1)
    
    #Convert to tensors
    nodes_tensor = torch.tensor(nodes.values, dtype=torch.float32)
    edges_tensor = torch.tensor(edges.values, dtype=torch.long).t().contiguous()
    y_edge_tensor = torch.tensor(y_edge_features.values, dtype=torch.float32).squeeze(dim=1)
    trace_lat_tensor = torch.tensor(trace_lat, dtype=torch.float32)
    graph = Data(x=nodes_tensor, edge_index=edges_tensor, y=y_edge_tensor, trace_lat=trace_lat_tensor)
    graph.node_names = node_names
    #inv_map = {v: k for k, v in node_to_int.items()}
    return graph

def preprocess(path, one_hot_enc = False, normalize_features= [], normalize_by_node_features = [], scale_features = []):
    tqdm.pandas()
    data, global_map, measures = prepare_data(path, normalize_features, normalize_by_node_features, scale_features)
    #inv_maps = []
    print("\n********************************")
    print("********Preparing Graphs**********")
    print("********************************\n")
    graphs = data.progress_apply(lambda trace: prepare_graph(trace, global_map, one_hot_enc, normalize_by_node_features), axis=1)
    graphs = graphs.to_list()
    #graphs = []
    '''
    for index, trace in tqdm(data.iterrows(), total=data.shape[0]):
        graph, inv_map = prepare_graph(trace, global_map, one_hot_enc, normalize_by_node_features)
        graphs.append(graph)
        inv_maps.append(inv_map)
    '''
    return data, graphs, global_map, measures

if __name__ == "__main__":   
    data, graphs, global_map, measures = preprocess('./A/microservice/test/', one_hot_enc=False, normalize_features=['cpu_use', 'mem_use_percent'], \
    normalize_by_node_features=['cpu_use', 'mem_use_percent'])
    graph = graphs[386]
    #inv_map = inv_maps[386]
    #g = to_networkx(graph, to_undirected=False)
    #plt.figure()
    #nx.draw(g, labels = inv_map, with_labels = True)
    #plt.show
    #data['p_latency'] = data['latency']
    #data = data.explode('latency')
    #visualize(data, 'latency')
