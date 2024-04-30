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
import torch_geometric.transforms as T
from scipy.stats import norm, t
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from torch_geometric.utils import to_networkx


def prepare_data(path, normalize_features= [], normalize_by_node_features = [], scale_features = []):
    data = pd.DataFrame()
    data_dir = Path(path)
    file_list = list(map(str, data_dir.glob("assurance*.pkl")))
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
    trace_to_integer = {}
    for data_file in tqdm(file_list):
        with open(data_file, 'rb') as file:
            file_data = pickle.load(file)
        df = pd.DataFrame(file_data)
        df['latency'] = df['latency'].apply(lambda latencies: micro_to_mili(latencies))
        df['original_latency'] = df['latency']
        #df = df[df['max_latency'] >= 150000]
        #df['starttime'] = df.apply(lambda row: get_start_times(row['timestamp'], row['latency']), axis=1)
        df['timestamp'] = df['timestamp'].apply(lambda stamps: stamps_to_time(stamps))
        df['trace_integer'] = df.apply(lambda row: get_trace_integer(row, trace_to_integer), axis=1)
        data = pd.concat([data,df])
    
    counts = data['label'].value_counts()
    ##################################################
    print("\n***********Fault Distribution************")
    print(counts)
    print("*****************************************\n")
    ##################################################
    #data = data[data['label'] != 1]
    
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
    '''
    occurrences = data['trace_integer'].value_counts()
    # Filter out values with counts less than 10
    trace_integers_to_keep = occurrences[occurrences >= 30].index.tolist()
    # Filter original DataFrame based on the condition
    data = data[data['trace_integer'].isin(trace_integers_to_keep)]
    '''
    measures = {}
    # Apply the normalization function to each group based on trace_integer
    #stats = data.groupby('trace_integer')['max_latency'].agg(['mean', 'std'])
    #stats.fillna(0, inplace=True)
    #data = data.groupby('trace_integer').apply(normalize_cluster)
    data = data.reset_index(drop=True)
    transformation_features = normalize_by_node_features + normalize_features + scale_features
    for feature in transformation_features:
        measures[feature] = {}
    for feature in scale_features:
        data, feature_max, feature_min = scale(data, feature)
        measures[feature]['scale'] = [feature_max, feature_min]
    for feature in normalize_by_node_features:
        data, feature_measures = normalize_by_node(data, feature)
        measures[feature]['norm_by_node'] = feature_measures
    for feature in normalize_features:
        data, feature_mean, feature_std = normalize(data, feature)
        measures[feature]['norm'] = [feature_mean, feature_std]
    data, stats = normalize_by_trace(data)
    #measures['latency'] = {}
    #measures['latency']['norm_by_trace'] = stats
    #data = data[data['trace_length'] > 1]
    
    for column in ['mean', 'std', 'maximum', 'minimum']:
        data, feature_mean, feature_std = normalize(data, column)
    
    global_map = prepare_global_map(data)
    return data, global_map, measures

def order_data(data_row):
    latencies = data_row['original_latency']
    sorted_indices = sorted(range(len(latencies)), key=lambda i: latencies[i])
    data_row['latency'] = [data_row['latency'][i] for i in sorted_indices]
    data_row['original_latency'] = [data_row['original_latency'][i] for i in sorted_indices]
    data_row['max_latency'] = data_row['latency'][-1]
    data_row['s_t'] = [data_row['s_t'][i] for i in sorted_indices]
    data_row['cpu_use'] = [data_row['cpu_use'][i] for i in sorted_indices]
    data_row['mem_use_percent'] = [data_row['mem_use_percent'][i] for i in sorted_indices]
    data_row['mem_use_amount'] = [data_row['mem_use_amount'][i] for i in sorted_indices]
    data_row['net_send_rate'] = [data_row['net_send_rate'][i] for i in sorted_indices]
    data_row['net_receive_rate'] = [data_row['net_receive_rate'][i] for i in sorted_indices]
    data_row['file_read_rate'] = [data_row['file_read_rate'][i] for i in sorted_indices]
    data_row['file_write_rate'] = [data_row['file_write_rate'][i] for i in sorted_indices]
    data_row['trace_length'] = len(sorted_indices)
    return data_row

def normalize_by_trace(data, grouped_data=True):    
    if grouped_data:
        values = data.apply(pd.Series.explode)
    else:
        values = data
        
    print("\n********************************")
    print("***Normalizing latency by trace***")
    print("********************************\n")
   
    # Group by 'node_name' and calculate mean and std of column values
    result = values.groupby(['trace_integer', 's_t'])['latency'].agg(['mean', 'std', 'max', 'min', 'count'])
    result = result.reset_index()
    result[['source', 'target']] = result['s_t'].apply(pd.Series)
    result.drop(columns=['s_t'], inplace=True)
    result.set_index(['trace_integer', 'source', 'target'], inplace=True)

    # If you want to reset the index and have 'node_name' as a regular column:
    #result.reset_index(inplace=True)
    result.fillna(1, inplace=True)
    
    columns = ['latency']
    values = values.groupby(['trace_integer', 's_t']).apply(normalize_cluster, columns=columns, center=False)
    values = values.reset_index(drop=True)
    if grouped_data:
        print("Grouping by trace_id")
        values = values.groupby(['trace_id', 'trace_integer']).agg(lambda x: x.tolist())
        values = values.reset_index()
        print("Ordering Rows")
        values = values.apply(lambda row: order_data(row), axis=1)
    return values, result

def normalize_cluster(cluster, columns, center=True):
    #data = cluster[cluster['label'] != 1]
    data = cluster
    for column in columns:
        mean = data[column].mean()
        std = data[column].std()
        if column == 'latency':
            count = data[column].count()
            maximum = data[column].max()
            minimum = data[column].min()
            cluster['mean'] = mean
            cluster['maximum'] = maximum
            cluster['minimum'] = minimum
            cluster['count'] = count
        df = cluster.shape[0] - 1
        if df == 0:
            df = 1
        '''
        if len(cluster[column]) > 1:
            mean, _, _ = gamma.fit(cluster[column].astype('float32'))
        else:
            std = 1
        '''
        if std == 0 or np.isnan(std):
            if center:
                cluster[column] = 0
                z_scores = cluster[column]
            cluster['std'] = 1
        else:
            if center:
                z_scores = (cluster[column] - mean) / std
            cluster['std'] = std
        if center:
            #probabilities = gamma.cdf(cluster[column].astype('float32'), a=mean)
            probabilities = norm.cdf(z_scores.to_numpy().astype(float))
            cluster[column] = probabilities
            '''
            mean = cluster[column].mean()
            std = cluster[column].std()
            maximum = cluster[column].max()
            minimum = cluster[column].min()
            cluster['mean'] = mean
            cluster['maximum'] = maximum
            cluster['minimum'] = minimum
            '''
    return cluster

def recover_by_trace(values, trace_integers, edges, measures, original=False):
    recovered_values = []
    edges = pd.concat(edges).reset_index(drop=True)
    for (value, trace_integer, edge) in zip(values, trace_integers, edges):
        value = value.item()
        
        if value > 0.999999999:
            value = 0.999999999
        if value < 0.000000001:
            value = 0.000000001
        
        mean = measures.loc[(trace_integer.item(), edge[0], edge[1]), 'mean']
        std = measures.loc[(trace_integer.item(), edge[0], edge[1]), 'std']
        df = measures.loc[(trace_integer.item(), edge[0], edge[1]), 'count']
        if df == 0:
            df = 1
        #recovered_value = [gamma.ppf(value, a=mean)]
        recovered_value = norm.ppf(value)
        recovered_value = (recovered_value * std) + mean
        recovered_values = recovered_values + [recovered_value]
    return torch.tensor(recovered_values, dtype=torch.float32)

def recover_by_cluster(values, trace_integers, measures):
    recovered_values = []
    for (value, trace_integer) in zip(values, trace_integers):
        mean = measures.loc[trace_integer.item(), 'mean']
        std = measures.loc[trace_integer.item(), 'std']
        recovered_value = [(value * std) + mean]
        recovered_values = recovered_values + [recovered_value]
    return torch.tensor(recovered_values, dtype=torch.float32)

def get_trace_integer(row, trace_to_integer):
    s_t = row['s_t']
    trace = frozenset(s_t)
    if trace not in trace_to_integer:
        next_integer = max(trace_to_integer.values()) + 1 if trace_to_integer else 0
        trace_to_integer[trace] = next_integer
        next_integer += 1
    return trace_to_integer[trace]

def stamps_to_time(stamps):
    time = []
    for stamp in stamps:
        time = time + [datetime.fromtimestamp(stamp/1000000)\
                       .strftime("%d/%m/%Y %H:%M:%S.%f")]
    return time

def micro_to_mili(latencies):
    return [latency / 1000 for latency in latencies]

def scale(data, column):
    values = pd.DataFrame()
    values = data.explode(column)
    
    maximum = values[column].max()
    minimum = values[column].min()
    print(maximum)
    print(minimum)
    data[column] = data[column].apply(lambda row: scale_values(row, maximum, minimum))
    return data, maximum, minimum

def log(value):
    if value != 0:
        scaled_value = np.log10(value)
    else:
        scaled_value = 0
    return scaled_value

def scale_values(values, maximum, minimum):
    scaled_values = []
    maximum = log(maximum)
    minimum = log(minimum)
    for value in values:
        scaled_value = log(value)
        #scaled_value = (scaled_value - minimum) / (maximum-minimum)
        scaled_values = scaled_values + [scaled_value]
    return scaled_values

def recover_value(values, maximum, minimum):
    recovered_values = []
    maximum = log(maximum)
    minimum = log(minimum)
    #recovered_values = inv_boxcox(values, maximum)
    for value in values:
        #value = ((maximum - minimum) * value) + minimum
        recovered_value = 10 ** value
        recovered_values = recovered_values + [recovered_value]
    
    return torch.tensor(recovered_values, dtype=torch.float32)

def normalize_by_edge(data):
    values = pd.DataFrame()
    filtered_data = data[['latency', 's_t']].copy()
    
    values['latency'] = data['latency'].explode('latency')
    df_edges = data['s_t'].explode('s_t')
    
    values['node_name'] = df_edges
    
    print("\n********************************")
    print("***Normalizing latency by node***")
    print("********************************\n")
   
    # Group by 'node_name' and calculate mean and std of column values
    result = values.groupby('node_name')['latency'].agg(['mean', 'std'])

    # Rename the columns for clarity
    result.columns = ['average', 'std_dev']
    
    result = pd.DataFrame(result)
    result.index = pd.MultiIndex.from_tuples(result.index)


    # If you want to reset the index and have 'node_name' as a regular column:
    #result.reset_index(inplace=True)
    result.fillna(1, inplace=True)
    result['std_dev'].replace(0,1, inplace=True)
    data['latency'] = filtered_data.progress_apply(lambda row: centre_by_node(row['latency'], row['s_t'], result), axis=1)
    
    return data, result

def normalize_by_node(data, column):
    values = pd.DataFrame()
    filtered_data = data[[column, 's_t']].copy()
    
    values[column] = data[column].explode(column)
    df_edges = data['s_t'].explode('s_t')
    nodes = df_edges.apply(lambda x: x[1])
    
    if column == 'latency':
        values['node_name'] = df_edges
    else:
        values['node_name'] = nodes
    
    print("\n********************************")
    print("***Normalizing " + column + " by node***")
    print("********************************\n")
   
    # Group by 'node_name' and calculate mean and std of column values
    result = values.groupby('node_name')[column].agg(['mean', 'std'])

    # Rename the columns for clarity
    result.columns = ['average', 'std_dev']
    
    result = pd.DataFrame(result)
    if column == 'latency':
        result.index = pd.MultiIndex.from_tuples(result.index)


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
        if measures.index.nlevels != 2:
            node = node[1]
        mean = measures.loc[node, 'average']
        std = measures.loc[node, 'std_dev']
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
    '''
    if column == 'latency':
        latencies = data['latency'].explode()
        mean, std, _ = gamma.fit(latencies.astype('float32'))
    '''
    data[column] = data[column].apply(lambda values: centre(values, mean, std, column))
    return data, mean, std

def centre(values, mean, std, column):
    centred_values = []
    if std == 0: std=1
    for value in values:
        centred_value = (value - mean) / std
        centred_values = centred_values + [centred_value]
    return centred_values

def recover(values, mean, std):
    recovered_values = []
    for value in values:
        value = value.item()
        if value >= 0.99:
            value = 0.99
        elif value <= 0.01:
            value = 0.01
        value = norm.ppf(value)
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
             'net_receive_rate': trace['net_receive_rate'],
             'file_read_rate': trace['file_read_rate']}
        

    for feature in normalize_by_node_features:
        if feature != 'latency':
            nodes[feature+'_normalized'] = trace[feature+'_normalized']
    
    nodes = pd.DataFrame(nodes)

    #Create dataframe of edges
    edges = pd.DataFrame(trace['s_t'], columns = ['source', 'target'])
    nedges = pd.DataFrame({'edges': trace['s_t']})['edges']

    trace_integers = [trace['trace_integer']] * len(edges)
    trace_integer = trace['trace_integer']
    
    
    edge_attr = {'mean': trace['mean'],\
                 'std': trace['std'],\
                 'max': trace['maximum'],\
                 'min': trace['minimum']}  
    
    edge_attr = pd.DataFrame(edge_attr)
    #Assume that the metrics belong to the target node in the edge. Store the 
    #node name of the target with the metrics
    nodes['node_name'] = edges['target']
    node_names = edges['target']
    
    y_edge_features = {'latency': trace['latency']}
    y_edge_features = pd.DataFrame(y_edge_features)
    
    original = {'latency': trace['original_latency']}
    original = pd.DataFrame(original)
    
    #trace_lat = y_edge_features.iloc[-1]
    trace_lat = trace['max_latency']
    #Find all unique node names
    unique_nodes = pd.concat([edges['source'], edges['target']]).unique()
    
    #Check nodes that only occur as source in edges, therefore will have no metric
    #values in the node feature matrix.
    not_in_target = edges[~edges['source'].isin(edges['target'])]

    if len(not_in_target) > 0:
    # Create a DataFrame filled with zeros with the same number of rows as 'not_in_target'
    # and one less column than 'nodes'
        zero_df = pd.DataFrame(0, index=range(len(not_in_target)), columns=nodes.columns[:-1])
        #zero_df['trace_integer'] = trace_integer
        source_df = pd.DataFrame(not_in_target['source'].values, columns=['node_name'])
        
        # Reset the index of zero_df and source_df to align them properly
        zero_df.reset_index(drop=True, inplace=True)
        source_df.reset_index(drop=True, inplace=True)
        # Add 'node_name' column to zero_df and populate it with values from 'source' column of not_in_target
        zero_df = pd.concat([zero_df, source_df], axis = 1)
        # Append zero_df to the bottom of nodes DataFrame
        nodes = pd.concat([nodes, zero_df])

    nodes = nodes.groupby('node_name').mean().reset_index()
    node_name_column = nodes.pop('node_name')  # Remove 'node_name' from the DataFrame
    nodes['node_name'] = node_name_column  # Add 'node_name' as the last column
    nodes = nodes.reset_index(drop=True)
    
    #Map node names to integers
    node_to_int = {node: i for i, node in enumerate(unique_nodes)}
    nodes['node_id'] = nodes['node_name'].map(node_to_int)
    edges['source'] = edges['source'].map(node_to_int)
    edges['target'] = edges['target'].map(node_to_int)
    
    nodes = nodes.sort_values(by='node_id')
    nodes = nodes.drop(columns=['node_id'])
    
    nodes['node_name'] = nodes['node_name'].map(global_map)
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
    original = torch.tensor(original.values, dtype=torch.float32)
    trace_lat_tensor = torch.tensor(trace_lat, dtype=torch.float32)
    trace_integers_tensor = torch.tensor(trace_integers, dtype=torch.long)
    trace_integer_tensor = torch.tensor(trace_integer, dtype=torch.long)
    edge_attr_tensor = torch.tensor(edge_attr.values, dtype=torch.float32)
    graph = Data(x=nodes_tensor, edge_index=edges_tensor, edge_attr=edge_attr_tensor, y=y_edge_tensor, trace_lat=trace_lat_tensor, trace_integer=trace_integer_tensor)
    graph.trace_integers = trace_integers_tensor
    graph.node_names = nedges
    graph.first_node = nedges[-1:]
    graph.original = original

    return graph

def preprocess(path, one_hot_enc = False, normalize_features = [], normalize_by_node_features = [], scale_features = []):
    tqdm.pandas()
    data, global_map, measures = prepare_data(path, normalize_features, normalize_by_node_features, scale_features)
    print("\n********************************")
    print("********Preparing Graphs**********")
    print("********************************\n")
    graphs = data.progress_apply(lambda trace: prepare_graph(trace, global_map, one_hot_enc, normalize_by_node_features), axis=1)
    graphs = graphs.to_list()
    return data, graphs, global_map, measures

if __name__ == "__main__":   
    data, graphs, global_map, measures = preprocess('./A/microservice/test/', one_hot_enc=False, normalize_features=[], \
    normalize_by_node_features=[])
    #normal = data[data['label'] != 1]
    #abnormal = data[data['label'] == 1]
    latencies = data['latency'].explode()
    # Plot a histogram of the latencies
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.hist(latencies, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Latencies')
    plt.xlabel('Latency')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    '''
    latencies = abnormal['latency'].explode()
    # Plot a histogram of the latencies
    plt.figure(2, figsize=(10, 6))  # Adjust the figure size as needed
    plt.hist(latencies, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Latencies')
    plt.xlabel('Latency')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    '''
    graph = graphs[386]
    #inv_map = inv_maps[386]
    #g = to_networkx(graph, to_undirected=False)
    #plt.figure()
    #nx.draw(g, labels = inv_map, with_labels = True)
    #plt.show
    #data['p_latency'] = data['latency']
    #data = data.explode('latency')
