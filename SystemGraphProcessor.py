# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:15:58 2024

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
from Preprocess import get_trace_integer, normalize_cluster

def prepare_data(path, normalize_features= [], normalize_by_node_features = [], scale_features = []):
    data = pd.DataFrame()
    data_dir = Path(path)
    file_list = list(map(str, data_dir.glob("*.pkl")))
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
        df['original_latency'] = df['latency']
        df['timestamp'] = df['timestamp'].apply(lambda stamps: stamps_to_time(stamps))
        df['latency'] = df['latency'].apply(lambda latencies: micro_to_mili(latencies))
        df = df.apply(lambda row: order_data(row), axis=1)
        #df = df[df['max_latency'] >= 150000]
        #df['starttime'] = df.apply(lambda row: get_start_times(row['timestamp'], row['latency']), axis=1)
        df['trace_integer'] = df.apply(lambda row: get_trace_integer(row, trace_to_integer), axis=1)
        df = df.apply(pd.Series.explode)
        df[['source', 'target']] = df['s_t'].apply(pd.Series)
        df.drop('s_t', axis=1, inplace=True)
        data = pd.concat([data,df])
    data = data[data['label'] != 1]
    data.reset_index(drop=True, inplace=True)
    outliers = ['d3fdfb558dfb754de55b9e8d80eeb7a3', \
                'f8398b6b1ad61f915ff275141eb345e7', \
                'd50503eb258fcf371b719b716555f55d', \
                '9f6fb14ccb19fc48668c1898c4835905', \
                '6b479c5de1a70eb50b1ea151c93b6181']
    data = data[~data['trace_id'].isin(outliers)]
    data.reset_index(drop=True, inplace=True)
    # Group by trace_integer and calculate mean and standard deviation
    stats = data.groupby('trace_integer')['max_latency'].agg(['mean', 'std'])
    stats.fillna(0, inplace=True)
    data = data.sort_values(by='timestamp')
    data.reset_index(drop=True, inplace=True)
    # Apply the normalization function to each group based on trace_integer
    data = data.groupby('trace_integer').apply(normalize_cluster)
    # Remove duplicates based on 'timestamp' and 'target'
    edges = data[['timestamp', 'trace_id', 'source', 'target', 'max_latency', 'trace_integer']]
    edges.rename(columns={'max_latency': 'latency'}, inplace=True)
    #edges['latency'] = edges.latency.apply(lambda value: log(value))
    data = data.drop_duplicates(subset=['timestamp', 'target'])
    
    s_t = pd.concat([edges['source'], edges['target']])
    not_in_nodes = s_t[~s_t.isin(data['target'])]
    not_in_nodes = not_in_nodes.drop_duplicates()
    '''
    if len(not_in_nodes) > 0:
        # Create a DataFrame filled with zeros with the same number of rows as 'not_in_target'
        zero_df = pd.DataFrame(0, index=range(len(not_in_nodes)), columns=data.columns)
        # Add 'node_name' column to zero_df and populate it with values from 'source' column of not_in_target
        zero_df['target'] = not_in_nodes.values
        # Append zero_df to the bottom of nodes DataFrame
        data = pd.concat([data, zero_df])
    '''
    
    # Create a DataFrame with all possible combinations of timestamps and targets
    all_timestamps = pd.DataFrame(pd.date_range(data['timestamp'].min(), data['timestamp'].max(), freq='T').strftime('%Y-%m-%d %H:%M'), columns=['timestamp'])
    all_timestamps['timestamp'] = all_timestamps['timestamp'].astype(str)
    all_targets = pd.DataFrame(data['target'].unique(), columns=['target'])
    if len(not_in_nodes) > 0:
        all_targets = all_targets.append([{'target': value} for value in not_in_nodes], ignore_index=True)

    all_combinations = all_timestamps.merge(all_targets, how='cross')

    # Left merge the original DataFrame with all_combinations DataFrame
    merged_df = all_combinations.merge(data, on=['timestamp', 'target'], how='left')
    # Fill missing values with values from the previous timestamp
    nodes = merged_df.fillna(value=0)
    nodes = nodes.drop(columns=['source', 'endtime', 'latency'])
    
    
    for feature in scale_features:
        nodes, _, _ = scale(nodes, feature)
    
    #edges, maximum, minimum = scale(edges, 'latency')
    maximum = 0
    minimum = 0
    return nodes, edges, stats, trace_to_integer

def scale(data, column):
    values = data
    
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

def scale_values(value, maximum, minimum):
    maximum = log(maximum)
    minimum = log(minimum)
    
    scaled_value = log(value)
    scaled_value = (scaled_value - minimum) / (maximum-minimum)
    return scaled_value

def recover_values(values, maximum, minimum):
    recovered_values = []
    maximum = log(maximum)
    minimum = log(minimum)
    #recovered_values = inv_boxcox(values, maximum)
    for value in values:
        value = ((maximum - minimum) * value) + minimum
        recovered_value = 10 ** value
        recovered_values = recovered_values + [recovered_value]
    
    return torch.tensor(recovered_values, dtype=torch.float32)

def prepare_graphs(nodes, edges, features):
    nodes_grouped = nodes.groupby('timestamp')
    edges_grouped = edges.groupby('timestamp')
    graphs = []
    for timestamp, (nodes_data, edges_data) in tqdm(zip(nodes_grouped.groups.keys(), zip(nodes_grouped, edges_grouped))):
        nodes = nodes_data[1]
        edges = edges_data[1]
        '''
        average_latency_per_node = edges.groupby('target')['latency'].mean().reset_index()
        nodes = pd.merge(nodes, average_latency_per_node, on='target', how='left')
        nodes = nodes.fillna(value=0)
      
        s_t = pd.concat([edges['source'], edges['target']])
        not_in_nodes = s_t[~s_t.isin(nodes['target'])]
        not_in_nodes = not_in_nodes.drop_duplicates()
        if len(not_in_nodes) > 0:
            # Create a DataFrame filled with zeros with the same number of rows as 'not_in_target'
            zero_df = pd.DataFrame(0, index=range(len(not_in_nodes)), columns=nodes.columns)
            # Add 'node_name' column to zero_df and populate it with values from 'source' column of not_in_target
            zero_df['target'] = not_in_nodes.values
            # Append zero_df to the bottom of nodes DataFrame
            nodes = pd.concat([nodes, zero_df])
        '''
        #Find all unique node names
        unique_nodes = nodes['target'].unique()
        node_to_int = {node: i for i, node in enumerate(unique_nodes)}
        edges['source'] = edges['source'].map(node_to_int)
        edges['target'] = edges['target'].map(node_to_int)
        # Calculate the count of each unique source-target pair
        edge_counts = edges.groupby(['source', 'target']).size().reset_index(name='weight')
        # Merge pair_counts back to the original DataFrame
        edges = pd.merge(edges, edge_counts, on=['source', 'target'], how='left')
        latencies = edges['latency'].astype(float)
        
        edges_by_traces = edges.groupby('trace_id')
        trace_graphs = []
        max_latencies = []
        for trace_id, trace_edges in edges_by_traces:
            global_to_local = {}
            trace_nodes = pd.concat([trace_edges['source'], trace_edges['target']]).drop_duplicates()
            global_to_local = {value: idx for idx, value in enumerate(trace_nodes) if value not in global_to_local}
            trace_edges['source'] = trace_edges['source'].map(global_to_local)
            trace_edges['target'] = trace_edges['target'].map(global_to_local)
            trace_latencies = trace_edges['latency'].astype(float)
            total_lat = trace_latencies.max()
            max_latencies = max_latencies + [total_lat]
            trace_nodes_tensor = torch.tensor(trace_nodes.values, dtype=torch.long)
            edge_weights = trace_edges['trace_integer']
            trace_integer = trace_edges['trace_integer'].iloc[0]
            trace_edges = trace_edges.drop(columns=['timestamp', 'trace_id', 'latency', 'weight', 'trace_integer'])
            trace_edges_tensor = torch.tensor(trace_edges.values, dtype=torch.long).t().contiguous()
            trace_latency_tensor = torch.tensor(trace_latencies.values, dtype=torch.float32)
            targets = torch.tensor(total_lat, dtype=torch.float32)
            edge_weights = torch.tensor(edge_weights.values, dtype=torch.long)
            trace_integer = torch.tensor(trace_integer, dtype=torch.long)
            graph = Data(x=trace_nodes_tensor,edge_index=trace_edges_tensor,edge_attr=edge_weights,trace_integer=trace_integer,y=trace_latency_tensor,targets=targets)
            trace_graphs = trace_graphs + [graph]
        
        
        nodes = nodes[features]
        edges = edges.drop_duplicates(subset=['source', 'target'])
        edge_weights = edges['trace_integer']
        edges = edges.drop(columns=['timestamp', 'trace_id', 'latency', 'weight', 'trace_integer'])
        latency_tensor = torch.tensor(latencies.values, dtype=torch.float32)
        nodes_tensor = torch.tensor(nodes.values, dtype=torch.float32)
        edges_tensor = torch.tensor(edges.values, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights.values, dtype=torch.long)
        targets = torch.tensor(max_latencies, dtype=torch.float32)
        graph = Data(x=nodes_tensor, edge_index=edges_tensor, edge_attr=edge_weights, trace_graphs=trace_graphs, y=latency_tensor, targets=targets)
        graphs = graphs + [graph]
        
        num_nodes = len(node_to_int)
    return graphs, num_nodes
        
def order_data(data_row):
    latencies = data_row['latency']
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
    return data_row

def stamps_to_time(stamps):
    time = []
    for stamp in stamps:
        time = time + [datetime.fromtimestamp(stamp/1000000)\
                       .strftime("%Y-%m-%d %H:%M")]
    return time

def micro_to_mili(latencies):
    return [latency / 1000 for latency in latencies]

def system_graph_processor(path, features):
    nodes, edges, measures, _ = prepare_data(path)
    graphs, num_nodes = prepare_graphs(nodes, edges, features)
    return nodes, graphs, num_nodes, measures

if __name__ == "__main__":
    features = ['cpu_use', 'mem_use_percent', 'mem_use_amount', 'net_send_rate', 'net_receive_rate', 'file_read_rate']
    nodes, edges, measures, trace_to_integer = prepare_data('./A/microservice/test/', scale_features=features)
    
    graphs, num_nodes = prepare_graphs(nodes, edges, features)