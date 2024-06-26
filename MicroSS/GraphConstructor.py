# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:32:25 2024

@author: Hamza
"""

import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('../')
from Preprocess import normalize_by_trace, normalize
import pickle
from torch_geometric.data import Data


def load_metrics(path):
    data_dir = Path(path)
    file_list = list(map(str, data_dir.glob("*07-15.csv")))
    metric_dict = {}
    for file_name in tqdm(file_list):
        file_name = Path(file_name)
        service_name = file_name.stem.split('_')[0]
        df = pd.read_csv(file_name)
        df.set_index('timestamp', inplace=True)
        metric_dict[service_name] = df
    return metric_dict

def load_traces(path):
    data_dir = Path(path)
    patterns = ["*2021-07-01.pkl"]
    file_list = []
    for pattern in patterns:
        matching_files = list(map(str, data_dir.glob(pattern)))
        file_list.extend(matching_files)
    traces = pd.DataFrame()
    for file_name in tqdm(file_list):
        with open(file_name, 'rb') as file:
            file_data = pd.read_pickle(file)
        df = pd.DataFrame(file_data)
        traces = pd.concat([traces,df])
    traces = traces.reset_index(drop=True)
    
    print('Normalizing traces')
    traces, stats = normalize_by_trace(traces)
    for column in ['mean', 'std', 'maximum', 'minimum']:
        traces, feature_mean, feature_std = normalize(traces, column)
    return traces, stats
        
def construct_graph(metrics, trace, global_map):
    #Create dataframe of edges
    edges = pd.DataFrame(trace['s_t'], columns = ['source', 'target'])
    edges['timestamp'] = trace['rounded_timestamp']
    
    edge_attr = {'mean': trace['mean'],\
                 'std': trace['std'],\
                 'max': trace['maximum'],\
                 'min': trace['minimum']} 
        
    edge_attr = pd.DataFrame(edge_attr)
    
    y_edge_features = pd.DataFrame({
         'latency': trace['latency']
     })
    
    original = pd.DataFrame({'orignal': trace['original_latency']})
    trace_lat = y_edge_features['latency'].max()
    
    nodes = pd.DataFrame([])
    nodes_list = []
    for timestamp, node in zip(edges['timestamp'], edges['target']):
        node_metrics = metrics[node]
        try:
            current_metrics = node_metrics.loc[str(timestamp)].to_frame().T
        except KeyError:
            # Handle the KeyError by returning immediately from the function
            return 
        current_metrics['node_name'] = node
        nodes_list.append(current_metrics)
    nodes = pd.concat(nodes_list)
    
    edges = edges.drop(columns=['timestamp'])
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
    
    #Find all unique node names
    unique_nodes = pd.concat([edges['source'], edges['target']]).unique()
    #Map node names to integers
    node_to_int = {node: i for i, node in enumerate(unique_nodes)}
    nodes['node_id'] = nodes['node_name'].map(node_to_int)
    edges['source'] = edges['source'].map(node_to_int)
    edges['target'] = edges['target'].map(node_to_int)
    nodes = nodes.sort_values(by='node_id')
    nodes = nodes.drop(columns=['node_id'])
    
    
    
    nodes['node_name'] = nodes['node_name'].map(global_map)
    
    #Convert to tensors
    nodes_tensor = torch.tensor(nodes.values, dtype=torch.float32)
    x = nodes_tensor[:,:-1]
    norms = torch.norm(x, dim=1, keepdim=True)
    norms[norms == 0] = 1e-8
    # Normalize each row
    x = x.div(norms)
    nodes_tensor = torch.cat((x, nodes_tensor[:,-1].unsqueeze(1)), axis=1)
    
    edges_tensor = torch.tensor(edges.values, dtype=torch.long).t().contiguous()
    max_value = edges_tensor.max().item()
    node_size = nodes_tensor.shape[0]
    #Check for Invalid Traces
    if max_value >= node_size:
        return
    y_edge_tensor = torch.tensor(y_edge_features.values, dtype=torch.float32).squeeze(dim=1)
    original = torch.tensor(original.values, dtype=torch.float32)
    trace_lat_tensor = torch.tensor(trace_lat, dtype=torch.float32)
    edge_attr_tensor = torch.tensor(edge_attr.values, dtype=torch.float32)
    graph = Data(x=nodes_tensor, edge_index=edges_tensor, edge_attr=edge_attr_tensor, y=y_edge_tensor, trace_lat=trace_lat_tensor)
    graph.original = original
    return graph

def get_global_map(metric_dict):
    global_map = {key: idx for idx, key in enumerate(metric_dict.keys())}
    num_of_services  = max(global_map.values())
    global_map['user'] = num_of_services + 1
    return global_map

def process_micross(path):
    tqdm.pandas()
    metric_dict = load_metrics(path+'data/')
    global_map = get_global_map(metric_dict)
    traces, stats = load_traces(path+'traces/')
    graphs = traces.progress_apply(lambda trace: construct_graph(metric_dict, trace, global_map), axis=1)
    # Filter out None values
    graphs = graphs[graphs.notna()].reset_index(drop=True)
    return traces, graphs, global_map, stats

if __name__ == "__main__":      
    traces, graphs, global_map, stats = process_micross('./')

        
        
    