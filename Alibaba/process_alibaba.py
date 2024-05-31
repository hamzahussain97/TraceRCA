# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:05:05 2024

@author: Hamza
"""
import torch
from tqdm import tqdm
import numpy as np
from collections import Counter


def process_alibaba(path):
    data_list = torch.load(path+"/data/full_span_data_list.pt")
    runtime2graph = torch.load(path+"/data/runtime2spangraph_map.pt")
    
    unique_ms_ids = np.unique(
        np.array(
            [
                ms_id.item()
                for runtime_id in runtime2graph.keys()
                for ms_id in runtime2graph[runtime_id]["ms_id"]
            ]
        ).ravel()
    )
    
    # Create a dictionary mapping each unique ms_id to a unique integer
    ms_id_to_int = {ms_id: i for i, ms_id in enumerate(unique_ms_ids)}
    graphs = []
    entry_ids = []
    for graph in tqdm(data_list):
        x = graph.x
        ms_ids = graph.cat_X
        # Map ms_ids to their corresponding integers
        mapped_ms_ids = torch.tensor([ms_id_to_int[ms_id.item()] for ms_id in ms_ids], dtype=torch.float)
        
        # add ms_ids at the end of the node features
        x = torch.cat((x, mapped_ms_ids.unsqueeze(1)), axis=1)
        graph.x = x
        graph.y = np.log10(graph.y)
        graph.trace_lat = graph.y
        #if graph.entry_id == 1:
        graphs.append(graph)
        entry_ids.append(graph.entry_id.item())
    total_traces = len(graphs)
    print(f"Total number of traces: {total_traces}")
    entry_ids = Counter(entry_ids)

    return data_list, graphs, ms_id_to_int, ms_id_to_int
'''
path = './Alibaba/'
data, graphs, global_map, measures, entry_ids = process_alibaba('.')
'''