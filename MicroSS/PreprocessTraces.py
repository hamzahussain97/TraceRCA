# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:03:36 2024

@author: Hamza
"""

import pickle
import os
import re
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

def process_traces(path):
    data_dir = Path(path)
    data = pd.DataFrame()
    all_files = os.listdir(path)    
    for trace_file in tqdm(all_files):
        df = pd.read_csv(os.path.join(path, trace_file))
        df = df[['timestamp', 'trace_id', 'service_name', 'span_id', 'parent_id', 'start_time', 'end_time']]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        df['latency'] = (df['end_time'] - df['start_time']).dt.total_seconds() * 1000
        df = df.drop(columns=['start_time', 'end_time'])
        data = pd.concat([data,df])
    print("Creating Dictionary")
    span_id_to_service = dict(zip(data['span_id'], data['service_name']))
    span_id_to_service[0] = 'user'
    print("Mapping Parent ID to Names")
    data['parent_id'] = data['parent_id'].map(span_id_to_service)
    data.reset_index(drop=True, inplace=True)
    data = data.drop(columns=['span_id'])
    print("Creating s_t column")
    data['s_t'] = pd.Series(zip(data['parent_id'], data['service_name']))
    data.drop(columns=['parent_id', 'service_name'])
    print("Grouping by Trace IDs")
    data = data.groupby('trace_id').agg(lambda x: x.tolist())
    print("Fixing timestamp column")
    data['timestamp'] = data['timestamp'].apply(lambda x: x[0])
    # Reset index to bring 'trace_id' back as a column
    data.reset_index(inplace=True)
    return data

if __name__ == "__main__":
    data = process_traces('./trace_split/trace')
    # Extract date from 'timestamp' column
    
    data['date'] = data['timestamp'].dt.date
    
    # Group DataFrame by date
    grouped = data.groupby('date')
    
    # Iterate over groups and write to separate files
    for date, group in grouped:
        print(date)
        # Remove the 'date' column as it's no longer needed
        group.drop(columns=['date'], inplace=True)
        
        # Write group to a file
        csv_file = f"data_{date}.csv"  # Adjust the filename format as needed
        pkl_file = f"data_{date}.pkl"
        group.to_csv(os.path.join('./traces/', csv_file), index=False, header=group.columns)
        group.to_pickle(pkl_file)
