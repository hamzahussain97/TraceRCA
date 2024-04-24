# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:43:28 2024

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

def prepare_metric_data(path, output_path, daterange):
    service_names = ['dbservice1', 'dbservice2', 'logservice1', 'logservice2',\
                     'mobservice1', 'mobservice2', 'redis', \
                     'redisservice1', 'redisservice2', 'webservice1',\
                     'webservice2', 'zookeeper']
    # List all files in the directory
    all_files = os.listdir(path)
    dfs = []
    file_names = []
    for service in service_names:
        # Filter files belonging to the specified service
        service_files = [file for file in all_files if service in file]
        # Filter files within the specified date range
        date_range_files = [file for file in service_files if daterange in file]
        for idx, file in enumerate(date_range_files):
            df = read_file(os.path.join(path, file))
            dataframes.append(df)
        
        concatenated_df = pd.concat(dataframes, axis=1)
        # Reset the index to move 'timestamp' column back to regular column
        concatenated_df.reset_index(inplace=True)
        concatenated_df = concatenated_df.sort_values(by='timestamp')
        concatenated_df.fillna(0,inplace=True)
        # Construct the output file name
        output_file_name = f"{service}_{daterange}.csv"
        # Write the concatenated DataFrame to a CSV file
        #concatenated_df.to_csv(os.path.join(output_path, output_file_name), index=False)
        dfs.append(concatenated_df) 
        file_names.append(output_file_name)
    return dfs, file_names

def read_file(file):
    df = pd.read_csv(file)
    # Set 'timestamp' column as index
    df = df.drop_duplicates(['timestamp'])
    df.set_index('timestamp', inplace=True)
    # Define the regex pattern to extract the metric name
    pattern = r'\d+\.\d+\.\d+\.\d+_(\w+)_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}\.csv'
    
    # Search for the pattern in the file name
    match = re.search(pattern, file)
    if match:
        metric_name = match.group(1)
    else:
        # Handle the case when the pattern is not found
        metric_name = "UnknownMetricName"
    df.rename(columns={'value': metric_name}, inplace=True)
    return df

def filter_common_columns(dataframes):
    column_sets = [set(df.columns) for df in dataframes]
    
    # Find the intersection of column names across all dataframes
    common_columns = set.intersection(*column_sets)
    # Convert the set to a sorted list
    sorted_common_columns = sorted(common_columns)
    
    # Filter each dataframe to keep only the common columns
    dfs_filtered = [df.loc[:, sorted_common_columns] for df in dataframes]
    return dfs_filtered

def filter_time_range(dataframes):
    # List to store the max and min timestamps for each DataFrame
    max_timestamps = []
    min_timestamps = []
    # Iterate through each DataFrame in dataframes
    for df in dataframes:
        # Extract the 'timestamp' column
        timestamps = df['timestamp']
        # Find the maximum and minimum timestamps
        max_timestamp = timestamps.max()
        min_timestamp = timestamps.min()
        # Append the max and min timestamps to the timestamps_info list
        max_timestamps.append(max_timestamp)
        min_timestamps.append(min_timestamp)
    # Find the maximum of all minimum values and minimum of all maximum values
    max_of_min_timestamps = max(min_timestamps)
    min_of_max_timestamps = min(max_timestamps)
    
    # Filter all the DataFrames to have the timestamp range accordingly
    dfs_filtered = [df[(df['timestamp'] >= max_of_min_timestamps) & (df['timestamp'] <= min_of_max_timestamps)] for df in dataframes]
    return dfs_filtered

# Define a function to round timestamps to the nearest 30-second interval
def round_to_30_seconds(timestamp):
    return pd.Timestamp(timestamp, unit='ms').round('30s')

def sum_by_timewindow(dataframes):
    # Iterate through each DataFrame in dfs_filtered
    summed_dfs = []
    for df in dataframes:
        # Round the timestamps to the nearest 30-second interval
        df['rounded_timestamp'] = df['timestamp'].apply(round_to_30_seconds)
        
        # Group rows based on rounded timestamps and sum the values within each group
        summed_df = df.groupby('rounded_timestamp').sum()
        
        # Reset index to convert the grouped timestamp column back to a regular column
        summed_df.reset_index(inplace=True)
        
        # Reorder columns to have 'timestamp' as the first column
        summed_df = summed_df[['rounded_timestamp'] + [col for col in summed_df.columns if col != 'rounded_timestamp']]
        
        summed_df.drop(columns=['timestamp'], inplace=True)
        
        # Rename the 'rounded_timestamp' column to 'timestamp'
        summed_df.rename(columns={'rounded_timestamp': 'timestamp'}, inplace=True)
        
        # Append the summed DataFrame to the list
        summed_dfs.append(summed_df)
    return summed_dfs

def write_to_file(dataframes, file_names, output_path):
    for df, file_name in zip(dataframes, file_names):
        #Write the DataFrame to a CSV file
        df.to_csv(os.path.join(output_path, file_name), header = df.columns, index=False)

def concat_dateranges(dfs1, dfs2):
    dataframes = []
    for df1, df2 in zip(dfs1, dfs2):
        df = pd.concat([df1, df2], sort=True)
        df = df.drop_duplicates(['timestamp'])
        dataframes.append(df)
    return dataframes

if __name__ == "__main__":
    daterange = '2021-07-01_2021-07-15'
    dataframes, file_names = prepare_metric_data('./metric_split/metric', './data/', daterange)
    dataframes = filter_common_columns(dataframes)
    dataframes = filter_time_range(dataframes)
    dataframes = sum_by_timewindow(dataframes)
    write_to_file(dataframes, file_names, './data/')
    daterange = '2021-07-15_2021-07-31'
    dataframes_2, file_names_2 = prepare_metric_data('./metric_split/metric', './data/', daterange)
    dataframes_2 = filter_common_columns(dataframes_2)
    dataframes_2 = filter_time_range(dataframes_2)
    dataframes_2 = sum_by_timewindow(dataframes_2)
    write_to_file(dataframes_2, file_names_2, './data/')
    dataframes = concat_dateranges(dataframes, dataframes_2)
    