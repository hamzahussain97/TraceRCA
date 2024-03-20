# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:20:20 2024

@author: Hamza
"""

from torch.utils.data import Dataset
from torch_geometric.transforms import NormalizeFeatures

class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.transform = NormalizeFeatures()
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx): 
        return self.data_list[idx]
        #return self.transform(data)
