# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:20:20 2024

@author: Hamza
"""

from torch.utils.data import Dataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import DataLoader


class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.transform = NormalizeFeatures()
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx): 
        return self.data_list[idx]
        #return self.transform(data)


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super(CustomDataLoader, self).__init__(dataset, *args, **kwargs)

    def __iter__(self):
        return CustomIterator(self.dataset, self.batch_size, collate_fn=self.collate_fn)

class CustomIterator:
    def __init__(self, dataset, batch_size, collate_fn):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx + self.batch_size > len(self.dataset):
            raise StopIteration
        else:
            # Define the start and end indices for the current batch
            start_idx = self.idx
            end_idx = min(start_idx + self.batch_size, len(self.dataset))

            # Get the data objects within the window
            batch = [self.dataset[i] for i in range(start_idx, end_idx)]

            # Update the index for the next batch
            self.idx += 1

            return self.collate_fn(batch)