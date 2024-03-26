# import the necessary packages
from torch.utils.data import Dataset
import torch
from core import open_minian
import numpy as np
from math import ceil

class GRUDataset(Dataset):
    def __init__(self, path, test_split=0.1, val_split=0.1, section_len=200):
        '''
        We want to create a truncated version of the dataset for training purposes. For the time being, we will
        split the dataset into chunks of length of 200. We will slide the window by 200.

        However since we are using a bidirectional GRU, we will need to keep the hidden states of the forward and backward
        passes. We would be able to pass the hidden states easily if we were just doing a forward pass. Unfortunately due to bidirecionality
        we have a chicken and egg problem were the backward pass will be less reliable because it will be dependent on cell outputs
        that have not been calculated yet.

        To address this we will have to separate epochs. A small_epoch that will denote passing through the data of a single cell and
        a large_epoch that will denote a pass through all cells. After each small_epoch we will save the hidden states of the forward and backward
        passes of the updated model. The intuition here is that over time the hidden states will converge to a stable representation of past and future
        events.
        '''
        data = open_minian(path)
        # Loading into memory may take up some space but it is necessary for fast access during training
        self.E = data['E'].load()
        self.YrA = data['YrA'].load()
        self.C = data['C'].load()
        self.DDF = data['DDF'].load()

        self.section_len = section_len

        self.hidden_states = None

        self.samples = {}

        self.unit_ids = self.E.unit_ids.values
        self.test_unit_ids = np.random.choice(self.unit_ids, int(test_split * len(self.unit_ids)), replace=False)
        self.test_unit_ids.sort()
        self.unit_ids = np.setdiff1d(self.unit_ids, self.test_unit_ids)

        self.small_epoch = 0
        for unit_id in self.unit_ids:
            self.samples[unit_id] = []
            self.val_samples[unit_id] = []
            # Pick val_split indices to be used for validation
            val_indices = np.random.choice(np.arange(ceil(self.E.shape[1] / section_len)), int(val_split * self.E.shape[1] / section_len), replace=False)
            val_indices.sort()
            indices = np.setdiff1d(np.arange(ceil(self.E.shape[1] / section_len)), val_indices)
            for start in indices:
                self.samples[unit_id].append((start * section_len, (start + 1) * section_len))
            for start in val_indices:
                self.val_samples[unit_id].append((start * section_len, (start + 1) * section_len))
        
        

    def __len__(self):
        return len(self.samples[self.small_epoch])
    
    def __getitem__(self, idx):
        unit_id = self.unit_ids[self.small_epoch]
        start, end = self.samples[unit_id][idx]

        Yra_sample = self.YrA.sel(unit_id=unit_id, frame=slice(start, end)).values
        C_sample = self.C.sel(unit_id=unit_id, frame=slice(start, end)).values
        DDF_sample = self.DDF.sel(unit_id=unit_id, frame=slice(start, end)).values

        sample = torch.as_tensor(np.concatenate([Yra_sample, C_sample, DDF_sample], axis=0))
        forward_hidden = self.hidden_states[start, 0, :]
        backward_hidden = self.hidden_states[end, 1, :]

        return sample, forward_hidden, backward_hidden

    def update_hidden_states(self, hidden_states):
        # Shape (sequence_len, 2, hidden_size)
        self.hidden_states = hidden_states

    def get_event_ratio(self):
        E = self.E.values
        return np.sum(E) / E.size
    
    def get_current_sample(self):
        # Necessary for getting the initial hidden states
        unit_id = self.unit_ids[self.small_epoch]

        Yra_sample = self.YrA.sel(unit_id=unit_id).values
        C_sample = self.C.sel(unit_id=unit_id).values
        DDF_sample = self.DDF.sel(unit_id=unit_id).values

        sample = torch.as_tensor(np.concatenate([Yra_sample, C_sample, DDF_sample], axis=0))

        return sample
    
    def get_data(self):
        return self.E, self.YrA, self.C, self.DDF

    
class TestDataset(Dataset):
    def __init__(self, data, unit_ids):
        '''
        This dataset will be provided by unit_ids through random sub-selection from GRUDataset
        '''
        self.E, self.YrA, self.C, self.DDF = data

        self.unit_ids = unit_ids

        self.samples = {}

        for unit_id in self.unit_ids:
            Yra_sample = self.YrA.sel(unit_id=unit_id).values
            C_sample = self.C.sel(unit_id=unit_id).values
            DDF_sample = self.DDF.sel(unit_id=unit_id).values

            self.samples[unit_id] = np.concatenate([Yra_sample, C_sample, DDF_sample], axis=0)
        

        
        

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        unit_id = self.unit_ids[idx]
        
        sample = torch.as_tensor(self.samples[unit_id])

        return sample
    
class ValDataset(Dataset):
    def __init__(self, data, indices, unit_ids):
        '''
        This dataset will be provided through random sub-selection from GRUDataset
        '''
        self.E, self.YrA, self.C, self.DDF = data

        self.indices = indices
        self.unit_ids = unit_ids    

        self.hidden_states = None
        

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        unit_id = self.indices[idx]    

        start, end = self.indices[unit_id]

        Yra_sample = self.YrA.sel(unit_id=unit_id, frame=slice(start, end)).values
        C_sample = self.C.sel(unit_id=unit_id, frame=slice(start, end)).values
        DDF_sample = self.DDF.sel(unit_id=unit_id, frame=slice(start, end)).values

        sample = torch.as_tensor(np.concatenate([Yra_sample, C_sample, DDF_sample], axis=0))
        forward_hidden = self.hidden_states[start, 0, :]
        backward_hidden = self.hidden_states[end, 1, :]

        return sample, forward_hidden, backward_hidden
    
    def update_hidden_states(self, hidden_states):
        # Shape (sequence_len, 2, hidden_size)
        self.hidden_states = hidden_states

    def get_current_sample(self):
        # Necessary for getting the initial hidden states
        unit_id = self.unit_ids[self.small_epoch]

        Yra_sample = self.YrA.sel(unit_id=unit_id).values
        C_sample = self.C.sel(unit_id=unit_id).values
        DDF_sample = self.DDF.sel(unit_id=unit_id).values

        sample = torch.as_tensor(np.concatenate([Yra_sample, C_sample, DDF_sample], axis=0))

        return sample