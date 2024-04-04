# import the necessary packages
from torch.utils.data import Dataset
import torch
from core.backend import open_minian
import numpy as np
from math import ceil

class GRUDataset(Dataset):
    def __init__(self, paths: list[str], test_split=0.1, val_split=0.1, section_len=200):
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
        self.paths = paths
        self.data = []
        self.hidden_states = None
        self.section_len = section_len

        self.intermediate_epoch = 0
        self.small_epoch = 0

        for path in self.paths:
            data = open_minian(path)

            all_unit_ids = data['E'].unit_id.values
            verified = data['E'].verified.values.astype(int)
            unit_ids = all_unit_ids[verified==1]

            # Loading into memory may take up some space but it is necessary for fast access during training
            E = data['E'].sel(unit_id=unit_ids).load()
            YrA = data['YrA'].sel(unit_id=unit_ids).load()
            C = data['C'].sel(unit_id=unit_ids).load()
            DFF = data['DFF'].sel(unit_id=unit_ids).load()

            # Normalize the datasets locally
            YrA = YrA / YrA.max(dim='frame')
            C = C / C.max(dim='frame')
            DFF = DFF / DFF.max(dim='frame')

            self.data.append(MouseData(path, C, DFF, E, YrA, unit_ids))

        
        # Iterate through the MouseData objects and allocate sets for training, validation and testing
        total = 0
        cell_counts = [0]
        for mouse_data in self.data:
            total += len(mouse_data)
            cell_counts.append(total)
        # Total contains the total number of cells in the dataset
        test_unit_indices = np.random.choice(np.arange(total), int(test_split * total), replace=False)
        test_unit_indices.sort()

        test_unit_ids = [[] for _ in range(len(self.data))]
        for index in test_unit_indices:
            for i, count in enumerate(cell_counts):
                if index < count:
                    test_unit_ids[i-1].append(index - cell_counts[i])
                    break
        
        total_0 = 0
        total_1 = 0
        for i, mouse_data in enumerate(self.data):
            mouse_data.create_test_ids(test_unit_ids[i])
            mouse_data.create_samples(val_split, section_len)

            sub_total_0, sub_total_1 = mouse_data.get_counts()
            total_0 += sub_total_0
            total_1 += sub_total_1


        self.weight = torch.tensor([total_0 / total_1]).to(torch.float32)
        
        

    def __len__(self):
        mouse = self.data[self.intermediate_epoch]
        unit_id = mouse.unit_ids[self.small_epoch]
        return len(mouse.samples[unit_id])
    
    def __getitem__(self, idx):
        mouse = self.data[self.intermediate_epoch]
        unit_id = mouse.unit_ids[self.small_epoch]
        start, end = mouse.samples[unit_id][idx]

        Yra_sample = mouse.YrA.sel(unit_id=unit_id, frame=slice(start, end)).values
        C_sample = mouse.C.sel(unit_id=unit_id, frame=slice(start, end)).values
        DFF_sample = mouse.DFF.sel(unit_id=unit_id, frame=slice(start, end)).values

        sample = torch.as_tensor(np.stack([Yra_sample, C_sample, DFF_sample]).T).to(torch.float32)
        forward_hidden = self.hidden_states[start, 0, :].to(torch.float32)
        backward_hidden = self.hidden_states[end, 1, :].to(torch.float32)
        hidden = torch.stack([forward_hidden, backward_hidden])

        target = torch.as_tensor(mouse.E.sel(unit_id=unit_id, frame=slice(start, end)).values).unsqueeze(-1).to(torch.float32)

        return sample, hidden, target

    def update_hidden_states(self, hidden_states):
        # Shape (sequence_len, 2, hidden_size)
        self.hidden_states = hidden_states
    
    def get_current_sample(self):
        # Necessary for getting the initial hidden states
        mouse = self.data[self.intermediate_epoch]
        unit_id = mouse.unit_ids[self.small_epoch]

        Yra_sample = mouse.YrA.sel(unit_id=unit_id).values
        C_sample = mouse.C.sel(unit_id=unit_id).values
        DFF_sample = mouse.DFF.sel(unit_id=unit_id).values

        sample = torch.as_tensor(np.stack([Yra_sample, C_sample, DFF_sample]).T).to(torch.float32)

        return sample
    
    def get_data(self):
        return self.data
    
    def get_mouse_cell_count(self):
        return len(self.data[self.intermediate_epoch])
    
    def get_training_steps(self):
        total = 0
        for mouse in self.data:
            for unit_id in mouse.unit_ids:
                total += len(mouse.samples[unit_id])
        return total

    
class TestDataset(Dataset):
    def __init__(self, data):
        '''
        This dataset will be provided by unit_ids through random sub-selection from GRUDataset
        '''
        self.data = data

        self.samples = []
        self.targets = []

        for mouse in self.data:
            for unit_id in mouse.test_unit_ids:
                Yra_sample = mouse.YrA.sel(unit_id=unit_id).values
                C_sample = mouse.C.sel(unit_id=unit_id).values
                DFF_sample = mouse.DFF.sel(unit_id=unit_id).values

                self.samples.append(np.stack([Yra_sample, C_sample, DFF_sample]).T)
                self.targets.append(torch.as_tensor(mouse.E.sel(unit_id=unit_id).values).unsqueeze(-1).to(torch.float32))
        

        
        

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = torch.as_tensor(self.samples[idx]).to(torch.float32)
        target = self.targets[idx]
        return sample, target
    
class ValDataset(Dataset):
    def __init__(self, data):
        '''
        This dataset will be provided through random sub-selection from GRUDataset
        '''
        self.data= data

        self.small_epoch = 0
        self.intermediate_epoch = 0
        

    def __len__(self):
        mouse = self.data[self.intermediate_epoch]
        unit_id = mouse.unit_ids[self.small_epoch]
        return len(mouse.samples[unit_id])
    
    def __getitem__(self, idx):
        mouse = self.data[self.intermediate_epoch]
        unit_id = mouse.unit_ids[self.small_epoch]
        start, end = mouse.samples[unit_id][idx]

        Yra_sample = mouse.YrA.sel(unit_id=unit_id, frame=slice(start, end)).values
        C_sample = mouse.C.sel(unit_id=unit_id, frame=slice(start, end)).values
        DFF_sample = mouse.DFF.sel(unit_id=unit_id, frame=slice(start, end)).values

        sample = torch.as_tensor(np.stack([Yra_sample, C_sample, DFF_sample]).T).to(torch.float32)
        forward_hidden = self.hidden_states[start, 0, :].to(torch.float32)
        backward_hidden = self.hidden_states[end, 1, :].to(torch.float32)
        hidden = torch.stack([forward_hidden, backward_hidden])

        target = torch.as_tensor(mouse.E.sel(unit_id=unit_id, frame=slice(start, end)).values).unsqueeze(-1).to(torch.float32)

        return sample, hidden, target

    def update_hidden_states(self, hidden_states):
        # Shape (sequence_len, 2, hidden_size)
        self.hidden_states = hidden_states
    
    def get_current_sample(self):
        # Necessary for getting the initial hidden states
        mouse = self.data[self.intermediate_epoch]
        unit_id = mouse.unit_ids[self.small_epoch]

        Yra_sample = mouse.YrA.sel(unit_id=unit_id).values
        C_sample = mouse.C.sel(unit_id=unit_id).values
        DFF_sample = mouse.DFF.sel(unit_id=unit_id).values

        sample = torch.as_tensor(np.stack([Yra_sample, C_sample, DFF_sample]).T).to(torch.float32)

        return sample
    
    def get_mouse_cell_count(self):
        return len(self.data[self.intermediate_epoch])
    

    def get_val_steps(self):
        total = 0
        for mouse in self.data:
            for unit_id in mouse.unit_ids:
                total += len(mouse.val_samples[unit_id])
        return total
    


class MouseData:
    def __init__(self, path, C, DFF, E, YrA, unit_ids):
        self.path = path

        self.C = C
        self.DFF = DFF
        self.E = E
        self.YrA = YrA

        self.unit_ids = unit_ids
        self.test_unit_ids = []

        self.samples = {}
        self.val_samples = {}
    
    def __len__(self):
        return len(self.unit_ids)

    def create_test_ids(self, indices):
        self.test_unit_ids = self.unit_ids[indices]
        self.unit_ids = np.setdiff1d(self.unit_ids, self.test_unit_ids)

    def create_samples(self, val_split, section_len):
        for unit_id in self.unit_ids:
            self.samples[unit_id] = []
            self.val_samples[unit_id] = []
            # Pick val_split indices to be used for validation
            val_indices = np.random.choice(np.arange(ceil(self.E.shape[1] / section_len)), int(val_split * self.E.shape[1] / section_len), replace=False)
            val_indices.sort()
            indices = np.setdiff1d(np.arange(ceil(self.E.shape[1] / section_len)), val_indices)
            for start in indices:
                self.samples[unit_id].append((start * section_len, (start + 1) * section_len - 1)) # -1 as xarray slicing is inclusive
            for start in val_indices:
                self.val_samples[unit_id].append((start * section_len, (start + 1) * section_len - 1))

    def get_counts(self):
        total_0 = 0
        total_1 = 0
        for unit_id in self.unit_ids:
            E_sample = self.E.sel(unit_id=unit_id).values
            for sample in self.samples[unit_id]:
                total_0 += np.sum(E_sample[sample[0]:sample[1]] == 0)
                total_1 += np.sum(E_sample[sample[0]:sample[1]] == 1)

        return total_0, total_1