# import the necessary packages
from torch.utils.data import Dataset
import torch
from core.backend import open_minian
import numpy as np
from math import ceil
from ml_training import config

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
            E = data['E'].sel(unit_id=unit_ids)
            YrA = data['YrA'].sel(unit_id=unit_ids)
            C = data['C'].sel(unit_id=unit_ids)
            DFF = data['DFF'].sel(unit_id=unit_ids)

            # Normalize the datasets locally
            YrA = YrA / YrA.max(dim='frame')
            C = C / C.max(dim='frame')
            DFF = DFF / DFF.max(dim='frame')

            # We want the data to be preloaded into memory for faster access.
            # Convert into float32 tensors
            YrA = torch.tensor(YrA.values.astype(np.float32)).to(config.DEVICE)
            C = torch.tensor(C.values.astype(np.float32)).to(config.DEVICE)
            DFF = torch.tensor(DFF.values.astype(np.float32)).to(config.DEVICE)
            E = torch.tensor(E.values.astype(np.float32)).to(config.DEVICE)

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

        self.hidden_zeros = torch.zeros(config.HIDDEN_SIZE).to(torch.float32).to(config.DEVICE)
        

    def __len__(self):
        mouse = self.data[self.intermediate_epoch]
        unit_id = mouse.unit_ids[self.small_epoch]
        return len(mouse.samples[unit_id])
    
    def __getitem__(self, idx):
        mouse = self.data[self.intermediate_epoch]
        unit_id = mouse.unit_ids[self.small_epoch]
        start, end = mouse.samples[unit_id][idx]

        sample, target = mouse.get_sample(unit_id, start, end)
        # These are called hidden states but in fact they are the outputs of the GRU
        # Therefore we need to pick the last output of the forward and backward passes, whilst accounting
        # for the fact that we may have 0 inputs.
        hidden = []
        for i in range(len(self.hidden_states)):
            if start == 0:
                forward_hidden = self.hidden_zeros
            else:
                forward_hidden = self.hidden_states[i][start-1, 0, :]
            if end == mouse.E.shape[1]:
                backward_hidden = self.hidden_zeros
            else:
                backward_hidden = self.hidden_states[i][end+1, 1, :]
            hidden.append(torch.stack([forward_hidden, backward_hidden]))

        return sample, hidden, target

    def update_hidden_states(self, hidden_states):
        # Shape (sequence_len, 2, hidden_size)
        self.hidden_states = hidden_states
    
    def get_current_sample(self):
        # Necessary for getting the initial hidden states
        mouse = self.data[self.intermediate_epoch]
        unit_id = mouse.unit_ids[self.small_epoch]

        sample, _ = mouse.get_sample(unit_id, 0, -1)

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
                sample, target = mouse.get_sample(unit_id, 0, -1)

                self.samples.append(sample)
                self.targets.append(target)
        

        
        

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
        self.hidden_zeros = torch.zeros(config.HIDDEN_SIZE).to(torch.float32).to(config.DEVICE)
        

    def __len__(self):
        mouse = self.data[self.intermediate_epoch]
        unit_id = mouse.unit_ids[self.small_epoch]
        return len(mouse.samples[unit_id])
    
    def __getitem__(self, idx):
        mouse = self.data[self.intermediate_epoch]
        unit_id = mouse.unit_ids[self.small_epoch]
        start, end = mouse.samples[unit_id][idx]

        sample, target = mouse.get_sample(unit_id, start, end)

        hidden = []
        for i in range(len(self.hidden_states)):
            if start == 0:
                forward_hidden = self.hidden_zeros
            else:
                forward_hidden = self.hidden_states[i][start-1, 0, :]
            if end == mouse.E.shape[1]:
                backward_hidden = self.hidden_zeros
            else:
                backward_hidden = self.hidden_states[i][end+1, 1, :]
            hidden.append(torch.stack([forward_hidden, backward_hidden]))

        return sample, hidden, target

    def update_hidden_states(self, hidden_states):
        # Shape (sequence_len, 2, hidden_size)
        self.hidden_states = hidden_states
    
    def get_current_sample(self):
        # Necessary for getting the initial hidden states
        mouse = self.data[self.intermediate_epoch]
        unit_id = mouse.unit_ids[self.small_epoch]

        sample, _ = mouse.get_sample(unit_id, 0, -1)

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

        self.unit_to_index = {unit_id: i for i, unit_id in enumerate(self.unit_ids)}
    
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
                self.samples[unit_id].append((start * section_len, (start + 1) * section_len))
            for start in val_indices:
                self.val_samples[unit_id].append((start * section_len, (start + 1) * section_len))

    def get_counts(self):
        total_0 = 0
        total_1 = 0
        for idx, unit_id in enumerate(self.unit_ids):
            E_sample = self.E[idx]
            for sample in self.samples[unit_id]:
                total_0 += torch.sum(E_sample[sample[0]:sample[1]] == 0)
                total_1 += torch.sum(E_sample[sample[0]:sample[1]] == 1)

        return total_0, total_1
    
    def get_sample(self, unit_id, start, end):
        idx = self.unit_to_index[unit_id]
        Yra_sample = self.YrA[idx, start:end]
        C_sample = self.C[idx, start:end]
        DFF_sample = self.DFF[idx, start:end]
        target = self.E[idx, start:end].unsqueeze(-1)

        sample = torch.stack([Yra_sample, C_sample, DFF_sample]).T

        return sample, target

class LocalTransformerDataset:
    def __init__(self, paths_to_unit_ids, section_len=200, rolling=50, slack=50):
        '''
        We will do a bit different approach here the size of each sample will be section_len + 2 * slack. We will slide the window by rolling.
        For the final and first sample we'll pad the input with zeros. We no longer need to keep track of hidden states.
        '''
        self.paths_to_unit_ids = paths_to_unit_ids
        self.data = []
        total_0 = 0
        total_1 = 0
        
        for path, unit_ids in paths_to_unit_ids.items():
            for unit_id in unit_ids:
                minian_data = open_minian(path)
                E = minian_data['E'].sel(unit_id=unit_id)
                YrA = minian_data['YrA'].sel(unit_id=unit_id)
                C = minian_data['C'].sel(unit_id=unit_id)
                DFF = minian_data['DFF'].sel(unit_id=unit_id)

                # Normalize the datasets locally
                YrA = YrA / YrA.max(dim='frame')
                C = C / C.max(dim='frame')
                DFF = DFF / DFF.max(dim='frame')

                # We want the data to be preloaded into memory for faster access.
                # Convert into float32 tensors
                YrA = torch.tensor(YrA.values.astype(np.float32)).to(config.DEVICE)
                C = torch.tensor(C.values.astype(np.float32)).to(config.DEVICE)
                DFF = torch.tensor(DFF.values.astype(np.float32)).to(config.DEVICE)

                input_data = torch.stack([YrA, C, DFF]).T
                output = torch.tensor(E.values.astype(np.float32)).to(config.DEVICE)

                for i in range(0, len(input_data) - (section_len+2*slack), rolling):
                    sample = input_data[i:i+section_len+2*slack]
                    target = output[i+slack:i+section_len+slack]
                    self.data.append((sample, target))
                    total_0 += torch.sum(target == 0)
                    total_1 += torch.sum(target == 1)

        self.weight = torch.tensor([total_0 / total_1]).to(torch.float32)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample, target = self.data[idx]
        return sample, target
    
            
    

def train_val_test_split(paths, val_split=0.1, test_split=0.1) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]]:
    total_len = 0
    path_to_ids = {}
    for path in paths:
        data = open_minian(path)

        all_unit_ids = data['E'].unit_id.values
        verified = data['E'].verified.values.astype(int)
        unit_ids = all_unit_ids[verified==1]
        total_len += len(unit_ids)
        path_to_ids[path] = unit_ids
    
    total_val = int(val_split * total_len)
    total_test = int(test_split * total_len)
    random_picks = np.random.choice(np.arange(total_len), total_val + total_test, replace=False)
    val_picks, test_picks = random_picks[:total_val], random_picks[total_val:]

    train_unit_ids = {}
    val_unit_ids = {}
    test_unit_ids = {}

    boost = 0
    for path, unit_ids in path_to_ids.items():
        for i in range(len(unit_ids)):
            adjusted_idx = i + boost
            if adjusted_idx in val_picks:
                if path not in val_unit_ids:
                    val_unit_ids[path] = []
                val_unit_ids[path].append(unit_ids[i])
            elif adjusted_idx in test_picks:
                if path not in test_unit_ids:
                    test_unit_ids[path] = []
                test_unit_ids[path].append(unit_ids[i])
            else:
                if path not in train_unit_ids:
                    train_unit_ids[path] = []
                train_unit_ids[path].append(unit_ids[i])

        boost += len(unit_ids)

    return train_unit_ids, val_unit_ids, test_unit_ids