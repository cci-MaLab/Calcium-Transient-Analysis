# import the necessary packages
from torch.utils.data import Dataset
import torch
from core.backend import open_minian
import numpy as np
from math import ceil
from ml_training import config
from torch.nn import ConstantPad2d

class TrainDataset(Dataset):
    def __init__(self, paths: list[str], train_size=None, test_split=0.1, val_split=0.1, section_len=200, stratification=False, experiment_type="cross_session", model_type="gru"):
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

        for i, path in enumerate(self.paths):
            data = open_minian(path)

            all_unit_ids = data['E'].unit_id.values
            verified = data['E'].verified.values.astype(int)
            unit_ids = all_unit_ids[verified==1]
            if train_size is not None and test_split >= 1 and i == 0:
                # randomly select unit_ids of size train_size+test_size
                if experiment_type == "within_session":
                    unit_ids = np.random.choice(unit_ids, train_size+test_split, replace=False)
                else:
                    unit_ids = np.random.choice(unit_ids, train_size, replace=False)
            if train_size is not None and test_split >= 1 and i == 1 and experiment_type == "cross_session":
                unit_ids = np.random.choice(unit_ids, test_split, replace=False)


            # Loading into memory may take up some space but it is necessary for fast access during training
            E = data['E'].sel(unit_id=unit_ids)
            E = torch.tensor(E.values.astype(np.float32)).to(config.DEVICE)
            training_types = config.INPUT
            YrA = None
            C = None
            DFF = None
            for training_type in training_types:
                if training_type == "YrA":
                    YrA = data['YrA'].sel(unit_id=unit_ids)
                    YrA = YrA / YrA.max(dim='frame')
                    YrA = torch.tensor(YrA.values.astype(np.float32)).to(config.DEVICE)
                elif training_type == "C":
                    C = data['C'].sel(unit_id=unit_ids)
                    C = C / C.max(dim='frame')
                    C = torch.tensor(C.values.astype(np.float32)).to(config.DEVICE)
                elif training_type == "DFF":
                    DFF = data['DFF'].sel(unit_id=unit_ids)
                    DFF = DFF / DFF.max(dim='frame')
                    DFF = torch.tensor(DFF.values.astype(np.float32)).to(config.DEVICE)

            self.data.append(MouseData(path, C, DFF, E, YrA, unit_ids))

        
        # Iterate through the MouseData objects and allocate sets for training, validation and testing
        total = 0
        cell_counts = [0]
        for mouse_data in self.data:
            total += len(mouse_data)
            cell_counts.append(total)
        # Total contains the total number of cells in the dataset
        test_count = int(test_split * total) if test_split < 1 else test_split
        test_unit_indices = np.random.choice(np.arange(total), test_count, replace=False)
        test_unit_indices.sort()

        test_unit_ids = [[] for _ in range(len(self.data))]
        for index in test_unit_indices:
            for i, count in enumerate(cell_counts):
                if index < count:
                    test_unit_ids[i-1].append(index - cell_counts[i])
                    break
        
        if experiment_type == "cross_session":
            test_unit_ids = [[] for _ in range(len(self.data))]
            indices = np.arange(len(self.data[-1]))
            test_unit_ids[-1] = indices

        total_0 = 0
        total_1 = 0
        slack = config.SLACK if model_type != "gru" else 0
        for i, mouse_data in enumerate(self.data):
            mouse_data.create_test_ids(test_unit_ids[i])
            mouse_data.create_samples(val_split, section_len, slack=slack, stratification=stratification)

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
        # With non-GRUs we don't need to keep track of hidden states
        if self.hidden_states is not None:
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
    def __init__(self, data, model_type="gru"):
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
    
    def get_test_indices(self):
        indices = {}
        for mouse in self.data:
            if len(mouse.test_unit_ids) > 0:
                indices[mouse.path] = mouse.test_unit_ids
        
        return indices
    
class ValDataset(Dataset):
    def __init__(self, data):
        '''
        This dataset will be provided through random sub-selection from GRUDataset
        '''
        self.data= data
        self.hidden_states = None

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
        if self.hidden_states is not None:
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

    def create_samples(self, val_split, section_len, slack=0, stratification=False):
        length = self.E.shape[1]
        self.slack = slack
        for unit_id in self.unit_ids:
            self.samples[unit_id] = []
            self.val_samples[unit_id] = []
            all_samples = []
            for i in range(ceil(self.E.shape[1] / section_len)):
                start, end = i * section_len, (i + 1) * section_len

                if stratification:
                    E = self.E[self.unit_to_index[unit_id], start:end]
                    C = self.C[self.unit_to_index[unit_id], start:end]
                    EC = E + C
                    if torch.sum(EC) == 0:
                        continue
                if start-slack < 0 or end+slack > length:
                    continue
                all_samples.append((start, end))

            # Pick val_split indices to be used for validation
            val_indices = np.random.choice(np.arange(len(all_samples)), int(val_split * len(all_samples)), replace=False)
            val_indices.sort()
            indices = np.setdiff1d(np.arange(len(all_samples)), val_indices)
            train_samples = [all_samples[i] for i in indices]
            val_samples = [all_samples[i] for i in val_indices]
            for start, end in train_samples:
                self.samples[unit_id].append((start, end))
            for start, end in val_samples:
                self.val_samples[unit_id].append((start, end))

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
        list_of_samples = []
        slack = self.slack if (start != 0 and end != -1) else 0
        if self.YrA is not None:
            Yra_sample = self.YrA[idx, start-slack:end+slack]
            list_of_samples.append(Yra_sample)
        if self.C is not None:
            C_sample = self.C[idx, start-slack:end+slack]
            list_of_samples.append(C_sample)
        if self.DFF is not None:
            DFF_sample = self.DFF[idx, start-slack:end+slack]
            list_of_samples.append(DFF_sample)
        target = self.E[idx, start:end].unsqueeze(-1)

        sample = torch.stack(list_of_samples).T

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


def extract_data(minian_data, unit_id, slack=50):
    padding = ConstantPad2d((0, 0, slack, slack), 0)
    E = minian_data['E'].sel(unit_id=unit_id)
    
    training_types = config.INPUT
    YrA = None
    C = None
    DFF = None
    input_data = []
    for training_type in training_types:
        if training_type == "YrA":
            YrA = minian_data['YrA'].sel(unit_id=unit_id)
            YrA = YrA / YrA.max(dim='frame')
            YrA = torch.tensor(YrA.values.astype(np.float32)).to(config.DEVICE)
            input_data.append(YrA)
        elif training_type == "C":
            C = minian_data['C'].sel(unit_id=unit_id)
            C = C / C.max(dim='frame')
            C = torch.tensor(C.values.astype(np.float32)).to(config.DEVICE)
            input_data.append(C)
        elif training_type == "DFF":
            DFF = minian_data['DFF'].sel(unit_id=unit_id)
            DFF = DFF / DFF.max(dim='frame')
            DFF = torch.tensor(DFF.values.astype(np.float32)).to(config.DEVICE)
            input_data.append(DFF)

    input_data = torch.stack(input_data).T
    input_data = padding(input_data)
    output = torch.tensor(E.values.astype(np.float32)).to(config.DEVICE)

    return input_data, output