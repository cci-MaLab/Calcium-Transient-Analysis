# import the necessary packages
from torch.utils.data import Dataset
import torch
from caltrig.core.backend import open_minian
import numpy as np
from ml_training import config
from torch.nn import ConstantPad2d

class CustomDataset(Dataset):
    def __init__(self, paths_to_unit_ids, section_len=200, rolling=50, slack=50, only_events=False):
        '''
        We will do a bit different approach here the size of each sample will be section_len + 2 * slack. We will slide the window by rolling.
        For the final and first sample we'll pad the input with zeros. We no longer need to keep track of hidden states.
        '''
        self.paths_to_unit_ids = paths_to_unit_ids
        self.data = []
        total_0 = 0
        total_1 = 0
        

        for path, unit_ids in paths_to_unit_ids.items():
            minian_data = open_minian(path)
            for unit_id in unit_ids:
                input_data, output = extract_data(minian_data, unit_id, slack)

                for i in range(0, len(input_data) - (section_len+2*slack)+1, rolling):
                    sample = input_data[i:i+section_len+2*slack]
                    target = output[i:i+section_len]
                    if only_events:
                        if torch.sum(target) == 0:
                            continue
                    self.data.append((sample, target))
                    total_0 += torch.sum(target == 0)
                    total_1 += torch.sum(target == 1)

        self.weight = torch.tensor([total_0 / total_1]).to(torch.float32)
                                    


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample, target = self.data[idx]
        return sample, target
    


    
def train_val_test_split_custom(train_paths, test_paths, total_train=1, val_split=0.1, total_test=5) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]]:
    # This is slightly different from the non-custom version, train_paths will be used for training and validation and test_paths will be used for testing
    total_len = 0
    path_to_ids = {}
    # The val will be 10% of the total train with a minimum of 1
    total_val = max(int(val_split * total_train), 1)
    total_train -= total_val

    for path in train_paths:
        data = open_minian(path)

        all_unit_ids = data['E'].unit_id.values
        verified = data['E'].verified.values.astype(int)
        unit_ids = all_unit_ids[verified==1]
        total_len += len(unit_ids)
        path_to_ids[path] = unit_ids
    
    total_picks = np.random.choice(np.arange(total_len), total_train+total_val, replace=False)
    train_picks, val_picks = total_picks[:total_train], total_picks[total_train:]

    train_unit_ids = {}
    val_unit_ids = {}

    boost = 0
    for path, unit_ids in path_to_ids.items():
        for i in range(len(unit_ids)):
            adjusted_idx = i + boost
            if adjusted_idx in train_picks:
                if path not in train_unit_ids:
                    train_unit_ids[path] = []
                train_unit_ids[path].append(unit_ids[i])
            elif adjusted_idx in val_picks:
                if path not in val_unit_ids:
                    val_unit_ids[path] = []
                val_unit_ids[path].append(unit_ids[i])

        boost += len(unit_ids)

    path_to_ids = {}
    total_len = 0
    for path in test_paths:
        data = open_minian(path)

        all_unit_ids = data['E'].unit_id.values
        verified = data['E'].verified.values.astype(int)
        unit_ids = all_unit_ids[verified==1]
        total_len += len(unit_ids)
        path_to_ids[path] = unit_ids
    
    test_picks = np.random.choice(np.arange(total_len), total_test, replace=False)

    test_unit_ids = {}
    boost = 0
    for path, unit_ids in path_to_ids.items():
        for i in range(len(unit_ids)):
            adjusted_idx = i + boost
            if adjusted_idx in test_picks:
                if path not in test_unit_ids:
                    test_unit_ids[path] = []
                test_unit_ids[path].append(unit_ids[i])

        boost += len(unit_ids)

    return train_unit_ids, val_unit_ids, test_unit_ids

def train_val_test_split(paths, train_total=None, val_split=0.1, test_split=0.1) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]]:
    total_len = 0
    path_to_ids = {}
    for path in paths:
        data = open_minian(path)

        all_unit_ids = data['E'].unit_id.values
        verified = data['E'].verified.values.astype(int)
        unit_ids = all_unit_ids[verified==1]
        total_len += len(unit_ids)
        path_to_ids[path] = unit_ids
    
    # Val will be 10% of the train_total with a minimum of 1 otherwise it will be val_split of the total_len
    if train_total is None:
        total_val = int(val_split * total_len)
    else:
        total_val = max(int(val_split * train_total), 1)
    total_test = int(test_split * total_len) if test_split < 1 else test_split
    train_total = total_len - total_val - total_test if train_total is None else train_total - total_val
    random_picks = np.random.choice(np.arange(total_len), total_val + total_test, replace=False)
    val_picks, test_picks = random_picks[:total_val], random_picks[total_val:]
    # All the rest will be used for training
    train_picks = np.setdiff1d(np.arange(total_len), random_picks)
    if train_total < len(train_picks):
        train_picks = np.random.choice(train_picks, train_total, replace=False)

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
            elif adjusted_idx in train_picks:
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