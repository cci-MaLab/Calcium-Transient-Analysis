import torch
import numpy as np
from ml_training import config
from math import ceil

def sequence_to_predictions(model, signal, rolling=50, voting="average", threshold=config.THRESHOLD):
    # The assumption is that signal is already normalized and padded
    model.eval()
    slack = model.slack
    sequence_len = model.sequence_len
    
    max_value_repeats = int(ceil(sequence_len / rolling)) # Calculate how many times a single frame will appear in different sequences
    preds = np.ones((max_value_repeats,len(signal)-2*slack)) * -1
    with torch.no_grad():
        for row, i in enumerate(range(0, len(signal) - (sequence_len + 2*slack)+1, rolling)):
            sample = signal[i:i+sequence_len+2*slack].unsqueeze(0)
            pred = model(sample)
            pred = torch.sigmoid(pred)
            pred = pred.cpu().detach().numpy()
            preds[row%max_value_repeats, i:sequence_len+i] = pred
    
    if voting == "average":
        preds[preds==-1] = 0
        preds = np.mean(preds, axis=0)
    elif voting == "max":
        preds = np.max(preds, axis=0)
    elif voting == "min":
        preds[preds==-1] = 1
        preds = np.min(preds, axis=0)
    else:
        raise ValueError("Invalid type")
    preds = np.where(preds > threshold, 1, 0)
    return preds