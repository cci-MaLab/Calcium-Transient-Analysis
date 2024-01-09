import pandas as pd
from backend import DataInstance
import numpy as np


class calculations:
    def __init__(
        self,
        mice,
        preBinNum,
        postBinNum,
        binSize, 
        action
        ) -> None:
        self.mice = mice
        self.preBinNum = preBinNum
        self.postBinNum = postBinNum
        self.binSize = binSize
        self.action = action
        pass

    def auc(self):
        '''
        todo:
        after update the timewindow part, I will update the AUC for each bin
        '''
        auc = []
        mouse = []
        bins = np.zeros(shape = (self.preBinNum+self.postBinNum,1))
        print(bins)    
        for di in self.mice:
            unit_ids = di.data['unit_ids']
            timestamps = di.get_timestep(self.action)
            for index, time in enumerate(timestamps):    
                bin_list = di.events[self.action].get_binList(time, self.preBinNum,self.postBinNum,self.binSize)
                for b in bin_list:
                    neuron_auc = []  
                    for uid in unit_ids:
                        sum = int(b.sel(unit_id = uid).sum())
                        print(sum)
                        neuron_auc.append(sum)
                    bins = np.c_[bins,np.array(neuron_auc).T]
                    print(bins)
            #     mouse.append(bins)
            # auc.append(mouse)
        return auc
            
    def action_num(self):
        '''
        todo:
        calculate for each bin
        '''
        actionNum = []
        return actionNum

    def avg_single_action_auc(self,auc,actionNum):
        '''
        todo:
        for each bin
        '''
        avg_single_auc = []
        for i,j in zip(auc,actionNum):
            avg = i/j
            avg_single_auc.append(avg)
        return avg_single_auc

    def iei(self):
        '''
        todo:
        for each bin
        '''
        return
