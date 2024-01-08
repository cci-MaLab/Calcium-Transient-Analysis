import pandas as pd
from backend import DataInstance


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
        bins = []      
        for di in self.mice:
            timestamps = di.get_timestep(self.action)
            for time in timestamps:    
                bin_list = di.events[self.action].get_binList(time, self.preBinNum,self.postBinNum,self.binSize)
                for b in bin_list:
                    sum = b['C'].sum()
                    bins.append(sum)
                mouse.append(bins)
            auc.append(mouse)
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
