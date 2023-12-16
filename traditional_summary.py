import pandas as pd
from backend import DataInstance


class calculations:
    def __init__(
        self,
        events:list
        ) -> None:
        self.events = events
        pass

    def auc(self):
        '''
        todo:
        after update the timewindow part, I will update the AUC for each bin
        '''
        auc = []
        for i in self.events:
            sum = i['C'].sum()
            auc.append(sum)
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
