from sklearn import svm
import pandas as pd
from backend import DataInstance 
from typing import List

class advanced:
    def __init__(
        self
        ) -> None:
        self.behavior_timewindow_dict = {'binSize':5,'preNum':3,'postNum':5}

    def chooseDataInstance(self, dataInstances:List[DataInstance]):
        self.dataInstances = dataInstances
        self.action_list = self.dataInstances[0].events.keys()

    def set_behavior(self, binSize, preNum, postNum):
        self.behavior_timewindow_dict = {'binSize':binSize,'preNum':preNum,'postNum':postNum}

    def get_features(self,action ='RNFS'):
        features = []
        labels = []
        for instance in self.dataInstances:
            timestamps = instance.get_timestep[action]# need to double check frame number or real time
            for time in timestamps:
                labels.append(instance.group)
                start_time = time - self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['preNum']*1000
                end_time = time + self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['postNum']*1000
                feature = instance.data['C'].sel(frame = (start_time,end_time))
                features.append(feature)
        return features, labels

    def generate_model(self):
        # svm model
        features, lables = self.get_features()
        clf = svm.SVC()
        clf.fit(features, lables)
        return 