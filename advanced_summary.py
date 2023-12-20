from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
from backend import DataInstance 
from typing import List
import numpy as np

class advanced:
    def __init__(
        self,
        preNum,
        postNum,
        binSize,
        samples:List[DataInstance]
        ) -> None:
        self.behavior_timewindow_dict = {'binSize':binSize,'preNum':preNum,'postNum':postNum}
        self.dataInstances = samples
        self.action_list = self.dataInstances[0].events.keys()

    def set_behavior(self, binSize, preNum, postNum):
        self.behavior_timewindow_dict = {'binSize':binSize,'preNum':preNum,'postNum':postNum}

    def get_features(self,action ='RNFS'):
        features_group = []
        features = []
        labels = []
        for instance in self.dataInstances:
            timestamps = instance.get_timestep(action) # need to double check frame number or real time
            print(timestamps)
            for time in timestamps:
                labels.append(instance.group)
                delay = self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['preNum']
                duration = self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['postNum']+ self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['preNum']
                single_event, start_frame, end_frame = instance.events[action].get_section(time,duration,delay)
                # start_time = time - self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['preNum']*1000
                # end_time = time + self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['postNum']*1000
                # feature = instance.data['C'].sel(frame = (start_time,end_time))
                features.append(list(single_event))
            
        # uniform the size
        min_l = 0
        for i in features:
            min_l = min(min_l,len(i[0]))
        all_features = []
        for i in features:
            for j in i:
                all_features.append(j[:min_l])
        all_features = np.array(all_features)
        labels = np.array(labels)
        print(len(labels),len(all_features))
        return all_features, labels

    def generate_model(self):
        # svm model
        features, labels = self.get_features()
        print(features)
        feature_train, feature_test, label_train,lable_test = train_test_split(features,labels,test_size = 0.33,random_state = 10)
        clf = svm.SVC()
        clf.fit(feature_train, label_train)
        clf.score(feature_test,lable_test)
        print(clf.score(feature_test,lable_test))
        return 