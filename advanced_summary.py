from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
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
        # features_group = []
        features = []
        labels = []
        # traces = {}
        traces = []
        for instance in self.dataInstances:
            timestamps = instance.get_timestep(action) # need to double check frame number or real time
            print(instance.group)
            unit_ids = instance.A.keys()
            # print(timestamps)
            for time in timestamps:
                delay = 0-self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['preNum']
                duration = self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['postNum']+ self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['preNum']
                single_event_feature, start_frame, end_frame,integrity = instance.events[action].get_interval_section(event_frame = time, duration = duration, delay = delay,interval = 100,type = 'C')
                single_event_trace, start_frame, end_frame = instance.events[action].get_section(event_frame = time, duration = duration, delay = delay,type = 'C')
                # start_time = time - self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['preNum']*1000
                # end_time = time + self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['postNum']*1000
                # feature = instance.data['C'].sel(frame = (start_time,end_time))
                if integrity == False:
                    continue
                for uid in unit_ids:
                    single_feature = single_event_feature.sel(unit_id = uid)
                    single_trace = single_event_trace.sel(unit_id = uid)
                    features.append(np.array(single_feature))
                    
                    labels.append(instance.group)
                    traces.append(single_trace)
                    # traces[instance.group].append(single_trace)
            # features_group.append(features)
        return features, labels, traces

    def generate_model(self):
        # svm model
        features, labels,traces = self.get_features()
        min_l = len(features[0])
        for i in features:
            min_l = min(min_l,len(i))
        # print(features)
        print(min_l)
        final_features = []
        for i in features:
            i = i[:min_l]
            final_features.append(i)
        print(len(final_features))
        feature_train, feature_test, label_train,lable_test = train_test_split(final_features,labels,test_size = 0.33,random_state = 20)
        clf = svm.SVC()
        clf.fit(feature_train, label_train)
        clf.score(feature_test,lable_test)
        print(clf.score(feature_test,lable_test))
        return clf.score(feature_test,lable_test), traces,labels