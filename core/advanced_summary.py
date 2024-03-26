import sys
sys.path.insert(0, ".")
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
from .backend import DataInstance 
from typing import List
import numpy as np
import time

class advanced:
    def __init__(
        self,
        preNum,
        postNum,
        binSize,
        samples:List[DataInstance],
        event_type,
        value_type,
        feature_type
        ) -> None:
        self.behavior_timewindow_dict = {'binSize':binSize,'preNum':preNum,'postNum':postNum}
        self.dataInstances = samples
        self.action_list = self.dataInstances[0].events.keys()
        self.event_type = event_type
        self.value_type = value_type
        self.feature_type = feature_type

    def set_behavior(self, binSize, preNum, postNum):
        self.behavior_timewindow_dict = {'binSize':binSize,'preNum':preNum,'postNum':postNum}

    def get_features(self, event_type, value_type, feature_type):
        # features_group = []
        features = []
        feature_A = []
        feature_B = []
        labels = []
        label_A = []
        label_B = []
        traces = {}
        framelines = {}
        # traces = []
        if feature_type == "Signal":
            for instance_index,instance in enumerate(self.dataInstances):
                timestamps = instance.get_timestep(event_type) # need to double check frame number or real time
                print(instance_index,':',instance.group)
                unit_ids = instance.A.keys()
                # print(timestamps)
                for single_time in timestamps:
                    print(single_time)
                    delay = 0-self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['preNum']
                    duration = self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['postNum']+ self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['preNum']
                    single_event_feature, start_frame, end_frame,integrity = instance.events[event_type].get_interval_section(event_frame = single_time, duration = duration, delay = delay,interval = 0,type = value_type)
                    if integrity == False:
                        continue
                    single_event_trace, start_frame, end_frame = instance.events[event_type].get_section(event_frame = single_time, duration = duration, delay = delay,type = 'C')
                    frameline = list(range(start_frame-single_time,end_frame-single_time+1))
                    print('frameline:',len(frameline), 'event:',len(single_event_trace.coords['frame'].values))
                    # start_time = time - self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['preNum']*1000
                    # end_time = time + self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['postNum']*1000
                    # feature = instance.data['C'].sel(frame = (start_time,end_time))
                    for uid in unit_ids:
                        single_feature = single_event_feature.sel(unit_id = uid)
                        single_trace = single_event_trace.sel(unit_id = uid)
                        features.append(np.array(single_feature))                    
                        labels.append(instance.group)
                        # traces.append(single_trace)
                        if instance.group not in traces.keys():
                            traces[instance.group] = []
                        traces[instance.group].append(single_trace)
                        if instance.group not in framelines.keys():
                            framelines[instance.group] = []
                        framelines[instance.group].append(frameline)
                # features_group.append(features)
        elif feature_type == 'AUC':
            for instance_index,instance in enumerate(self.dataInstances):
                timestamps = instance.get_timestep(event_type) # need to double check frame number or real time
                print(instance_index,':',instance.group)
                unit_ids = instance.A.keys()
                # print(timestamps)
                for single_time in timestamps:
                    delay = 0-self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['preNum']
                    duration = self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['postNum']+ self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['preNum']
                    start = time.time()
                    single_event_feature, start_frame, end_frame,integrity = instance.events[event_type].get_interval_section(event_frame = single_time, duration = duration, delay = delay,interval = 0,type = value_type)
                    end = time.time()
                    print('Running time is: '+str(end-start))
                    if integrity == False:
                        continue
                    single_event_trace, start_frame, end_frame = instance.events[event_type].get_section(event_frame = single_time, duration = duration, delay = delay,type = 'C')
                    frameline = list(range(start_frame-single_time,end_frame-single_time+1))
                    # print('frameline:',len(frameline), 'event:',len(single_event_trace.coords['frame'].values))

                    for uid in unit_ids:
                        auc = np.sum(np.asarray(single_event_feature.sel(unit_id = uid)))
                        # print('uid: '+ str(uid)+' auc: '+str(auc))
                        # print('group: ' + instance.group)
                        single_feature = [auc]
                        single_trace = single_event_trace.sel(unit_id = uid)
                        if instance.group == 'Cocaine':
                            feature_A.append(single_feature)
                            label_A.append(instance.group)
                        elif instance.group =='Saline':
                            feature_B.append(single_feature)    
                            label_B.append(instance.group)
                        if instance.group not in traces.keys():
                            traces[instance.group] = []
                        traces[instance.group].append(single_trace)
                        if instance.group not in framelines.keys():
                            framelines[instance.group] = []
                        framelines[instance.group].append(frameline)
            amount = min(len(feature_A), len(feature_B))
            feature_A = feature_A[:amount]
            label_A = label_A[:amount]
            feature_B = feature_B[:amount]
            label_B = label_B[:amount]
            features = feature_A+feature_B
            labels = label_A + label_B
            print("features len:", len(features), " labels len:",len(labels))
        elif feature_type == 'Frequency' and value_type =='S':
            for instance_index,instance in enumerate(self.dataInstances):
                timestamps = instance.get_timestep(event_type) # need to double check frame number or real time
                print(instance_index,':',instance.group)
                unit_ids = instance.A.keys()
                # print(timestamps)
                for single_time in timestamps:
                    delay = 0-self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['preNum']
                    duration = self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['postNum']+ self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['preNum']
                    single_event_feature, start_frame, end_frame,integrity = instance.events[event_type].get_interval_section(event_frame = single_time, duration = duration, delay = delay,interval = 0,type = 'C')
                    # start_time = time.time()
                    single_S_feature, Sstart_frame, Send_frame,Sintegrity = instance.events[event_type].get_interval_section(event_frame = single_time, duration = duration, delay = delay,interval = 0,type = 'S')
                    # end_time = time.time()
                    # print('Running time is: '+str(end_time-start_time))
                    if integrity == False:
                        continue
                    single_event_trace, start_frame, end_frame = instance.events[event_type].get_section(event_frame = single_time, duration = duration, delay = delay,type = 'C')
                    frameline = list(range(start_frame-single_time,end_frame-single_time+1))
                    # print('frameline:',len(frameline), 'event:',len(single_event_trace.coords['frame'].values))
                    # start_time = time - self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['preNum']*1000
                    # end_time = time + self.behavior_timewindow_dict['binSize']*self.behavior_timewindow_dict['postNum']*1000
                    # feature = instance.data['C'].sel(frame = (start_time,end_time))
                    for uid in unit_ids:                  
                        # c_AUC = np.sum(np.asarray(single_event_feature.sel(unit_id = uid)))
                        # s_AUC = np.sum(np.asarray(single_S_feature.sel(unit_id = uid)))
                        start_time = time.time()
                        position_S = np.where(single_S_feature > 0)[0]
                        split_position = np.where(np.diff(position_S) != 1)[0] + 1
                        event_S = np.split(single_S_feature[position_S], split_position)
                        freq = len(event_S)
                        end_time = time.time()
                        print('Calculate freq time is: '+str(end_time-start_time))
                        single_feature = [freq]
                        single_trace = single_event_trace.sel(unit_id = uid)
                        if instance.group == 'Cocaine':
                            feature_A.append(single_feature)
                            label_A.append(instance.group)
                        elif instance.group =='Saline':
                            feature_B.append(single_feature)    
                            label_B.append(instance.group)
                        # features.append([single_feature])                    
                        # labels.append(instance.group)
                        # traces.append(single_trace)
                        if instance.group not in traces.keys():
                            traces[instance.group] = []
                        traces[instance.group].append(single_trace)
                        if instance.group not in framelines.keys():
                            framelines[instance.group] = []
                        framelines[instance.group].append(frameline)
                # features_group.append(features)
            amount = min(len(feature_A), len(feature_B))
            feature_A = feature_A[:amount]
            label_A = label_A[:amount]
            feature_B = feature_B[:amount]
            label_B = label_B[:amount]
            features = feature_A+feature_B
            labels = label_A + label_B
            print("features len:", len(features), " labels len:",len(labels))
        return features, labels, traces,framelines

    def generate_model(self,event_type, value_type, feature_type):
        # svm model
        features, labels,traces,framelines = self.get_features(event_type= event_type,value_type = value_type,feature_type=feature_type)
        if feature_type == "Signal":
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
        else:
            final_features = features
        feature_train, feature_test, label_train,lable_test = train_test_split(final_features,labels,test_size = 0.33,random_state = 20)
        clf = svm.SVC()
        clf.fit(feature_train, label_train)
        clf.score(feature_test,lable_test)
        print(clf.score(feature_test,lable_test))
        prediction = clf.predict(feature_test)
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        # negative: Saline
        # positive: Coke
        # specifity = (number of true negatives)/(number of true negatives+number of false positive)
        # sensitivity = (number of true positives)/(number of true positives + number of false negatives)
        for i,j in zip(prediction,lable_test):
            if i == j and i == 'Saline':
                true_negative += 1
            elif i == j and i == 'Cocaine':
                true_positive += 1
            elif i != j and i == 'Saline':
                false_negative += 1
            else:
                false_positive += 1
        specifity = true_negative/(true_negative+false_positive)
        sensitivity = true_positive/(true_positive+false_negative)
        return clf.score(feature_test,lable_test), traces,framelines,labels, specifity, sensitivity