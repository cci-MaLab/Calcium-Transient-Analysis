import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from .backend import DataInstance
from .advanced_summary import advanced
from .traditional_summary import calculations
import random

DNA_PREBINNUM_SIZE = 4
DNA_POSTBINNUM_SIZE = 4
DNA_BINSIZE_SIZE = 5


DNA_PREBINNUM_BOUND = [0,9]
DNA_POSTBINNUM_BOUND = [0,9]
DNA_BINSIZE_BOUND = [0,19]



class Genetic_Algorithm:
    
    def __init__(
            self,
            mice = None,
            max_generation = 1,
            population_size = 20,
            cross_rate = 0.5,
            mutation_rate = 0.15,
            event_type = 'RNFS',
            value_type = 'C',
            feature_type = 'AUC'
            
    ):
        self.mice = mice
        if(mice is None):
            self.mice = self.mice_demo()
        self.max_generation = max_generation
        self.population_size = population_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.event_type = event_type
        self.feature_type = feature_type
        self.value_type = value_type

    def setMice(self,mice):
        self.mice = mice

    def mice_demo(self):
        di1= DataInstance("/N/project/Cortical_Calcium_Image/Miniscope data/05.2023_Tenth_group/AA058_D1/2023_05_05/11_02_42/Miniscope_2/S4/config.ini") # Coke demo
        di2= DataInstance("/N/project/Cortical_Calcium_Image/Miniscope data/12.2022_Seventh_group/AA042_D1/2022_12_12/12_35_11/Miniscope_2/S1/config.ini") # Saline demo
        di3= DataInstance("/N/project/Cortical_Calcium_Image/Miniscope data/05.2023_Tenth_group/AA058_D6/2023_05_10/09_49_50/Miniscope_2/S1/config.ini") # Coke demo
        di4= DataInstance("/N/project/Cortical_Calcium_Image/Miniscope data/03.2023_Eighth_group/AA048_D8/2023_03_13/12_27_08/Miniscope_2/S1/config.ini") # Saline demo
        mice = [di1,di2,di3,di4]
        return mice
    
    def addLog(self,filename):
        self.logfile = filename
        return

    def decoded_dna(self, population):
        preBinNum_DNA = population[:, 0 : DNA_PREBINNUM_SIZE]
        postBinNum_DNA = population[:, DNA_PREBINNUM_SIZE:DNA_PREBINNUM_SIZE + DNA_POSTBINNUM_SIZE]
        binSize_DNA = population[:, DNA_PREBINNUM_SIZE + DNA_POSTBINNUM_SIZE : ]
        preBinNum = preBinNum_DNA.dot(2 ** np.arange(DNA_PREBINNUM_SIZE)[::-1])
        postBinNum = postBinNum_DNA.dot(2 ** np.arange(DNA_POSTBINNUM_SIZE)[::-1])
        binSize = binSize_DNA.dot(2 ** np.arange(DNA_BINSIZE_SIZE)[::-1])
        return preBinNum,postBinNum,binSize

    
    def get_fitness(self, population,mice):
        fitness = np.empty((0,2))
        all_traces = []
        all_framelines = []
        preBinNum_DNA = population[:, 0 : DNA_PREBINNUM_SIZE]
        postBinNum_DNA = population[:, DNA_PREBINNUM_SIZE:DNA_PREBINNUM_SIZE + DNA_POSTBINNUM_SIZE]
        binSize_DNA = population[:, DNA_PREBINNUM_SIZE + DNA_POSTBINNUM_SIZE : ]
        preBinNum = preBinNum_DNA.dot(2 ** np.arange(DNA_PREBINNUM_SIZE)[::-1])
        postBinNum = postBinNum_DNA.dot(2 ** np.arange(DNA_POSTBINNUM_SIZE)[::-1])
        binSize = binSize_DNA.dot(2 ** np.arange(DNA_BINSIZE_SIZE)[::-1])
        print(preBinNum)
        print(postBinNum)
        print(binSize)
        # RNFS_time = mice[0].events[self.event_type].timesteps
        for i in range(len(population)):
            log = open(self.logfile, mode = 'a')
            log.write('No. '+ str(i)+'\n')
            log.write('preBinNum: ' + str(preBinNum[i]) + ' postBinNum: ' + str(postBinNum[i]) + ' binSize: ' + str(binSize[i]) +'\n')
            log.close()
            advanced_calculator = advanced(preBinNum[i],postBinNum[i],binSize[i],mice,self.event_type,self.value_type,self.feature_type)
            scores,traces,framelines,labels,specifity, sensitivity = advanced_calculator.generate_model(event_type = self.event_type, value_type = self.value_type,feature_type = self.feature_type)
            fitness = np.append(fitness,[scores],axis = 0)
            all_traces.append(traces)
            all_framelines.append(framelines)
            log = open(self.logfile, mode = 'a')
            log.write('Score:' + str(scores)+'\n')
            log.write('Specifity: '+str(specifity)+' Sensitivity: '+ str(sensitivity)+'\n')
            log.close()
        return np.array(fitness), all_traces, all_framelines

    def crossover(self, population, CROSSOVER_RATE=0.8):
        next_generation = []
        for parent1 in population:
            child  = parent1
            if np.random.rand() < CROSSOVER_RATE:
                parent2 = population[np.random.randint(self.population_size)]
                cross_points = np.random.randint(low=0, high=(DNA_PREBINNUM_SIZE + DNA_POSTBINNUM_SIZE + DNA_BINSIZE_SIZE) * 2)
                child[cross_points:] = parent2[cross_points:]
            next_generation.append(child)
        return np.array(next_generation)

    def mutation(self, dna, mutation_rate):
        if np.random.rand() < self.mutation_rate:
            mutate_point = np.random.randint(0, DNA_PREBINNUM_SIZE + DNA_POSTBINNUM_SIZE + DNA_BINSIZE_SIZE)
            dna[mutate_point] = dna[mutate_point] ^ 1
        return dna

    def encoded_dna(self, preBinNum_input,postBinNum_input,binSize_input):
        preBinNum = []
        postBinNum = []
        binSize = []
        while(preBinNum_input>0 or len(preBinNum) < DNA_PREBINNUM_SIZE):
            preBinNum.insert(0, preBinNum_input % 2)
            preBinNum_input = int(preBinNum_input / 2)
        while(postBinNum_input > 0 or len(postBinNum) < DNA_POSTBINNUM_SIZE):
            postBinNum.insert(0, postBinNum_input % 2)
            postBinNum_input = int(postBinNum_input / 2)
        while(binSize_input>0 or len(binSize) < DNA_BINSIZE_SIZE):
            binSize.insert(0,binSize_input % 2)
            binSize_input = int(binSize_input/2)
        dna = preBinNum + postBinNum + binSize
        return np.array(dna)

        

    def select(self, population,fitness):
        index = np.random.choice(np.arange(self.population_size), size=self.population_size, replace=True, p=(fitness[:,0]) / (fitness[:,0].sum()))
        print(index)
        # index = np.argsort(fitness)
        # print("1:",index)
        
        # index[0] = index[-1]
        # print("2:",index)
        return population[index]
        
    def output_results(self, population, fitness,traces, framelines,number:int = 5):
        index = np.argsort(fitness)[-number:]       # temporary method
        index = index[::-1]
        f_traces = []
        f_framelines = []
        for i in index:
            f_traces.append(traces[i])
            f_framelines.append(framelines[i])
        return population[index],fitness[index],f_traces,f_framelines

    def execute(self):
        population = np.random.randint(0,2,(self.population_size,DNA_PREBINNUM_SIZE+DNA_POSTBINNUM_SIZE+DNA_BINSIZE_SIZE))
        self.curve = []
        self.curve_max = []
        self.curve_min = []
        self.f1Curve = []
        self.f1Curve_max = []
        self.f1Curve_min = []
        var = []
        log = open(self.logfile,mode = 'a')
        log.write('Max generation: '+str(self.max_generation)+' Population: '+ str(self.population_size)+'\n')
        log.close()
        for i in range(self.max_generation):         
            print("Generation: ",i)
            print("Before:------")
            print(population)
            log = open(self.logfile, mode = 'a')
            log.write('Generation: '+ str(i)+'\n')
            log.close()
            population = self.crossover(population, self.cross_rate)
            for j in range(self.population_size):
                population[j] = self.mutation(population[j],self.mutation_rate)
            print("After:------------")
            print(population)
            fitness, traces,framelines = self.get_fitness(population,self.mice)
            print(fitness)
            mean_value = np.mean(fitness,axis = 0)
            max_value = np.max(fitness,axis = 0)
            min_value = np.min(fitness,axis = 0)
            self.curve.append(mean_value[0])
            self.f1Curve.append(mean_value[1])
            self.curve_max.append(max_value[0])
            self.curve_min.append(min_value[0])
            self.f1Curve_max.append(max_value[1])
            self.f1Curve_min.append(min_value[1])
            var.append(np.var(fitness))
            population = self.select(population,fitness)
        log = open(self.logfile, mode = 'a')
        log.write('Average score:' + str(self.curve)+'\n')
        log.write('var:' + str(var)+'\n')
        log.close()
        examples = []
        xvalues = []
        AUCs = []
        good_number = min(5, self.population_size)
        number_of_samples = 20
        best_windows, best_fitness,best_traces,best_frameline = self.output_results(population,fitness[:,0],traces,framelines,good_number)
        preBinNum,postBinNum,binSize = self.decoded_dna(best_windows)
        print('w:',len(best_windows),'t:',len(best_traces))
        for i in range(good_number):
            print(i)
            example_features = {}
            example_framelines = {}
            # calculation = calculations(self.mice,preBinNum[i],postBinNum[i],binSize[i],self.event)   
            # auc = calculation.auc()
            example_features['Cocaine'] = []
            example_features['Saline'] = []
            example_framelines['Cocaine'] = []
            example_framelines['Saline'] = []
            for j in list(np.random.randint(low = 0, high = len(best_traces[i]['Cocaine'])-1,size = number_of_samples)):
                example_features['Cocaine'].append(best_traces[i]['Cocaine'][j])
                example_framelines['Cocaine'].append(best_frameline[i]['Cocaine'][j])
            for j in list(np.random.randint(low = 0, high = len(best_traces[i]['Saline'])-1,size = number_of_samples)):
                example_features['Saline'].append(best_traces[i]['Saline'][j])
                example_framelines['Saline'].append(best_frameline[i]['Saline'][j])
            examples.append(example_features)
            xvalues.append(example_framelines)
            # AUCs.append(auc)
        self.preBinNum = preBinNum
        self.postBinNum = postBinNum
        self.binSize = binSize
        self.examples = examples # Make sure they are all the same size in timeline
        self._best_fitness = best_fitness
        self.example_xvalues = xvalues
        self.traces = best_traces
        self.xvalues = best_frameline
        # self.AUCs = AUCs
        # return preBinNum,postBinNum,binSize
            #print traces and footprint
         
    # def return_results(self,rank:int):
    #     return self.preBinNum[rank],self.postBinNum[rank],self.binSize[rank], self.fitness[rank], self.examples[rank]

    def calculate_data(self, preBinNum:int,postBinNum:int,binSize:int,event_type,value_type:str = 'C'):
        print(preBinNum)
        print(postBinNum)
        mouseID_list = []
        day_list = []
        session_list = []
        unitID_list = []
        group_list = []
        auc_list = []
        freq_list = []
        list_title = []
        for i in range(preBinNum-1,-1,-1):
            list_title.append('pre bin '+ str(i+1))
        for i in range(postBinNum):
            list_title.append('post bin '+ str(i+1))
        bin_auc = {}
        for i in range(preBinNum+postBinNum):
            bin_auc[i] = []
        for instance in self.mice:
            print(instance.group+": ",end=' ')
            unit_ids = instance.data['unit_ids']
            print(unit_ids,end= ' ')
            print('{}: {}'.format(event_type,instance.get_timestep(event_type)))
            for timestamp in instance.get_timestep(event_type):
                print(timestamp)
                bin_list = instance.events[event_type].get_binList(timestamp,preBinNum,postBinNum,binSize,value_type)   
                for uid in unit_ids:
                    for idx,bin in enumerate(bin_list):
                        auc = np.sum(np.asarray(bin.sel(unit_id = uid)))
                        bin_auc[idx].append(auc)
                    mouseID_list.append(instance.mouseID)
                    day_list.append(instance.day)
                    session_list.append(instance.session)
                    unitID_list.append(uid)       
                    group_list.append(instance.group)      
                    # print(instance.group,end = ',')
            # print(" ")
        res_data = {'mouse_id' : mouseID_list, 'group':group_list,'day':day_list,'session':session_list,'unit_id':unitID_list}
        for i,j in zip(list_title,bin_auc.keys()):
            res_data[i] = bin_auc[j]
        res_df = pd.DataFrame(res_data)
        return res_df

    ##steps : select mouse first
    ## step 2 : fixed the parameter
    # step 3 : machine learning
    # step 4 : parameter genetic algorithm
    ## time duration start time /AUC or Freq
    # bin number
    # define training set
    # rising part AUC or freqency
    # try no filter first
    # then filter the datapoints
    # IEI AUC Freq 
    # foot print 
    # sort data
    '''
    scattered data interpolation
    1sec 2-D map 
    '''
    '''
    time stamp events 
    '''



# todo
# save as file
# specifity = (number of true negatives)/(number of true negatives+number of false positive)
# sensitivity = (number of true positives)/(number of true positives + number of false negatives)