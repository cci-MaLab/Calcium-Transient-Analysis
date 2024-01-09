import numpy as np
from backend import DataInstance
from advanced_summary import advanced
from traditional_summary import calculations
import random

DNA_PREBINNUM_SIZE = 4
DNA_POSTBINNUM_SIZE = 4
DNA_BINSIZE_SIZE = 5
POPULATION_SIZE = 3


DNA_PREBINNUM_BOUND = [0,9]
DNA_POSTBINNUM_BOUND = [0,9]
DNA_BINSIZE_BOUND = [0,19]



class Genetic_Algorithm:
    
    def __init__(
            self,
            mice = None,
            max_generation = 1,
            cross_rate = 0.5,
            mutation_rate = 0.15,
            event = 'RNFS'
    ):
        self.mice = mice
        if(mice is None):
            self.mice = self.mice_demo()
        self.max_generation = max_generation
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.event = event

    def mice_demo(self):
        di1= DataInstance("/N/project/Cortical_Calcium_Image/Miniscope data/05.2023_Tenth_group/AA058_D1/2023_05_05/11_02_42/Miniscope_2/S4/config.ini",['ALP','IALP','RNFS'] ) # Coke demo
        di1.gourp = 'Cocaine'
        di2= DataInstance("/N/project/Cortical_Calcium_Image/Miniscope data/12.2022_Seventh_group/AA042_D1/2022_12_12/12_35_11/Miniscope_2/S1/config.ini",['ALP','IALP','RNFS'] ) # Saline demo
        di2.group = 'Saline'
        di3= DataInstance("/N/project/Cortical_Calcium_Image/Miniscope data/05.2023_Tenth_group/AA058_D6/2023_05_10/09_49_50/Miniscope_2/S1/config.ini",['ALP','IALP','RNFS'] ) # Coke demo
        di3.group = 'Cocaine'
        di4= DataInstance("/N/project/Cortical_Calcium_Image/Miniscope data/03.2023_Eighth_group/AA048_D8/2023_03_13/12_27_08/Miniscope_2/S1/config.ini",['ALP','IALP','RNFS'] ) # Saline demo
        di4.group = 'Saline'
        mice = [di1,di2,di3,di4]
        print(di1.group)
        return mice

    def decoded_dna(self, population):
        preBinNum_DNA = population[:, 0 : DNA_PREBINNUM_SIZE]
        postBinNum_DNA = population[:, DNA_PREBINNUM_SIZE:DNA_PREBINNUM_SIZE + DNA_POSTBINNUM_SIZE]
        binSize_DNA = population[:, DNA_PREBINNUM_SIZE + DNA_POSTBINNUM_SIZE : ]
        preBinNum = preBinNum_DNA.dot(2 ** np.arange(DNA_PREBINNUM_SIZE)[::-1])
        postBinNum = postBinNum_DNA.dot(2 ** np.arange(DNA_POSTBINNUM_SIZE)[::-1])
        binSize = binSize_DNA.dot(2 ** np.arange(DNA_BINSIZE_SIZE)[::-1])
        return preBinNum,postBinNum,binSize

    
    def get_fitness(self, population,mice):
        fitness = []
        all_traces = []
        preBinNum_DNA = population[:, 0 : DNA_PREBINNUM_SIZE]
        postBinNum_DNA = population[:, DNA_PREBINNUM_SIZE:DNA_PREBINNUM_SIZE + DNA_POSTBINNUM_SIZE]
        binSize_DNA = population[:, DNA_PREBINNUM_SIZE + DNA_POSTBINNUM_SIZE : ]
        preBinNum = preBinNum_DNA.dot(2 ** np.arange(DNA_PREBINNUM_SIZE)[::-1])
        postBinNum = postBinNum_DNA.dot(2 ** np.arange(DNA_POSTBINNUM_SIZE)[::-1])
        binSize = binSize_DNA.dot(2 ** np.arange(DNA_BINSIZE_SIZE)[::-1])
        print(preBinNum)
        print(postBinNum)
        print(binSize)
        RNFS_time = mice[0].events[self.event].timesteps
        for i in range(len(population)):
            advanced_calculator = advanced(preBinNum[i],postBinNum[i],binSize[i],mice)
            score,traces,labels = advanced_calculator.generate_model()
            fitness.append(score)
            all_traces.append(traces)
        return np.array(fitness), all_traces

    def crossover(self, population, CROSSOVER_RATE=0.8):
        next_generation = []
        for parent1 in population:
            child  = parent1
            if np.random.rand() < CROSSOVER_RATE:
                parent2 = population[np.random.randint(POPULATION_SIZE)]
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
        # print(preBinNum)
        # print(postBinNum)
        # print(binSize)
        return np.array(dna)

    # def decoded_dna(population):
    #     preBinNum_dna = population[:,:DNA_PREBINNUM_SIZE]
    #     postBinNum_dna = population[:,]
    #     binSize_dna = population[:]

        

    def select(self, population,fitness):
        index = np.random.choice(np.arange(POPULATION_SIZE), size=POPULATION_SIZE, replace=True, p=(fitness) / (fitness.sum()))
        return population[index]
        
    def output_results(self, population, fitness,traces, number:int = 5):
        index = np.argsort(fitness)[-number:]
        index = index[::-1]
        f_traces = []
        for i in index:
            f_traces.append(traces[i])
        return population[index],f_traces

    def execute(self):
        population = np.random.randint(0,2,(POPULATION_SIZE,DNA_PREBINNUM_SIZE+DNA_POSTBINNUM_SIZE+DNA_BINSIZE_SIZE))
        for i in range(self.max_generation):
            print("Generation: ",i)
            print("Before:------")
            print(population)
            population = self.crossover(population, self.cross_rate)
            for j in range(POPULATION_SIZE):
                population[j] = self.mutation(population[j],self.mutation_rate)
            print("After:------------")
            print(population)
            fitness, traces = self.get_fitness(population,self.mice)
            population = self.select(population,fitness)
        examples = []
        AUCs = []
        good_number = 5
        number_of_samples = 20
        best_windows, features = self.output_results(population,fitness,traces,good_number)
        preBinNum,postBinNum,binSize = self.decoded_dna(best_windows)
        # for i in range(good_number):
            # example_features = {}
            # calculation = calculations(self.mice,preBinNum[i],postBinNum[i],binSize[i],self.event)   
            # auc = calculation.auc()
            # example_features['Cocaine'] = random.sample(features['Cocaine'],number_of_samples)
            # example_features['Saline'] = random.sample(features['Saline'],number_of_samples)
            # examples.append(example_features)
            # AUCs.append(auc)
        self.preBinNum = preBinNum
        self.postBinNum = postBinNum
        self.binSize = binSize
        # self.examples = examples
        # self.AUCs = AUCs
        return preBinNum,postBinNum,binSize
            #print traces and footprint
         

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