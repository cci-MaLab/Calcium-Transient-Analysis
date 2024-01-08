import numpy as np
from backend import DataInstance
from advanced_summary import advanced
from traditional_summary import calculations
import random

DNA_PREBINNUM_SIZE = 4
DNA_POSTBINNUM_SIZE = 4
DNA_BINSIZE_SIZE = 5
POPULATION_SIZE = 20


DNA_PREBINNUM_BOUND = [0,9]
DNA_POSTBINNUM_BOUND = [0,9]
DNA_BINSIZE_BOUND = [0,19]



class Genetic_Algorithm:
    
    def __init__(
            self,
            mice,
            max_generation = 50,
            cross_rate = 0.5,
            mutation_rate = 0.15,
            event = 'RNFS'
    ):
        self.mice = mice
        self.max_generation = max_generation
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.event = event
        pass
    #DNA [preBinNum,postBinNum,binSize]

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
        final_features_list = []
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
            score,final_features,labels = advanced_calculator.generate_model()
            fitness.append(score)
            final_features_list.append(final_features)
        return np.array(fitness),final_features_list

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
        while(binSize_input>0 or len(binSize)<DNA_BINSIZE_SIZE):
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
        
    def output_results(self, population, fitness,features, number:int = 5):
        index = np.argsort(fitness)[:number]
        return population[index],features[index,:]

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
            fitness, f_features = self.get_fitness(population,self.mice)
            population = self.select(population,fitness)
        examples = []
        good_number = 5
        best_window, features = self.output_results(population,fitness,f_features,good_number)
        preBinNum,postBinNum,binSize = self.decoded_dna(best_window)
        for i in range(good_number):
            calculation = calculations(self.mice,preBinNum[i],postBinNum[i],binSize[i],self.event)   
            AUCs = calculation.auc()
            number_of_samples = 20
            example_features = random.sample(features,number_of_samples)
            examples.append(example_features)
        return preBinNum,postBinNum,binSize,examples,AUCs
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