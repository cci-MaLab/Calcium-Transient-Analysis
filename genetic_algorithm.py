import numpy as np
from backend import DataInstance
from advanced_summary import advanced

DNA_PREBINNUM_SIZE = 4
DNA_POSTBINNUM_SIZE = 4
DNA_BINSIZE_SIZE = 5
POPULATION_SIZE = 10
CROSS_RATE = 0.5
MUTATION_RATE = 0.15
DNA_PREBINNUM_BOUND = [0,9]
DNA_POSTBINNUM_BOUND = [0,9]
DNA_BINSIZE_BOUND = [0,19]
EVENT = 'RNFS'


class Genetic_Algorithm:
    
    def __init__(
            self,
            mice
    ):
        self.mice = mice
        pass
    #DNA [preBinNum,postBinNum,binSize]


    def choose_mice():# demo
        di1= DataInstance("/N/project/Cortical_Calcium_Image/Miniscope data/05.2023_Tenth_group/AA058_D1/2023_05_05/11_02_42/Miniscope_2/S1/config.ini",['ALP','IALP','RNFS'] ) # Coke demo
        di2= DataInstance("/N/project/Cortical_Calcium_Image/Miniscope data/12.2022_Seventh_group/AA042_D1/2022_12_12/12_35_11/Miniscope_2/S1/config.ini",['ALP','IALP','RNFS'] ) # Saline demo
        return di1, di2
    
    def get_fitness(self, population,mice):
        fitness = []
        preBinNum_DNA = population[:, 0 : DNA_PREBINNUM_SIZE]
        postBinNum_DNA = population[:, DNA_PREBINNUM_SIZE:DNA_PREBINNUM_SIZE + DNA_POSTBINNUM_SIZE]
        binSize_DNA = population[:, DNA_PREBINNUM_SIZE + DNA_POSTBINNUM_SIZE : ]
        preBinNum = preBinNum_DNA.dot(2 ** np.arange(DNA_PREBINNUM_SIZE)[::-1])
        postBinNum = postBinNum_DNA.dot(2 ** np.arange(DNA_POSTBINNUM_SIZE)[::-1])
        binSize = binSize_DNA.dot(2 ** np.arange(DNA_BINSIZE_SIZE)[::-1])
        print(preBinNum)
        print(postBinNum)
        print(binSize)
        RNFS_time = mice[0].events[EVENT].timesteps
        for i in range(len(population)):
            advanced_calculator = advanced(preBinNum[i],postBinNum[i],binSize[i],mice)
            score = advanced_calculator.generate_model()
            fitness.append(score)
        return np.array(fitness)

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
        if np.random.rand() < MUTATION_RATE:
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
        


    def excute(self):

        MAX_GENERATION = 50

        # test demo
        # di1= DataInstance("/N/project/Cortical_Calcium_Image/Miniscope data/05.2023_Tenth_group/AA058_D1/2023_05_05/11_02_42/Miniscope_2/S4/config.ini",['ALP','IALP','RNFS'] ) # Coke demo
        # di2= DataInstance("/N/project/Cortical_Calcium_Image/Miniscope data/12.2022_Seventh_group/AA042_D1/2022_12_12/12_35_11/Miniscope_2/S1/config.ini",['ALP','IALP','RNFS'] ) # Saline demo
        # di3= DataInstance("/N/project/Cortical_Calcium_Image/Miniscope data/05.2023_Tenth_group/AA058_D6/2023_05_10/09_49_50/Miniscope_2/S1/config.ini",['ALP','IALP','RNFS'] ) # Coke demo
        # di4= DataInstance("/N/project/Cortical_Calcium_Image/Miniscope data/03.2023_Eighth_group/AA048_D8/2023_03_13/12_27_08/Miniscope_2/S1/config.ini",['ALP','IALP','RNFS'] ) # Saline demo

        # init parents(user input)
        # preBinNum_input = int(input('Input preBinNum (0-9)'))
        # postBinNum_input = int(input('Input postBinNum (0-9)'))
        # binSize_input = int(input('Input binSize (0-19)'))


        preBinNum_input = 1
        postBinNum_input = 1
        binSize_input = 5


        # parentA = encoded_dna(preBinNum_input,postBinNum_input,binSize_input)
        # preBinNum_input2 = int(input('Input preBinNum (0-9)'))
        # postBinNum_input2 = int(input('Input postBinNum (0-9)'))
        # binSize_input2 = int(input('Input binSize (0-19)'))
        # parentB = encoded_dna(preBinNum_input2,postBinNum_input2,binSize_input2)

        dna = self.encoded_dna(preBinNum_input,postBinNum_input,binSize_input)
        population = np.array([dna])
        for i in range(POPULATION_SIZE-1):
            population = np.append(population,[dna],axis = 0)
        
        for i in range(MAX_GENERATION):
            print("Generation: ",i)
            print("Before:------")
            print(population)
            population = self.crossover(population, CROSS_RATE)
            for j in range(POPULATION_SIZE):
                population[j] = self.mutation(population[j],MUTATION_RATE)
            print("After:------------")
            print(population)
            fitness = self.get_fitness(population,self.mice)
            population = self.select(population,fitness)
            
         
        # test SVM
        # advanced_calculator = advanced(preBinNum_input,postBinNum_input,binSize_input,mice)
        # score = advanced_calculator.generate_model()
        # 
        #     select()



        
        # di1,di2 = choose_mice()
        # symbol = 1
        # population = [parentA,parentB]
        # while(symbol==1):
        #     while(len(population) < POPULATION_SIZE):

                
        #         continue  
        #     symbol = input("If you want to quit, input 0 else input 1")

    # if __name__ == "__main__":
    #     main()
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