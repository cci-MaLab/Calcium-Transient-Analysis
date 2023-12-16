import numpy as np
from backend import DataInstance
from advanced_summary import advanced

DNA_PREBINNUM_SIZE = 5
DNA_POSTBINNUM_SIZE = 5
DNA_BINSIZE_SIZE = 10
POPULATION_SIZE = 10
CROSS_RATE = 0.5
MUTATION_RATE = 0.15
DNA_PREBINNUM_BOUND = [0,9]
DNA_POSTBINNUM_BOUND = [0,9]
DNA_BINSIZE_BOUND = [0,19]

#DNA [preBinNum,postBinNum,binSize]


def choose_mice():# demo
    di1= DataInstance("/N/project/Cortical_Calcium_Image/Miniscope data/05.2023_Tenth_group/AA058_D1/2023_05_05/11_02_42/Miniscope_2/S1/config.ini",['ALP','IALP','RNFS'] ) # Coke demo
    di2= DataInstance("/N/project/Cortical_Calcium_Image/Miniscope data/12.2022_Seventh_group/AA042_D1/2022_12_12/12_35_11/Miniscope_2/S1/config.ini",['ALP','IALP','RNFS'] ) # Saline demo


def get_fitness(population):
    preBinNum_DNA = population[:, 0:DNA_PREBINNUM_SIZE]
    postBinNum_DNA = population[:, DNA_PREBINNUM_SIZE:DNA_PREBINNUM_SIZE + DNA_POSTBINNUM_SIZE]
    binSize_DNA = population[:, DNA_PREBINNUM_SIZE : DNA_PREBINNUM_SIZE + DNA_POSTBINNUM_SIZE:]
    preBinNum = int(2**preBinNum_DNA-1)
    postBinNum = int(2**postBinNum_DNA-1)
    binSize = int(2**binSize_DNA-1)
    di= DataInstance("/N/project/Cortical_Calcium_Image/Miniscope data/05.2023_Tenth_group/AA058_D1/2023_05_05/11_02_42/Miniscope_2/S4/config.ini",['ALP','IALP','RNFS'] )
    RNFS_time = di.events['RNFS'].timesteps
    binList_list = []
    for i in RNFS_time:
        binList = di.events['RNFS'].get_binList(i,preBinNum,postBinNum,binSize)
        binList_list.append(binList)

def mutation(dna, mutation_rate):
    if np.random.rand() < MUTATION_RATE:
        mutate_point = np.random.randint(0, DNA_PREBINNUM_SIZE + DNA_POSTBINNUM_SIZE + DNA_BINSIZE_SIZE)
        dna[mutate_point] = dna[mutate_point] ^ 1

def encoded_dna(preBinNum_input,postBinNum_input,binSize_input):
    preBinNum = []
    postBinNum = []
    binSize = []
    while(preBinNum_input>0 & len(preBinNum) < DNA_PREBINNUM_SIZE):
        preBinNum.insert(0, preBinNum_input % 2)
        preBinNum_input = int(preBinNum_input / 2)
    while(postBinNum_input>0):
        postBinNum.insert(0, postBinNum_input % 2)
        postBinNum_input = int(postBinNum_input / 2)
    while(binSize_input>0):
        binSize.insert(0,binSize_input % 2)
        binSize_input = int(binSize_input/2)
    dna = preBinNum+postBinNum+binSize
    return np.array(dna)

# def decoded_dna(population):
#     preBinNum_dna = population[:,:DNA_PREBINNUM_SIZE]
#     postBinNum_dna = population[:,]
#     binSize_dna = population[:]

    

def select(population):
    return


def main():
    preBinNum_input = input('Input preBinNum (0-9)')
    postBinNum_input = input('Input postBinNum (0-9)')
    binSize_input = input('Input binSize (0-19)')
    parentA = encoded_dna(preBinNum_input,postBinNum_input,binSize_input)
    preBinNum_input2 = input('Input preBinNum (0-9)')
    postBinNum_input2 = input('Input postBinNum (0-9)')
    binSize_input2 = input('Input binSize (0-19)')
    parentB = encoded_dna(preBinNum_input2,postBinNum_input2,binSize_input2)

    symbol = 1
    population = [parentA,parentB]
    while(symbol==1):
        while(len(population) < POPULATION_SIZE):

            
            continue  
        symbol = input("If you want to quit, input 0 else input 1")

if __name__ == "__main__":
    main()
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