import sys
import os
sys.path.insert(0, ".")
from backend import DataInstance
from backend import Event
from genetic_algorithm import Genetic_Algorithm
import matplotlib.pyplot as plt



def main():
    # events = ['ALP','IALP']
    ga = Genetic_Algorithm()
    ga.execute()
    # preBinNum,postBinNum,binSize,fitness, examples= ga.return_results(0)
    preBinNum = ga.preBinNum
    postBinNum = ga.postBinNum
    binSize = ga.binSize
    fitness = ga._best_fitness
    examples = ga.examples
    frameline = ga.xvalues
    print('preBinNum: ',preBinNum)
    print('postBinNum: ' ,postBinNum)
    print('binSize: ' ,binSize)
    # print('xvalues',frameline[0])
    # print('examples:',examples)
        # print('AUCs: ' ,AUCs[i])

    

if __name__ == "__main__":
    main()