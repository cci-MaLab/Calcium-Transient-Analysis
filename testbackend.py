import sys
import os
sys.path.insert(0, ".")
from backend import DataInstance
from backend import Event
import matplotlib.pyplot as plt

def main():
    events = ['ALP','IALP']
    newsession = DataInstance("/N/project/Cortical_Calcium_Image/Miniscope data/05.2023_Tenth_group/AA058_D1/2023_05_05/11_02_42/Miniscope_2/S4/config.ini",events )
    for i in events:
        newsession.events[i].set_delay_and_duration(0,20)
        newsession.events[i].set_values()
    # newsession.outliers_list = [1,2]
    newsession.set_vector()
    print(newsession.values)  
    newsession.compute_clustering()
    # print(newsession.linkage_data)
    # plt.imshow(newsession.clustering_result)

if __name__ == "__main__":
    main()