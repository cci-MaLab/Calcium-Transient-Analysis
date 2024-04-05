# import the necessary packages
import torch
import os
# base path of the dataset
DATASET_PATH = ["/N/project/Cortical_Calcium_Image/Miniscope data/05.2023_Tenth_group/AA058_D1/2023_05_05/11_02_42/Miniscope_2/S1/minian",
                "/N/project/Cortical_Calcium_Image/Miniscope data/12.2022_Seventh_group/AA044_D1_2/2022_12_17/11_47_59/Miniscope_2/S1/minian"]
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 1
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = "ml_training/output"
# define the path to the output serialized model, model training
# plot, and testing image paths
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

TEST_SIZE = 0.1
VAL_SIZE = 0.1
SECTION_LEN=200
HIDDEN_SIZE=64