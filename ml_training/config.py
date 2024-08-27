# import the necessary packages
import torch
# base path of the dataset
DATASET_PATH = ["./data/PL010/PL010_D1S1", "./data/PL010/PL010_D1S4", "./data/PL010/PL010_D8S1", "./data/PL010/PL010_D8S4",
                 "./data/AA058/AA058_D1S1", "./data/AA058/AA058_D1S4", "./data/AA058/AA058_D5S1", "./data/AA058/AA058_D5S4", 
                 "./data/AA036/AA036_D2S1", "./data/AA036/AA036_D2S4", "./data/AA036/AA036_D6S1", "./data/AA036/AA036_D6S4",
                 "./data/AA034/AA034_D1S1", "./data/AA034/AA034_D1S4", "./data/AA034/AA034_D7S1", "./data/AA034/AA034_D7S4"]
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = False #True if DEVICE == "cuda" else False
STRATIFICATION = True

# initialize learning rate, number of epochs to train for, and the
# batch size
INPUT = ["C", "DFF"]
INIT_LR = 0.001
NUM_EPOCHS = 1
BATCH_SIZE = 1
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = "ml_training/output"
CUSTOM_TEST = False

TEST_SIZE = 0.1
TRAIN_SIZE = None
VAL_SIZE = 0.1
SECTION_LEN=200
HIDDEN_SIZE=30 
NUM_LAYERS=3
ROLLING=50
SLACK=50
HEADS=1

WEIGHT_MULTIPLIER = 1

MODEL_TYPE = "gru"


def update_config(config_dict):
    for key in config_dict:
        globals()[key] = config_dict[key]