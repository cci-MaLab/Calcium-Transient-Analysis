# import the necessary packages
import torch
# base path of the dataset
DATASET_PATH = ["./data/ml_data1",
                "./data/ml_data2",
                "./data/ml_data3"]
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = False #True if DEVICE == "cuda" else False

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 1
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = "ml_training/output"

TEST_SIZE = 0.1
VAL_SIZE = 0.1
SECTION_LEN=200
HIDDEN_SIZE=32
NUM_LAYERS=3
ROLLING=50
SLACK=50
HEADS=1

WEIGHT_MULTIPLIER = 1.5

MODEL_TYPE = "LSTM"