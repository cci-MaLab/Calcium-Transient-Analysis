# import the necessary packages
import torch
import os
# base path of the dataset
DATASET_PATH = os.path.join("archive", "chest_xray")
# define the path to the train, test and validation
TRAIN_PATH_NORMAL = os.path.join(DATASET_PATH, "train", "NORMAL")
TRAIN_PATH_PNEUMONIA = os.path.join(DATASET_PATH, "train", "PNEUMONIA")
TEST_PATH_NORMAL = os.path.join(DATASET_PATH, "test", "NORMAL")
TEST_PATH_PNEUMONIA = os.path.join(DATASET_PATH, "test", "PNEUMONIA")
VALIDATION_PATH_NORMAL = os.path.join(DATASET_PATH, "val", "NORMAL")
VALIDATION_PATH_PNEUMONIA = os.path.join(DATASET_PATH, "val", "PNEUMONIA")
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 5
BATCH_SIZE = 20
# define the input image dimensions
INPUT_IMAGE_WIDTH = 512
INPUT_IMAGE_HEIGHT = 512
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "cnn_pneumonia.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

TEST_SIZE = 0.1
VAL_SIZE = 0.1