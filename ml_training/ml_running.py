from ml_training.train import train
import matplotlib.pyplot as plt
#import seaborn as sns
import os
import numpy as np
import pickle

"""MOVE THIS FILE TO THE ROOT DIRECTORY OF THE PROJECT TO RUN"""

def standard_run():
    train_sizes = [1,2,5,10,15, 20]
    repeats = 5
    outputs = {}
    for train_size in train_sizes:
        outputs[train_size] = {}
        for i in range(repeats):
            output = train(train_size=train_size, test_size=5, cross_session=True)
            outputs[train_size][i] = output
    
    # Create a boxplot of the f1 scores where x is the train size
    f1_scores = {train_size: [outputs[train_size][i]["f1"] for i in range(repeats)] for train_size in train_sizes}
    # Clear the current figure
    plt.clf()
    sns.boxplot(data=f1_scores)
    plt.xlabel("Train Size (No. of Cells)")
    plt.ylabel("F1 Score")
    plt.title("F1 Score by Train Size")

    plt.show()



def cross_session_same_day():
    all_paths = [["./data/PL010/PL010_D1S1", "./data/PL010/PL010_D1S4"], ["./data/AA058/AA058_D1S1", "./data/AA058/AA058_D1S4"], ["./data/AA036/AA036_D2S1", "./data/AA036/AA036_D2S4"], ["./data/AA034/AA034_D1S1", "./data/AA034/AA034_D1S4"]]
    save_path = "./ml_training/test_results/cross_session_same_day/"
    # Check if the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for paths in all_paths:
        session_name = paths[0].split("/")[-1]
        session_path = os.path.join(save_path, session_name)
        if not os.path.exists(session_path):
            os.makedirs(session_path)
        train_sizes = [2,3,5,10,15,20]
        repeats = 10
        outputs = {}
        for train_size in train_sizes:
            outputs[train_size] = {}
            for i in range(repeats):
                output = train(**{"DATASET_PATH": paths,  "MODEL_TYPE": "GRU", "HIDDEN_SIZE": 30, "TRAIN_SIZE": train_size, "TEST_SIZE": 5, "CUSTOM_TEST": True})
                outputs[train_size][i] = output
        
        f1_scores = {train_size: [outputs[train_size][i]["f1"] for i in range(repeats)] for train_size in train_sizes}
        plt.clf()
        sns.boxplot(data=f1_scores)
        plt.xlabel("Train Size (No. of Cells)")
        plt.ylabel("F1 Score")
        plt.title("F1 Score by Train Size")

        plt.savefig(os.path.join(session_path, "f1_scores.png"))

        results = []
        for train_size in train_sizes:
            for i in range(repeats):
                output = outputs[train_size][i]
                gt = output["gt"]
                preds = output["preds"]
                results.append(gt)
                results.append(preds)
        
        # Save to csv
        results = np.array(results)
        np.savetxt(os.path.join(session_path, session_name + "_results.csv"), results, delimiter=",")

        # Since the result lengths can be different we need to save them line by line
        for i in range(len(results)):
            with open(os.path.join(save_path, "all_results.csv"), "a") as f:
                np.savetxt(f, np.array([results[i]]), delimiter=",")

        # Pickle Save the output["test_indices"]
        test_indices = {"all": {"GRU": output["test_indices"]}}
        with open(os.path.join(session_path, "gru_all_test_indices.pkl"), "wb") as f:
            pickle.dump(test_indices, f)

        train_indices = {"all": {"GRU": output["train_indices"]}}
        with open(os.path.join(session_path, "gru_all_train_indices.pkl"), "wb") as f:
            pickle.dump(train_indices, f)

def cross_day_same_session():
    all_paths = [["./data/PL010/PL010_D1S1", "./data/PL010/PL010_D8S1"], ["./data/AA058/AA058_D1S1", "./data/AA058/AA058_D5S1"], ["./data/AA036/AA036_D2S1", "./data/AA036/AA036_D6S1"], ["./data/AA034/AA034_D1S1", "./data/AA034/AA034_D7S1"]]
    save_path = "./ml_training/test_results/cross_day_same_session/"
    for paths in all_paths:
        session_name = paths[0].split("/")[-1]
        session_path = os.path.join(save_path, session_name)
        if not os.path.exists(session_path):
            os.makedirs(session_path)
        train_sizes = [2,3,5,10,15,20]
        repeats = 10
        outputs = {}
        for train_size in train_sizes:
            outputs[train_size] = {}
            for i in range(repeats):
                output = train(**{"DATASET_PATH": paths,  "MODEL_TYPE": "GRU", "HIDDEN_SIZE": 30, "TRAIN_SIZE": train_size, "TEST_SIZE": 5, "CUSTOM_TEST": True})
                outputs[train_size][i] = output
        
        f1_scores = {train_size: [outputs[train_size][i]["f1"] for i in range(repeats)] for train_size in train_sizes}
        plt.clf()
        sns.boxplot(data=f1_scores)
        plt.xlabel("Train Size (No. of Cells)")
        plt.ylabel("F1 Score")
        plt.title("F1 Score by Train Size")

        plt.savefig(os.path.join(session_path, "f1_scores.png"))

        results = []
        for train_size in train_sizes:
            for i in range(repeats):
                output = outputs[train_size][i]
                gt = output["gt"]
                preds = output["preds"]
                results.append(gt)
                results.append(preds)
        
        # Save to csv
        results = np.array(results)
        np.savetxt(os.path.join(session_path, session_name + "_results.csv"), results, delimiter=",")

        # Pickle Save the output["test_indices"]
        test_indices = {"all": {"GRU": output["test_indices"]}}
        with open(os.path.join(session_path, "gru_all_test_indices.pkl"), "wb") as f:
            pickle.dump(test_indices, f)

        train_indices = {"all": {"GRU": output["train_indices"]}}
        with open(os.path.join(session_path, "gru_all_train_indices.pkl"), "wb") as f:
            pickle.dump(train_indices, f)

def cross_day_cross_session():
    all_paths = [["./data/PL010/PL010_D1S1", "./data/PL010/PL010_D8S4"], ["./data/AA058/AA058_D1S1", "./data/AA058/AA058_D5S4"], ["./data/AA036/AA036_D2S1", "./data/AA036/AA036_D6S4"], ["./data/AA034/AA034_D1S1", "./data/AA034/AA034_D7S4"]]
    save_path = "./ml_training/test_results/cross_day_cross_session/"
    for paths in all_paths:
        session_name = paths[0].split("/")[-1]
        session_path = os.path.join(save_path, session_name)
        if not os.path.exists(session_path):
            os.makedirs(session_path)
        train_sizes = [2,3,5,10,15,20]
        repeats = 10
        outputs = {}
        for train_size in train_sizes:
            outputs[train_size] = {}
            for i in range(repeats):
                output = train(**{"DATASET_PATH": paths,  "MODEL_TYPE": "GRU", "HIDDEN_SIZE": 30, "TRAIN_SIZE": train_size, "TEST_SIZE": 5, "CUSTOM_TEST": True})
                outputs[train_size][i] = output
        
        f1_scores = {train_size: [outputs[train_size][i]["f1"] for i in range(repeats)] for train_size in train_sizes}
        plt.clf()
        sns.boxplot(data=f1_scores)
        plt.xlabel("Train Size (No. of Cells)")
        plt.ylabel("F1 Score")
        plt.title("F1 Score by Train Size")

        plt.savefig(os.path.join(session_path, "f1_scores.png"))

        results = []
        for train_size in train_sizes:
            for i in range(repeats):
                output = outputs[train_size][i]
                gt = output["gt"]
                preds = output["preds"]
                results.append(gt)
                results.append(preds)
        
        # Save to csv
        results = np.array(results)
        np.savetxt(os.path.join(session_path, session_name + "_results.csv"), results, delimiter=",")

        # Pickle Save the output["test_indices"]
        test_indices = {"all": {"GRU": output["test_indices"]}}
        with open(os.path.join(session_path, "gru_all_test_indices.pkl"), "wb") as f:
            pickle.dump(test_indices, f)

        train_indices = {"all": {"GRU": output["train_indices"]}}
        with open(os.path.join(session_path, "gru_all_train_indices.pkl"), "wb") as f:
            pickle.dump(train_indices, f)

def within_session():
    all_paths = ["./data/PL010/PL010_D1S1", "./data/AA058/AA058_D1S1", "./data/AA036/AA036_D2S1", "./data/AA034/AA034_D1S1"]
    save_path = "./ml_training/test_results/within_session/"
    for path in all_paths:
        session_name = path.split("/")[-1]
        session_path = os.path.join(save_path, session_name)
        if not os.path.exists(session_path):
            os.makedirs(session_path)
        train_sizes = [2,3,5,10,15,20]
        repeats = 10
        outputs = {}
        for train_size in train_sizes:
            outputs[train_size] = {}
            for i in range(repeats):
                output = train(**{"DATASET_PATH": [path], "MODEL_TYPE": "GRU", "HIDDEN_SIZE": 30, "TRAIN_SIZE": train_size, "TEST_SIZE": 5})
                outputs[train_size][i] = output
        
        f1_scores = {train_size: [outputs[train_size][i]["f1"] for i in range(repeats)] for train_size in train_sizes}
        plt.clf()
        sns.boxplot(data=f1_scores)
        plt.xlabel("Train Size (No. of Cells)")
        plt.ylabel("F1 Score")
        plt.title("F1 Score by Train Size")

        plt.savefig(os.path.join(session_path, "f1_scores.png"))

        results = []
        for train_size in train_sizes:
            for i in range(repeats):
                output = outputs[train_size][i]
                gt = output["gt"]
                preds = output["preds"]
                results.append(gt)
                results.append(preds)
        
        # Save to csv
        results = np.array(results)
        np.savetxt(os.path.join(session_path, session_name + "_results.csv"), results, delimiter=",")

        # Since the result lengths can be different we need to save them line by line
        for i in range(len(results)):
            with open(os.path.join(session_path, "all_results.csv"), "a") as f:
                np.savetxt(f, np.array([results[i]]), delimiter=",")

        # Pickle Save the output["test_indices"]
        test_indices = {"all": {"GRU": output["test_indices"]}}
        with open(os.path.join(session_path, "gru_all_test_indices.pkl"), "wb") as f:
            pickle.dump(test_indices, f)

        train_indices = {"all": {"GRU": output["train_indices"]}}
        with open(os.path.join(session_path, "gru_all_train_indices.pkl"), "wb") as f:
            pickle.dump(train_indices, f)


def cross_animal():
    all_paths = [["./data/PL010/PL010_D1S1", "./data/AA058/AA058_D1S1"], ["./data/AA058/AA058_D1S1", "./data/AA036/AA036_D2S1"], ["./data/AA036/AA036_D2S1", "./data/AA034/AA034_D1S1"], ["./data/AA034/AA034_D1S1", "./data/PL010/PL010_D1S1"]]
    save_path = "./ml_training/test_results/cross_animal/"
    # Check if the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for paths in all_paths:
        session_name = paths[0].split("/")[-1]
        session_path = os.path.join(save_path, session_name)
        if not os.path.exists(session_path):
            os.makedirs(session_path)
        train_sizes = [1,2,5,10,15,20]
        repeats = 5
        outputs = {}
        for train_size in train_sizes:
            outputs[train_size] = {}
            for i in range(repeats):
                output = train(paths=paths, train_size=train_size, test_size=5, cross_session=True)
                outputs[train_size][i] = output
        
        f1_scores = {train_size: [outputs[train_size][i]["f1"] for i in range(repeats)] for train_size in train_sizes}
        plt.clf()
        sns.boxplot(data=f1_scores)
        plt.xlabel("Train Size (No. of Cells)")
        plt.ylabel("F1 Score")
        plt.title("F1 Score by Train Size")

        plt.savefig(os.path.join(session_path, "f1_scores.png"))

        results = []
        for train_size in train_sizes:
            for i in range(repeats):
                output = outputs[train_size][i]
                gt = output["gt"]
                preds = output["preds"]
                results.append(gt)
                results.append(preds)
        
        # Save to csv
        results = np.array(results)
        np.savetxt(os.path.join(session_path, session_name + "_results.csv"), results, delimiter=",")

def train_on_all_GRU():
    all_paths = ["./data/PL010/PL010_D1S1", "./data/PL010/PL010_D1S4", "./data/PL010/PL010_D8S1", "./data/PL010/PL010_D8S4",
                 "./data/AA058/AA058_D1S1", "./data/AA058/AA058_D1S4", "./data/AA058/AA058_D5S1", "./data/AA058/AA058_D5S4", 
                 "./data/AA036/AA036_D2S1", "./data/AA036/AA036_D2S4", "./data/AA036/AA036_D6S1", "./data/AA036/AA036_D6S4",
                 "./data/AA034/AA034_D1S1", "./data/AA034/AA034_D1S4", "./data/AA034/AA034_D7S1", "./data/AA034/AA034_D7S4"]
    
    save_path = "./ml_training/test_results/all/GRU/"
    # Check if the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    repeats = 1
    outputs = {"all": {}}
    for i in range(repeats):
        output = train(**{"PATHS": all_paths, "MODEL_TYPE": "GRU", "HIDDEN_SIZE": 30})
        outputs["all"][i] = output
    
    f1_scores = [outputs["all"][i]["f1"] for i in range(repeats)]
    #plt.clf()
    #sns.boxplot(data=f1_scores)
    #plt.ylabel("F1 Score")
    #plt.title("F1 Score by Train Size")

    #plt.savefig(os.path.join(save_path, "f1_scores.png"))

    results = []
    for i in range(repeats):
        output = outputs["all"][i]
        gt = output["gt"]
        preds = output["preds"]
        results.append(gt)
        results.append(preds)

    # Since the result lengths can be different we need to save them line by line
    for i in range(len(results)):
        with open(os.path.join(save_path, "all_results.csv"), "a") as f:
            np.savetxt(f, np.array([results[i]]), delimiter=",")

    # Pickle Save the output["test_indices"]
    test_indices = {"all": {"GRU": output["test_indices"]}}
    with open(os.path.join(save_path, "gru_all_test_indices.pkl"), "wb") as f:
        pickle.dump(test_indices, f)

def train_on_all_LSTM():
    all_paths = ["./data/PL010/PL010_D1S1", "./data/PL010/PL010_D1S4", "./data/PL010/PL010_D8S1", "./data/PL010/PL010_D8S4",
                 "./data/AA058/AA058_D1S1", "./data/AA058/AA058_D1S4", "./data/AA058/AA058_D5S1", "./data/AA058/AA058_D5S4", 
                 "./data/AA036/AA036_D2S1", "./data/AA036/AA036_D2S4", "./data/AA036/AA036_D6S1", "./data/AA036/AA036_D6S4",
                 "./data/AA034/AA034_D1S1", "./data/AA034/AA034_D1S4", "./data/AA034/AA034_D7S1", "./data/AA034/AA034_D7S4"]
    
    save_path = "./ml_training/test_results/all/LSTM/"
    # Check if the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    repeats = 10
    outputs = {"all": {}}
    for i in range(repeats):
        output = train(**{"PATHS": all_paths, "MODEL_TYPE": "LSTM"})
        outputs["all"][i] = output
    
    f1_scores = [outputs["all"][i]["f1"] for i in range(repeats)]
    plt.clf()
    sns.boxplot(data=f1_scores)
    plt.ylabel("F1 Score")
    plt.title("F1 Score by Train Size")

    plt.savefig(os.path.join(save_path, "f1_scores.png"))

    results = []
    for i in range(repeats):
        output = outputs["all"][i]
        gt = output["gt"]
        preds = output["preds"]
        results.append(gt)
        results.append(preds)

    # Since the result lengths can be different we need to save them line by line
    for i in range(len(results)):
        with open(os.path.join(save_path, "all_results.csv"), "a") as f:
            np.savetxt(f, np.array([results[i]]), delimiter=",")

    # Pickle Save the output["test_indices"]
    test_indices = {"all": {"LSTM": output["test_indices"]}}
    with open(os.path.join(save_path, "lstm_all_test_indices.pkl"), "wb") as f:
        pickle.dump(test_indices, f)


def train_on_all_Transformer():
    all_paths = ["./data/PL010/PL010_D1S1", "./data/PL010/PL010_D1S4", "./data/PL010/PL010_D8S1", "./data/PL010/PL010_D8S4",
                 "./data/AA058/AA058_D1S1", "./data/AA058/AA058_D1S4", "./data/AA058/AA058_D5S1", "./data/AA058/AA058_D5S4", 
                 "./data/AA036/AA036_D2S1", "./data/AA036/AA036_D2S4", "./data/AA036/AA036_D6S1", "./data/AA036/AA036_D6S4",
                 "./data/AA034/AA034_D1S1", "./data/AA034/AA034_D1S4", "./data/AA034/AA034_D7S1", "./data/AA034/AA034_D7S4"]
    
    save_path = "./ml_training/test_results/all/Transformer/"
    # Check if the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    repeats = 10
    outputs = {"all": {}}
    for i in range(repeats):
        output = train(**{"PATHS": all_paths, "MODEL_TYPE": "BasicTransformer", "NUM_EPOCHS": 1,
                          "HIDDEN_SIZE": 42, "NUM_LAYERS": 3, "HEADS": 2})
        outputs["all"][i] = output
    
    f1_scores = [outputs["all"][i]["f1"] for i in range(repeats)]
    plt.clf()
    sns.boxplot(data=f1_scores)
    plt.ylabel("F1 Score")
    plt.title("F1 Score by Train Size")

    plt.savefig(os.path.join(save_path, "f1_scores.png"))

    results = []
    for i in range(repeats):
        output = outputs["all"][i]
        gt = output["gt"]
        preds = output["preds"]
        results.append(gt)
        results.append(preds)

    # Since the result lengths can be different we need to save them line by line
    for i in range(len(results)):
        with open(os.path.join(save_path, "all_results.csv"), "a") as f:
            np.savetxt(f, np.array([results[i]]), delimiter=",")

    # Pickle Save the output["test_indices"]
    test_indices = {"all": {"Transformer": output["test_indices"]}}
    with open(os.path.join(save_path, "transformer_all_test_indices.pkl"), "wb") as f:
        pickle.dump(test_indices, f)

def cross_animal_comprehensive():
    all_paths = {"PL010-AA058": [["./data/PL010/PL010_D1S1", "./data/PL010/PL010_D1S4", "./data/PL010/PL010_D8S1", "./data/PL010/PL010_D8S4"],["./data/AA058/AA058_D1S1", "./data/AA058/AA058_D1S4", "./data/AA058/AA058_D5S1", "./data/AA058/AA058_D5S4"]],
                 "PL010-AA036": [["./data/PL010/PL010_D1S1", "./data/PL010/PL010_D1S4", "./data/PL010/PL010_D8S1", "./data/PL010/PL010_D8S4"],["./data/AA036/AA036_D2S1", "./data/AA036/AA036_D2S4", "./data/AA036/AA036_D6S1", "./data/AA036/AA036_D6S4"]],
                 "PL010-AA034": [["./data/PL010/PL010_D1S1", "./data/PL010/PL010_D1S4", "./data/PL010/PL010_D8S1", "./data/PL010/PL010_D8S4"],["./data/AA034/AA034_D1S1", "./data/AA034/AA034_D1S4", "./data/AA034/AA034_D7S1", "./data/AA034/AA034_D7S4"]],
                 "AA058-AA036": [["./data/AA058/AA058_D1S1", "./data/AA058/AA058_D1S4", "./data/AA058/AA058_D5S1", "./data/AA058/AA058_D5S4"],["./data/AA036/AA036_D2S1", "./data/AA036/AA036_D2S4", "./data/AA036/AA036_D6S1", "./data/AA036/AA036_D6S4"]],
                 "AA058-AA034": [["./data/AA058/AA058_D1S1", "./data/AA058/AA058_D1S4", "./data/AA058/AA058_D5S1", "./data/AA058/AA058_D5S4"],["./data/AA034/AA034_D1S1", "./data/AA034/AA034_D1S4", "./data/AA034/AA034_D7S1", "./data/AA034/AA034_D7S4"]],
                 "AA058-PL010": [["./data/AA058/AA058_D1S1", "./data/AA058/AA058_D1S4", "./data/AA058/AA058_D5S1", "./data/AA058/AA058_D5S4"],["./data/PL010/PL010_D1S1", "./data/PL010/PL010_D1S4", "./data/PL010/PL010_D8S1", "./data/PL010/PL010_D8S4"]],
                 "AA036-AA034": [["./data/AA036/AA036_D2S1", "./data/AA036/AA036_D2S4", "./data/AA036/AA036_D6S1", "./data/AA036/AA036_D6S4"],["./data/AA034/AA034_D1S1", "./data/AA034/AA034_D1S4", "./data/AA034/AA034_D7S1", "./data/AA034/AA034_D7S4"]],
                 "AA036-PL010": [["./data/AA036/AA036_D2S1", "./data/AA036/AA036_D2S4", "./data/AA036/AA036_D6S1", "./data/AA036/AA036_D6S4"],["./data/PL010/PL010_D1S1", "./data/PL010/PL010_D1S4", "./data/PL010/PL010_D8S1", "./data/PL010/PL010_D8S4"]],
                 "AA036-AA058": [["./data/AA036/AA036_D2S1", "./data/AA036/AA036_D2S4", "./data/AA036/AA036_D6S1", "./data/AA036/AA036_D6S4"],["./data/AA058/AA058_D1S1", "./data/AA058/AA058_D1S4", "./data/AA058/AA058_D5S1", "./data/AA058/AA058_D5S4"]],
                 "AA034-PL010": [["./data/AA034/AA034_D1S1", "./data/AA034/AA034_D1S4", "./data/AA034/AA034_D7S1", "./data/AA034/AA034_D7S4"],["./data/PL010/PL010_D1S1", "./data/PL010/PL010_D1S4", "./data/PL010/PL010_D8S1", "./data/PL010/PL010_D8S4"]],
                 "AA034-AA058": [["./data/AA034/AA034_D1S1", "./data/AA034/AA034_D1S4", "./data/AA034/AA034_D7S1", "./data/AA034/AA034_D7S4"],["./data/AA058/AA058_D1S1", "./data/AA058/AA058_D1S4", "./data/AA058/AA058_D5S1", "./data/AA058/AA058_D5S4"]],
                 "AA034-AA036": [["./data/AA034/AA034_D1S1", "./data/AA034/AA034_D1S4", "./data/AA034/AA034_D7S1", "./data/AA034/AA034_D7S4"],["./data/AA036/AA036_D2S1", "./data/AA036/AA036_D2S4", "./data/AA036/AA036_D6S1", "./data/AA036/AA036_D6S4"]]}
    
    for name, paths in all_paths.items():
        save_path = "./ml_training/test_results/cross_animal_comprehensive/" + name + "/"
        # Check if the save path exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        train_sizes = [2,3,5,10,15,20]
        repeats = 10
        outputs = {}
        for train_size in train_sizes:
            print(f"TRAIN SIZE: {train_size}")
            outputs[train_size] = {}
            for i in range(repeats):
                output = train(**{"DATASET_PATH": paths, "MODEL_TYPE": "GRU", "HIDDEN_SIZE": 30,
                                   "TEST_SIZE": 5, "TRAIN_SIZE": train_size, "CUSTOM_TEST": True})
                outputs[train_size][i] = output
        
        f1_scores = {train_size: [outputs[train_size][i]["f1"] for i in range(repeats)] for train_size in train_sizes}
        plt.clf()
        sns.boxplot(data=f1_scores)
        plt.xlabel("Train Size (No. of Cells)")
        plt.ylabel("F1 Score")
        plt.title("F1 Score by Train Size")

        plt.savefig(os.path.join(save_path, "f1_scores.png"))

        results = []
        for train_size in train_sizes:
            for i in range(repeats):
                output = outputs[train_size][i]
                gt = output["gt"]
                preds = output["preds"]
                results.append(gt)
                results.append(preds)
        
        # Since the result lengths can be different we need to save them line by line
        for i in range(len(results)):
            with open(os.path.join(save_path, "all_results.csv"), "a") as f:
                np.savetxt(f, np.array([results[i]]), delimiter=",")

        test_indices = {"all": {"GRU": output["test_indices"]}}
        with open(os.path.join(save_path, "gru_all_test_indices.pkl"), "wb") as f:
            pickle.dump(test_indices, f)
        
        train_indices = {"all": {"GRU": output["train_indices"]}}
        with open(os.path.join(save_path, "gru_all_train_indices.pkl"), "wb") as f:
            pickle.dump(train_indices, f)


def train_on_all_LocalTransformer():
    all_paths = ["./data/PL010/PL010_D1S1", "./data/PL010/PL010_D1S4", "./data/PL010/PL010_D8S1", "./data/PL010/PL010_D8S4",
                 "./data/AA058/AA058_D1S1", "./data/AA058/AA058_D1S4", "./data/AA058/AA058_D5S1", "./data/AA058/AA058_D5S4", 
                 "./data/AA036/AA036_D2S1", "./data/AA036/AA036_D2S4", "./data/AA036/AA036_D6S1", "./data/AA036/AA036_D6S4",
                 "./data/AA034/AA034_D1S1", "./data/AA034/AA034_D1S4", "./data/AA034/AA034_D7S1", "./data/AA034/AA034_D7S4"]
    
    save_path = "./ml_training/test_results/all/LocalTransformer/"
    # Check if the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    repeats = 10
    outputs = {"all": {}}
    for i in range(repeats):
        output = train(**{"PATHS": all_paths, "MODEL_TYPE": "LocalTransformer", "NUM_EPOCHS": 1,
                          "HIDDEN_SIZE": 10, "NUM_LAYERS": 3, "HEADS": 2})
        outputs["all"][i] = output
    
    f1_scores = [outputs["all"][i]["f1"] for i in range(repeats)]
    plt.clf()
    sns.boxplot(data=f1_scores) 
    plt.ylabel("F1 Score")
    plt.title("F1 Score by Train Size")

    plt.savefig(os.path.join(save_path, "f1_scores.png"))

    results = []
    for i in range(repeats):
        output = outputs["all"][i]
        gt = output["gt"]
        preds = output["preds"]
        results.append(gt)
        results.append(preds)

    # Since the result lengths can be different we need to save them line by line
    for i in range(len(results)):
        with open(os.path.join(save_path, "all_results.csv"), "a") as f:
            np.savetxt(f, np.array([results[i]]), delimiter=",")

    # Pickle Save the output["test_indices"]
    test_indices = {"all": {"Transformer": output["test_indices"]}}
    with open(os.path.join(save_path, "local_transformer_all_test_indices.pkl"), "wb") as f:
        pickle.dump(test_indices, f)

if __name__ == "__main__": 
    train_on_all_GRU()