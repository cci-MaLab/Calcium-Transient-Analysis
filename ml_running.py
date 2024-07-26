from ml_training.train_hidden import train
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pickle

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

def cross_day_same_session():
    all_paths = [["./data/PL010/PL010_D1S1", "./data/PL010/PL010_D8S1"], ["./data/AA058/AA058_D1S1", "./data/AA058/AA058_D5S1"], ["./data/AA036/AA036_D2S1", "./data/AA036/AA036_D6S1"], ["./data/AA034/AA034_D1S1", "./data/AA034/AA034_D7S1"]]
    save_path = "./ml_training/test_results/cross_day_same_session/"
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

def cross_day_cross_session():
    all_paths = [["./data/PL010/PL010_D1S1", "./data/PL010/PL010_D8S4"], ["./data/AA058/AA058_D1S1", "./data/AA058/AA058_D5S4"], ["./data/AA036/AA036_D2S1", "./data/AA036/AA036_D6S4"], ["./data/AA034/AA034_D1S1", "./data/AA034/AA034_D7S4"]]
    save_path = "./ml_training/test_results/cross_day_cross_session/"
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

def within_session():
    all_paths = ["./data/PAE-1030524-S8"]
    save_path = "./ml_training/test_results/within_session/"
    # Check if the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for path in all_paths:
        session_name = path.split("/")[-1]
        session_path = os.path.join(save_path, session_name)
        if not os.path.exists(session_path):
            os.makedirs(session_path)
        train_sizes = [1,2,5,10,15,20]
        repeats = 5
        outputs = {}
        for train_size in train_sizes:
            outputs[train_size] = {}
            for i in range(repeats):
                output = train(paths=[path], train_size=train_size, test_size=5, experiment_type="within_session")
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
    
    repeats = 5
    outputs = {"all": {}}
    for i in range(repeats):
        output = train(paths=all_paths, test_size=0.1, experiment_type="all", model_type="gru")
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
    
    repeats = 5
    outputs = {"all": {}}
    for i in range(repeats):
        output = train(paths=all_paths, test_size=0.1, experiment_type="all", model_type="lstm")
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
    with open(os.path.join(save_path, "gru_all_test_indices.pkl"), "wb") as f:
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
    
    repeats = 5
    outputs = {"all": {}}
    for i in range(repeats):
        output = train(paths=all_paths, test_size=0.1, experiment_type="all", model_type="transformer")
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
    with open(os.path.join(save_path, "gru_all_test_indices.pkl"), "wb") as f:
        pickle.dump(test_indices, f)

if __name__ == "__main__": 
    train_on_all_Transformer()