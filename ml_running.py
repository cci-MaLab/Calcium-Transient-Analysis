from ml_training.train_hidden import train

if __name__ == "__main__":
    train_sizes = [1,2,5,10,15, 20]
    repeats = 5
    outputs = {}
    for train_size in train_sizes:
        outputs[train_size] = {}
        for i in range(repeats):
            output = train(train_size=train_size, test_size=5, cross_session=True)
            outputs[train_size][i] = output
    
    # Create a boxplot of the f1 scores where x is the train size
    import matplotlib.pyplot as plt
    import seaborn as sns

    f1_scores = {train_size: [outputs[train_size][i]["f1"] for i in range(repeats)] for train_size in train_sizes}
    # Clear the current figure
    plt.clf()
    sns.boxplot(data=f1_scores)
    plt.xlabel("Train Size (No. of Cells)")
    plt.ylabel("F1 Score")
    plt.title("F1 Score by Train Size")

    plt.show()