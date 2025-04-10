from ml_training.dataset import CustomDataset, train_val_test_split, extract_data, train_val_test_split_custom
from ml_training.model import LocalTransformer, GRU, BasicTransformer, LSTM
from ml_training import config
from ml_training.config import update_config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from ml_training.ml_util import sequence_to_predictions
from caltrig.core.backend import open_minian
import matplotlib.pyplot as plt
import torch
import time
import numpy as np
import os
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_score, recall_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay


def train(**kwargs):
	update_config(kwargs)
	# For saving purposes get the hour and minute and date of the run
	t = time.localtime()
	current_time = time.strftime("%m_%d_%H_%M", t)
	output_path = os.path.sep.join([config.BASE_OUTPUT, current_time])
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	plot_path = os.path.sep.join([output_path, "plots"])
	if not os.path.exists(plot_path):
		os.makedirs(plot_path)
	
	# load the image and mask filepaths in a sorted manner
	paths = config.DATASET_PATH
	if config.CUSTOM_TEST:
		if type(paths[0]) == str:
			paths[0] = [paths[0]]
			paths[1] = [paths[1]]
		train_unit_ids, val_unit_ids, test_unit_ids = train_val_test_split_custom(paths[0], paths[1], config.TRAIN_SIZE, config.VAL_SIZE, config.TEST_SIZE)
	else:
		train_unit_ids, val_unit_ids, test_unit_ids = train_val_test_split(paths, train_total=config.TRAIN_SIZE, test_split=config.TEST_SIZE, val_split=config.VAL_SIZE)

	# create the train and test datasets
	trainDS = CustomDataset(train_unit_ids, section_len=config.SECTION_LEN, rolling=config.ROLLING, slack=config.SLACK, only_events=False)
	valDS = CustomDataset(val_unit_ids, section_len=config.SECTION_LEN, rolling=config.ROLLING, slack=config.SLACK, only_events=False)
	
	# create the training and test data loaders
	trainLoader = DataLoader(trainDS, shuffle=True,
		batch_size=config.BATCH_SIZE, num_workers=0)
	valLoader = DataLoader(valDS, shuffle=False,
		batch_size=config.BATCH_SIZE, num_workers=0)

	# initialize our model
	if config.MODEL_TYPE == "LocalTransformer":
		model = LocalTransformer(inputs=config.INPUT, local_attn_window_size=config.HIDDEN_SIZE, 
											max_seq_len=2*config.SLACK+config.SECTION_LEN, depth=config.NUM_LAYERS, 
											causal=False, look_forward=5, look_backward=5, 
											exact_windowsize=True, slack=config.SLACK,
											sequence_len=config.SECTION_LEN, heads=config.HEADS).to(config.DEVICE)
		model_name = "local_transformer_"

	elif config.MODEL_TYPE == "BasicTransformer":
		model = BasicTransformer(sequence_len=config.SECTION_LEN, slack=config.SLACK, inputs=config.INPUT, 
						   hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, 
						   num_heads=config.HEADS, classes=1).to(config.DEVICE)
		model_name = "basic_transformer_"
	elif config.MODEL_TYPE == "lstm":
		model = LSTM(inputs=config.INPUT, hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, sequence_len=config.SECTION_LEN, slack=config.SLACK).to(config.DEVICE)
		model_name = "lstm_"
	else:
		model = GRU(inputs=config.INPUT, hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, sequence_len=config.SECTION_LEN, slack=config.SLACK).to(config.DEVICE)
		model_name = "gru_"
	# initialize loss function and optimizer

	lossFunc = BCEWithLogitsLoss()
	opt = Adam(model.parameters(), lr=config.INIT_LR)
	# calculate steps per epoch for training and validation set
	trainSteps = len(trainDS) // config.BATCH_SIZE
	valSteps = np.max([len(valDS) // config.BATCH_SIZE, 1])
	lowest_loss = np.inf
	# initialize a dictionary to store training history
	H = {"train_loss": [], "val_loss": []}

	# loop over epochs
	print("[INFO] training the network...")
	startTime = time.time()
	"""
	Due to the way we have set up our dataset, we have both a small and large epoch. The small epoch is per cell and the large epoch is per dataset.
	This is necessary as we need to save the hidden states on each intermediate pass
	"""
	for e in tqdm(range(config.NUM_EPOCHS), leave=False):
		# initialize the total training and validation loss
		totalTrainLoss = 0
		totalValLoss = 0
		model.train()
		# loop over the training set
		for (i, (inputs, target)) in enumerate(tqdm(trainLoader, leave=False)):
			# unpack the data and make sure they are on the same device
			x, y = inputs, target
			# perform a forward pass and calculate the training loss
			pred = model(x)
			loss = lossFunc(pred, y)
			# first, zero out any previously accumulated gradients, then
			# perform backpropagation, and then update model parameters
			opt.zero_grad()
			loss.backward()
			opt.step()
			# add the loss to the total training loss so far
			totalTrainLoss += loss
			
		# switch off autograd
		with torch.no_grad():
			# set the model in evaluation mode
			model.eval()
			for (i, (inputs, target)) in enumerate(tqdm(valLoader, leave=False)):
				# unpack the data and make sure they are on the same device
				x, y = inputs.to(config.DEVICE), target.to(config.DEVICE)
				# perform a forward pass and calculate the training loss
				pred = model(x)
				totalValLoss += lossFunc(pred, y)
		# calculate the average training and validation loss
		avgTrainLoss = totalTrainLoss / trainSteps
		avgValLoss = totalValLoss / valSteps
		# update our training history
		H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
		H["val_loss"].append(avgValLoss.cpu().detach().numpy())
		# print the model training and validation information
		print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
		print("Train loss: {:.6f}, Val loss: {:.4f}".format(
			avgTrainLoss, avgValLoss))

		# save the model if the validation loss has decreased
		if avgValLoss < lowest_loss:
			print("[INFO] saving the model...")
			name = model_name + "model_val_" + current_time + ".pth"
			model_val_path = os.path.sep.join([output_path, name])
			torch.save(model, model_val_path)
			lowest_loss = avgValLoss
		
		
	
	# display the total time needed to perform the training
	endTime = time.time()
	print("[INFO] total time taken to train the model: {:.2f}s".format(
		endTime - startTime))

	# plot the training loss
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H["train_loss"], label="train_loss")
	plt.plot(H["val_loss"], label="val_loss")
	plt.title("Training Loss on Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.legend(loc="lower left")
	loss_path = os.path.sep.join([plot_path, "loss.png"])
	plt.savefig(loss_path)
	# Load the best model
	model = torch.load(model_val_path)

	# Start testing
	model.eval()
	
	# Test the model
	preds = []
	gt = []
	for path, unit_ids in test_unit_ids.items():
		minian_data = open_minian(path)
		for unit_id in unit_ids:
			input_data, output = extract_data(minian_data, unit_id, config.SLACK)
			pred = sequence_to_predictions(model, input_data, config.ROLLING, voting="max")
			preds.append(pred)
			gt.append(output.cpu().detach().numpy())
	
	preds = np.concatenate(preds)
	gt = np.concatenate(gt)

	# calculate the accuracy
	acc = accuracy_score(gt, preds.round())
	print("[INFO] Accuracy: {:.4f}".format(acc))
	# calculate Precision and Recall per class
	precision1 = precision_score(gt, preds.round())
	recall1 = recall_score(gt, preds.round())
	print("[INFO] Transient Event Precision: {:.4f}".format(precision1))
	print("[INFO] Transient Event Recall: {:.4f}".format(recall1))
	# precision and recall for other class
	precision2 = precision_score(gt, preds.round(), pos_label=0)
	recall2 = recall_score(gt, preds.round(), pos_label=0)
	print("[INFO] No Transient Event Precision: {:.4f}".format(precision2))
	print("[INFO] No Transient Event Recall: {:.4f}".format(recall2))
	# Get confusion Matrix plot
	cm = confusion_matrix(gt, preds.round())
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No TE", "TE"])
	# Save confusion matrix plot
	cm_path = os.path.sep.join([plot_path, "confusion_matrix.png"])
	disp.plot()
	plt.savefig(cm_path)	
	# calculate the F1 score and AUC ROC score	
	f1 = f1_score(gt, preds.round())
	print("[INFO] F1 score: {:.4f}".format(f1))
	# calculate the AUC ROC score
	auc = roc_auc_score(gt, preds)
	print("[INFO] AUC ROC score: {:.4f}".format(auc))
	# Visualize ROC
	fpr, tpr, _ = roc_curve(gt, preds)
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(fpr, tpr, label="ROC curve")
	plt.title("ROC Curve")
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.legend(loc="lower right")
	roc_path = os.path.sep.join([plot_path, "roc_curve.png"])
	plt.savefig(roc_path)

	# Create a text file with the parameters used
	if not config.CUSTOM_TEST:
		with open(os.path.sep.join([output_path, "parameters.txt"]), "w") as file:
			file.write("TYPE: Local Transformer\n")
			file.write("INIT_LR: {}\nNUM_EPOCHS: {}\nBATCH_SIZE: {}\nTHRESHOLD: {}\nTEST_SIZE: {}\nVAL_SIZE: {}\nSECTION_LEN: {}\nHIDDEN_SIZE: {}\nNUM_LAYERS: {}\n SLACK: {} \n ROLLING: {}\n".format(
				config.INIT_LR, config.NUM_EPOCHS, config.BATCH_SIZE, config.THRESHOLD, config.TEST_SIZE, config.VAL_SIZE, config.SECTION_LEN, config.HIDDEN_SIZE, config.NUM_LAYERS, config.SLACK, config.ROLLING))
			file.write("Accuracy: {:.4f}\n".format(acc))
			file.write("Transient Event Precision: {:.4f}\n".format(precision1))
			file.write("Transient Event Recall: {:.4f}\n".format(recall1))
			file.write("No Transient Event Precision: {:.4f}\n".format(precision2))
			file.write("No Transient Event Recall: {:.4f}\n".format(recall2))
			file.write("F1 score: {:.4f}\n".format(f1))
			file.write("AUC ROC score: {:.4f}\n".format(auc))
			file.write("Data used for training: \n")
			for data_type in config.INPUT:
				file.write(data_type + "\n")
			# Write the data used for the training
			file.write("Data paths for training: \n")
			for path in paths:
				file.write(path + "\n")

	# Create dict that keeps all the outputs
	outputs = {"accuracy": acc, "precision1": precision1, "recall1": recall1,
			 "precision0": precision2, "recall0": recall2, "f1": f1, "auc": auc,
			 "preds": preds, "gt": gt}
	
	# Extract the indices from the test set
	outputs["test_indices"] = test_unit_ids
	outputs["train_indices"] = train_unit_ids

	return outputs