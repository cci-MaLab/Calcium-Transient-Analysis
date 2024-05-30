from ml_training.dataset_hidden import (GRUDataset, TestDataset, ValDataset)
from ml_training.model import GRU_Hidden
from ml_training import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import numpy as np
import os
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_score, recall_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay


def train(paths=None, train_size=None, test_size=None, cross_session=False):
	test_size = test_size if test_size is not None else config.TEST_SIZE 
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
	paths = config.DATASET_PATH if paths is None else paths

	# create the train and test datasets
	trainDS = GRUDataset(paths=paths, train_size=train_size, test_split=test_size, cross_session=cross_session,
					     val_split=config.VAL_SIZE, section_len=config.SECTION_LEN, stratification=config.STRATIFICATION)
	valDS = ValDataset(data=trainDS.get_data())
	testDS = TestDataset(data=trainDS.get_data())
	# create the training and test data loaders
	trainLoader = DataLoader(trainDS, shuffle=True,
		batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
		num_workers=0)
	testLoader = DataLoader(testDS, shuffle=False,
		batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
		num_workers=0)
	valLoader = DataLoader(valDS, shuffle=False,
		batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
		num_workers=0)

	# initialize our CNN model
	gru = GRU_Hidden(hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, inputs=config.INPUT).to(config.DEVICE)
	# initialize loss function and optimizer

	lossFunc = BCEWithLogitsLoss()
	opt = Adam(gru.parameters(), lr=config.INIT_LR)
	# calculate steps per epoch for training and validation set
	trainSteps = trainDS.get_training_steps() // config.BATCH_SIZE
	valSteps = np.max([valDS.get_val_steps() // config.BATCH_SIZE, 1])
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
		for m in tqdm(range(len(trainDS.data)), leave=False):
			trainDS.intermediate_epoch = m
			for u in tqdm(range(trainDS.get_mouse_cell_count()), leave=False):
				trainDS.small_epoch = u
				# We need to get the local hidden states for the current unit
				with torch.no_grad():
					gru.eval()
					sample = trainDS.get_current_sample()
					trainDS.hidden_states = gru.forward_hidden(sample.to(config.DEVICE))

				# set the model in training mode
				gru.train()
				# loop over the training set
				for (i, (inputs, hidden, target)) in enumerate(tqdm(trainLoader, leave=False)):
					# unpack the data and make sure they are on the same device
					x, h0, y = inputs.to(config.DEVICE), hidden, target.to(config.DEVICE)
					# Batch dimension has to be second for hidden
					h0 = [h.to(config.DEVICE).swapaxes(0, 1) for h in h0]
					# perform a forward pass and calculate the training loss
					pred = gru(x, h0)
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
			gru.eval()
			for m in tqdm(range(len(valDS.data)), leave=False):
				valDS.moderate_epoch = m
				for u in tqdm(range(valDS.get_mouse_cell_count()), leave=False):
					valDS.small_epoch = u
					sample = valDS.get_current_sample()
					valLoader.dataset.update_hidden_states(gru.forward_hidden(sample))
					# loop over the validation set
					for (i, (inputs, hidden, target)) in enumerate(tqdm(valLoader, leave=False)):
						# unpack the data and make sure they are on the same device
						x, h0, y = inputs.to(config.DEVICE), hidden, target.to(config.DEVICE)
						# Batch dimension has to be second for hidden
						h0 = [h.to(config.DEVICE).swapaxes(0, 1) for h in h0]
						# perform a forward pass and calculate the training loss
						pred = gru(x, h0)
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
			name ="gru_hidden_model_val_" + current_time + ".pth"
			model_val_path = os.path.sep.join([output_path, name])
			torch.save(gru, model_val_path)
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
	gru = torch.load(model_val_path)

	# Start testing
	gru.eval()
	# initialize lists to store predictions and ground-truth
	preds = []
	gt = []
	# switch off autograd
	with torch.no_grad():
		# loop over the test set
		for (i, (inputs, target)) in enumerate(tqdm(testLoader, leave=True)):
			# unpack the data and make sure they are on the same device
			x, y = inputs.to(config.DEVICE), target.to(config.DEVICE)
			# perform a forward pass and calculate the training loss
			pred = gru(x)
			pred = torch.sigmoid(pred)
			pred = pred.cpu().detach().numpy()
			preds.extend(pred)
			# add the ground-truth to the list
			gt.extend(y.cpu().detach().numpy())

	preds = np.array(preds).flatten()
	gt = np.array(gt).flatten()
	# calculate the accuracy
	acc = accuracy_score(gt, preds.round())
	print("[INFO] Accuracy: {:.4f}".format(acc))
	# calculate Precision and Recall per class
	precision1 = precision_score(gt, preds.round())
	recall1 = recall_score(gt, preds.round())
	print("[INFO] Transient Event Precision: {:.4f}".format(precision1))
	print("[INFO] Transient Event Recall: {:.4f}".format(recall1))
	# precision and recall for other class
	precision0 = precision_score(gt, preds.round(), pos_label=0)
	recall0 = recall_score(gt, preds.round(), pos_label=0)
	print("[INFO] No Transient Event Precision: {:.4f}".format(precision0))
	print("[INFO] No Transient Event Recall: {:.4f}".format(recall0))
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
	with open(os.path.sep.join([output_path, "parameters.txt"]), "w") as file:
		file.write("INIT_LR: {}\nNUM_EPOCHS: {}\nBATCH_SIZE: {}\nTHRESHOLD: {}\nTEST_SIZE: {}\nVAL_SIZE: {}\nSECTION_LEN: {}\nHIDDEN_SIZE: {}\nNUM_LAYERS: {}\n".format(
			config.INIT_LR, config.NUM_EPOCHS, config.BATCH_SIZE, config.THRESHOLD, config.TEST_SIZE, config.VAL_SIZE, config.SECTION_LEN, config.HIDDEN_SIZE, config.NUM_LAYERS))
		file.write("Accuracy: {:.4f}\n".format(acc))
		file.write("Transient Event Precision: {:.4f}\n".format(precision1))
		file.write("Transient Event Recall: {:.4f}\n".format(recall1))
		file.write("No Transient Event Precision: {:.4f}\n".format(precision0))
		file.write("No Transient Event Recall: {:.4f}\n".format(recall0))
		file.write("F1 score: {:.4f}\n".format(f1))
		file.write("AUC ROC score: {:.4f}\n".format(auc))
		# Write the data used for the training
		file.write("Data used for training: \n")
		for path in paths:
			file.write(path + "\n")

	# Create dict that keeps all the outputs
	outputs = {"accuracy": acc, "precision1": precision1, "recall1": recall1,
			 "precision0": precision0, "recall0": recall0, "f1": f1, "auc": auc,
			 "preds": preds, "gt": gt}
	return outputs