#!/usr/bin/env python
""" Train Model Script

This executable script trains the MASViT model using GTSRB Train samples and
validates with GTSRB Test samples

Example of usage:
	$ python train_model.py 

MIT License

Copyright (c) 2024 Luiz Gustavo da Rocha Charamba

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from utils import test_model, printlog, init_dataloader, calculate_mean_std_by_folder
from utils import to_polar, circular_shift_angle, random_pad
from torchvision import datasets, transforms as T
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Adam
from torch import nn
import torchinfo
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from models import MultiAngleScaleVisionTransformer
from sklearn.metrics import classification_report
__author__ = "Luiz Gustavo da Rocha Charamba"
__credits__ = "Luiz Gustavo da Rocha Charamba"
__copyright__ = "Copyright (c) 2024 Luiz Gustavo da Rocha Charamba"
__emails__ = ["charamba.lgr@gmail.com", "lgrc@cin.ufpe.br"]
__github__ = "github.com/Charamba"
__license__ = "MIT"
__maintainer__ = "developer"
__status__ = "Production"
__version__ = "0.0.1"

import matplotlib
matplotlib.use("Agg")

torch.manual_seed(33)
torch.cuda.manual_seed(33)

# define training hyperparameters
INIT_LR = 1e-4
BATCH_SIZE = 16
EPOCHS = 300
LR_DECAY = True

# set the device
device = torch.device("cuda")

# Paths
# sugested dataset paths
train_folder_path = "../data/GTRSB/Train"
valid_folder_path = "../data/GTRSB/Test"
test_folder_path = "../data/GTRSB/Test"
# output paths
acc_img_path = "train&val_acc.png"  # image of accuracy plot
loss_img_path = "train&val_loss.png"  # image of loss plot
trained_model_path = "trained_model"  # weights of trained model


# Polar Conversion
max_radius = 34  # 24*sqrt(2)=34
polar_conversion = T.Lambda(lambda x: to_polar(x, max_radius, center=(24, 24)))

# Cyclic-Angular Shift
cyclic_shift = T.Lambda(lambda x: circular_shift_angle(x))

# Right-Side Paddings
padding_values = list(range(0, 41))
right_paddings = T.Lambda(lambda x: random_pad(
	x, padding_values=padding_values, right=1))

# Mean and Std for Normalization
mean, std = calculate_mean_std_by_folder(train_folder_path, BATCH_SIZE)

# Augmentation transform: for train data
augmentation_transform = T.Compose([T.ToTensor(),
									T.ColorJitter(
										brightness=0.8, contrast=0.8, saturation=0.8),
									polar_conversion, cyclic_shift, right_paddings,
									T.Normalize(mean, std)])

# Default transform: for validation and test data
default_transform = T.Compose(
	[T.ToTensor(), polar_conversion, T.Normalize(mean, std)])

# initializing datasets
train_dataset = datasets.ImageFolder(train_folder_path, augmentation_transform)
valid_dataset = datasets.ImageFolder(valid_folder_path, default_transform)
test_dataset = datasets.ImageFolder(test_folder_path, default_transform)

# initializing data loaders
train_dataloader = init_dataloader(train_dataset, batch_size=BATCH_SIZE)
valid_dataloader = init_dataloader(valid_dataset, batch_size=BATCH_SIZE)
test_dataloader = init_dataloader(test_dataset)

# calculate steps per epoch for training and validation set
train_steps = 0
# train_steps = len(train_dataloader.dataset) // BATCH_SIZE
valid_steps = len(valid_dataloader.dataset) // BATCH_SIZE

# initialize model
printlog("[INFO] initializing the model...", append=False)

printlog("[INFO] normalization (mean, std) = " + str((mean, std)))

printlog("-------------------")
printlog("[INFO] DATASETS SIZES: ")
printlog("train_dataset size: " + str(len(train_dataloader.dataset)))
printlog("valid_dataset size: " + str(len(valid_dataloader.dataset)))
printlog("test_dataset size: " + str(len(test_dataloader.dataset)))
printlog("-------------------\n")
printlog("[INFO] DATASETS PATHS: ")
printlog("train_dataset path: " + train_folder_path)
printlog("valid_dataset path: " + valid_folder_path)
printlog("test_dataset path: " + test_folder_path)
printlog("-------------------\n")
printlog("[INFO] HYPER-PARAMETERS: ")
printlog("INIT_LR:   " + str(INIT_LR))
printlog("LR_DECAY:   " + str(LR_DECAY))
printlog("BATCH_SIZE: " + str(BATCH_SIZE))
printlog("EPOCHS:     " + str(EPOCHS))

model = MultiAngleScaleVisionTransformer(input_channel=3,
										 hidden_channel_1=64,
										 hidden_channel_2=128,
										 hidden_channel_3=256,
										 output_channel=512,
										 num_patches=8,  # 48 -> 8 rows
										 num_heads=256,
										 mlp_size=512,
										 mlp_dropout=0.5,
										 embedding_dropout=0.0,
										 angle_dropout=0.5,
										 num_transformer_layers=8,
										 num_classes=43).cuda()

printlog(str(torchinfo.summary(model, (3, 48, 48), batch_dim=0,
							   col_names=("input_size", "output_size",
										  "num_params", "kernel_size", "mult_adds"),
							   verbose=0)))

# initialize optimizer, learning rate scheduler and loss function
optimizer = Adam(model.parameters(), lr=INIT_LR)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
loss_function = nn.CrossEntropyLoss()
printlog("optimizer:     " + str(optimizer))
printlog("scheduler:     " + str(scheduler))
printlog("loss_function: " + str(loss_function))
printlog("-------------------\n")

# initialize a dictionary to store training history
H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}

# measure how long training is going to take
printlog("[INFO] training the network...")
start_train_time = time.time()

# loop over epochs
for e in range(0, EPOCHS):
	train_steps = 0
	start_epoch_time = time.time()
	# set the model in training mode
	model.train()
	# initialize the total training and validation loss
	total_train_loss = 0
	total_val_loss = 0
	# initialize the number of correct predictions in the training
	# and validation step
	train_correct = 0
	val_correct = 0
	# loop over the training set
	for (x, y) in train_dataloader:
		train_steps += 1

		# send the input to the device
		(x, y) = (x.to(device), y.to(device))
		# perform a forward pass and calculate the training loss
		pred = model(x)
		loss = loss_function(pred, y)
		# zero out the gradients, perform the backpropagation step,
		# and update the weights
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
		total_train_loss += loss.cpu().detach().numpy()
		train_correct += (pred.argmax(1) == y).type(
			torch.float).sum().item()

# switch off autograd for evaluation
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()

		avg_val_loss, val_acc = test_model(
			model, loss_function, valid_dataloader, valid_steps, device)

	end_epoch_time = time.time()
	# calculate the average training and validation loss
	avg_train_loss = total_train_loss / train_steps
	# calculate the training and validation accuracy
	train_acc = train_correct / len(train_dataloader.dataset)
	# update training history
	H["train_loss"].append(avg_train_loss)  # .cpu().detach().numpy())
	H["train_acc"].append(train_acc)
	H["val_loss"].append(avg_val_loss)  # .cpu().detach().numpy())
	H["val_acc"].append(val_acc)

	# print the model training and validation information
	printlog("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
	printlog("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
		avg_train_loss, train_acc))
	printlog("Val loss: {:.6f}, Val accuracy: {:.4f}".format(
		avg_val_loss, val_acc))

	if LR_DECAY:
		before_lr = optimizer.param_groups[0]["lr"]
		scheduler.step()
		after_lr = optimizer.param_groups[0]["lr"]
		printlog("Learning rate: %.6f -> %.6f" % (before_lr, after_lr))

	printlog("Epoch time: {:.2f}s".format(end_epoch_time - start_epoch_time))
	printlog("-------------------\n")

# finish measuring how long training took
end_train_time = time.time()
printlog("[INFO] total time taken to train the model: {:.2f}s".format(
	end_train_time - start_train_time))
# we can now evaluate the network on the test set
printlog("[INFO] evaluating network...")

# initialize lists to store our predictions
preds = []
targets = []
# turn off autograd for testing evaluation
with torch.no_grad():
	# set the model in evaluation mode
	model.eval()

	# loop over the test set
	for (x, y) in test_dataloader:

		targets.extend(y.cpu().numpy())
		# send the input to the device
		x = x.to(device)
		# make the predictions and add them to the list
		pred = model(x)
		preds.extend(pred.argmax(axis=1).cpu().numpy())

# generate a classification report
target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
				'13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
				'23', '24', '25', '26', '27', '28', '29', '30', '31', '32',
				'33', '34', '35', '36', '37', '38', '39', '40', '41', '42']
printlog(classification_report(targets,
							   np.array(preds), target_names=target_names))


# plot the training loss and accuracy curves
# Loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.title("Loss")
plt.xlabel("Epoch #")
plt.legend(loc="upper right")
plt.savefig(loss_img_path)

# Accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Accuracy")
plt.xlabel("Epoch #")
plt.legend(loc="lower right")
plt.savefig(acc_img_path)

# serialize the model to disk
torch.save(model, trained_model_path)
