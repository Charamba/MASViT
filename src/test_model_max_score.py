#!/usr/bin/env python
""" Test Model Script for Max Score approach

This executable script tests MASViT using the Max Score approach, accepting 
the following parameters as arguments: the path to the pretrained model 
weights file (-m, --model), the test dataset (-d, --dataset), and the log 
file path (-l, --log), which contains the classification report of the test.

Example of usage:
	$ python test_model_max_score.py -m ../model/trained_model
	-d ../data/proj-GTSRB/el45 -l test_fixed_centroid_proj-GTSRB-el45.log  

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

__author__ = "Luiz Gustavo da Rocha Charamba"
__credits__ = "Luiz Gustavo da Rocha Charamba"
__copyright__ = "Copyright (c) 2024 Luiz Gustavo da Rocha Charamba"
__emails__ = ["charamba.lgr@gmail.com", "lgrc@cin.ufpe.br"]
__github__ = "github.com/Charamba"
__license__ = "MIT"
__maintainer__ = "developer"
__status__ = "Production"
__version__ = "0.0.1"

import argparse
import time
from sklearn.metrics import classification_report
import numpy as np
import torch
from torchvision import datasets, transforms as T
from utils import printlog, init_dataloader, to_polar

torch.manual_seed(33)
torch.cuda.manual_seed(33)

# set the device we will be using to train the model
device = torch.device("cuda")

BATCH_SIZE = 1

# centroid coordinates: we used in our experiments
# 49 centroids (7x7 centered pixels) instead all 48x48 pixels
x_min, y_min = (21, 21)
x_max, y_max = (27, 27)

# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
				help="path to input trained model")
ap.add_argument("-d", "--dataset", type=str, required=False,
				help="dataset folder path")
ap.add_argument("-l", "--log", type=str, required=False,
				help="log file name")
args = vars(ap.parse_args())

model_path = args["model"]
test_folder_path = args["dataset"]
logfilename = args["log"]


(mean, std) = (torch.tensor([0.2217, 0.1983, 0.2105]),
			   torch.tensor([0.1880, 0.1719, 0.1816]))
tensor_transform = T.Compose([T.ToTensor()])

# initializing dataset
test_dataset = datasets.ImageFolder(test_folder_path, tensor_transform)

# initialize test data loader
test_dataloader = init_dataloader(test_dataset, batch_size=BATCH_SIZE)

printlog("[INFO] loading the model...", append=False, filename=logfilename)
printlog("[INFO] normalization (mean, std) = " +
		 str((mean, std)), filename=logfilename)

# load the model's weigths
model = torch.load(args["model"])

# testing the model
printlog("[INFO] testing network...", filename=logfilename)

# initialize lists to store our predictions
preds = []
targets = []

# generating several centroid coordinates
centroids = []
for i in range(y_min, y_max+1):
	for j in range(x_min, x_max+1):
		centroids.append((i, j))

start_train_time = time.time()
# turn off autograd for testing evaluation
with torch.no_grad():
	# set the model in evaluation mode
	model.eval()

	# loop over the test set
	for (x, y) in test_dataloader:

		targets.extend(y.cpu().numpy())

		max_score = 0.0
		best_pred = None

		for centroid in centroids:
			polar_conversion = T.Lambda(lambda sample: to_polar(
				sample, max_radius=34, center=centroid))
			polar_transform = T.Compose(
				[polar_conversion, T.Normalize(mean, std)])

			x_ = x[0]

			x_ = polar_transform(x_)

			x_ = x_.unsqueeze(0)  # put in batch format again

			x_ = x_.to(device)

			# make the predictions and add them to the list
			pred = model(x_)
			if max_score < pred.max().cpu().numpy():
				max_score = pred.max().cpu().numpy()
				best_pred = pred

		preds.extend([best_pred.argmax(axis=1).cpu().numpy()])


end_train_time = time.time()
printlog("[INFO] total time taken to test the model: {:.2f}s".format(
	end_train_time - start_train_time))


# generate a classification report
target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
				'13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
				'23', '24', '25', '26', '27', '28', '29', '30', '31', '32',
				'33', '34', '35', '36', '37', '38', '39', '40', '41', '42']
printlog(classification_report(targets,
							   np.array(preds), target_names=target_names, digits=4), filename=logfilename)
