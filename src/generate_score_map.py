#!/usr/bin/env python
""" Score Map Generation Script

This executable script generates and saves a score map, accepting the 
following parameters as arguments: an input image path (-img, --image), 
a class index (-idx, --class_idx) and the path to the pretrained model 
weights file (-m, --model).

Example of usage:
	$ python generate_score_map.py -m ../model/trained_model -img 
	../data/aff-GTSRB_el45_az45_01279.png -idx 0

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

from utils import to_polar
from skimage.io import imread
from torchvision import transforms as T
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
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

# set the device
device = torch.device("cuda")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
				help="path to input trained model")
ap.add_argument("-img", "--image", type=str, required=False,
				help="image path")
ap.add_argument("-idx", "--class_index", type=int, required=False,
				help="class index")
args = vars(ap.parse_args())

model_path = args["model"]
image_path = args["image"]
class_idx = args["class_index"]

# pre computed mean and std
(mean, std) = (torch.tensor([0.2217, 0.1983, 0.2105]),
			   torch.tensor([0.1880, 0.1719, 0.1816]))

max_radius = 34  # 24*sqrt(2)=34

# reading and converting image to pytorch tensor
tensor_transform = T.Compose([T.ToTensor()])
orig_img = imread(image_path)
img = tensor_transform(orig_img)

print("[INFO] loading the model...")
print("[INFO] normalization (mean, std) = " + str((mean, std)))

# load the model's weigths
model = torch.load(model_path)

print("[INFO] generating score map of class index " +
	  str(class_idx) + " from " + image_path)


# scan full image
centroids = []
for i in range(0, 48):
	for j in range(0, 48):
		centroids.append((i, j))

score_map = np.zeros((48, 48))

with torch.no_grad():
	# set the model in evaluation mode
	model.eval()

	# scan the full image
	for centroid in centroids:
		polar_conversion = T.Lambda(lambda sample: to_polar(
			sample, max_radius=max_radius, center=centroid))
		default_transform = T.Compose(
			[polar_conversion, T.Normalize(mean, std)])

		x = default_transform(img)

		x = x.unsqueeze(0)  # put in batch format again

		x = x.to(device)

		pred = model(x)

		i, j = centroid
		score_map[i, j] = pred[0][class_idx]


print("[INFO] saving plot of original image")
plt.imshow(orig_img)
plt.savefig("plot_" + image_path.split('/')[-1])

print("[INFO] saving score map")
plt.imshow(score_map, interpolation='gaussian', cmap=plt.cm.jet)
plt.savefig("plot_score_map_" + str(class_idx) +
			"_" + image_path.split('/')[-1])
