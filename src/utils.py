#!/usr/bin/env python
""" Utils

This script contains the implementation of some auxiliar functions 

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

import torch
import numpy as np
from torchvision import datasets, transforms
from skimage.transform import warp_polar, resize


def printlog(message, append=True, filename="train.log"):
	"""
	Writes the message in a log file.

	Args:
		message (str): Log info message.
		append (bool): Boolean indicating if the message should be appended to the file. Defaults to True.
		filename (str): Path of log file. Defaults to "train.log".

	Returns:
		None 
	"""
	file_parameter = "w"
	if append:
		file_parameter = "a"
	logfile = open(filename, file_parameter)
	logfile.write(message + "\n")
	logfile.close()


def init_dataloader(dataset, batch_size=11, shuffle=True, num_workers=8):
	"""
	Initialize a Torch DataLoader.

	Args:
		dataset (torch.utils.data.Dataset): The dataset to load.
		batch_size (int): Number of samples per batch. Defaults to 11.
		shuffle (bool): Boolean indicating if dataset samples should be shuffled. Defaults to True.
		num_workers (int): Number of workers for data loading. Defaults to 8.

	Returns:
		torch.utils.data.DataLoader: An initialized Torch DataLoader 
	"""

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
											 shuffle=shuffle, num_workers=num_workers)
	return dataloader


def calculate_mean_std_by_loader(loader):
	"""
	Calculates the average and standard deviation by data loader.

	Args:
		loader (torch.utils.data.DataLoader): An initialized Torch DataLoader

	Returns:
		mean (torch.Tensor): The mean of the dataset.
		std (torch.Tensor): The standard deviation of the dataset.
	"""
	images, labels = next(iter(loader))
	mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
	return mean, std


def calculate_mean_std_by_folder(folder_path, batch_size, custom_tranform=None):
	"""
	Calculates the average and standard deviation by a data folder path.

	Args:
		folder_path (str): Path to the data folder.
		batch_size (int): Batch size for loading data.
		custom_tranform (torchvision.transforms.Compose, optional): A custom transformation to apply. Defaults to None.

	Returns:
		mean (torch.Tensor): The mean of the dataset.
		std (torch.Tensor): The standard deviation of the dataset.
	"""
	compose_transform = transforms.Compose([transforms.ToTensor()])

	if custom_tranform is not None:
		compose_transform = transforms.Compose(
			[transforms.ToTensor(), custom_tranform])

	img_dataset = datasets.ImageFolder(folder_path, compose_transform)
	auxiliar_dataloader = init_dataloader(img_dataset, batch_size=batch_size)
	mean, std = calculate_mean_std_by_loader(auxiliar_dataloader)
	return mean, std


tensor2img = transforms.ToPILImage()


def to_polar(x, max_radius, center):
	"""
	Converts a euclidean image to the polar domain.

	Args:
		x (torch.Tensor): Input image tensor in euclidean domain 
		max_radius (float): Maximum radius for the polar transformation.
		center (tuple): Center of conversion in (x,y) coordinates.

	Returns:
		torch.Tensor: A torch tensor of the image converted to polar domain 
	"""

	img = tensor2img(x)
	img = np.array(img)
	h, w, c = img.shape  # numpy array image shape notation
	polar_img = img
	polar_img = warp_polar(img, center=center, radius=max_radius, output_shape=(
		h, w), scaling='linear', channel_axis=-1)
	polar_img = resize(polar_img, (h, w), anti_aliasing=True)
	polar_img = polar_img[:, :, :3]  # removing alpha channel
	polar_img = polar_img.astype(np.float32)
	polar_img_tensor = torch.tensor(polar_img)
	polar_img_tensor = polar_img_tensor.permute(2, 0, 1)
	return polar_img_tensor


def circular_shift_angle(x):
	"""
	Performs a circular shift vertically in the input image 
	with random shift size.

	Args:
		x (torch.Tensor): Input image tensor.

	Returns:
		torch.Tensor: A vertically shifted image tensor.
	"""
	h = x.shape[1]  # H dimension
	ang_shift = torch.randint(0, h - 1, (1,)).item()
	return torch.roll(x, ang_shift, dims=1)


def random_pad(x, padding_values=[0], left=0, top=0, right=0, bottom=0):
	"""
	Performs paddings randomly on the input image based on padding values
	and desired sides.

	Args:
		x (torch.Tensor): Input image tensor.
		padding_values (list of int): List of integer values for padding.
		left (int): Amount of padding to add to the left side.
		top (int): Amount of padding to add to the top side.
		right (int): Amount of padding to add to the right side.
		bottom (int): Amount of padding to add to the bottom side.

	Returns:
		torch.Tensor: A padded image tensor.
	"""

	if padding_values:
		_, orig_h, orig_w = x.shape
		len_values = len(padding_values)
		random_idxs = torch.randint(0, len_values, (4,))
		left_idx = random_idxs[0].item()
		right_idx = random_idxs[1].item()
		top_idx = random_idxs[2].item()
		bottom_idx = random_idxs[3].item()
		left_pad_size = padding_values[left_idx]
		right_pad_size = padding_values[right_idx]
		top_pad_size = padding_values[top_idx]
		bottom_pad_size = padding_values[bottom_idx]
		x = transforms.Pad(padding=(left*left_pad_size, top*top_pad_size,
						   right*right_pad_size, bottom*bottom_pad_size))(x)
		x = transforms.Resize(size=(orig_h, orig_w))(x)
	return x


def test_model(model, loss_function, test_dataloader, steps, device):
	"""
	Tests the model with the full dataset and returns average loss and accuracy.

	Args:
		model (torch.nn.Module): The model to be tested.
		loss_function (callable): The loss function.
		test_dataloader (torch.utils.data.DataLoader): A Torch DataLoader for the test dataset.
		steps (int): Number of steps to infer the entire dataset.
		device (str): Device to run the inference (e.g., 'cpu' or 'cuda').

	Returns:
		float: Average loss.
		float: Accuracy.
	"""
	correct_predictions = 0
	total_loss = 0

	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()

		for (x, y) in test_dataloader:
			# send the input to the device
			(x, y) = (x.to(device), y.to(device))
			# make the predictions and calculate the validation loss
			pred = model(x)
			loss = loss_function(pred, y)
			total_loss += loss.cpu().detach().numpy()
			# calculate the number of correct predictions
			correct_predictions += (pred.argmax(1) == y).type(
				torch.float).sum().item()

	# calculate avg loss
	avg_loss = total_loss / steps
	# calculate accuracy
	accuracy = correct_predictions / len(test_dataloader.dataset)

	return avg_loss, accuracy
