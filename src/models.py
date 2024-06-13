#!/usr/bin/env python
""" Models

This script contains the implementation of Multi-Angle-Scale Vision Transformer class

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
import torch.nn as nn
from torch import relu
from angle_dropout import AngleDropout


class MultiAngleScaleVisionTransformer(nn.Module):
	"""
	A Multi-Angle-Scale Vision Transformer implementation for image classification tasks.

	Args:
			input_channel (int): Number of input channels in the images.
			hidden_channel_1 (int): Number of output channels for the first convolutional layer.
			hidden_channel_2 (int): Number of output channels for the second convolutional layer.
			hidden_channel_3 (int): Number of output channels for the third convolutional layer.
			output_channel (int): Number of output channels for the final convolutional layer.
			num_patches (int): Number of patches to divide the input image into.
			num_heads (int): Number of attention heads in the Transformer encoder.
			mlp_size (int): Size of the feedforward layer in the Transformer encoder.
			mlp_dropout (float): Dropout rate for the MLP layers in the Transformer encoder.
			embedding_dropout (float): Dropout rate for the embedding layer.
			angle_dropout (float): Dropout rate for the angle dropout layer.
			num_transformer_layers (int): Number of Transformer encoder layers.
			num_classes (int): Number of output classes for classification.

	Attributes:
			conv1, conv2, conv3, conv4 (nn.Conv2d): Convolutional layers for patch embedding.
			vert_conv1, vert_conv2, vert_conv3, vert_conv4 (nn.Conv2d): Vertical convolutional layers for feature extraction.
			flatten (nn.Flatten): Layer to flatten the feature maps.
			class_embedding (nn.Parameter): Learnable class token embedding.
			position_embedding (nn.Parameter): Learnable position embedding.
			transformer_encoder (nn.TransformerEncoder): Transformer encoder composed of multiple layers.
			classifier (nn.Sequential): Classification head consisting of a layer normalization and a linear layer.
			bn_conv1, bn_conv2, bn_conv3, bn_conv4 (nn.BatchNorm2d): Batch normalization layers for the convolutional layers.
			bn_vert_conv1, bn_vert_conv2, bn_vert_conv3, bn_vert_conv4 (nn.BatchNorm2d): Batch normalization layers for the vertical convolutional layers.
			embedding_dropout (nn.Dropout): Dropout layer for embeddings.
			angle_dropout (AngleDropout): Dropout layer that applies angle-based dropout.
	"""

	def __init__(self,
				 input_channel,
				 hidden_channel_1,
				 hidden_channel_2,
				 hidden_channel_3,
				 output_channel,
				 num_patches,
				 num_heads,
				 mlp_size,
				 mlp_dropout,
				 embedding_dropout,
				 angle_dropout,
				 num_transformer_layers,
				 num_classes):
		super().__init__()

		# Convolutional layers
		self.conv1 = nn.Conv2d(
			input_channel, hidden_channel_1, kernel_size=(1, 1))
		self.conv2 = nn.Conv2d(
			hidden_channel_1, hidden_channel_2, kernel_size=(1, 3))
		self.conv3 = nn.Conv2d(
			hidden_channel_2, hidden_channel_3, kernel_size=(1, 5))
		self.conv4 = nn.Conv2d(
			hidden_channel_3, output_channel, kernel_size=(1, 6), stride=(1, 6))

		self.vert_conv1 = nn.Conv2d(
			output_channel, output_channel, kernel_size=(3, 1))
		self.vert_conv2 = nn.Conv2d(
			output_channel, output_channel, kernel_size=(3, 1))
		self.vert_conv3 = nn.Conv2d(
			output_channel, output_channel, kernel_size=(3, 1))
		self.vert_conv4 = nn.Conv2d(
			output_channel, output_channel, kernel_size=(3, 1))

		# only flatten the feature map dimensions into a single vector
		self.flatten = nn.Flatten(start_dim=2,
								  end_dim=3)

		self.num_patches = num_patches
		self.embedding_dim = output_channel

		# Class token embedding
		self.class_embedding = nn.Parameter(data=torch.randn(1, 1, self.embedding_dim),
											requires_grad=True)

		# Position token embedding
		self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, self.embedding_dim),
											   requires_grad=True)
		# Transformer Encoder
		encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim,
												   nhead=num_heads,
												   dim_feedforward=mlp_size,
												   dropout=mlp_dropout,
												   activation="gelu",
												   batch_first=True,
												   norm_first=True)

		self.transformer_encoder = nn.TransformerEncoder(
			encoder_layer, num_layers=num_transformer_layers)

		# Classifier head
		self.classifier = nn.Sequential(
			nn.LayerNorm(normalized_shape=self.embedding_dim),
			nn.Linear(in_features=self.embedding_dim,
					  out_features=num_classes))

		# Batch Normalization layers
		self.bn_conv1 = nn.BatchNorm2d(hidden_channel_1)
		self.bn_conv2 = nn.BatchNorm2d(hidden_channel_2)
		self.bn_conv3 = nn.BatchNorm2d(hidden_channel_3)
		self.bn_conv4 = nn.BatchNorm2d(output_channel)

		self.bn_vert_conv1 = nn.BatchNorm2d(output_channel)
		self.bn_vert_conv2 = nn.BatchNorm2d(output_channel)
		self.bn_vert_conv3 = nn.BatchNorm2d(output_channel)
		self.bn_vert_conv4 = nn.BatchNorm2d(output_channel)

		# Dropout
		self.embedding_dropout = nn.Dropout(embedding_dropout)

		self.angle_dropout = AngleDropout(num_patches, angle_dropout)

		# Initializing weights
		self.apply(self.init_weights_xavier)

	def init_weights_xavier(self, module):
		"""
		Initialize the weights of the module using Xavier uniform distribution.

		Args:
			module (nn.Module): The module to initialize.
		"""
		if isinstance(module, nn.Linear):
			nn.init.xavier_uniform_(module.weight)
			module.bias.data.fill_(0.01)

	def forward(self, x):
		"""
		Forward pass of the model.

		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, input_channel, height, width).

		Returns:
			torch.Tensor: Output tensor of shape (batch_size, num_classes) containing the class scores.
		"""
		# Create class token embedding and expand it to match the batch size
		batch_size = x.shape[0]
		class_token = self.class_embedding.expand(batch_size, -1, -1)

		# Conv Layers (create patch embedding)
		x = relu(self.bn_conv1(self.conv1(x)))      # (48,48) --> (48,48)
		x = relu(self.bn_conv2(self.conv2(x)))      # (48,48) --> (48,46)
		x = relu(self.bn_conv3(self.conv3(x)))      # (48,46) --> (48,42)
		x = nn.MaxPool2d((1, 7), stride=(1, 7))(x)  # (48,42) --> (48,6)
		x = relu(self.bn_conv4(self.conv4(x)))      # (48,6)  --> (48,1)

		x = relu(self.bn_vert_conv1(self.vert_conv1(x)))  # (48,1) --> (46,1)
		x = relu(self.bn_vert_conv2(self.vert_conv2(x)))  # (46,1) --> (44,1)
		x = relu(self.bn_vert_conv3(self.vert_conv3(x)))  # (44,1) --> (42,1)
		x = relu(self.bn_vert_conv4(self.vert_conv4(x)))  # (42,1) --> (40,1)
		x = nn.MaxPool2d((5, 1), stride=(5, 1))(x)        # (40,1) --> (8,1)

		# Performs angle dropout
		x = self.angle_dropout(x)

		# Adjust so the embedding is on the final dimension:
		# [batch_size, row_patches*C, N] -> [batch_size, N, row_patches*C]
		x = self.flatten(x)
		x = x.permute(0, 2, 1)

		# Concat class embedding and patch embedding
		x = torch.cat((class_token, x), dim=1)

		# Add position embedding to patch embedding
		x = self.position_embedding + x

		# Run embedding dropout (optional)
		x = self.embedding_dropout(x)

		# Pass patch, position and class embedding through transformer encoder layers
		x = self.transformer_encoder(x)

		# Put 0 index logit through classifier
		# run on each sample in a batch at 0 index
		x = self.classifier(x[:, 0])

		return x
