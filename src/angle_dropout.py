#!/usr/bin/env python
""" Angle Dropout

This script contains the implementation of Angle Dropout Layer class

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
import torch.nn.functional as F
from torch import nn, Tensor


class AngleDropout(nn.Module):
	"""
	Applies a dropout that randomly selects certain rows of the 
	input tensors (N,C,H,W) along the height dimension and sets 
	them to zero based on a dropout probability p

	Args:
		num_angles (int): Number of angles (rows) of the input tensor
		p (float): Probability of dropout

	Example:
		>>> x = torch.ones(1,1,8,4).requires_grad_(True).cuda() 
		>>> x # before angle dropout
		>>> drop_layer = AngleDropout(8, p=0.5).cuda()
		>>> drop_layer(x) # after angle dropout

	"""

	def __init__(self, num_angles: int, p: float = 0.5):
		super().__init__()
		self.num_angles = num_angles
		self.p = p  # probability

	def forward(self, x: Tensor) -> Tensor:
		"""
		Forward pass of the AngleDropout layer.

		Args:
			x (Tensor): Input tensor of shape (N, C, H, W).

		Returns:
			Tensor: Output tensor with dropout applied along the height dimension.
		"""
		if self.training:
			b, c, h, w = x.shape

			# dropout angles randomly
			mask_angle = F.dropout(torch.ones(
				b, 1, self.num_angles, 1).cuda(), p=self.p, training=self.training)
			# repeat in c dimensions
			mask_angle = mask_angle.repeat(1, c, 1, 1)

			mask_data = mask_angle.expand(b, c, self.num_angles, w)

			mask_data = mask_data.repeat(1, 1, 1, int(
				h/self.num_angles)).reshape(b, c, h, w)

			x = mask_data * x
		return x
