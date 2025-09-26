# -*- coding: utf-8 -*-
"""
Functions to train/test model
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
## PyTorch dependencies
import torch.nn as nn
import pdb
import torch
import numpy as np


class Base_Lacunarity(nn.Module):
    def __init__(self, dim=2, eps=1e-6, model_name=None, kernel=None, stride=None, padding=None):
        super(Base_Lacunarity, self).__init__()
        self.dim = dim
        self.eps = eps
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.model_name = model_name
        self.normalize = nn.Tanh()
        if self.kernel is None:
            if self.dim == 1:
                self.gap_layer = nn.AdaptiveAvgPool1d(1)
            elif self.dim == 2:
                self.gap_layer = nn.AdaptiveAvgPool2d(1)
            elif self.dim == 3:
                self.gap_layer = nn.AdaptiveAvgPool3d(1)
            else:
                raise RuntimeError('Invalid dimension for global lacunarity layer')
        else:
            if self.dim == 1:
                self.gap_layer = nn.AvgPool1d(kernel[0], stride=stride[0], padding=0)
            elif self.dim == 2:
                self.gap_layer = nn.AvgPool2d((kernel[0], kernel[1]), stride=(stride[0], stride[1]), padding=0)
            elif self.dim == 3:
                self.gap_layer = nn.AvgPool3d((kernel[0], kernel[1], kernel[2]), stride=(stride[0], stride[1], stride[2]), padding=0)
            else:
                raise RuntimeError('Invalid dimension for local lacunarity layer')

    def forward(self, x):
        x = ((self.normalize(x) + 1) / 2) * 255
        squared_x_tensor = x ** 2
        n_pts = np.prod(np.asarray(x.shape[-2:]))
        L_numerator = (n_pts ** 2) * self.gap_layer(squared_x_tensor)
        L_denominator = (n_pts * self.gap_layer(x)) ** 2
        L_r = (L_numerator / (L_denominator + self.eps)) - 1
        return L_r

class lacunarity_pooling(nn.Module):
    def __init__(self, lacunarity_layer=None, Params=None):
        super(lacunarity_pooling, self).__init__()
        if lacunarity_layer is None:
            lacunarity_layer = Base_Lacunarity(dim=2)
        self.lacunarity_layer = lacunarity_layer
        self.model_name = Params["Model_name"] if Params is not None else None
        self.dataset = Params["Dataset"] if Params is not None else None
        self.num_classes = Params["num_classes"][self.dataset] if Params is not None else None
        self.feature_extraction = Params['feature_extraction'] if Params is not None and 'feature_extraction' in Params else None
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x_pool = self.lacunarity_layer(x)
        x_avg = self.avgpool(x)
        # Ensure shapes match for multiplication
        if x_pool.shape != x_avg.shape:
            raise ValueError(f"Shape mismatch: x_pool {x_pool.shape}, x_avg {x_avg.shape}")
        pool_layer = x_pool * x_avg
        pool_layer = pool_layer.view(pool_layer.size(0), -1)  # Flatten for FC layer
        return pool_layer

# Example usage:
# lacunarity_layer = Base_Lacunarity(dim=2)
# pooling = lacunarity_pooling(lacunarity_layer, Params)
# output = pooling(input_tensor)