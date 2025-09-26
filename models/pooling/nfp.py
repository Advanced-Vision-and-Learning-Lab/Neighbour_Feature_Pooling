"""
Neighborhood Feature Pooling (NFP) implementation in PyTorch
Fixed to properly compute similarity between center and pure neighbors
"""

from numpy import pi
import torch
import torch.nn as nn  
import torch.nn.functional as F
from torch import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import pdb

class NFPPooling(nn.Module):
    def __init__(self, in_channels, R=1, measure='norm', p=1, stride=1, padding=0,
                 dilation=1, bias=False, padding_mode='reflect', similarity=True,
                 eps=1e-6, input_size=224, q_scs=1e-6):
        super(NFPPooling, self).__init__()
        self.in_size = input_size
        self.measure = measure.lower()

        # Define layer properties
        self.in_channels = in_channels
        self.R = R
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.similarity = similarity
        self.p = p
        self.dilation = dilation
        self.bias = bias
        self.eps = eps
        self.q_scs = q_scs


        # Compute kernel size and out channels
        self.kernel_size = int(2*self.R + 1)
        self.out_channels = int(self.kernel_size**2 - 1)

        # Define convolution layer for neighbor extraction
        self.comp_neighbors = nn.Conv2d(self.in_channels, self.out_channels*self.in_channels,
                                      self.kernel_size, stride=self.stride,
                                      padding=self.padding,
                                      padding_mode=self.padding_mode,
                                      dilation=self.dilation, bias=self.bias,
                                      groups=self.in_channels)

        # Initialize weights for neighbor extraction
        self.comp_neighbors.weight.data.fill_(0)

        # Center value extraction layer
        self.center_value = nn.Conv2d(self.in_channels, self.in_channels,
                                    self.kernel_size, stride=self.stride,
                                    padding=self.padding,
                                    padding_mode=self.padding_mode,
                                    dilation=self.dilation, bias=self.bias,
                                    groups=self.in_channels)
        self.center_value.weight.data.fill_(0)
        self.center_value.weight.data[:,:,self.R,self.R] = torch.tensor(1)
        self.center_value.weight.requires_grad = False

        # Get indices for square ring neighborhood
        index = torch.arange(0, self.kernel_size)
        neighbors = torch.cartesian_prod(index, index)
        center = neighbors.shape[0]//2
        neighbors = torch.cat((neighbors[:center], neighbors[center+1:]))
        neighbors = neighbors.repeat(self.in_channels, 1)

        # Set weights for neighbor comparison
        sim_space = torch.arange(0, self.out_channels*self.in_channels)
        
        # Distance measures: compute center - neighbor
        if measure in ['norm', 'rmse', 'mahalanobis']:
            self.comp_neighbors.weight.data[:,:,self.R,self.R] = torch.tensor(1)  # Center = 1
            self.comp_neighbors.weight.data[sim_space,:,neighbors[:,0],neighbors[:,1]] = -1  # Neighbor = -1
        # Similarity measures: extract pure neighbor values (center = 0, neighbor = 1)
        else:
            self.comp_neighbors.weight.data[:,:,self.R,self.R] = torch.tensor(0)  # Center = 0 (KEY FIX!)
            self.comp_neighbors.weight.data[sim_space,:,neighbors[:,0],neighbors[:,1]] = 1  # Neighbor = 1

        self.comp_neighbors.weight.requires_grad = False

        # Set similarity measure
        if self.measure == "norm":
            self.similarity_measure = self.Norm
        elif self.measure == 'cosine':
            self.similarity_measure = self.Cosine
        elif self.measure == "dot":
            self.similarity_measure = self.DotProduct
        elif self.measure == "rmse":
            self.similarity_measure = self.RMSE
        elif self.measure == 'geman':
            self.similarity_measure = self.GMC
        elif self.measure == 'attention':
            self.similarity_measure = self.Attention
        elif self.measure == 'emd':
            self.similarity_measure = self.EMD
        elif self.measure == 'canberra':
            self.similarity_measure = self.Canberra
        elif self.measure == 'hellinger':
            self.similarity_measure = self.Hellinger
        elif self.measure == 'chisquared1':
            self.similarity_measure = self.ChiSquared1
        elif self.measure == 'chisquared2':
            self.similarity_measure = self.ChiSquared2
        elif self.measure == 'gfc':
            self.similarity_measure = self.GFC
        elif self.measure == 'pearson':
            self.similarity_measure = self.Pearson
        elif self.measure == 'jeffrey':
            self.similarity_measure = self.Jeffrey
        elif self.measure == 'squaredchord':
            self.similarity_measure = self.SquaredChord
        elif self.measure == 'smith':
            self.similarity_measure = self.Smith
        elif self.measure == 'sharpened_cosine' or self.measure == 'scs':
            self.similarity_measure = self.SharpenedCosine
        else:
            raise RuntimeError(f'Similarity measure {self.measure} not implemented')

        # Initialize output size property
        self.__output_size = None

    @property
    def output_size(self):
        """Get output feature map size based on convolution parameters"""
        self.__output_size = (self.in_size + 2 * self.padding - 
                            self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        return self.__output_size

    def forward(self, x):
        """Forward pass with automatic measure selection"""
        return self.similarity_measure(x)

    def reshape_tensor(self, x):
        """Reshape tensor for similarity computation"""
        n, c, h, w = x.size()
        return x.reshape(n, c//self.out_channels, self.out_channels, h, w)

    def Norm(self, x):
        """Compute L-p norm similarity"""
        x = self.comp_neighbors(x)
        x = self.reshape_tensor(x)
        x = LA.norm(x, ord=self.p, dim=1)
        if self.similarity:
            x = -x
        return x

    def Cosine(self, x):
        """Compute cosine similarity between center and pure neighbors"""
        x_center = self.center_value(x)
        x_neighbors = self.comp_neighbors(x)  # Now extracts pure neighbors!
        x_neighbors = self.reshape_tensor(x_neighbors)
        x_center = x_center.unsqueeze(2)
        x = F.cosine_similarity(x_center, x_neighbors, dim=1, eps=self.eps)
        if not self.similarity:
            x = 1 - x
        return x

    def DotProduct(self, x):
        """Compute dot product similarity"""
        x_center = self.center_value(x)
        x_neighbors = self.comp_neighbors(x)  # Pure neighbors
        x_neighbors = self.reshape_tensor(x_neighbors)
        x_center = x_center.unsqueeze(2)
        x = torch.sum(x_center * x_neighbors, dim=1)
        if not self.similarity:
            x = -x
        return x

    def RMSE(self, x):
        """Compute Root Mean Square Error"""
        x = self.comp_neighbors(x)  # This gives center - neighbor
        x = self.reshape_tensor(x)
        x = torch.sqrt(torch.mean(x**2, dim=1))
        if self.similarity:
            x = -x
        return x

    def GMC(self, x):
        """Compute Geman-McClure similarity"""
        x_center = self.center_value(x)                    # (B, C, H, W)
        x_neighbors = self.comp_neighbors(x)               # (B, C*N, H, W)
        x_neighbors = self.reshape_tensor(x_neighbors)     # (B, C, N, H, W)
        x_center = x_center.unsqueeze(2)                   # (B, C, 1, H, W)
        diff = (x_center - x_neighbors) ** 2               # (B, C, N, H, W)
        gm = diff / (diff + self.eps)                      # (B, C, N, H, W)
        # Aggregate over channel dimension to match Cosine output: [B, N, H, W]
        x = gm.mean(dim=1)                                 # (B, N, H, W)
        if not self.similarity:
            x = 1 - x
        return x

    def Attention(self, x):
        """Compute attention-based similarity (without normalization)"""
        x_center = self.center_value(x)
        x_neighbors = self.comp_neighbors(x)  # Pure neighbors
        x_neighbors = self.reshape_tensor(x_neighbors)
        x_center = x_center.unsqueeze(2)
        x = torch.sum(x_center * x_neighbors, dim=1)
        x = F.softmax(x, dim=1)
        if not self.similarity:
            x = -x
        return x

    def EMD(self, x):
        """Compute Earth Mover's Distance (simplified L1)"""
        x_center = self.center_value(x)
        x_neighbors = self.comp_neighbors(x)  # Pure neighbors
        x_neighbors = self.reshape_tensor(x_neighbors)
        x_center = x_center.unsqueeze(2)
        x = torch.sum(torch.abs(x_center - x_neighbors), dim=1)
        if self.similarity:
            x = -x
        return x

    def Canberra(self, x):
        """Compute Canberra distance"""
        x_center = self.center_value(x)
        x_neighbors = self.comp_neighbors(x)  # Pure neighbors
        x_neighbors = self.reshape_tensor(x_neighbors)
        x_center = x_center.unsqueeze(2)
        x = torch.sum(torch.abs(x_center - x_neighbors) / (torch.abs(x_center) + torch.abs(x_neighbors) + self.eps), dim=1)
        if self.similarity:
            x = -x
        return x

    def Hellinger(self, x):
        """Compute Hellinger distance"""
        x_center = self.center_value(x)
        x_neighbors = self.comp_neighbors(x)  # Pure neighbors
        x_neighbors = self.reshape_tensor(x_neighbors)
        x_center = x_center.unsqueeze(2)
        # Use absolute values + eps to avoid sqrt of negative numbers
        x_center_abs = torch.abs(x_center) + self.eps
        x_neighbors_abs = torch.abs(x_neighbors) + self.eps
        x = torch.sqrt(0.5 * torch.sum((torch.sqrt(x_center_abs) - torch.sqrt(x_neighbors_abs))**2, dim=1))
        if self.similarity:
            x = -x
        return x

    def ChiSquared1(self, x):
        """Compute Chi-Squared distance (type 1)"""
        x_center = self.center_value(x)
        x_neighbors = self.comp_neighbors(x)  # Pure neighbors
        x_neighbors = self.reshape_tensor(x_neighbors)
        x_center = x_center.unsqueeze(2)
        x = torch.sum((x_center - x_neighbors)**2 / (torch.abs(x_center) + torch.abs(x_neighbors) + self.eps), dim=1)
        if self.similarity:
            x = -x
        return x

    def ChiSquared2(self, x):
        """Compute Chi-Squared distance (type 2)"""
        x_center = self.center_value(x)
        x_neighbors = self.comp_neighbors(x)  # Pure neighbors
        x_neighbors = self.reshape_tensor(x_neighbors)
        x_center = x_center.unsqueeze(2)
        x = torch.sum((x_center - x_neighbors)**2 / (torch.abs(x_center) + self.eps), dim=1)
        if self.similarity:
            x = -x
        return x

    def GFC(self, x):
        """Compute Generalized Feature Correlation"""
        x_center = self.center_value(x)
        x_neighbors = self.comp_neighbors(x)  # Pure neighbors
        x_neighbors = self.reshape_tensor(x_neighbors)
        x_center = x_center.unsqueeze(2)
        numerator = torch.sum(x_center * x_neighbors, dim=1)
        denominator = torch.norm(x_center, dim=1) * torch.norm(x_neighbors, dim=1) + self.eps
        x = numerator / denominator
        if not self.similarity:
            x = -x
        return x

    def Pearson(self, x):
        """Compute Pearson correlation"""
        x_center = self.center_value(x)
        x_neighbors = self.comp_neighbors(x)  # Pure neighbors
        x_neighbors = self.reshape_tensor(x_neighbors)
        x_center = x_center.unsqueeze(2)
        x_center_mean = torch.mean(x_center, dim=1, keepdim=True)
        x_neighbors_mean = torch.mean(x_neighbors, dim=1, keepdim=True)
        x_center_centered = x_center - x_center_mean
        x_neighbors_centered = x_neighbors - x_neighbors_mean
        numerator = torch.sum(x_center_centered * x_neighbors_centered, dim=1)
        denominator = torch.sqrt(torch.sum(x_center_centered**2, dim=1) * torch.sum(x_neighbors_centered**2, dim=1) + self.eps)
        x = numerator / denominator
        if not self.similarity:
            x = -x
        return x

    def Jeffrey(self, x):
        """Compute Jeffrey divergence"""
        x_center = self.center_value(x)
        x_neighbors = self.comp_neighbors(x)  # Pure neighbors
        x_neighbors = self.reshape_tensor(x_neighbors)
        x_center = x_center.unsqueeze(2)
        # Use absolute values + eps to avoid log of negative/zero values
        x_center_abs = torch.abs(x_center) + self.eps
        x_neighbors_abs = torch.abs(x_neighbors) + self.eps
        x = torch.sum(x_center_abs * torch.log(x_center_abs/x_neighbors_abs) + 
                     x_neighbors_abs * torch.log(x_neighbors_abs/x_center_abs), dim=1)
        if self.similarity:
            x = -x
        return x

    def SquaredChord(self, x):
        """Compute Squared Chord distance"""
        x_center = self.center_value(x)
        x_neighbors = self.comp_neighbors(x)  # Pure neighbors
        x_neighbors = self.reshape_tensor(x_neighbors)
        x_center = x_center.unsqueeze(2)
        # Use absolute values + eps to avoid sqrt of negative numbers
        x_center_abs = torch.abs(x_center) + self.eps
        x_neighbors_abs = torch.abs(x_neighbors) + self.eps
        value = torch.sqrt(x_center_abs) - torch.sqrt(x_neighbors_abs)
        value = torch.square(value)
        x = torch.sum(value, dim=1)
        if self.similarity:
            x = -x
        return x

    def Smith(self, x):
        """Compute Smith similarity"""
        x_center = self.center_value(x)
        x_neighbors = self.comp_neighbors(x)  # Pure neighbors
        x_neighbors = self.reshape_tensor(x_neighbors)
        x_center = x_center.unsqueeze(2)
        # Use absolute values to avoid negative issues in Smith similarity
        x_center_abs = torch.abs(x_center)
        x_neighbors_abs = torch.abs(x_neighbors)
        min_value = torch.minimum(x_center_abs, x_neighbors_abs)
        min_sum = torch.sum(min_value, dim=1)
        sum_center = torch.sum(x_center_abs, dim=1)
        sum_neighbors = torch.sum(x_neighbors_abs, dim=1)
        x = 1 - (min_sum / (torch.minimum(sum_center, sum_neighbors) + self.eps))
        if not self.similarity:
            x = -x
        return x

    def SharpenedCosine(self, x):
        """
        Compute Sharpened Cosine Similarity.
        Output shape is always [B, N, H, W], identical to Cosine and other similarities.
        Only the values differ due to the sharpening operation.
        """
        p = self.p
        q = self.q_scs  # Small constant to stabilize denominator

        x_center = self.center_value(x)                    # (B, C, H, W)
        x_neighbors = self.comp_neighbors(x)               # (B, C * N, H, W)
        x_neighbors = self.reshape_tensor(x_neighbors)     # (B, C, N, H, W)
        x_center = x_center.unsqueeze(2)                   # (B, C, 1, H, W)

        # Normalize
        x_center_norm = torch.norm(x_center, dim=1, keepdim=True) + q   # (B, 1, 1, H, W)
        x_neighbors_norm = torch.norm(x_neighbors, dim=1, keepdim=True) + q  # (B, 1, N, H, W)

        # Cosine similarity per neighbor, per channel
        cosine = torch.sum(x_center * x_neighbors, dim=1) / (x_center_norm * x_neighbors_norm)  # (B, N, H, W)

        # Apply sharpening (raise to power p, preserve sign)
        scs = torch.sign(cosine) * (torch.abs(cosine) ** p)  # (B, N, H, W)

        # NaN protection: replace NaNs with zero
        scs = torch.nan_to_num(scs, nan=0.0, posinf=0.0, neginf=0.0)

        if not self.similarity:
            scs = 1 - scs

        return scs.mean(dim=1)  # (B, N, H, W)

