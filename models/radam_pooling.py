import torch
import torch.nn as nn
import torchvision.transforms as T
from models.RNN import RAE  # You must also copy RNN.py to your project
from einops.layers.torch import Rearrange


class lp_norm_layer(nn.Module):
    def __init__(self, p=2.0, dim=(1, 2), eps=1e-10):
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return torch.nn.functional.normalize(x, p=self.p, dim=self.dim, eps=self.eps)


class RADAMPooling(nn.Module):
    def __init__(self, spatial_size, in_channels, M=4, pos_encoding=True, device=None):
        super().__init__()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.M = M
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.Q = 1  # number of hidden neurons in RAE (as in the paper)
        self.pos_encoding = pos_encoding

        self.rearrange = nn.Sequential(
            lp_norm_layer(p=2.0, dim=(2, 3), eps=1e-10),
            T.Resize(spatial_size),
            Rearrange('b c h w -> b c (h w)')
        )

        # Initialize M randomized autoencoders (RAEs)
        self.RAEs = [
            RAE(Q=self.Q, P=in_channels, N=spatial_size ** 2,
                device=self.device, seed=i * (self.Q * in_channels),
                pos_encoding=pos_encoding)
            for i in range(self.M)
        ]

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]  # handle a single feature map as list

        B = x[0].size(0)
        device = self.device
        out = []

        for i in range(len(x)):
            x[i] = self.rearrange(x[i].to(device))

        for b in range(B):
            feature_stack = torch.vstack([x[i][b] for i in range(len(x))])  # (C_total, N)
            pooled = torch.zeros(self.Q, self.in_channels, device=device)

            for rae in self.RAEs:
                pooled += rae.fit_AE(feature_stack)

            pooled = torch.nan_to_num(pooled)
            out.append(pooled)

        return torch.stack(out)  # shape: [B, Q, C] → you may reshape it afterward
