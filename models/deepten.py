# models/deepten.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepTENEncoding(nn.Module):
    """
    PyTorch-only version of the Encoding Layer from Deep-TEN.
    This layer aggregates residuals from input features to a learnable codebook.
    Reference: CVPR 2017 - Hang Zhang et al.
    """
    def __init__(self, in_channels, num_codes):
        """
        Args:
            in_channels (int): Feature channels D.
            num_codes (int): Number of codewords K.
        """
        super().__init__()
        self.D = in_channels
        self.K = num_codes
        self.codewords = nn.Parameter(torch.randn(self.K, self.D))
        self.scale = nn.Parameter(torch.randn(self.K))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / (self.K * self.D) ** 0.5
        self.codewords.data.uniform_(-std, std)
        self.scale.data.uniform_(-1, 0)

    def scaled_l2(self, X, C, S):
        # X: [B, N, D], C: [K, D], S: [K]
        # Output: [B, N, K]
        B, N, D = X.size()
        C = C.unsqueeze(0).unsqueeze(0)  # [1, 1, K, D]
        X = X.unsqueeze(2)               # [B, N, 1, D]
        r = X - C                        # residuals
        s = S.view(1, 1, self.K, 1)      # [1, 1, K, 1]
        dist = torch.sum(r ** 2, dim=3)  # [B, N, K]
        return -s.squeeze(-1) * dist

    def aggregate(self, A, X, C):
        # A: [B, N, K], X: [B, N, D], C: [K, D]
        B, N, D = X.size()
        C = C.unsqueeze(0)  # [1, K, D]
        A = A.transpose(1, 2)  # [B, K, N]
        R = X.unsqueeze(1) - C.unsqueeze(2)  # [1, K, N, D]
        E = (A.unsqueeze(3) * R).sum(2)  # [B, K, D]
        return E

    def forward(self, x):
        # Input x: [B, D, H, W]
        B, D, H, W = x.size()
        assert D == self.D
        x = x.view(B, D, -1).transpose(1, 2)  # [B, N, D]
        A = F.softmax(self.scaled_l2(x, self.codewords, self.scale), dim=2)
        E = self.aggregate(A, x, self.codewords)  # [B, K, D]
        return E.view(B, -1)  # flatten to [B, K*D]
