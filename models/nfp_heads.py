# Shared NFP heads, fusion, and utility modules for Neighbour Feature Pooling
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.pooling.enhanced_nfp import EnhancedNFPPooling



####  1 - NFPHead
class NFPHead(nn.Module):
    """
    Combines GAP with a single Enhanced-NFP feature map (R=1),
    fuses them using an MLP, returns a 512-D vector.
    """
    def __init__(self, in_c=512, bottleneck_dim=512, R=1, measure="cosine"):
        super().__init__()
        self.nfp = EnhancedNFPPooling(
            in_channels=in_c,
            R=R,
            measure=measure,
            padding=R,
        )
        with torch.no_grad():
            dummy = torch.randn(1, in_c, 7, 7)
            nfp_out = self.nfp(dummy)
            self.nfp_out_channels = nfp_out.shape[1]
        self.compress = nn.Sequential(
            nn.Conv2d(self.nfp_out_channels, bottleneck_dim, 1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(in_c + bottleneck_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, bottleneck_dim),
        )
    def forward(self, fmap):
        gap_vec = torch.flatten(self.gap(fmap), 1)
        nfp_map = self.nfp(fmap)
        nfp_cmp = self.compress(nfp_map)
        nfp_vec = torch.flatten(self.gap(nfp_cmp), 1)
        fusion_input = torch.cat([gap_vec, nfp_vec], dim=1)
        fused = self.fusion_mlp(fusion_input)
        return fused


## 2 - NFPHead_NoConv
class NFPHead_NoConv(nn.Module):
    """
    Combines GAP with Enhanced-NFP feature map (R=1),
    concatenates directly after GAP without any Conv layer,
    fuses them using an MLP, returns a 512-D vector.
    """
    def __init__(self, in_c=512, R=1, measure="cosine"):
        super().__init__()
        self.nfp = EnhancedNFPPooling(
            in_channels=in_c,
            R=R,
            measure=measure,
            padding=R,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(in_c + in_c, 512),  # GAP 512 + NFP GAP 512
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )

    def forward(self, fmap):
        gap_vec = torch.flatten(self.gap(fmap), 1)         # [B, 512]
        nfp_map = self.nfp(fmap)                           # [B, 512, H, W]
        nfp_vec = torch.flatten(self.gap(nfp_map), 1)      # [B, 512]
        fusion_input = torch.cat([gap_vec, nfp_vec], dim=1)  # [B, 1024]
        fused = self.fusion_mlp(fusion_input)              # [B, 512]
        return fused

####  2 - MultiRadiusNFPHead
class MultiRadiusNFPHead(nn.Module):
    """
    Fuses GAP with a concatenation of Enhanced-NFP feature maps
    computed at multiple radii (default R = 1, 2), returning a 512-D vector.
    """
    def __init__(self, in_c=512, bottleneck_dim=512, R_list=(1, 2), measure="cosine"):
        super().__init__()
        self.nfp_blocks = nn.ModuleList([
            EnhancedNFPPooling(
                in_channels=in_c,
                R=R,
                measure=measure,
                padding=R,
            ) for R in R_list
        ])
        with torch.no_grad():
            dummy = torch.randn(1, in_c, 7, 7)
            total_c = sum(b(dummy).shape[1] for b in self.nfp_blocks)
        self.compress = nn.Sequential(
            nn.Conv2d(total_c, bottleneck_dim, 1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        self.se_gate = nn.Sequential(
            nn.Linear(in_c + bottleneck_dim, (in_c + bottleneck_dim) // 2),
            nn.ReLU(inplace=True),
            nn.Linear((in_c + bottleneck_dim) // 2, 1),
            nn.Sigmoid(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
    def forward(self, fmap):
        gap_vec = torch.flatten(self.gap(fmap), 1)
        nfp_maps = [blk(fmap) for blk in self.nfp_blocks]
        nfp_cat = torch.cat(nfp_maps, dim=1)
        nfp_cmp = self.compress(nfp_cat)
        nfp_vec = torch.flatten(self.gap(nfp_cmp), 1)
        alpha = self.se_gate(torch.cat([gap_vec, nfp_vec], dim=1))
        fused = gap_vec + alpha * nfp_vec
        return fused

####  3 - PositionalEncoding2D
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        self.height = height
        self.width = width
        pe = torch.zeros(d_model, height, width)
        y_pos = torch.arange(0, height).unsqueeze(1).float()
        x_pos = torch.arange(0, width).unsqueeze(0).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[0::2, :, :] = torch.sin(y_pos * div_term.view(-1, 1, 1))
        pe[1::2, :, :] = torch.cos(x_pos * div_term.view(-1, 1, 1))
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

####  4 - AttentionFusion
class AttentionFusion(nn.Module):
    def __init__(self, gap_dim, nfp_dim, fusion_dim=512):
        super().__init__()
        self.gap_proj = nn.Linear(gap_dim, fusion_dim)
        self.nfp_proj = nn.Linear(nfp_dim, fusion_dim)
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )
    def forward(self, gap, nfp):
        gap_proj = self.gap_proj(gap)
        nfp_proj = self.nfp_proj(nfp)
        fused = torch.cat([gap_proj, nfp_proj], dim=1)
        weights = self.gate(fused)
        gap_weight = weights[:, 0].unsqueeze(1)
        nfp_weight = weights[:, 1].unsqueeze(1)
        fused_output = gap_weight * gap_proj + nfp_weight * nfp_proj
        return fused_output
####  5 - NFPBottleneck
class NFPBottleneck(nn.Module):
    expansion = 1
    def __init__(self, in_channels: int, out_channels: int, stride=1):
        super().__init__()
        mid = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=False, stride=stride)
        self.bn1   = nn.BatchNorm2d(mid)
        self.nfp = EnhancedNFPPooling(mid, R=1, measure="cosine", padding=0)
        with torch.no_grad():
            mid2 = self.nfp(torch.randn(1, mid, 5, 5)).shape[1]
        self.conv2 = nn.Conv2d(mid2, out_channels, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
    def _match(self, ident, target_hw):
        if ident.shape[-1] == target_hw:
            return ident
        k = ident.shape[-1] - target_hw + 1
        return F.avg_pool2d(ident, kernel_size=k, stride=1, padding=0)
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.nfp(out)
        out = self.bn2(self.conv2(out))
        identity = self._match(identity, out.shape[-1])
        out += identity
        return self.relu(out)
####  6 - PositionalEncoding2D
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        self.height = height
        self.width = width

        pe = torch.zeros(d_model, height, width)
        y_pos = torch.arange(0, height).unsqueeze(1).float()
        x_pos = torch.arange(0, width).unsqueeze(0).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[0::2, :, :] = torch.sin(y_pos * div_term.view(-1, 1, 1))
        pe[1::2, :, :] = torch.cos(x_pos * div_term.view(-1, 1, 1))

        self.register_buffer('pe', pe.unsqueeze(0))  # shape: [1, C, H, W]

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2), :x.size(3)] 
####  7 - SimilarityAwarePooling
class SimilarityAwarePooling(nn.Module):
    def __init__(self, in_channels=512, R=1, measure='cosine', **kwargs):
        super().__init__()
        from models.pooling.enhanced_nfp import EnhancedNFPPooling
        self.nfp = EnhancedNFPPooling(
            in_channels=in_channels,
            R=R,
            measure=measure,
            padding=0,
            **kwargs
        )
        # Determine output channels after NFP
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, 7, 7)
            nfp_out = self.nfp(dummy)
        nfp_channels = nfp_out.shape[1]
        self.att_proj = nn.Conv2d(nfp_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.nfp_channels = nfp_channels  # Save for use in parent

    def forward(self, x):
        x = self.nfp(x)  # (B, C, H, W)
        B, C, H, W = x.shape
        att = self.att_proj(x)         # (B, 1, H, W)
        att = att.flatten(2)           # (B, 1, H*W)
        att = self.softmax(att)        # (B, 1, H*W)
        att = att.view(B, 1, H, W)     # (B, 1, H, W)
        pooled = (x * att).sum(dim=[2, 3])  # (B, C)
        return pooled
####  8 - NFPBottleneck
class NFPBottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride=1):
        super().__init__()
        mid = out_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(mid)
        self.nfp = EnhancedNFPPooling(mid, R=1, measure="cosine", padding=0)

        with torch.no_grad():
            mid2 = self.nfp(torch.randn(1, mid, 5, 5)).shape[1]

        self.conv2 = nn.Conv2d(mid2, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 🔧 Fix for residual shape mismatch
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def _match(self, ident, target_hw):
        if ident.shape[-1] == target_hw:
            return ident
        k = ident.shape[-1] - target_hw + 1
        return F.avg_pool2d(ident, kernel_size=k, stride=1, padding=0)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.nfp(out)
        out = self.bn2(self.conv2(out))

        identity = self._match(identity, out.shape[-1])
        identity = self.downsample(identity)  # ✅ Channel match here

        out += identity
        return self.relu(out)



####  9 - AdaptiveFusionNFP
class AdaptiveFusionNFP(nn.Module):
    """
    Combines GAP and Enhanced NFP features using attention-based fusion.
    """
    def __init__(self, in_channels=512, bottleneck_dim=512, R=1, measure='cosine', dropout_p=0.2):
        super().__init__()

        # GAP branch
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # NFP branch
        self.nfp = EnhancedNFPPooling(in_channels=in_channels, R=R, measure=measure, padding=R)
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, 7, 7)
            nfp_out = self.nfp(dummy)
            self.nfp_dim = nfp_out.view(1, -1).size(1)

        # Compression (Conv1x1) on NFP
        self.compress = nn.Sequential(
            nn.Conv2d(nfp_out.shape[1], bottleneck_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
        )

        # SE-style attention gate to fuse GAP and NFP adaptively
        fusion_dim = in_channels + bottleneck_dim
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim // 2, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        # Global Average Pooling
        gap_feat = torch.flatten(self.gap(x), 1)  # [B, 512]

        # NFP map → compress → GAP
        nfp_map = self.compress(self.nfp(x))  # [B, bottleneck_dim, H, W]
        nfp_feat = torch.flatten(F.adaptive_avg_pool2d(nfp_map, 1), 1)  # [B, bottleneck_dim]

        # Concatenate and fuse via attention
        fusion_input = torch.cat([gap_feat, nfp_feat], dim=1)
        alpha = self.fusion_gate(fusion_input)  # [B, 1]

        # Final fused vector
        fused = gap_feat + alpha * nfp_feat
        return self.dropout(fused)  # Final output size: [B, 512]