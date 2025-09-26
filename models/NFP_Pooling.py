import torch
import torch.nn as nn
from models.pooling.nfp import NFPPooling

class nfp_pooling(nn.Module):
    def __init__(self, nfp_layer=None, Params=None):
        super(nfp_pooling, self).__init__()
        if nfp_layer is None:
            dense_feature_dim = Params["num_ftrs"][Params["Model_name"]] if Params else 2048
            nfp_layer = NFPPooling(
                in_channels=dense_feature_dim,
                R=1,
                measure='cosine',
                padding=1,
                input_size=Params.get('input_size', 7) if Params else 7
            )
        self.nfp_layer = nfp_layer
        self.model_name = Params["Model_name"] if Params is not None else None
        self.dataset = Params["Dataset"] if Params is not None else None
        self.num_classes = Params["num_classes"][self.dataset] if Params is not None else None
        self.feature_extraction = Params['feature_extraction'] if Params is not None and 'feature_extraction' in Params else None
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.nfp_proj = nn.Linear(self.nfp_layer.out_channels, dense_feature_dim) if Params else None

    def forward(self, x):
        # Global average pooling: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        x_avg = self.avgpool(x).view(x.size(0), -1)
        # NFP pooling: (B, C, H, W) -> (B, C', H', W')
        x_nfp = self.nfp_layer(x)
        # Global average pooling on NFP output: (B, C', H', W') -> (B, C', 1, 1) -> (B, C')
        x_nfp = nn.functional.adaptive_avg_pool2d(x_nfp, (1, 1)).view(x_nfp.size(0), -1)
        if self.nfp_proj is not None:
            x_nfp = self.nfp_proj(x_nfp)
        # Elementwise multiplication
        pool_layer = x_avg * x_nfp
        return pool_layer