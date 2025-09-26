# MobileNetV3-based models for Neighbour Feature Pooling
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from models.nfp_heads import NFPHead, MultiRadiusNFPHead, PositionalEncoding2D, AttentionFusion, NFPBottleneck, SimilarityAwarePooling
from models.pooling.enhanced_nfp import EnhancedNFPPooling

class ViTTinyWithNFPHead(nn.Module):
    """
    ViT-Tiny backbone with single-radius (R=1) Enhanced-NFP head fused with GAP using MLP.
    """
    def __init__(self, num_classes=21, bottleneck_dim=None, R=1, measure="cosine", dropout_p=0.2):
        super().__init__()
        from models.nfp_heads import NFPHead
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0, global_pool='')
        for p in self.backbone.parameters():
            p.requires_grad = True
            
        # ViT-Tiny has 192 output channels (not 256 as incorrectly stated before)
        out_channels = 192
        if bottleneck_dim is None:
            bottleneck_dim = out_channels
        
        self.nfp_head = NFPHead(in_c=out_channels, bottleneck_dim=bottleneck_dim, R=R, measure=measure)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(bottleneck_dim, num_classes)
        
    def forward(self, x):
        # ViT-Tiny returns [B, 197, 192] where 197 = 1 (class token) + 196 (patch tokens)
        feats = self.backbone.forward_features(x)  # [B, 197, 192]
        
        # Extract patch tokens (exclude class token)
        patch_tokens = feats[:, 1:]  # [B, 196, 192]
        
        # Reshape to 4D feature map: [B, 196, 192] -> [B, 192, 14, 14]
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)  # 196 = 14 * 14
        fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)  # [B, 192, 14, 14]
        
        fused = self.nfp_head(fmap)
        logits = self.fc(self.dropout(fused))
        return logits

class MobileNetV3WithNFPHead(nn.Module):
    """
    MobileNetV3 backbone with single-radius (R=1) Enhanced-NFP head fused with GAP using MLP.
    """
    def __init__(self, num_classes=21, bottleneck_dim=None, R=1, measure="cosine", dropout_p=0.2):
        super().__init__()
        from models.nfp_heads import NFPHead
        self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0, global_pool='')
        for p in self.backbone.parameters():
            p.requires_grad = True
            
        # MobileNetV3-Large has 960 output channels
        out_channels = 960
        if bottleneck_dim is None:
            bottleneck_dim = out_channels
        
        self.nfp_head = NFPHead(in_c=out_channels, bottleneck_dim=bottleneck_dim, R=R, measure=measure)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(bottleneck_dim, num_classes)
        
    def forward(self, x):
        fmap = self.backbone.forward_features(x)  # [B, 960, H, W]
        fused = self.nfp_head(fmap)
        logits = self.fc(self.dropout(fused))
        return logits 