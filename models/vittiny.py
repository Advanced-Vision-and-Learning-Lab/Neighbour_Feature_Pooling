import torch
import torch.nn as nn
import timm
from models.pooling.nfp import NFPPooling

# 1. Only GAP, no NFP, no fusion
class VITTINY_GAP_ONLY(nn.Module):
    """
    ViTTiny backbone with Global Average Pooling (GAP) and a final fully connected layer.
    No NFP, no conv, no MLP, no concat; just classic classification.
    """
    def __init__(self, num_classes=21, input_shape=(3, 224, 224)):
        super().__init__()
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0, global_pool='')
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feats = self.backbone.forward_features(dummy)
            if feats.dim() == 3:
                patch_tokens = feats[:, 1:]
                B, N, C = patch_tokens.shape
                H = W = int(N ** 0.5)
                fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
            else:
                fmap = feats
            self.backbone_out_dim = fmap.shape[1]
        self.fc = nn.Linear(self.backbone_out_dim, num_classes)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 3:
            patch_tokens = feats[:, 1:]
            B, N, C = patch_tokens.shape
            H = W = int(N ** 0.5)
            fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        else:
            fmap = feats
        gap_out = self.gap(fmap)
        gap_out = gap_out.view(fmap.size(0), -1)
        out = self.fc(gap_out)
        return out

# 2. Only GAP with MLP
class VITTINY_GAP_MLP(nn.Module):
    """
    ViTTiny backbone with GAP, modulated by an MLP gate (self-attention on GAP feature).
    No NFP, no conv, but the GAP feature is adaptively gated by an MLP. No concat.
    """
    def __init__(self, num_classes=21, dropout_p=0.2, input_shape=(3, 224, 224)):
        super().__init__()
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0, global_pool='')
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p)
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feats = self.backbone.forward_features(dummy)
            if feats.dim() == 3:
                patch_tokens = feats[:, 1:]
                B, N, C = patch_tokens.shape
                H = W = int(N ** 0.5)
                fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
            else:
                fmap = feats
            self.backbone_out_dim = fmap.shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(self.backbone_out_dim, self.backbone_out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone_out_dim // 2, self.backbone_out_dim),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(self.backbone_out_dim, num_classes)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 3:
            patch_tokens = feats[:, 1:]
            B, N, C = patch_tokens.shape
            H = W = int(N ** 0.5)
            fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        else:
            fmap = feats
        gap_out = self.gap(fmap).view(fmap.size(0), -1)
        gated = gap_out * self.mlp(gap_out)
        out = self.fc(self.dropout(gated))
        return out

# 3. Only NFP with conv
class VITTINY_NFP_CONV_ONLY(nn.Module):
    """
    ViTTiny backbone with NFPPooling and conv compression.
    Only NFP features are used (with conv), no GAP, no MLP, no concat.
    """
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), nfp_kwargs=None, bottleneck_dim=None, in_channels=None):
        super().__init__()
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0, global_pool='')
        for p in self.backbone.parameters():
            p.requires_grad = True
        nfp_kwargs = nfp_kwargs or dict(R=1, measure='cosine', input_size=input_shape[-1])
        self.nfp = NFPPooling(in_channels, **nfp_kwargs)
        if self.nfp.in_channels != 1:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                feats = self.backbone.forward_features(dummy)
                if feats.dim() == 3:
                    patch_tokens = feats[:, 1:]
                    B, N, C = patch_tokens.shape
                    H = W = int(N ** 0.5)
                    fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
                else:
                    fmap = feats
                self.nfp.in_channels = fmap.shape[1]
                nfp_out = self.nfp(fmap)
                nfp_channels = nfp_out.shape[1]
            if bottleneck_dim is None:
                bottleneck_dim = 192
            self.compress = nn.Sequential(
                nn.Conv2d(nfp_channels, bottleneck_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(bottleneck_dim),
                nn.ReLU(inplace=True),
            )
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(bottleneck_dim, num_classes)
    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 3:
            patch_tokens = feats[:, 1:]
            B, N, C = patch_tokens.shape
            H = W = int(N ** 0.5)
            fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        else:
            fmap = feats
        nfp_out = self.nfp(fmap)
        nfp_out = self.compress(nfp_out)
        pooled = torch.flatten(self.gap(nfp_out), 1)
        logits = self.fc(pooled)
        return logits

# 4. Only NFP with conv and MLP
class VITTINY_NFP_CONV_MLP(nn.Module):
    """
    ViTTiny backbone with NFPPooling, conv compression, and MLP gating.
    Only NFP features are used (with conv), compressed and modulated by an MLP. No GAP, no concat.
    """
    def __init__(self, num_classes=21, nfp_kwargs=None, bottleneck_dim=None, dropout_p=0.2, input_shape=(3, 224, 224), in_channels=None):
        super().__init__()
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0, global_pool='')
        for p in self.backbone.parameters():
            p.requires_grad = True
        nfp_kwargs = nfp_kwargs or dict(R=1, measure='cosine', input_size=input_shape[-1])
        self.nfp = NFPPooling(in_channels, **nfp_kwargs)
        if self.nfp.in_channels != 1:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                feats = self.backbone.forward_features(dummy)
                if feats.dim() == 3:
                    patch_tokens = feats[:, 1:]
                    B, N, C = patch_tokens.shape
                    H = W = int(N ** 0.5)
                    fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
                else:
                    fmap = feats
                self.nfp.in_channels = fmap.shape[1]
                nfp_out = self.nfp(fmap)
                nfp_channels = nfp_out.shape[1]
            if bottleneck_dim is None:
                bottleneck_dim = 192
            self.compress = nn.Sequential(
                nn.Conv2d(nfp_channels, bottleneck_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(bottleneck_dim),
                nn.ReLU(inplace=True),
            )
            self.mlp = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(bottleneck_dim, bottleneck_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(bottleneck_dim // 2, bottleneck_dim),
                nn.Sigmoid()
            )
            self.dropout = nn.Dropout(dropout_p)
            self.fc = nn.Linear(bottleneck_dim, num_classes)
    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 3:
            patch_tokens = feats[:, 1:]
            B, N, C = patch_tokens.shape
            H = W = int(N ** 0.5)
            fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        else:
            fmap = feats
        nfp_map = self.nfp(fmap)
        nfp_map = self.compress(nfp_map)
        mlp_weight = self.mlp(nfp_map)
        nfp_vec = torch.flatten(nn.functional.adaptive_avg_pool2d(nfp_map, 1), 1)
        gated = nfp_vec * mlp_weight
        out = self.fc(self.dropout(gated))
        return out

# 5. NFP (with conv) concat with GAP, no MLP
class VITTINY_GAP_NFP_CONV_NOMLP_CONCAT(nn.Module):
    """
    ViTTiny backbone with GAP and NFP (NFPPooling + Conv) features concatenated.
    NFP features are projected to match GAP dimension before concatenation. No MLP.
    """
    def __init__(self, num_classes=21, bottleneck_dim=None, R=1, measure="cosine", dropout_p=0.2, input_shape=(3, 224, 224), in_channels=None):
        super().__init__()
        self.backbone = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0, global_pool="")
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.nfp = NFPPooling(in_channels, R=R, measure=measure, padding=R)
        if self.nfp.in_channels != 1:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                feats = self.backbone.forward_features(dummy)
                if feats.dim() == 3:
                    patch_tokens = feats[:, 1:]
                    B, N, C = patch_tokens.shape
                    H = W = int(N ** 0.5)
                    fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
                else:
                    fmap = feats
                self.backbone_out_dim = fmap.shape[1]
                self.nfp.in_channels = self.backbone_out_dim
                nfp_out = self.nfp(fmap)
                self.nfp_out_channels = nfp_out.shape[1]
            if bottleneck_dim is None:
                bottleneck_dim = 192
            self.nfp_conv = nn.Sequential(
                nn.Conv2d(self.nfp_out_channels, bottleneck_dim, 1, bias=False),
                nn.BatchNorm2d(bottleneck_dim),
                nn.ReLU(inplace=True),
            )
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(dropout_p)
            self.fc = nn.Linear(self.backbone_out_dim + bottleneck_dim, num_classes)
    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 3:
            patch_tokens = feats[:, 1:]
            B, N, C = patch_tokens.shape
            H = W = int(N ** 0.5)
            fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        else:
            fmap = feats
        gap_feat = torch.flatten(self.gap(fmap), 1)
        nfp_map = self.nfp(fmap)
        nfp_feat_map = self.nfp_conv(nfp_map)
        nfp_feat = torch.flatten(self.gap(nfp_feat_map), 1)
        fused = torch.cat([gap_feat, nfp_feat], dim=1)
        logits = self.fc(self.dropout(fused))
        return logits

# 6. NFP (no conv) concat with GAP, no MLP
class VITTINY_GAP_NFP_NOCONV_NOMLP_CONCAT(nn.Module):
    """
    ViTTiny backbone with GAP and NFPPooling features concatenated (no conv projection).
    NFP features are concatenated directly with GAP features. No MLP.
    """
    def __init__(self, num_classes=21, R=1, measure="cosine", dropout_p=0.2, input_shape=(3, 224, 224), in_channels=None):
        super().__init__()
        self.backbone = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0, global_pool="")
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.nfp = NFPPooling(in_channels, R=R, measure=measure, padding=R)
        if self.nfp.in_channels != 1:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                feats = self.backbone.forward_features(dummy)
                if feats.dim() == 3:
                    patch_tokens = feats[:, 1:]
                    B, N, C = patch_tokens.shape
                    H = W = int(N ** 0.5)
                    fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
                else:
                    fmap = feats
                self.backbone_out_dim = fmap.shape[1]
                self.nfp.in_channels = self.backbone_out_dim
                nfp_out = self.nfp(fmap)
                self.nfp_out_channels = nfp_out.shape[1]
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(dropout_p)
            self.fc = nn.Linear(self.backbone_out_dim + self.nfp_out_channels, num_classes)
    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 3:
            patch_tokens = feats[:, 1:]
            B, N, C = patch_tokens.shape
            H = W = int(N ** 0.5)
            fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        else:
            fmap = feats
        gap_feat = torch.flatten(self.gap(fmap), 1)
        nfp_map = self.nfp(fmap)
        nfp_feat = torch.flatten(self.gap(nfp_map), 1)
        fused = torch.cat([gap_feat, nfp_feat], dim=1)
        logits = self.fc(self.dropout(fused))
        return logits

# 7. NFP (with conv) concat with GAP, with MLP
class VITTINY_GAP_NFP_CONV_MLP_CONCAT(nn.Module):
    """
    ViTTiny backbone with GAP and NFP (NFPPooling + Conv) features concatenated, then fused by MLP.
    NFP features are projected to match GAP dimension before concatenation. MLP fusion.
    """
    def __init__(self, num_classes=21, bottleneck_dim=None, R=1, measure="cosine", dropout_p=0.2, input_shape=(3, 224, 224), in_channels=None):
        super().__init__()
        self.backbone = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0, global_pool="")
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.nfp = NFPPooling(in_channels, R=R, measure=measure, padding=R)
        if self.nfp.in_channels != 1:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                feats = self.backbone.forward_features(dummy)
                if feats.dim() == 3:
                    patch_tokens = feats[:, 1:]
                    B, N, C = patch_tokens.shape
                    H = W = int(N ** 0.5)
                    fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
                else:
                    fmap = feats
                self.backbone_out_dim = fmap.shape[1]
                self.nfp.in_channels = self.backbone_out_dim
                nfp_out = self.nfp(fmap)
                self.nfp_out_channels = nfp_out.shape[1]
            if bottleneck_dim is None:
                bottleneck_dim = 192
            self.nfp_conv = nn.Sequential(
                nn.Conv2d(self.nfp_out_channels, bottleneck_dim, 1, bias=False),
                nn.BatchNorm2d(bottleneck_dim),
                nn.ReLU(inplace=True),
            )
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(dropout_p)
            self.mlp = nn.Sequential(
                nn.Linear(self.backbone_out_dim + bottleneck_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.backbone_out_dim + bottleneck_dim),
                nn.Sigmoid()
            )
            self.fc = nn.Linear(self.backbone_out_dim + bottleneck_dim, num_classes)
    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 3:
            patch_tokens = feats[:, 1:]
            B, N, C = patch_tokens.shape
            H = W = int(N ** 0.5)
            fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        else:
            fmap = feats
        gap_feat = torch.flatten(self.gap(fmap), 1)
        nfp_map = self.nfp(fmap)
        nfp_feat_map = self.nfp_conv(nfp_map)
        nfp_feat = torch.flatten(self.gap(nfp_feat_map), 1)
        fused = torch.cat([gap_feat, nfp_feat], dim=1)
        fused = fused * self.mlp(fused)
        logits = self.fc(self.dropout(fused))
        return logits

# 8. NFP (no conv) concat with GAP, with MLP
class VITTINY_GAP_NFP_NOCONV_MLP_CONCAT(nn.Module):
    """
    ViTTiny backbone with GAP and NFPPooling features concatenated (no conv), then fused by MLP.
    NFP features are concatenated directly with GAP features. MLP fusion.
    """
    def __init__(self, num_classes=21, R=1, measure="cosine", dropout_p=0.2, input_shape=(3, 224, 224), in_channels=None):
        super().__init__()
        self.backbone = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0, global_pool="")
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.nfp = NFPPooling(in_channels, R=R, measure=measure, padding=R)
        if self.nfp.in_channels != 1:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                feats = self.backbone.forward_features(dummy)
                if feats.dim() == 3:
                    patch_tokens = feats[:, 1:]
                    B, N, C = patch_tokens.shape
                    H = W = int(N ** 0.5)
                    fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
                else:
                    fmap = feats
                self.backbone_out_dim = fmap.shape[1]
                self.nfp.in_channels = self.backbone_out_dim
                nfp_out = self.nfp(fmap)
                self.nfp_out_channels = nfp_out.shape[1]
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(dropout_p)
            self.mlp = nn.Sequential(
                nn.Linear(self.backbone_out_dim + self.nfp_out_channels, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.backbone_out_dim + self.nfp_out_channels),
                nn.Sigmoid()
            )
            self.fc = nn.Linear(self.backbone_out_dim + self.nfp_out_channels, num_classes)
    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 3:
            patch_tokens = feats[:, 1:]
            B, N, C = patch_tokens.shape
            H = W = int(N ** 0.5)
            fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        else:
            fmap = feats
        gap_feat = torch.flatten(self.gap(fmap), 1)
        nfp_map = self.nfp(fmap)
        nfp_feat = torch.flatten(self.gap(nfp_map), 1)
        fused = torch.cat([gap_feat, nfp_feat], dim=1)
        fused = fused * self.mlp(fused)
        logits = self.fc(self.dropout(fused))
        return logits 