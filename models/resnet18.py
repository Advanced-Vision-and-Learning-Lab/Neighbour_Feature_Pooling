# ResNet18-based models for Neighbour Feature Pooling with MLP fusion
import torch
import torch.nn as nn
import timm
from models.pooling.nfp import NFPPooling

class NFPHead(nn.Module):
    """
    Combines GAP with a single Enhanced-NFP feature map (R=1),
    fuses them using an MLP, returns a 512-D vector.
    """
    def __init__(self, in_c=512, bottleneck_dim=512, R=1, measure="cosine"):
        super().__init__()
        print(f"NFPHead: in_c={in_c}, bottleneck_dim={bottleneck_dim}")
        
        self.nfp = NFPPooling(
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




class ResNet18_NFPHeadWithSEGate(nn.Module):
    """
    ResNet-18 with GAP + NFP features fused using SE-style gating.
    """
    def __init__(self, num_classes=21, bottleneck_dim=512, R=1, measure="cosine", dropout_p=0.2, input_shape=None):
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=True, num_classes=0, global_pool="")

        for p in self.backbone.parameters():
            p.requires_grad = True  # full finetuning

        self.nfp_head = NFPHead(in_c=512, bottleneck_dim=bottleneck_dim, R=R, measure=measure)
        self.dropout = nn.Dropout(dropout_p)

        # SE-style gate: input is concatenation of GAP and NFP features
        self.se_gate = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        fmap = self.backbone.forward_features(x)                   # [B, 512, 7, 7]
        gap_feat = nn.functional.adaptive_avg_pool2d(fmap, 1).view(x.size(0), -1)  # [B, 512]
        nfp_feat = self.nfp_head(fmap)                             # [B, 512]

        # Compute fusion weight
        fusion_input = torch.cat([gap_feat, nfp_feat], dim=1)      # [B, 1024]
        alpha = self.se_gate(fusion_input)                         # [B, 1]
        fused = (1 - alpha) * gap_feat + alpha * nfp_feat          # [B, 512]

        logits = self.fc(self.dropout(fused))
        return logits

# --- Simple NFP+Conv block (no fusion/MLP) ---
# (No class needed, just use  NFPPooling + Conv+BN+ReLU in each model)

# 1. Only GAP, no NFP, no fusion
class RESNET18_GAP_ONLY(nn.Module):
    """
    Standard ResNet18 backbone with Global Average Pooling (GAP) and a final fully connected layer.
    No NFP, no conv, no MLP, no concat; just classic CNN classification.
    """
    def __init__(self, num_classes=21, input_shape=(3, 224, 224)):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=True, num_classes=0, global_pool='')
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # Dynamically infer backbone output feature size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feat = self.backbone.forward_features(dummy)
            self.backbone_out_dim = feat.shape[1]
        self.fc = nn.Linear(self.backbone_out_dim, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        gap_out = self.gap(x)
        gap_out = gap_out.view(x.size(0), -1)
        out = self.fc(gap_out)
        return out

# 2. Only GAP with MLP
class RESNET18_GAP_MLP(nn.Module):
    """
    ResNet18 backbone with GAP, modulated by an MLP gate (self-attention on GAP feature).
    No NFP, no conv, but the GAP feature is adaptively gated by an MLP. No concat.
    """
    def __init__(self, num_classes=21, dropout_p=0.2, input_shape=(3, 224, 224)):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=True, num_classes=0, global_pool='')
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p)
        # Dynamically infer backbone output feature size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feat = self.backbone.forward_features(dummy)
            self.backbone_out_dim = feat.shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(self.backbone_out_dim, self.backbone_out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone_out_dim // 2, self.backbone_out_dim),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(self.backbone_out_dim, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        gap_out = self.gap(x).view(x.size(0), -1)
        gated = gap_out * self.mlp(gap_out)
        out = self.fc(self.dropout(gated))
        return out

# 3. Only NFP with conv
class RESNET18_NFP_CONV_ONLY(nn.Module):
    """
    ResNet18 backbone with NFPPooling and conv compression.
    Only NFP features are used (with conv), no GAP, no MLP, no concat.
    """
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), nfp_kwargs=None, bottleneck_dim=None):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=True, num_classes=0, global_pool='')
        for p in self.backbone.parameters():
            p.requires_grad = True
        nfp_kwargs = nfp_kwargs or dict(R=1, measure='cosine', input_size=input_shape[-1])
        self.nfp = NFPPooling(**nfp_kwargs)  # in_channels will be set dynamically
        # Dynamically infer backbone and NFP output feature size
        if self.nfp.in_channels != 1:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                feat = self.backbone.forward_features(dummy)
                self.nfp.in_channels = feat.shape[1]
                nfp_out = self.nfp(feat)
                nfp_channels = nfp_out.shape[1]
            if bottleneck_dim is None:
                bottleneck_dim = 512  # Always project to 512
            self.compress = nn.Sequential(
                nn.Conv2d(nfp_channels, bottleneck_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(bottleneck_dim),
                nn.ReLU(inplace=True),
            )
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(bottleneck_dim, num_classes)
        # Do not use nfp_channels, bottleneck_dim, self.compress, self.gap, or self.fc outside this block for the probing model

    def forward(self, x):
        feat = self.backbone.forward_features(x)
        nfp_out = self.nfp(feat)
        nfp_out = self.compress(nfp_out)
        pooled = torch.flatten(self.gap(nfp_out), 1)
        logits = self.fc(pooled)
        return logits

# 4. Only NFP with conv and MLP
class RESNET18_NFP_CONV_MLP(nn.Module):
    """
    ResNet18 backbone with NFPPooling, conv compression, and MLP gating.
    Only NFP features are used (with conv), compressed and modulated by an MLP. No GAP, no concat.
    """
    def __init__(self, num_classes=21, nfp_kwargs=None, bottleneck_dim=None, dropout_p=0.2, input_shape=(3, 224, 224)):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=True, num_classes=0, global_pool='')
        for p in self.backbone.parameters():
            p.requires_grad = True
        nfp_kwargs = nfp_kwargs or dict(R=1, measure='cosine', input_size=input_shape[-1])
        self.nfp = NFPPooling(**nfp_kwargs)
        # Dynamically infer backbone and NFP output feature size
        if self.nfp.in_channels != 1:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                feat = self.backbone.forward_features(dummy)
                self.nfp.in_channels = feat.shape[1]
                nfp_out = self.nfp(feat)
                nfp_channels = nfp_out.shape[1]
            if bottleneck_dim is None:
                bottleneck_dim = 512
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
        # Do not use nfp_channels, bottleneck_dim, self.compress, self.mlp, self.dropout, or self.fc outside this block for the probing model

    def forward(self, x):
        feat = self.backbone.forward_features(x)
        nfp_map = self.nfp(feat)
        nfp_map = self.compress(nfp_map)
        mlp_weight = self.mlp(nfp_map)
        nfp_vec = torch.flatten(nn.functional.adaptive_avg_pool2d(nfp_map, 1), 1)
        gated = nfp_vec * mlp_weight
        out = self.fc(self.dropout(gated))
        return out

# 5. NFP (with conv) concat with GAP, no MLP
class RESNET18_GAP_NFP_CONV_NOMLP_CONCAT(nn.Module):
    """
    ResNet18 backbone with GAP and NFP (NFPPooling + Conv) features concatenated.
    NFP features are projected to match GAP dimension before concatenation. No MLP.
    """
    def __init__(self, num_classes=21, bottleneck_dim=None, R=1, measure="cosine", dropout_p=0.2, input_shape=(3, 224, 224), in_channels=None):
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=True, num_classes=0, global_pool="")
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.nfp = NFPPooling(in_channels, R=R, measure=measure, padding=R)
        # Dynamically infer backbone and NFP output feature size
        if self.nfp.in_channels != 1:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                fmap = self.backbone.forward_features(dummy)
                self.backbone_out_dim = fmap.shape[1]
                self.nfp.in_channels = self.backbone_out_dim
                nfp_out = self.nfp(fmap)
                self.nfp_out_channels = nfp_out.shape[1]
            if bottleneck_dim is None:
                bottleneck_dim = 512
            self.nfp_conv = nn.Sequential(
                nn.Conv2d(self.nfp_out_channels, bottleneck_dim, 1, bias=False),
                nn.BatchNorm2d(bottleneck_dim),
                nn.ReLU(inplace=True),
            )
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(dropout_p)
            self.fc = nn.Linear(self.backbone_out_dim + bottleneck_dim, num_classes)
        # Do not use nfp_out_channels, bottleneck_dim, self.nfp_conv, self.gap, self.dropout, or self.fc outside this block for the probing model

    def forward(self, x):
        fmap = self.backbone.forward_features(x)
        gap_feat = torch.flatten(self.gap(fmap), 1)
        nfp_map = self.nfp(fmap)
        nfp_feat_map = self.nfp_conv(nfp_map)
        nfp_feat = torch.flatten(self.gap(nfp_feat_map), 1)
        fused = torch.cat([gap_feat, nfp_feat], dim=1)
        logits = self.fc(self.dropout(fused))
        return logits

# 6. NFP (no conv) concat with GAP, no MLP
class RESNET18_GAP_NFP_NOCONV_NOMLP_CONCAT(nn.Module):
    """
    ResNet18 backbone with GAP and NFPPooling features concatenated (no conv projection).
    NFP features are concatenated directly with GAP features. No MLP.
    """
    def __init__(self, num_classes=21, R=1, measure="cosine", dropout_p=0.2, input_shape=(3, 224, 224), in_channels=None):
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=True, num_classes=0, global_pool="")
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.nfp = NFPPooling(in_channels, R=R, measure=measure, padding=R)
        # Dynamically infer backbone and NFP output feature size
        if self.nfp.in_channels != 1:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                fmap = self.backbone.forward_features(dummy)
                self.backbone_out_dim = fmap.shape[1]
                self.nfp.in_channels = self.backbone_out_dim
                nfp_out = self.nfp(fmap)
                self.nfp_out_channels = nfp_out.shape[1]
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(dropout_p)
            # No projection layer here!
            self.fc = nn.Linear(self.backbone_out_dim + self.nfp_out_channels, num_classes)
        # Do not use nfp_out_channels, self.gap, self.dropout, or self.fc outside this block for the probing model

    def forward(self, x):
        fmap = self.backbone.forward_features(x)
        gap_feat = torch.flatten(self.gap(fmap), 1)
        nfp_map = self.nfp(fmap)
        nfp_feat = torch.flatten(self.gap(nfp_map), 1)
        fused = torch.cat([gap_feat, nfp_feat], dim=1)
        logits = self.fc(self.dropout(fused))
        return logits

# 7. NFP (with conv) concat with GAP, with MLP
class RESNET18_GAP_NFP_CONV_MLP_CONCAT(nn.Module):
    """
    ResNet18 backbone with GAP and NFP (NFPPooling + Conv) features concatenated, then fused by MLP.
    NFP features are projected to match GAP dimension before concatenation. MLP fusion.
    """
    def __init__(self, num_classes=21, bottleneck_dim=None, R=1, measure="cosine", dropout_p=0.2, input_shape=(3, 224, 224), in_channels=None):
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=True, num_classes=0, global_pool="")
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.nfp = NFPPooling(in_channels, R=R, measure=measure, padding=R)
        # Dynamically infer backbone and NFP output feature size
        if self.nfp.in_channels != 1:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                fmap = self.backbone.forward_features(dummy)
                self.backbone_out_dim = fmap.shape[1]
                self.nfp.in_channels = self.backbone_out_dim
                nfp_out = self.nfp(fmap)
                self.nfp_out_channels = nfp_out.shape[1]
            if bottleneck_dim is None:
                bottleneck_dim = 512
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
        # Do not use nfp_out_channels, bottleneck_dim, self.nfp_conv, self.gap, self.dropout, self.mlp, or self.fc outside this block for the probing model

    def forward(self, x):
        fmap = self.backbone.forward_features(x)
        gap_feat = torch.flatten(self.gap(fmap), 1)
        nfp_map = self.nfp(fmap)
        nfp_feat_map = self.nfp_conv(nfp_map)
        nfp_feat = torch.flatten(self.gap(nfp_feat_map), 1)
        fused = torch.cat([gap_feat, nfp_feat], dim=1)
        fused = fused * self.mlp(fused)
        logits = self.fc(self.dropout(fused))
        return logits

# 8. NFP (no conv) concat with GAP, with MLP
class RESNET18_GAP_NFP_NOCONV_MLP_CONCAT(nn.Module):
    """
    ResNet18 backbone with GAP and NFPPooling features concatenated (no conv), then fused by MLP.
    NFP features are concatenated directly with GAP features. MLP fusion.
    """
    def __init__(self, num_classes=21, R=1, measure="cosine", dropout_p=0.2, input_shape=(3, 224, 224), in_channels=None):
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=True, num_classes=0, global_pool="")
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.nfp = NFPPooling(in_channels, R=R, measure=measure, padding=R)
        # Dynamically infer backbone and NFP output feature size
        if self.nfp.in_channels != 1:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                fmap = self.backbone.forward_features(dummy)
                self.backbone_out_dim = fmap.shape[1]
                self.nfp.in_channels = self.backbone_out_dim
                nfp_out = self.nfp(fmap)
                self.nfp_out_channels = nfp_out.shape[1]
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(dropout_p)
            # No projection layer here!
            self.mlp = nn.Sequential(
                nn.Linear(self.backbone_out_dim + self.nfp_out_channels, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.backbone_out_dim + self.nfp_out_channels),
                nn.Sigmoid()
            )
            self.fc = nn.Linear(self.backbone_out_dim + self.nfp_out_channels, num_classes)
        # Do not use nfp_out_channels, self.gap, self.dropout, self.mlp, or self.fc outside this block for the probing model

    def forward(self, x):
        fmap = self.backbone.forward_features(x)
        gap_feat = torch.flatten(self.gap(fmap), 1)
        nfp_map = self.nfp(fmap)
        nfp_feat = torch.flatten(self.gap(nfp_map), 1)
        fused = torch.cat([gap_feat, nfp_feat], dim=1)
        fused = fused * self.mlp(fused)
        logits = self.fc(self.dropout(fused))
        return logits

class RESNET18_NFP_AT_LAYER(nn.Module):
    """
    ResNet18 backbone with NFP applied to the output of any main layer (layer1, layer2, layer3, layer4).
    Use layer_idx (0-3) to select which layer's output to use for NFP.
    """
    def __init__(self, num_classes=21, nfp_kwargs=None, bottleneck_dim=None, layer_idx=3, input_shape=(3, 224, 224)):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=True, num_classes=0, global_pool='')
        for p in self.backbone.parameters():
            p.requires_grad = True

        self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        assert 0 <= layer_idx < 4, "layer_idx must be 0 (layer1), 1 (layer2), 2 (layer3), or 3 (layer4)"
        self.layer_idx = layer_idx

        # Probe to get output channels for each layer
        self.layer_out_channels = []
        x = torch.zeros(1, *input_shape)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        for name in self.layer_names:
            x = getattr(self.backbone, name)(x)
            self.layer_out_channels.append(x.shape[1])

        # NFP for each layer
        self.nfp_modules = nn.ModuleList([
            NFPPooling(in_channels=ch, **(nfp_kwargs or {}))
            for ch in self.layer_out_channels
        ])

        # Projection and classifier
        if bottleneck_dim is None:
            bottleneck_dim = self.layer_out_channels[layer_idx]
        self.compress = nn.Sequential(
            nn.Conv2d(self.nfp_modules[layer_idx].out_channels, bottleneck_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(bottleneck_dim, num_classes)

    def forward(self, x, layer_idx=None):
        if layer_idx is None:
            layer_idx = self.layer_idx
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        for i, name in enumerate(self.layer_names):
            x = getattr(self.backbone, name)(x)
            if i == layer_idx:
                break
        nfp_out = self.nfp_modules[layer_idx](x)
        nfp_out = self.compress(nfp_out)
        pooled = torch.flatten(self.gap(nfp_out), 1)
        logits = self.fc(pooled)
        return logits
