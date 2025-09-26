import torch
import torch.nn as nn
import timm
from timm import create_model
from models.pooling.nfp import NFPPooling

# 1. Only GAP, no NFP, no fusion
class MOBILENETV3_GAP_ONLY(nn.Module):
    """
    MobileNetV3 backbone with Global Average Pooling (GAP) and a final fully connected layer.
    No NFP, no conv, no MLP, no concat; just classic classification.
    """
    def __init__(self, num_classes=21, input_shape=(3, 224, 224)):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0, global_pool='')
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
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
class MOBILENETV3_GAP_MLP(nn.Module):
    """
    MobileNetV3 backbone with GAP, modulated by an MLP gate (self-attention on GAP feature).
    No NFP, no conv, but the GAP feature is adaptively gated by an MLP. No concat.
    """
    def __init__(self, num_classes=21, dropout_p=0.2, input_shape=(3, 224, 224)):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0, global_pool='')
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p)
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
class MOBILENETV3_NFP_CONV_ONLY(nn.Module):
    """
    MobileNetV3 backbone with NFPPooling and conv compression.
    Only NFP features are used (with conv), no GAP, no MLP, no concat.
    """
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), nfp_kwargs=None, bottleneck_dim=None, in_channels=None):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0, global_pool='')
        for p in self.backbone.parameters():
            p.requires_grad = True
        nfp_kwargs = nfp_kwargs or dict(R=1, measure='cosine', input_size=input_shape[-1])
        self.nfp = NFPPooling(in_channels, **nfp_kwargs)
        if self.nfp.in_channels != 1:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                feat = self.backbone.forward_features(dummy)
                self.nfp.in_channels = feat.shape[1]
                nfp_out = self.nfp(feat)
                nfp_channels = nfp_out.shape[1]
            if bottleneck_dim is None:
                bottleneck_dim = 576
            self.compress = nn.Sequential(
                nn.Conv2d(nfp_channels, bottleneck_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(bottleneck_dim),
                nn.ReLU(inplace=True),
            )
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(bottleneck_dim, num_classes)
    def forward(self, x):
        feat = self.backbone.forward_features(x)
        nfp_out = self.nfp(feat)
        nfp_out = self.compress(nfp_out)
        pooled = torch.flatten(self.gap(nfp_out), 1)
        logits = self.fc(pooled)
        return logits

# 4. Only NFP with conv and MLP
class MOBILENETV3_NFP_CONV_MLP(nn.Module):
    """
    MobileNetV3 backbone with NFPPooling, conv compression, and MLP gating.
    Only NFP features are used (with conv), compressed and modulated by an MLP. No GAP, no concat.
    """
    def __init__(self, num_classes=21, nfp_kwargs=None, bottleneck_dim=None, dropout_p=0.2, input_shape=(3, 224, 224), in_channels=None):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0, global_pool='')
        for p in self.backbone.parameters():
            p.requires_grad = True
        nfp_kwargs = nfp_kwargs or dict(R=1, measure='cosine', input_size=input_shape[-1])
        self.nfp = NFPPooling(in_channels, **nfp_kwargs)
        if self.nfp.in_channels != 1:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                feat = self.backbone.forward_features(dummy)
                self.nfp.in_channels = feat.shape[1]
                nfp_out = self.nfp(feat)
                nfp_channels = nfp_out.shape[1]
            if bottleneck_dim is None:
                bottleneck_dim = 576
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
        feat = self.backbone.forward_features(x)
        nfp_map = self.nfp(feat)
        nfp_map = self.compress(nfp_map)
        mlp_weight = self.mlp(nfp_map)
        nfp_vec = torch.flatten(nn.functional.adaptive_avg_pool2d(nfp_map, 1), 1)
        gated = nfp_vec * mlp_weight
        out = self.fc(self.dropout(gated))
        return out

# 5. NFP (with conv) concat with GAP, no MLP
class MOBILENETV3_GAP_NFP_CONV_NOMLP_CONCAT(nn.Module):
    """
    MobileNetV3 backbone with GAP and NFP (NFPPooling + Conv) features concatenated.
    NFP features are projected to match GAP dimension before concatenation. No MLP.
    """
    def __init__(self, num_classes=21, bottleneck_dim=None, R=1, measure="cosine", dropout_p=0.2, input_shape=(3, 224, 224), in_channels=None):
        super().__init__()
        self.backbone = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=0, global_pool="")
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.nfp = NFPPooling(in_channels, R=R, measure=measure, padding=R)
        if self.nfp.in_channels != 1:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                fmap = self.backbone.forward_features(dummy)
                self.backbone_out_dim = fmap.shape[1]
                self.nfp.in_channels = self.backbone_out_dim
                nfp_out = self.nfp(fmap)
                self.nfp_out_channels = nfp_out.shape[1]
            if bottleneck_dim is None:
                bottleneck_dim = 576
            self.nfp_conv = nn.Sequential(
                nn.Conv2d(self.nfp_out_channels, bottleneck_dim, 1, bias=False),
                nn.BatchNorm2d(bottleneck_dim),
                nn.ReLU(inplace=True),
            )
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(dropout_p)
            self.fc = nn.Linear(self.backbone_out_dim + bottleneck_dim, num_classes)
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
class MOBILENETV3_GAP_NFP_NOCONV_NOMLP_CONCAT(nn.Module):
    """
    MobileNetV3 backbone with GAP and NFPPooling features concatenated (no conv projection).
    NFP features are concatenated directly with GAP features. No MLP.
    """
    def __init__(self, num_classes=21, R=1, measure="cosine", dropout_p=0.2, input_shape=(3, 224, 224), in_channels=None):
        super().__init__()
        self.backbone = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=0, global_pool="")
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.nfp = NFPPooling(in_channels, R=R, measure=measure, padding=R)
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
            self.fc = nn.Linear(self.backbone_out_dim + self.nfp_out_channels, num_classes)
    def forward(self, x):
        fmap = self.backbone.forward_features(x)
        gap_feat = torch.flatten(self.gap(fmap), 1)
        nfp_map = self.nfp(fmap)
        nfp_feat = torch.flatten(self.gap(nfp_map), 1)
        fused = torch.cat([gap_feat, nfp_feat], dim=1)
        logits = self.fc(self.dropout(fused))
        return logits

# 7. NFP (with conv) concat with GAP, with MLP
class MOBILENETV3_GAP_NFP_CONV_MLP_CONCAT(nn.Module):
    """
    MobileNetV3 backbone with GAP and NFP (NFPPooling + Conv) features concatenated, then fused by MLP.
    NFP features are projected to match GAP dimension before concatenation. MLP fusion.
    """
    def __init__(self, num_classes=21, bottleneck_dim=None, R=1, measure="cosine", dropout_p=0.2, input_shape=(3, 224, 224), in_channels=None):
        super().__init__()
        self.backbone = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=0, global_pool="")
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.nfp = NFPPooling(in_channels, R=R, measure=measure, padding=R)
        if self.nfp.in_channels != 1:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                fmap = self.backbone.forward_features(dummy)
                self.backbone_out_dim = fmap.shape[1]
                self.nfp.in_channels = self.backbone_out_dim
                nfp_out = self.nfp(fmap)
                self.nfp_out_channels = nfp_out.shape[1]
            if bottleneck_dim is None:
                bottleneck_dim = 576
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
class MOBILENETV3_GAP_NFP_NOCONV_MLP_CONCAT(nn.Module):
    """
    MobileNetV3 backbone with GAP and NFPPooling features concatenated (no conv), then fused by MLP.
    NFP features are concatenated directly with GAP features. MLP fusion.
    """
    def __init__(self, num_classes=21, R=1, measure="cosine", dropout_p=0.2, input_shape=(3, 224, 224), in_channels=None):
        super().__init__()
        self.backbone = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=0, global_pool="")
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.nfp = NFPPooling(in_channels, R=R, measure=measure, padding=R)
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
            self.mlp = nn.Sequential(
                nn.Linear(self.backbone_out_dim + self.nfp_out_channels, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.backbone_out_dim + self.nfp_out_channels),
                nn.Sigmoid()
            )
            self.fc = nn.Linear(self.backbone_out_dim + self.nfp_out_channels, num_classes)
    def forward(self, x):
        fmap = self.backbone.forward_features(x)
        gap_feat = torch.flatten(self.gap(fmap), 1)
        nfp_map = self.nfp(fmap)
        nfp_feat = torch.flatten(self.gap(nfp_map), 1)
        fused = torch.cat([gap_feat, nfp_feat], dim=1)
        fused = fused * self.mlp(fused)
        logits = self.fc(self.dropout(fused))
        return logits

class MOBILENETV3_NFP_INSERT(nn.Module):
    """
    MobileNetV3 with NFPPooling inserted after a selected layer index.
    All layers are preserved. NFP is applied to the intermediate feature,
    and its output replaces the input to the next layer onward.
    """
    def __init__(self, num_classes=21, nfp_insert_idx=1, nfp_kwargs=None, input_shape=(3, 224, 224)):
        super().__init__()
        assert 0 <= nfp_insert_idx <= 6, "nfp_insert_idx must be between 0 and 6 (inclusive)"

        self.nfp_insert_idx = nfp_insert_idx
        self.backbone = create_model('mobilenetv3_large_100', pretrained=True, num_classes=0, global_pool='')

        # Stem = conv_stem + bn1
        self.stem = nn.Sequential(
            self.backbone.conv_stem,
            self.backbone.bn1
        )

        # Extract layers 0 through 6 from backbone.blocks
        self.layers = nn.ModuleList()
        for i in range(7):  # MobileNetV3 has 7 "layers" in blocks
            self.layers.append(self.backbone.blocks[i])

        # Assign final layers from backbone BEFORE dummy forward
        self.conv_head = self.backbone.conv_head
        self.act2 = self.backbone.act2
        self.global_pool = self.backbone.global_pool  # usually AdaptiveAvgPool2d
        self.flatten = self.backbone.flatten

        # Dummy forward to find feature dimensions at each layer and dynamically set NFP in_channels
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            x = self.stem(dummy)
            for i in range(len(self.layers)):
                x = self.layers[i](x)
                if i == nfp_insert_idx:
                    feat_channels = x.shape[1]
                    # Dynamically initialize NFP with correct in_channels
                    self.nfp = NFPPooling(in_channels=feat_channels, **(nfp_kwargs or {}))
                    self.nfp_proj = nn.Sequential(
                        nn.Conv2d(self.nfp.out_channels, feat_channels, kernel_size=1, bias=False),
                        nn.BatchNorm2d(feat_channels),
                        nn.ReLU(inplace=True),
                    )
                    # Pass through NFP and projection
                    x = self.nfp(x)
                    x = self.nfp_proj(x)
            # Continue with the rest of the layers
            for j in range(i+1, len(self.layers)):
                x = self.layers[j](x)
            x = self.conv_head(x)
            x = self.act2(x)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            print("Shape before FC (dummy forward):", x.shape)
            feat_dim = x.shape[1]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == self.nfp_insert_idx:
                x = self.nfp(x)
                x = self.nfp_proj(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        print("Shape before FC (real forward):", x.shape)
        logits = self.fc(x)
        return logits