import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from models.Fractal_Pooling import fractal_pooling
from models.NFP_Pooling import nfp_pooling
from models.Lacunarity_Pooling import lacunarity_pooling
from models.deepten import DeepTENEncoding  
from models.radam_pooling import RADAMPooling
from models.pooling.nfp import NFPPooling

class RESNET18_GAP_ONLY(nn.Module):
    """
    Standard ResNet18 backbone with Global Average Pooling (GAP) and a final fully connected layer.
    No NFP, no conv, no MLP, no concat; just classic CNN classification.
    """
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), num_input_channels=3):
        super().__init__()
        self.backbone = create_model('resnet18', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Dynamically infer backbone output feature size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feat = self.backbone.forward_features(dummy)
            self.backbone_out_dim = feat.shape[1]
        self.fc = nn.Linear(self.backbone_out_dim, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out



# 1. Only GAP, no NFP, no fusion
class VITTINY_GAP_ONLY(nn.Module):
    """
    ViTTiny backbone with Global Average Pooling (GAP) and a final fully connected layer.
    No NFP, no conv, no MLP, no concat; just classic classification.
    """
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), num_input_channels=3):
        super().__init__()
        self.backbone = create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, num_classes)

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



# 1. Only GAP, no NFP, no fusion
class MOBILENETV3_GAP_ONLY(nn.Module):
    """
    MobileNetV3 backbone with Global Average Pooling (GAP) and a final fully connected layer.
    No NFP, no conv, no MLP, no concat; just classic classification.
    """
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), num_input_channels=3):
        super().__init__()
        self.backbone = create_model('mobilenetv3_large_100', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.gap = nn.AdaptiveAvgPool2d(1)
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feat = self.backbone.forward_features(dummy)
            self.backbone_out_dim = feat.shape[1]
        self.fc = nn.Linear(self.backbone_out_dim, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


# Fractal Pooling
class ResNet18_FractalPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), Params=None, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('resnet18', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.pool = fractal_pooling(Params=Params)
        self.fc = nn.Linear(512, num_classes)
        self.Params = Params

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ViTTiny_FractalPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), Params=None, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.pool = fractal_pooling(Params=Params)
        self.fc = nn.Linear(192, num_classes)  # ViT-Tiny has 192 output channels
        self.Params = Params

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 3:
            patch_tokens = feats[:, 1:]
            B, N, C = patch_tokens.shape
            H = W = int(N ** 0.5)
            fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        else:
            fmap = feats
        x = self.pool(fmap)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# MobileNetV3
class MobileNetV3_FractalPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), Params=None, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('mobilenetv3_large_100', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.pool = fractal_pooling(Params=Params)
        self.fc = nn.Linear(960, num_classes)  # MobileNetV3 large has 576 output channels
        self.Params = Params

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ResNet18
class ResNet18_NFPPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), Params=None, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('resnet18', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.pool = nfp_pooling(Params=Params)
        self.fc = nn.Linear(512, num_classes)
        self.Params = Params

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ViT-Tiny
class ViTTiny_NFPPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), Params=None, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.pool = nfp_pooling(Params=Params)
        self.fc = nn.Linear(192, num_classes)  # ViT-Tiny has 192 output channels
        self.Params = Params

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 3:
            patch_tokens = feats[:, 1:]
            B, N, C = patch_tokens.shape
            H = W = int(N ** 0.5)
            fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        else:
            fmap = feats
        x = self.pool(fmap)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# MobileNetV3
class MobileNetV3_NFPPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), Params=None, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('mobilenetv3_large_100', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.pool = nfp_pooling(Params=Params)
        self.fc = nn.Linear(960, num_classes)  # MobileNetV3 large has 576 output channels
        self.Params = Params

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MobileNetV3_MultiStageNFP(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()

        # === 1) Full model to extract conv_head + act2 ===
        full = create_model('mobilenetv3_large_100', pretrained=True)
        self.conv_head = nn.Sequential(full.conv_head, full.act2)  # 960 → 1280 + Hardswish
        C_head = full.num_features  # =1280

        # === 2) Features-only model to get intermediate layers ===
        self.backbone = create_model(
            'mobilenetv3_large_100', pretrained=True, features_only=True
        )
        self.feat_info = self.backbone.feature_info
        self.nfp_indices = list(range(len(self.feat_info)))  # [0, 1, 2, 3, 4]

        # === 3) Create one NFP module per feature layer ===
        R = 1
        self.nfps = nn.ModuleList()
        for idx in self.nfp_indices:
            C = self.feat_info[idx]['num_chs']
            self.nfps.append(NFPPooling(in_channels=C, R=R, measure='cosine', padding=R))

        self.num_neighbors = (2*R + 1)**2 - 1  # 8 neighbors
        total_features = len(self.nfp_indices) * self.num_neighbors  # 5 × 8 = 40

        # === 4) Project concatenated NFP vector (B, 40) → (B, 1280) ===
        self.nfp_proj = nn.Linear(total_features, C_head)

        # === 5) Final GAP + classification ===
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(C_head, num_classes)

    def forward(self, x):
        # === Get all 5 intermediate feature maps ===
        feats = self.backbone(x)  # List of 5 tensors

        nfp_vectors = []
        for i, idx in enumerate(self.nfp_indices):
            feat = feats[idx]  # (B, C_i, H_i, W_i)
            sim_map = self.nfps[i](feat)  # → (B, 8, H, W)
            pooled = F.adaptive_avg_pool2d(sim_map, 1).view(x.size(0), -1)  # (B, 8)
            nfp_vectors.append(pooled)

        # === Concatenate all NFP pooled vectors: (B, 40) ===
        v_concat = torch.cat(nfp_vectors, dim=1)  # (B, 40)

        # === Project to 1280-dim vector ===
        x_mid_nfp = self.nfp_proj(v_concat)  # (B, 1280)

        # === Extract conv_head features and pool ===
        feat_last = feats[-1]  # (B, 960, 7, 7)
        head_feats = self.conv_head(feat_last)  # → (B, 1280, 7, 7)
        x_avg_last = self.avgpool(head_feats).view(x.size(0), -1)  # (B, 1280)

        # === Fuse semantic + texture (element-wise) and classify ===
        fused = x_avg_last * x_mid_nfp  # (B, 1280)
        return self.fc(fused)




class MobileNetV3_MidNFP(nn.Module):
    def __init__(self, num_classes=21, nfp_mid_layer_idx=1):
        super().__init__()
        # === 1) grab a “full” model so we can steal its conv_head + act2 ===
        full = create_model(
            'mobilenetv3_large_100',
            pretrained=True,     # has: conv_stem, blocks, conv_head, act2, flatten, classifier
        )
        # conv_head: 960→1280, then Hardswish
        self.conv_head = nn.Sequential(full.conv_head, full.act2)
        C_head = full.num_features   # 1280
        
        # === 2) build a features_only backbone to get every stage’s output ===
        self.backbone = create_model(
            'mobilenetv3_large_100',
            pretrained=True,
            features_only=True,     # now forward(x) → list of feature maps
        )
        feat_info = self.backbone.feature_info
        last_idx  = len(feat_info) - 1
        if not (0 <= nfp_mid_layer_idx <= last_idx):
            raise ValueError(f"nfp_mid_layer_idx must be in [0..{last_idx}], got {nfp_mid_layer_idx}")
        self.mid_idx, self.last_idx = nfp_mid_layer_idx, last_idx

        # how many channels in the mid feature?
        C_mid = feat_info[self.mid_idx]['num_chs']
        
        # === 3) NFP on the mid feature ===
        R, Nr = 1, (2*1+1)**2 - 1   # 8 neighbor‐maps
        self.nfp_mid      = NFPPooling(in_channels=C_mid, R=R, measure='cosine', padding=R)
        # project from Nr → C_head so we can fuse with the head’s output
        self.nfp_mid_proj = nn.Linear(Nr, C_head)

        # === 4) final pooling & classifier ===
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(C_head, num_classes)

    def forward(self, x):
        # get _all_ the feature maps
        feats     = self.backbone(x)
        feat_mid  = feats[self.mid_idx]   # (B, C_mid, H_mid, W_mid)
        feat_last = feats[self.last_idx]  # (B,  C_last=960, H_last, W_last)

        # A) run the “head” conv (960→1280 + Hardswish)
        head_feats = self.conv_head(feat_last)  # (B, 1280, H_last, W_last)

        # B) NFP on the mid‐level → pool → project → (B, 1280)
        sim_maps    = self.nfp_mid(feat_mid)                  # (B, 8, H_mid, W_mid)
        v           = F.adaptive_avg_pool2d(sim_maps, 1) \
                          .view(x.size(0), -1)             # (B, 8)
        x_mid_nfp   = self.nfp_mid_proj(v)                    # (B, 1280)

        # C) GAP on the head features → (B, 1280)
        x_avg_last  = self.avgpool(head_feats).view(x.size(0), -1)

        # D) fuse & classify
        fused = x_avg_last * x_mid_nfp                        # (B, 1280)
        return self.fc(fused)

class MobileNetV3_NFPPooling_Intermediate(nn.Module):
    def __init__(
        self,
        num_classes: int,
        Params: dict,
        layer_idx: int = None,   # which block to tap (0–6); None = final
    ):
        super().__init__()
        self.Params    = Params.copy()
        self.layer_idx = layer_idx

        # ── Backbone ────────────────────────────────────────────────────────────
        backbone = create_model(
            'mobilenetv3_large_100',
            pretrained=True,
            num_classes=0   # drop final classifier
        )
        # stem = ConvBnAct (conv_stem + bn1)
        self.stem      = nn.Sequential(
            backbone.conv_stem,
            backbone.bn1
        )
        # inverted‐residual blocks
        self.blocks    = backbone.blocks
        # head = ConvBnAct (conv_head)
        self.conv_head = backbone.conv_head
        # final global pooling
        self.global_pool = backbone.global_pool

        # ── Determine channel count at tap point ────────────────────────────────
        if layer_idx is None:
            in_ch = backbone.num_features
        else:
            blk   = self.blocks[layer_idx]
            last = blk[-1]
            if hasattr(last, "conv_pwl"):
                in_ch = last.conv_pwl.out_channels
            elif hasattr(last, "conv"):
                in_ch = last.conv.out_channels
            else:
                raise RuntimeError(f"Unknown block type: {type(last)}")

        # ── Override Params so wrapper builds NFPPooling for `in_ch` ────────────
        # Assumes Params["Model_name"] is 'mobilenetv3_large_100'
        self.Params["num_ftrs"][ self.Params["Model_name"] ] = in_ch

        # ── Pooling wrapper (will now use NFPPooling(in_ch, ...)) ─────────────
        self.pool = nfp_pooling(Params=self.Params)

        # ── Classifier ──────────────────────────────────────────────────────────
        self.fc = nn.Linear(in_ch, num_classes)


    def forward(self, x):
        # 1) Stem
        x = self.stem(x)

        # 2) Tap intermediate or run full head
        if self.layer_idx is not None:
            # Run until the tapped block
            for idx, blk in enumerate(self.blocks):
                x = blk(x)
                if idx == self.layer_idx:
                    feat = x
                    break
        else:
            # Run through all blocks + head + pool
            x = self.blocks(x)
            x = self.conv_head(x)
            feat = self.global_pool(x).flatten(1)

        # 3) NFP + classification
        pooled = self.pool(feat)       # [B, in_ch]
        # (nfp_pooling returns [B, C], so no extra flatten needed)
        return self.fc(pooled)


# ResNet18
class ResNet18_LacunarityPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), Params=None, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('resnet18', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.pool = lacunarity_pooling(Params=Params)
        self.fc = nn.Linear(512, num_classes)
        self.Params = Params

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ViT-Tiny
class ViTTiny_LacunarityPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), Params=None, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.pool = lacunarity_pooling(Params=Params)
        self.fc = nn.Linear(192, num_classes)  # ViT-Tiny has 192 output channels
        self.Params = Params

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 3:
            patch_tokens = feats[:, 1:]
            B, N, C = patch_tokens.shape
            H = W = int(N ** 0.5)
            fmap = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        else:
            fmap = feats
        x = self.pool(fmap)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# MobileNetV3
class MobileNetV3_LacunarityPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), Params=None, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('mobilenetv3_large_100', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.pool = lacunarity_pooling(Params=Params)
        self.fc = nn.Linear(960, num_classes)  # MobileNetV3 large has 576 output channels
        self.Params = Params

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ResNet18
class ResNet18_DeepTENPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), num_codes=32, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('resnet18', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True

        self.encoding = DeepTENEncoding(in_channels=512, num_codes=num_codes)
        self.bn = nn.BatchNorm1d(num_codes * 512)
        self.fc = nn.Linear(num_codes * 512, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.encoding(x)
        x = self.bn(x)
        return self.fc(x)


# MobileNetV3
class MobileNetV3_DeepTENPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), num_codes=32, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('mobilenetv3_large_100', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True

        self.encoding = DeepTENEncoding(in_channels=960, num_codes=num_codes)
        self.bn = nn.BatchNorm1d(num_codes * 960)
        self.fc = nn.Linear(num_codes * 960, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.encoding(x)
        x = self.bn(x)
        return self.fc(x)


# ViT-Tiny
class ViTTiny_DeepTENPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), num_codes=32, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True

        self.encoding = DeepTENEncoding(in_channels=192, num_codes=num_codes)
        self.bn = nn.BatchNorm1d(num_codes * 192)
        self.fc = nn.Linear(num_codes * 192, num_classes)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 3:
            patch_tokens = feats[:, 1:]
            B, N, C = patch_tokens.shape
            H = W = int(N ** 0.5)
            feats = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        x = self.encoding(feats)
        x = self.bn(x)
        return self.fc(x)



class ResNet50_FractalPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), Params=None, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('resnet50', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.pool = fractal_pooling(Params=Params)
        self.fc = nn.Linear(2048, num_classes)
        self.Params = Params

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResNet50_NFPPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), Params=None, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('resnet50', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.pool = nfp_pooling(Params=Params)
        self.fc = nn.Linear(2048, num_classes)
        self.Params = Params

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResNet50_LacunarityPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), Params=None, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('resnet50', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.pool = lacunarity_pooling(Params=Params)
        self.fc = nn.Linear(2048, num_classes)
        self.Params = Params

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResNet50_DeepTENPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), num_codes=32, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('resnet50', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.encoding = DeepTENEncoding(in_channels=2048, num_codes=num_codes)
        self.bn = nn.BatchNorm1d(num_codes * 2048)
        self.fc = nn.Linear(num_codes * 2048, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.encoding(x)
        x = self.bn(x)
        return self.fc(x)


class ResNet18_RADAMPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), M=4, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('resnet18', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True

        self.pool = RADAMPooling(spatial_size=7, in_channels=512, M=M, device='cuda')
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)  # shape: (B, 512, 7, 7)
        x = self.pool(x).squeeze(1)  # shape: (B, 512)
        return self.fc(x)


class MobileNetV3_RADAMPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), M=4, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('mobilenetv3_large_100', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True

        self.pool = RADAMPooling(spatial_size=7, in_channels=960, M=M, device='cuda')
        self.fc = nn.Linear(960, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)  # shape: (B, 960, 7, 7) or similar
        x = self.pool(x).squeeze(1)  # shape: (B, 960)
        return self.fc(x)


class ViTTiny_RADAMPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), M=4, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True

        self.pool = RADAMPooling(spatial_size=14, in_channels=192, M=M, device='cuda')
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)  # (B, N+1, C), N=196
        if x.dim() == 3:
            x = x[:, 1:, :]  # remove CLS token → (B, 196, 192)
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)  # → (B, 192, 14, 14)
        x = self.pool(x).squeeze(1)  # shape: (B, 192)
        return self.fc(x)

class ResNet50_RADAMPooling(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224), M=4, num_input_channels=3):
        super().__init__()
        self.backbone = create_model('resnet50', pretrained=True, num_classes=0, global_pool='', in_chans=num_input_channels)
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.pool = RADAMPooling(spatial_size=7, in_channels=2048, M=M, device='cuda')
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)  # shape: (B, 2048, 7, 7)
        x = self.pool(x).squeeze(1)  # shape: (B, 2048)
        return self.fc(x)




class ResNet50_GAPOnly(nn.Module):
    def __init__(self, num_classes=21, input_shape=(3, 224, 224)):
        super().__init__()
        self.backbone = create_model('resnet50', pretrained=True, num_classes=0, global_pool='')
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)



