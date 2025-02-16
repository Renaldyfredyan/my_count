import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class CGAModule(nn.Module):
    """Content-Guided Attention Module"""
    def __init__(self, in_channels):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x

class FusionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.cga = CGAModule(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, low_feat, high_feat):
        # Ensure spatial dimensions match
        if low_feat.shape[2:] != high_feat.shape[2:]:
            high_feat = F.interpolate(high_feat, size=low_feat.shape[2:], 
                                    mode='bilinear', align_corners=False)
        
        # Concatenate features
        concat_feat = torch.cat([low_feat, high_feat], dim=1)
        fused_feat = self.conv1x1(concat_feat)
        
        # Apply CGA
        enhanced_feat = self.cga(fused_feat)
        
        # Final processing
        out = self.relu(self.norm(enhanced_feat))
        return out

class HybridEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initial projections
        self.proj_layers = nn.ModuleDict({
            'stage3': nn.Sequential(
                nn.Conv2d(512, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            'stage4': nn.Sequential(
                nn.Conv2d(1024, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        })
        
        # Fusion modules
        self.fusion_s4_s3 = FusionModule(256)  # Fuse S4 features with S3
        
        # Final CGA module
        self.final_cga = CGAModule(256)
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        # Project features to common dimension
        S3 = self.proj_layers['stage3'](features['stage3'])  # 32x32x256
        S4 = self.proj_layers['stage4'](features['stage4'])  # 16x16x256
        
        # Upsample S3 to target size
        S3 = F.interpolate(S3, size=(64, 64), mode='bilinear', align_corners=False)
        
        # Progressive fusion (bottom-up pathway)
        # Fuse S4 with S3
        fused_features = self.fusion_s4_s3(S3, S4)
        
        # Final enhancement
        enhanced_features = self.final_cga(fused_features)
        Fi = self.final_proj(enhanced_features)
        
        return Fi  # Output: 64x64x256

if __name__ == "__main__":
    from backbone import FeatureExtractor
    
    # Test pipeline
    extractor = FeatureExtractor()
    encoder = HybridEncoder()
    
    dummy_input = torch.randn(2, 3, 512, 512)
    features = extractor(dummy_input)
    Fi = encoder(features)
    
    print("\nFinal output shape (Fi):", Fi.shape)  # Should be [2, 256, 64, 64]