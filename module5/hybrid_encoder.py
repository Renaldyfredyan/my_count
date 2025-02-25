import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncodingsFixed(nn.Module):
    def __init__(self, emb_dim, temperature=10000):
        super(PositionalEncodingsFixed, self).__init__()
        self.emb_dim = emb_dim
        self.temperature = temperature

    def _1d_pos_enc(self, mask, dim):
        temp = torch.arange(self.emb_dim // 2).float().to(mask.device)
        temp = self.temperature ** (2 * (temp.div(2, rounding_mode='floor')) / self.emb_dim)
        enc = (~mask).cumsum(dim).float().unsqueeze(-1) / temp
        enc = torch.stack([
            enc[..., 0::2].sin(), enc[..., 1::2].cos()
        ], dim=-1).flatten(-2)
        return enc

    def forward(self, bs, h, w, device):
        mask = torch.zeros(bs, h, w, dtype=torch.bool, requires_grad=False, device=device)
        x = self._1d_pos_enc(mask, dim=2)
        y = self._1d_pos_enc(mask, dim=1)
        return torch.cat([y, x], dim=3).permute(0, 3, 1, 2)

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

    def forward(self, x1, x2):
        if x1.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        concat_feat = torch.cat([x1, x2], dim=1)
        fused_feat = self.conv1x1(concat_feat)
        enhanced_feat = self.cga(fused_feat)
        out = self.relu(self.norm(enhanced_feat))
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        attn_out, _ = self.mhsa(x, x, x)
        attn_out = self.norm(attn_out)
        out = attn_out.permute(1, 2, 0).view(B, C, H, W)
        return out

class HybridEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Positional Encoding
        self.pos_enc = PositionalEncodingsFixed(256)
        
        # Initial projections to 256 channels (updated for paper dimensions)
        self.proj_layers = nn.ModuleDict({
            'stage3': nn.Sequential(
                nn.Conv2d(192, 256, 1),  # From 192 channels (paper spec)
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            'stage4': nn.Sequential(
                nn.Conv2d(384, 256, 1),  # From 384 channels (paper spec)
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            'stage5': nn.Sequential(
                nn.Conv2d(768, 256, 1),  # From 768 channels (paper spec)
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        })
        
        # Multi-Head Self-Attention for F5
        self.mhsa = MultiHeadSelfAttention(256)
        
        # Post MHSA conv
        self.post_mhsa_conv = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Fusion modules
        self.fusion_s5_s4 = FusionModule(256)  # Fuse F5 with S4
        self.fusion_s4_s3 = FusionModule(256)  # Fuse F4 with S3
        
        # CGA modules for each level
        self.cga_s3 = CGAModule(256)
        self.cga_s4 = CGAModule(256)
        self.cga_s5 = CGAModule(256)
        
        # Final fusion and projection
        self.final_projection = nn.Sequential(
            nn.Conv2d(256 * 3, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        """
        Process features through hybrid encoder
        Args:
            features: Dictionary containing:
                - 'stage3': [B, 192, 64, 64]
                - 'stage4': [B, 384, 32, 32]
                - 'stage5': [B, 768, 16, 16]
        Returns:
            Tensor of shape [B, 256, 64, 64]
        """
        # Project features to 256 channels
        S3 = self.proj_layers['stage3'](features['stage3'])  # 64x64x256
        S4 = self.proj_layers['stage4'](features['stage4'])  # 32x32x256
        S5 = self.proj_layers['stage5'](features['stage5'])  # 16x16x256
        
        # Add positional encodings
        B, _, H, W = S3.shape
        pos_enc = self.pos_enc(B, H, W, S3.device)
        S3 = S3 + F.interpolate(pos_enc, size=S3.shape[2:])
        S4 = S4 + F.interpolate(pos_enc, size=S4.shape[2:])
        S5 = S5 + F.interpolate(pos_enc, size=S5.shape[2:])
        
        # Apply CGA to each level
        S3 = self.cga_s3(S3)
        S4 = self.cga_s4(S4)
        S5 = self.cga_s5(S5)
        
        # Multi-Head Self-Attention on S5
        F5 = self.mhsa(S5)
        F5 = self.post_mhsa_conv(F5)
        
        # Progressive fusion (bottom-up)
        F4 = self.fusion_s5_s4(S4, F5)  # Fuse F5 into S4
        F3 = self.fusion_s4_s3(S3, F4)  # Fuse F4 into S3
        
        # Upsample all features to S3 size
        F3_final = F3  # Already at target size
        F4_final = F.interpolate(F4, size=F3.shape[2:], mode='bilinear', align_corners=False)
        F5_final = F.interpolate(F5, size=F3.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate and project
        multi_scale_features = torch.cat([F3_final, F4_final, F5_final], dim=1)
        Fi = self.final_projection(multi_scale_features)
        
        return Fi

if __name__ == "__main__":
    # Test code
    encoder = HybridEncoder()
    # Create dummy features with paper-specified dimensions
    dummy_features = {
        'stage3': torch.randn(2, 192, 64, 64),  # From backbone stage 3
        'stage4': torch.randn(2, 384, 32, 32),  # From backbone stage 4
        'stage5': torch.randn(2, 768, 16, 16)   # From backbone stage 5
    }
    
    output = encoder(dummy_features)
    print("Output shape:", output.shape)  # Should be [2, 256, 64, 64]