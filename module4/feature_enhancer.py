import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class ContentGuidedAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        sa = self.spatial_attention(x)
        x = x * sa
        
        return x

class CrossScaleFeatureFusion(nn.Module):
    def __init__(self, dims=(192, 384, 768)):
        super().__init__()
        self.s3_dim, self.s4_dim, self.s5_dim = dims
        
        # Dimension reduction for S3 and S4
        self.s3_conv = nn.Conv2d(self.s3_dim, self.s3_dim, 1)
        self.s4_conv = nn.Conv2d(self.s4_dim, self.s4_dim, 1)
        
        # Content-guided attention
        self.cga_s3 = ContentGuidedAttention(self.s3_dim)
        self.cga_s4 = ContentGuidedAttention(self.s4_dim)
        
        # Final fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.s3_dim + self.s4_dim + self.s5_dim, self.s5_dim, 1),
            nn.BatchNorm2d(self.s5_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, s3, s4, f5):
        # Ensure spatial dimensions match
        s3_size = s3.shape[-2:]
        s4 = F.interpolate(s4, size=s3_size, mode='bilinear', align_corners=False)
        f5 = F.interpolate(f5, size=s3_size, mode='bilinear', align_corners=False)
        
        # Apply dimension reduction and CGA
        s3 = self.cga_s3(self.s3_conv(s3))
        s4 = self.cga_s4(self.s4_conv(s4))
        
        # Concatenate and fuse
        concat_features = torch.cat([s3, s4, f5], dim=1)
        out = self.fusion_conv(concat_features)
        
        return out

class FeatureEnhancer(nn.Module):
    def __init__(self, dims=(256, 512, 1024)):
        super().__init__()
        self.s3_dim, self.s4_dim, self.s5_dim = dims
        
        # Multi-head self-attention for S5
        self.mha = MultiHeadAttention(dim=self.s5_dim)
        
        # Cross-scale feature fusion
        self.cff = CrossScaleFeatureFusion(dims)
        
        # Layer norm for MHA
        self.norm = nn.LayerNorm(self.s5_dim)
        
    def forward(self, s3, s4, s5):
        B, C, H, W = s5.shape
        
        # Apply MHA to S5
        s5_flat = s5.flatten(2).transpose(1, 2)  # B, HW, C
        s5_flat = self.norm(s5_flat)
        f5 = self.mha(s5_flat)
        f5 = f5.transpose(1, 2).reshape(B, C, H, W)
        
        # Apply CFF
        fi = self.cff(s3, s4, f5)
        
        return fi

if __name__ == "__main__":
    # Test the implementation
    batch_size = 2
    
    # Create dummy inputs with paper's dimensions
    s3 = torch.randn(batch_size, 192, 64, 64)
    s4 = torch.randn(batch_size, 384, 32, 32)
    s5 = torch.randn(batch_size, 768, 16, 16)
    
    # Initialize FeatureEnhancer
    enhancer = FeatureEnhancer()
    
    # Forward pass
    output = enhancer(s3, s4, s5)
    print(f"Input shapes:")
    print(f"S3: {s3.shape}")
    print(f"S4: {s4.shape}")
    print(f"S5: {s5.shape}")
    print(f"Output shape: {output.shape}")