import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentGuidedAttention(nn.Module):
    """
    Content-Guided Attention (CGA) module for effective feature fusion 
    as described in the paper. Generates channel-specific spatial importance maps.
    """
    def __init__(self, in_channels, reduction=16):
        super(ContentGuidedAttention, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention
        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights
        
        # Spatial attention
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weights = self.spatial_attention(spatial_input)
        
        # Apply spatial attention
        out = x_channel * spatial_weights
        return out


class CrossScaleFusion(nn.Module):
    """
    Cross-Scale Fusion (CSF) module that fuses features from adjacent scales
    using content-guided attention.
    """
    def __init__(self, high_channels, low_channels, out_channels):
        super(CrossScaleFusion, self).__init__()
        self.conv_high = nn.Conv2d(high_channels, out_channels, kernel_size=1)
        self.conv_low = nn.Conv2d(low_channels, out_channels, kernel_size=1)
        self.cga = ContentGuidedAttention(out_channels)
        self.fusion = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, high_feat, low_feat):
        # Resize high-level features to match low-level features size
        high_feat = self.conv_high(high_feat)
        high_feat_upsampled = F.interpolate(high_feat, size=low_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # Process low-level features
        low_feat = self.conv_low(low_feat)
        
        # Content-guided attention
        fused_feat = high_feat_upsampled + low_feat
        fused_feat = self.cga(fused_feat)
        
        # Final fusion
        out = self.fusion(fused_feat)
        return out


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module applied to high-level features only.
    """
    def __init__(self, in_channels, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        self.qkv_proj = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Generate query, key, value
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, 3, self.num_heads, self.head_dim, height * width)
        q, k, v = qkv.unbind(dim=1)  # Split along the second dimension
        
        # Reshape for efficient matrix multiplication
        q = q.transpose(-2, -1)  # [B, num_heads, height*width, head_dim]
        k = k.transpose(-2, -1)  # [B, num_heads, height*width, head_dim]
        v = v.transpose(-2, -1)  # [B, num_heads, height*width, head_dim]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, num_heads, height*width, height*width]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = attn @ v  # [B, num_heads, height*width, head_dim]
        out = out.transpose(-2, -1).reshape(batch_size, channels, height, width)
        out = self.out_proj(out)
        
        return out


class HybridEncoder(nn.Module):
    """
    Hybrid Encoder that applies self-attention to high-level features only
    and uses a two-path approach to fuse features across different levels.
    """
    def __init__(self, feature_dims=[192, 384, 768], out_dim=256):
        super(HybridEncoder, self).__init__()
        self.feature_dims = feature_dims
        self.out_dim = out_dim
        
        # Process S5 with MHSA (Multi-Head Self-Attention)
        self.high_level_attention = MultiHeadSelfAttention(feature_dims[2])
        
        # Top-down path (high to low resolution)
        self.td_s5_s4 = CrossScaleFusion(feature_dims[2], feature_dims[1], out_dim)
        self.td_s4_s3 = CrossScaleFusion(out_dim, feature_dims[0], out_dim)
        
        # Bottom-up path (low to high resolution)
        self.bu_s3_s4 = CrossScaleFusion(out_dim, out_dim, out_dim)
        self.bu_s4_s5 = CrossScaleFusion(out_dim, out_dim, out_dim)
        
        # Final projection
        self.final_proj = nn.Conv2d(out_dim * 3, out_dim, kernel_size=1)
        
    def forward(self, s3, s4, s5):
        # Apply self-attention to high-level features (S5)
        f5 = self.high_level_attention(s5)
        
        # Top-down path
        f4_td = self.td_s5_s4(f5, s4)
        f3_td = self.td_s4_s3(f4_td, s3)
        
        # Bottom-up path
        f4_bu = self.bu_s3_s4(f3_td, f4_td)
        f5_bu = self.bu_s4_s5(f4_bu, f5)
        
        # Resize all features to the same resolution (S3's size)
        # and concatenate for final projection
        f3_final = f3_td
        f4_final = F.interpolate(f4_bu, size=f3_final.shape[2:], mode='bilinear', align_corners=False)
        f5_final = F.interpolate(f5_bu, size=f3_final.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate and project
        concat_feat = torch.cat([f3_final, f4_final, f5_final], dim=1)
        out = self.final_proj(concat_feat)
        
        return out


# Usage example:
if __name__ == "__main__":
    # Example feature map sizes from a Swin Transformer backbone
    batch_size = 4
    s3 = torch.rand(batch_size, 192, 64, 64)   # 1/8 resolution
    s4 = torch.rand(batch_size, 384, 32, 32)   # 1/16 resolution
    s5 = torch.rand(batch_size, 768, 16, 16)   # 1/32 resolution
    
    hybrid_encoder = HybridEncoder()
    output = hybrid_encoder(s3, s4, s5)
    
    print(f"Input shapes: S3: {s3.shape}, S4: {s4.shape}, S5: {s5.shape}")
    print(f"Output shape: {output.shape}")