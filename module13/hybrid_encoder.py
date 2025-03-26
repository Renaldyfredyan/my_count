import torch
from torch import nn
import torch.nn.functional as F
from mlp import MLP
from positional_encoding import PositionalEncodingsFixed

class ContentGuidedAttention(nn.Module):
    """
    Content-Guided Attention (CGA) module that integrates both channel attention 
    and spatial attention mechanisms.
    """
    def __init__(self, channels, reduction=16, dropout=0.1):
        super().__init__()
        self.channels = channels
        
        # Channel Attention with Global Average Pooling
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling (GAP)
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial Attention with average and max pooling
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, low_features, high_features):
        # Ensure high_features match low_features dimension
        if high_features.shape[2:] != low_features.shape[2:]:
            high_features = F.interpolate(
                high_features, 
                size=low_features.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        # Combine features
        x = low_features + high_features
        
        # Channel Attention
        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights
        
        # Spatial Attention
        # Create spatial attention inputs using both average and max pooling
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        
        # Apply spatial attention
        spatial_weights = self.spatial_attention(spatial_input)
        x_spatial = x_channel * spatial_weights
        
        # Final output with dropout
        out = self.dropout(x_spatial)
        
        return out


class FusionModule(nn.Module):
    """
    Fusion module that integrates features from different levels
    using Content-Guided Attention.
    """
    def __init__(self, channels, reduction=16, dropout=0.1):
        super().__init__()
        self.cga = ContentGuidedAttention(channels, reduction, dropout)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)  # 1x1 conv for final projection
    
    def forward(self, low_features, high_features):
        fused = self.cga(low_features, high_features)
        out = self.conv(fused)
        return out


class HybridEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module = nn.GELU,
        norm: bool = True
    ):
        super().__init__()
        
        # Projections for different feature levels
        self.conv_low = nn.Conv2d(192, emb_dim, 1)  # S3
        self.conv_mid = nn.Conv2d(384, emb_dim, 1)  # S4
        self.conv_high = nn.Conv2d(768, emb_dim, 1)  # S5
        
        # Self-attention for high-level features
        self.self_attention = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Cross-scale Fusion Modules
        self.fusion_s4_s5 = ContentGuidedAttention(
            channels=emb_dim,
            reduction=16,
            dropout=dropout
        )
        
        self.fusion_s3_s4 = ContentGuidedAttention(
            channels=emb_dim,
            reduction=16,
            dropout=dropout
        )
        
        # Bottom-Up path fusion modules
        self.fusion_s4_s3 = ContentGuidedAttention(
            channels=emb_dim,
            reduction=16,
            dropout=dropout
        )
        
        self.fusion_s5_s4 = ContentGuidedAttention(
            channels=emb_dim,
            reduction=16,
            dropout=dropout
        )
        
        # Conv layers for each fusion step
        self.conv_s3_td = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1)
        self.conv_s4_td = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1)
        self.conv_s5_td = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1)
        
        self.conv_s3_bu = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1)
        self.conv_s4_bu = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1)
        self.conv_s5_bu = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1)
        
        # Normalization layers for self-attention
        self.norm1 = nn.LayerNorm(emb_dim) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(emb_dim) if norm else nn.Identity()
        
        # MLP for self-attention
        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)
        
        # Final projection
        self.final_proj = nn.Conv2d(emb_dim * 3, emb_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first
        self.emb_dim = emb_dim

    def forward(self, s3, s4, s5):
        """
        Forward pass through the hybrid encoder.
        
        Args:
            s3: Low-level features [B, C, H, W] or [B, H, W, C]
            s4: Mid-level features [B, C, H, W] or [B, H, W, C]
            s5: High-level features [B, C, H, W] or [B, H, W, C]
            
        Returns:
            Enhanced features [B, C, H, W]
        """
        # Check if inputs are in BHWC format and convert if needed
        if s3.shape[-1] == 192 and s3.shape[1] != 192:  # BHWC format
            s3 = s3.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
            s4 = s4.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
            s5 = s5.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
        
        # Get batch size
        bs = s3.size(0)
        
        # Project features to embedding dimension
        s3_proj = self.conv_low(s3)  # [B, emb_dim, H3, W3]
        s4_proj = self.conv_mid(s4)  # [B, emb_dim, H4, W4]
        s5_proj = self.conv_high(s5)  # [B, emb_dim, H5, W5]
        
        # Process S5 with self-attention
        h5, w5 = s5_proj.shape[-2:]
        s5_flat = s5_proj.flatten(2).permute(0, 2, 1).contiguous()  # [B, H5*W5, emb_dim]
        
        # Apply self-attention with proper pre-norm/post-norm
        if self.norm_first:
            # Pre-norm approach
            s5_norm = self.norm1(s5_flat)
            s5_attn, _ = self.self_attention(s5_norm, s5_norm, s5_norm)
            s5_flat = s5_flat + self.dropout(s5_attn)
            
            # Feed-forward
            s5_norm = self.norm2(s5_flat)
            s5_ff = self.mlp(s5_norm)
            s5_flat = s5_flat + self.dropout(s5_ff)
        else:
            # Post-norm approach
            s5_attn, _ = self.self_attention(s5_flat, s5_flat, s5_flat)
            s5_flat = self.norm1(s5_flat + self.dropout(s5_attn))
            
            # Feed-forward
            s5_ff = self.mlp(s5_flat)
            s5_flat = self.norm2(s5_flat + self.dropout(s5_ff))

        # Reshape back to spatial
        s5_attn = s5_flat.permute(0, 2, 1).reshape(bs, self.emb_dim, h5, w5).contiguous()
        s5_td = self.conv_s5_td(s5_attn)
        
        # Apply top-down path (TD) fusion
        # S5 -> S4 fusion
        s4_td = self.fusion_s4_s5(s4_proj, F.interpolate(s5_td, size=s4_proj.shape[-2:], 
                                                        mode='bilinear', align_corners=True))
        s4_td = self.conv_s4_td(s4_td)
        
        # S4 -> S3 fusion
        s3_td = self.fusion_s3_s4(s3_proj, F.interpolate(s4_td, size=s3_proj.shape[-2:], 
                                                        mode='bilinear', align_corners=True))
        s3_td = self.conv_s3_td(s3_td)
        s3_bu = self.conv_s3_bu(s3_td)  # Apply s3_bu convolution
        
        # Apply bottom-up path (BU) fusion
        # S3 -> S4 fusion
        s4_bu = self.fusion_s4_s3(s4_td, F.interpolate(s3_bu, size=s4_td.shape[-2:], 
                                                      mode='bilinear', align_corners=False))
        s4_bu = self.conv_s4_bu(s4_bu)
        
        # S4 -> S5 fusion
        s5_bu = self.fusion_s5_s4(s5_td, F.interpolate(s4_bu, size=s5_td.shape[-2:], 
                                                         mode='bilinear', align_corners=False))
        s5_bu = self.conv_s5_bu(s5_bu)
        
        # Upsample all features to S3's resolution for concatenation
        s3_out = s3_bu  # Use s3_bu rather than s3_td for output
        s4_out = F.interpolate(s4_bu, size=s3_out.shape[-2:], mode='bilinear', align_corners=True)
        s5_out = F.interpolate(s5_bu, size=s3_out.shape[-2:], mode='bilinear', align_corners=True)
        
        # Concatenate and project to final features
        concat_features = torch.cat([s3_out, s4_out, s5_out], dim=1)
        output = self.final_proj(concat_features)
        
        return output