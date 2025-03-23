import torch
from torch import nn
import torch.nn.functional as F
from mlp import MLP
from positional_encoding import PositionalEncodingsFixed

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Compute channel-wise average and max, and concatenate
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return self.sigmoid(y)

class ContentGuidedAttention(nn.Module):
    def __init__(self, channels, num_heads=8, dropout=0.1, norm_first=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm_first = norm_first
        
        # Channel Attention
        self.channel_attn = ChannelAttention(channels)
        
        # Spatial Attention
        self.spatial_attn = SpatialAttention()
        
        # 1x1 Conv for input projections
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        
        # 1x1 Conv for output projection
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
        # Scaling factor for attention
        self.scale = (channels // num_heads) ** -0.5
        
        # Normalization layers
        self.norm1 = nn.LayerNorm([channels])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, low_features, high_features):
        # Get dimensions
        B, C, H, W = low_features.shape
        _, _, H_high, W_high = high_features.shape
        
        # Ensure high_features has the same spatial dimensions as low_features
        if H_high != H or W_high != W:
            high_features = F.interpolate(high_features, size=(H, W), mode='bilinear', align_corners=True)
        
        if self.norm_first:
            # Pre-norm approach
            low_flat = low_features.flatten(2).transpose(-1, -2).contiguous()  # [B, H*W, C]
            high_flat = high_features.flatten(2).transpose(-1, -2).contiguous()  # [B, H*W, C]
            
            low_norm = self.norm1(low_flat)
            high_norm = self.norm1(high_flat)
            
            low_norm = low_norm.transpose(-1, -2).reshape(B, C, H, W).contiguous()
            high_norm = high_norm.transpose(-1, -2).reshape(B, C, H, W).contiguous()
            
            # Apply projections
            q = self.q_proj(low_norm)
            k = self.k_proj(high_norm)
            v = self.v_proj(high_norm)
        else:
            # Direct projection
            q = self.q_proj(low_features)
            k = self.k_proj(high_features)
            v = self.v_proj(high_features)
        
        # Apply channel attention
        channel_weights = self.channel_attn(v)
        v_channel = v * channel_weights
        
        # Apply spatial attention
        spatial_weights = self.spatial_attn(v_channel)
        v_weighted = v_channel * spatial_weights
        
        # Reshape for attention computation
        q_flat = q.flatten(2).permute(0, 2, 1).contiguous()  # [B, H*W, C]
        k_flat = k.flatten(2).contiguous()  # [B, C, H*W]
        v_flat = v_weighted.flatten(2).permute(0, 2, 1).contiguous()  # [B, H*W, C]
        
        # Compute attention scores
        attn = torch.bmm(q_flat, k_flat) * self.scale  # [B, H*W, H*W]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.bmm(attn, v_flat)  # [B, H*W, C]
        out = out.permute(0, 2, 1).reshape(B, C, H, W).contiguous()  # [B, C, H, W]
        
        # Output projection
        out = self.out_proj(out)
        
        # Add skip connection
        output = low_features + out
        
        # Apply normalization after residual if post-norm
        if not self.norm_first:
            output_flat = output.flatten(2).transpose(-1, -2).contiguous()
            output_flat = self.norm1(output_flat)
            output = output_flat.transpose(-1, -2).reshape(B, C, H, W).contiguous()
        
        return output

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
            emb_dim, num_heads, dropout=dropout
        )
        
        # Normalization layers for self-attention
        self.norm1 = nn.LayerNorm(emb_dim) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(emb_dim) if norm else nn.Identity()
        
        # MLP for self-attention
        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)
        
        # Cross-scale Fusion Modules (top-down path)
        self.fusion_s5_s4 = ContentGuidedAttention(
            emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            norm_first=norm_first
        )
        
        self.fusion_s4_s3 = ContentGuidedAttention(
            emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            norm_first=norm_first
        )
        
        # Cross-scale Fusion Modules (bottom-up path)
        self.fusion_s3_s4 = ContentGuidedAttention(
            emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            norm_first=norm_first
        )
        
        self.fusion_s4_s5 = ContentGuidedAttention(
            emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            norm_first=norm_first
        )
        
        # 1x1 convolutions for feature projection
        self.s3_proj = nn.Conv2d(emb_dim, emb_dim, 1)
        self.s4_proj = nn.Conv2d(emb_dim, emb_dim, 1)
        self.s5_proj = nn.Conv2d(emb_dim, emb_dim, 1)
        
        # Final projection
        self.final_proj = nn.Conv2d(emb_dim * 3, emb_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first
        self.emb_dim = emb_dim

    def forward(self, s3, s4, s5):
        """
        Forward pass through the hybrid encoder.
        
        Args:
            s3: Low-level features [B, C, H, W]
            s4: Mid-level features [B, C, H, W]
            s5: High-level features [B, C, H, W]
            
        Returns:
            Enhanced features [B, C, H, W]
        """
        # Convert S3, S4, S5 to BCHW format if they aren't already
        if len(s3.shape) == 4 and s3.shape[1] != s3.shape[3]:  # BHWC format
            s3 = s3.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
            s4 = s4.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
            s5 = s5.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
        
        # Get batch size
        bs = s3.size(0)
        
        # Store original spatial dimensions
        s3_size = s3.shape[-2:]
        s4_size = s4.shape[-2:]
        s5_size = s5.shape[-2:]
        
        # Project features to embedding dimension
        s3 = self.conv_low(s3)
        s4 = self.conv_mid(s4)
        s5 = self.conv_high(s5)
        
        # Process S5 with self-attention
        h5, w5 = s5.shape[-2:]
        s5_flat = s5.flatten(2).permute(2, 0, 1).contiguous()  # [B, C, H*W] -> [H*W, B, C]
        
        # Generate positional embeddings
        pos_encoder = PositionalEncodingsFixed(self.emb_dim)
        pos_emb_spatial = pos_encoder(bs, h5, w5, s5.device)
        pos_emb = pos_emb_spatial.flatten(2).permute(2, 0, 1).contiguous()  # [B, C, H*W] -> [H*W, B, C]
        
        # Apply self-attention with proper pre-norm/post-norm
        if self.norm_first:
            # Pre-norm approach
            s5_norm = self.norm1(s5_flat)
            # Add positional embeddings to normalized input for q and k
            q = k = s5_norm + pos_emb
            # Self attention call
            s5_attn = self.self_attention(
                query=q,
                key=k,
                value=s5_norm,  # No position embeddings for value
                need_weights=False
            )[0]
            s5_flat = s5_flat + self.dropout(s5_attn)
            
            # Feed-forward
            s5_norm = self.norm2(s5_flat)
            s5_ff = self.mlp(s5_norm)
            s5_flat = s5_flat + self.dropout(s5_ff)
        else:
            # Post-norm approach
            # Add positional embeddings to input for q and k
            q = k = s5_flat + pos_emb
            # Self attention call
            s5_attn = self.self_attention(
                query=q,
                key=k,
                value=s5_flat,  # No position embeddings for value
                need_weights=False
            )[0]
            s5_flat = self.norm1(s5_flat + self.dropout(s5_attn))
            
            # Feed-forward
            s5_ff = self.mlp(s5_flat)
            s5_flat = self.norm2(s5_flat + self.dropout(s5_ff))

        # Reshape back to spatial
        s5 = s5_flat.permute(1, 2, 0).contiguous()  # [H*W, B, C] -> [B, C, H*W]
        s5 = s5.reshape(bs, self.emb_dim, h5, w5).contiguous()  # [B, C, H*W] -> [B, C, H, W]
        
        # Top-down path: S5 -> S4 -> S3
        # Fuse S5 with S4
        s4_td = self.fusion_s5_s4(s4, s5)
        
        # Fuse S4 with S3
        s3_td = self.fusion_s4_s3(s3, s4_td)
        
        # Bottom-up path: S3 -> S4 -> S5
        # Fuse S3 with S4
        s4_bu = self.fusion_s3_s4(s4, s3_td)
        
        # Fuse S4 with S5
        s5_bu = self.fusion_s4_s5(s5, s4_bu)
        
        # Project features
        s3_out = self.s3_proj(s3_td)
        s4_out = self.s4_proj(s4_bu)
        s5_out = self.s5_proj(s5_bu)
        
        # Upsample all features to S3's resolution
        s4_out = F.interpolate(s4_out, size=s3_size, mode='bilinear', align_corners=True)
        s5_out = F.interpolate(s5_out, size=s3_size, mode='bilinear', align_corners=True)
        
        # Concatenate and project
        concat_features = torch.cat([s3_out, s4_out, s5_out], dim=1)
        output = self.final_proj(concat_features)
        
        return output