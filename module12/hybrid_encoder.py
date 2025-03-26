import torch
from torch import nn
import torch.nn.functional as F
from mlp import MLP
from positional_encoding import PositionalEncodingsFixed

class ContentGuidedAttention(nn.Module):
    """
    Content Guided Attention (CGA) module as shown in Figure 4(b)
    Implements both channel attention and spatial attention with cross-connections
    """
    def __init__(self, channels, reduction_ratio=16, norm_first=False):
        super().__init__()
        self.channels = channels
        self.norm_first = norm_first
        
        # Channel attention for low-level features
        self.low_gap = nn.AdaptiveAvgPool2d(1)
        self.low_1x1_conv1 = nn.Conv2d(channels, channels // reduction_ratio, 1)
        self.low_relu = nn.ReLU(inplace=True)
        self.low_1x1_conv2 = nn.Conv2d(channels // reduction_ratio, channels, 1)
        
        # Channel attention for high-level features
        self.high_gap = nn.AdaptiveAvgPool2d(1)
        self.high_1x1_conv1 = nn.Conv2d(channels, channels // reduction_ratio, 1)
        self.high_relu = nn.ReLU(inplace=True)
        self.high_1x1_conv2 = nn.Conv2d(channels // reduction_ratio, channels, 1)
        
        # Spatial attention
        self.spatial_conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.spatial_sigmoid = nn.Sigmoid()
        
        # Cross Output module
        self.cross_out_mix = nn.Conv2d(channels, channels, 1)
        self.cross_out_conv = nn.Conv2d(channels, channels, 1)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm([channels])
        
        # Dropout layer untuk konsistensi dengan encoder transformer
        self.dropout = nn.Dropout(0.1)

    def channel_attention(self, x, is_low=True):
        b, c, h, w = x.shape
        
        if is_low:
            y = self.low_gap(x)  # [B, C, 1, 1]
            y = self.low_1x1_conv1(y)  # [B, C/r, 1, 1]
            y = self.low_relu(y)
            y = self.low_1x1_conv2(y)  # [B, C, 1, 1]
        else:
            y = self.high_gap(x)
            y = self.high_1x1_conv1(y)
            y = self.high_relu(y)
            y = self.high_1x1_conv2(y)
            
        y = torch.sigmoid(y)
        return x * y.expand_as(x)

    def spatial_attention(self, x):
        # Compute average and max along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and pass through conv layer
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.spatial_conv1(y)
        y = self.spatial_sigmoid(y)
        
        return x * y
        
    def forward(self, low_features, high_features):
        # Ensure high_features has the same spatial dimensions as low_features
        B, C, H, W = low_features.shape
        _, _, H_high, W_high = high_features.shape
        
        if H_high != H or W_high != W:
            high_features = F.interpolate(high_features, size=(H, W), mode='bilinear', align_corners=True)
        
        # Implementasikan pre-norm atau post-norm mengikuti pola TransformerEncoderLayer
        if self.norm_first:
            # Pre-norm approach
            low_flat = low_features.flatten(2).transpose(-1, -2).contiguous()  # [B, H*W, C]
            high_flat = high_features.flatten(2).transpose(-1, -2).contiguous()  # [B, H*W, C]
            
            # Normalisasi input terlebih dahulu
            low_norm = self.norm1(low_flat)
            high_norm = self.norm1(high_flat)  # Menggunakan norm1 yang sama untuk konsistensi
            
            # Reshape kembali ke bentuk spatial
            low_norm = low_norm.transpose(-1, -2).reshape(B, C, H, W).contiguous()
            high_norm = high_norm.transpose(-1, -2).reshape(B, C, H, W).contiguous()
            
            # Lakukan attention pada input yang sudah dinormalisasi
            low_channel_attn = self.channel_attention(low_norm, is_low=True)
            high_channel_attn = self.channel_attention(high_norm, is_low=False)
            low_spatial_attn = self.spatial_attention(low_norm)
            high_spatial_attn = self.spatial_attention(high_norm)
            
            # Cross connections dengan input asli (residual)
            low_out = low_features + self.dropout(low_channel_attn + high_spatial_attn)
            high_out = high_features + self.dropout(high_channel_attn + low_spatial_attn)
            
        else:
            # Post-norm approach (lakukan attention dulu, baru normalisasi)
            low_channel_attn = self.channel_attention(low_features, is_low=True)
            high_channel_attn = self.channel_attention(high_features, is_low=False)
            low_spatial_attn = self.spatial_attention(low_features)
            high_spatial_attn = self.spatial_attention(high_features)
            
            # Cross connections dengan input asli
            low_out = low_features + low_channel_attn + high_spatial_attn
            high_out = high_features + high_channel_attn + low_spatial_attn
        
        # Final fusion
        mixed = self.cross_out_mix(low_out + high_out)
        sigmoid_mixed = torch.sigmoid(mixed)
        
        # Weighted feature fusion
        out = low_out * sigmoid_mixed + high_out * (1 - sigmoid_mixed)
        out = self.cross_out_conv(out)
        
        # Apply normalization after residual if post-norm
        if not self.norm_first:
            out_flat = out.flatten(2).transpose(-1, -2).contiguous()
            out_flat = self.norm1(out_flat)
            out = out_flat.transpose(-1, -2).reshape(B, C, H, W).contiguous()
        
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
        
        # Self-attention untuk fitur high-level
        self.self_attention = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout
        )
        
        # Cross-scale Fusion Modules (top-down pathway)
        self.fusion_s5_s4 = ContentGuidedAttention(
            emb_dim,
            reduction_ratio=16,
            norm_first=norm_first
        )
        
        self.fusion_s4_s3 = ContentGuidedAttention(
            emb_dim,
            reduction_ratio=16, 
            norm_first=norm_first
        )
        
        # Bottom-up pathway fusion modules
        self.fusion_s3_s4_up = ContentGuidedAttention(
            emb_dim,
            reduction_ratio=16,
            norm_first=norm_first
        )
        
        self.fusion_s4_s5_up = ContentGuidedAttention(
            emb_dim,
            reduction_ratio=16,
            norm_first=norm_first
        )
        
        # Normalization layers for self-attention - konsisten dengan TransformerEncoderLayer
        self.norm1 = nn.LayerNorm(emb_dim) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(emb_dim) if norm else nn.Identity()
        
        # MLP for self-attention
        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)
        
        # Final projection
        self.final_proj = nn.Conv2d(emb_dim * 3, emb_dim, 1)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
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
        
        # Project features to embedding dimension
        s3 = self.conv_low(s3)    # [B, emb_dim, H/8, W/8]
        s4 = self.conv_mid(s4)    # [B, emb_dim, H/16, W/16]
        s5 = self.conv_high(s5)   # [B, emb_dim, H/32, W/32]
        
        # Process S5 with self-attention
        h5, w5 = s5.shape[-2:]
        s5_flat = s5.flatten(2).permute(2, 0, 1).contiguous()  # [H*W, B, C]

        # Generate positional embeddings
        pos_encoder = PositionalEncodingsFixed(self.emb_dim)
        pos_emb_spatial = pos_encoder(bs, h5, w5, s5.device)
        pos_emb = pos_emb_spatial.flatten(2).permute(2, 0, 1).contiguous()  # [H*W, B, C]
        
        # Implementasi self-attention sesuai dengan pola TransformerEncoderLayer
        if self.norm_first:
            # Pre-norm approach
            s5_norm = self.norm1(s5_flat)
            # Add positional embeddings
            q = k = s5_norm + pos_emb
            # Self attention call
            s5_attn = self.self_attention(
                query=q,
                key=k,
                value=s5_norm,
                need_weights=False
            )[0]
            s5_flat = s5_flat + self.dropout1(s5_attn)
            
            # Feed-forward
            s5_norm = self.norm2(s5_flat)
            s5_ff = self.mlp(s5_norm)
            s5_flat = s5_flat + self.dropout2(s5_ff)
        else:
            # Post-norm approach
            # Add positional embeddings
            q = k = s5_flat + pos_emb
            # Self attention call
            s5_attn = self.self_attention(
                query=q,
                key=k,
                value=s5_flat,
                need_weights=False
            )[0]
            s5_flat = self.norm1(s5_flat + self.dropout1(s5_attn))
            
            # Feed-forward
            s5_ff = self.mlp(s5_flat)
            s5_flat = self.norm2(s5_flat + self.dropout2(s5_ff))

        # Reshape back to spatial
        s5 = s5_flat.permute(1, 2, 0).contiguous()  # [B, C, H*W]
        s5 = s5.reshape(bs, self.emb_dim, h5, w5).contiguous()  # [B, C, H, W]
        
        # Top-down pathway (as shown in Figure 4a)
        # First, fuse S5 and S4
        s4_td = self.fusion_s5_s4(s4, s5)
        
        # Then, fuse S4_td and S3
        s3_td = self.fusion_s4_s3(s3, s4_td)
        
        # Bottom-up pathway (as shown in Figure 4a)
        s4_bu = self.fusion_s3_s4_up(s4_td, s3_td)
        s5_bu = self.fusion_s4_s5_up(s5, s4_bu)
        
        # Upsample all features to same resolution (S3's resolution)
        s3_size = s3.shape[-2:]
        s4_out = F.interpolate(s4_bu, size=s3_size, mode='bilinear', align_corners=True)
        s5_out = F.interpolate(s5_bu, size=s3_size, mode='bilinear', align_corners=True)
        
        # Concatenate and project
        concat_features = torch.cat([s3_td, s4_out, s5_out], dim=1)
        output = self.final_proj(concat_features)
        
        return output