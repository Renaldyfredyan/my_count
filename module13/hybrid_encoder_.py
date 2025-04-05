import torch
from torch import nn
import torch.nn.functional as F
from mlp import MLP
from positional_encoding import PositionalEncodingsFixed


class ContentGuidedAttention(nn.Module):
    """Content Guided Attention (CGA) for cross-scale feature fusion."""
    def __init__(self, channels, groups=8):
        super().__init__()
        self.channels = channels
        self.groups = groups
        
        # Channel Attention path
        self.channel_gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.channel_conv1 = nn.Conv2d(channels, channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.channel_conv2 = nn.Conv2d(channels, channels, 1)
        
        # Spatial Attention path
        self.spatial_gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling for channels
        self.spatial_gmp = nn.AdaptiveMaxPool2d(1)  # Global Max Pooling for channels
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
        # Group Convolution for feature reorganization
        self.gconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=groups)
        
        # Final 1x1 convolution for output projection
        self.proj_conv = nn.Conv2d(channels, channels, 1)
        
    def channel_shuffle(self, x):
        batch, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        
        # Reshape
        x = x.view(batch, self.groups, channels_per_group, height, width)
        
        # Transpose
        x = x.transpose(1, 2).contiguous()
        
        # Flatten
        x = x.view(batch, -1, height, width)
        
        return x
        
    def forward(self, low_features, high_features):
        """
        Implementation exactly following Fig. 4(b) in paper
        """
        # Ensure high_features has the same spatial dimensions as low_features
        B, C, H, W = low_features.shape
        _, _, H_high, W_high = high_features.shape
        
        if H_high != H or W_high != W:
            high_features = F.interpolate(high_features, size=(H, W), mode='bilinear', align_corners=True)
        
        # Initial addition of low and high features (as shown in diagram)
        combined = low_features + high_features
        
        # Channel Attention path (top path in Fig. 4b)
        # GAP -> 1x1 Conv -> ReLU -> 1x1 Conv
        ca_gap = self.channel_gap(combined)
        ca_conv1 = self.channel_conv1(ca_gap)
        ca_relu = self.relu(ca_conv1)
        wc = self.channel_conv2(ca_relu)
        
        # Spatial Attention path (bottom path in Fig. 4b)
        # Channel-wise Average Pooling & Max Pooling
        avg_out = torch.mean(combined, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(combined, dim=1, keepdim=True)  # [B, 1, H, W]
        # Concatenate along channel dimension
        sa_concat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        # Apply 7x7 conv
        ws = self.spatial_conv(sa_concat)  # [B, 1, H, W]
        
        # Addition of features with attention (central "+" in Fig. 4b)
        attention_combined = combined + wc + ws
        
        # Channel shuffle for feature reorganization
        shuffled = self.channel_shuffle(attention_combined)
        
        # Group Convolution (7x7)
        gconv_out = self.gconv(shuffled)
        
        # Sigmoid for attention weights
        attention_weights = torch.sigmoid(gconv_out)
        
        # Apply weights to features (multiplicative fusion, "×" in Fig. 4b)
        weighted_low = low_features * attention_weights
        weighted_high = high_features * (1 - attention_weights)
        
        # Final addition ("+" at right side in Fig. 4b)
        fused = weighted_low + weighted_high
        
        # Final 1x1 convolution for output projection
        output = self.proj_conv(fused)
        
        return output


class HybridEncoder(nn.Module):
    """
    Hybrid Encoder implementation exactly following Fig. 4(a) in the paper.
    Uses self-attention only on high-level features and cross-scale fusion with Content Guided Attention.
    """
    def __init__(
        self,
        num_layers: int = 1,
        emb_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        mlp_factor: int = 4,
        norm_first: bool = True,
        activation: nn.Module = nn.GELU,
        norm: bool = True,
        groups: int = 8
    ):
        super().__init__()
        
        # Projections for different feature levels (S3, S4, S5)
        self.conv_low = nn.Conv2d(192, emb_dim, 1)    # S3 projection
        self.conv_mid = nn.Conv2d(384, emb_dim, 1)    # S4 projection
        self.conv_high = nn.Conv2d(768, emb_dim, 1)   # S5 projection
        
        # Positional encoding for self-attention
        self.pos_encoder = PositionalEncodingsFixed(emb_dim)
        
        # Self-attention for high-level features only (S5)
        self.self_attention = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout
        )
        
        # Normalization layers for self-attention
        self.norm1 = nn.LayerNorm(emb_dim) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(emb_dim) if norm else nn.Identity()
        
        # MLP after self-attention
        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)
        
        # 1x1 Conv after self-attention (as shown in Fig. 4a)
        self.conv_after_sa = nn.Conv2d(emb_dim, emb_dim, 1)
        
        # Top-down pathway fusion modules and 1x1 convs
        self.fusion_s5_s4 = ContentGuidedAttention(emb_dim, groups)
        self.conv_after_fusion_s5_s4 = nn.Conv2d(emb_dim, emb_dim, 1)
        
        self.fusion_s4_s3 = ContentGuidedAttention(emb_dim, groups)
        self.conv_after_fusion_s4_s3 = nn.Conv2d(emb_dim, emb_dim, 1)
        
        # Bottom-up pathway fusion modules and 1x1 convs
        self.fusion_s3_s4 = ContentGuidedAttention(emb_dim, groups)
        self.conv_after_fusion_s3_s4 = nn.Conv2d(emb_dim, emb_dim, 1)
        
        self.fusion_s4_s5 = ContentGuidedAttention(emb_dim, groups)
        self.conv_after_fusion_s4_s5 = nn.Conv2d(emb_dim, emb_dim, 1)
        
        # Final projection
        self.final_proj = nn.Conv2d(emb_dim * 3, emb_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first
        self.emb_dim = emb_dim

    def forward(self, s3, s4, s5):
        """
        Forward pass through the hybrid encoder.
        Implementation exactly following Fig. 4(a) in the paper.
        
        Args:
            s3: Low-level features [B, 192, H, W]
            s4: Mid-level features [B, 384, H/2, W/2]
            s5: High-level features [B, 768, H/4, W/4]
            
        Returns:
            Enhanced features [B, emb_dim, H, W]
        """
        # Convert inputs to BCHW format if needed
        if len(s3.shape) == 4 and s3.shape[1] != s3.shape[3]:  # BHWC format
            s3 = s3.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
            s4 = s4.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
            s5 = s5.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
        
        # Get dimensions
        bs = s3.size(0)
        h3, w3 = s3.shape[-2:]
        h4, w4 = s4.shape[-2:]
        h5, w5 = s5.shape[-2:]
        
        # Project features to embedding dimension
        s3 = self.conv_low(s3)     # [B, emb_dim, H, W]
        s4 = self.conv_mid(s4)     # [B, emb_dim, H/2, W/2]
        s5 = self.conv_high(s5)    # [B, emb_dim, H/4, W/4]
        
        # 1. Process S5 with self-attention (high-level features only)
        # Flatten and permute for attention
        s5_flat = s5.flatten(2).permute(2, 0, 1).contiguous()  # [H*W/16, B, emb_dim]
        
        # Generate positional embeddings
        pos_emb = self.pos_encoder(bs, h5, w5, s5.device)
        pos_emb_flat = pos_emb.flatten(2).permute(2, 0, 1).contiguous()  # [H*W/16, B, emb_dim]
        
        # Apply self-attention with pre-norm/post-norm handling
        if self.norm_first:
            # Pre-norm approach
            s5_norm = self.norm1(s5_flat)
            q = k = s5_norm + pos_emb_flat  # Add position embeddings to query and key
            s5_attn = self.self_attention(q, k, s5_norm, need_weights=False)[0]
            s5_flat = s5_flat + self.dropout(s5_attn)
            
            s5_norm = self.norm2(s5_flat)
            s5_ff = self.mlp(s5_norm)
            s5_flat = s5_flat + self.dropout(s5_ff)
        else:
            # Post-norm approach
            q = k = s5_flat + pos_emb_flat  # Add position embeddings to query and key
            s5_attn = self.self_attention(q, k, s5_flat, need_weights=False)[0]
            s5_flat = self.norm1(s5_flat + self.dropout(s5_attn))
            
            s5_ff = self.mlp(s5_flat)
            s5_flat = self.norm2(s5_flat + self.dropout(s5_ff))
        
        # Reshape back to spatial format
        s5_attn = s5_flat.permute(1, 2, 0).reshape(bs, self.emb_dim, h5, w5).contiguous()
        
        # Apply 1x1 Conv after self-attention (shown in Fig. 4a)
        s5_processed = self.conv_after_sa(s5_attn)
        
        # 2. Top-down pathway with fusion (following Fig. 4a)
        # Upsample S5 to S4 size and fuse with S4
        s5_up = F.interpolate(s5_processed, size=(h4, w4), mode='bilinear', align_corners=True)
        s4_fused_td = self.fusion_s5_s4(s4, s5_up)
        s4_processed = self.conv_after_fusion_s5_s4(s4_fused_td)
        
        # Upsample fused S4 to S3 size and fuse with S3
        s4_up = F.interpolate(s4_processed, size=(h3, w3), mode='bilinear', align_corners=True)
        s3_fused_td = self.fusion_s4_s3(s3, s4_up)
        s3_processed = self.conv_after_fusion_s4_s3(s3_fused_td)
        
        # 3. Bottom-up pathway with fusion (following Fig. 4a)
        # Downsample S3 to S4 size and fuse with S4_processed
        s3_down = F.interpolate(s3_processed, size=(h4, w4), mode='bilinear', align_corners=True)
        s4_fused_bu = self.fusion_s3_s4(s4_processed, s3_down)
        s4_bu_processed = self.conv_after_fusion_s3_s4(s4_fused_bu)
        
        # Downsample fused S4 to S5 size and fuse with S5_processed
        s4_down = F.interpolate(s4_bu_processed, size=(h5, w5), mode='bilinear', align_corners=True)
        s5_fused_bu = self.fusion_s4_s5(s5_processed, s4_down)
        s5_bu_processed = self.conv_after_fusion_s4_s5(s5_fused_bu)
        
        # 4. Final feature integration (following Fig. 4a)
        # Upsample all features to S3 resolution for concatenation
        s4_final = F.interpolate(s4_bu_processed, size=(h3, w3), mode='bilinear', align_corners=True)
        s5_final = F.interpolate(s5_bu_processed, size=(h3, w3), mode='bilinear', align_corners=True)
        
        # Concatenate features and apply final projection
        concat_features = torch.cat([s3_processed, s4_final, s5_final], dim=1)
        output = self.final_proj(concat_features)
        
        return output


# Example usage:
if __name__ == "__main__":
    # Initialize the model
    model = HybridEncoder(
        num_layers=1,
        emb_dim=256,
        num_heads=8,
        dropout=0.1,
        layer_norm_eps=1e-5,
        mlp_factor=4,
        norm_first=True,
        groups=8
    )
    
    # Create random input tensors
    batch_size = 2
    s3 = torch.randn(batch_size, 192, 64, 64)  # Low-level features
    s4 = torch.randn(batch_size, 384, 32, 32)  # Mid-level features
    s5 = torch.randn(batch_size, 768, 16, 16)  # High-level features
    
    # Forward pass
    output = model(s3, s4, s5)
    
    # Print output shape
    print(f"Output shape: {output.shape}")  # Expected: [2, 256, 64, 64]