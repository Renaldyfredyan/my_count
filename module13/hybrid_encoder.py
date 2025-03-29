import torch
from torch import nn
import torch.nn.functional as F

class ContentGuidedAttention(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.channels = channels
        self.groups = groups
        
        # Channel Attention
        self.channel_gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.channel_conv1 = nn.Conv2d(channels, channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.channel_conv2 = nn.Conv2d(channels, channels, 1)
        
        # Spatial Attention
        # Menggunakan GAP dan GMP seperti di diagram
        self.spatial_gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.spatial_gmp = nn.AdaptiveMaxPool2d(1)  # Global Max Pooling
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
        # Group Convolution
        self.gconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=groups)
        
        # Final 1x1 convolution
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
        # Ensure high_features has the same spatial dimensions as low_features
        B, C, H, W = low_features.shape
        _, _, H_high, W_high = high_features.shape
        
        if H_high != H or W_high != W:
            high_features = F.interpolate(high_features, size=(H, W), mode='bilinear', align_corners=True)
        
        # Initial addition of low and high features (seperti di diagram CGA)
        combined = low_features + high_features
        
        # Channel Attention path (sesuai diagram)
        # GAP -> 1x1 Conv -> ReLU -> 1x1 Conv
        ca_gap = self.channel_gap(combined)
        ca_conv1 = self.channel_conv1(ca_gap)
        ca_relu = self.relu(ca_conv1)
        wc = self.channel_conv2(ca_relu)
        
        # Spatial Attention path (implementasi yang lebih sesuai dengan diagram)
        # Channel-wise Average Pooling / Channel Attention Map
        avg_out = torch.mean(combined, dim=1, keepdim=True)  # [B, 1, H, W]
        # Channel-wise Max Pooling / Spatial Attention Map
        max_out, _ = torch.max(combined, dim=1, keepdim=True)  # [B, 1, H, W]
        # Concatenate along channel dimension
        sa_concat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        # Apply 7x7 conv
        ws = self.spatial_conv(sa_concat)  # [B, 1, H, W]
        
        # Addition of channel and spatial attention outputs (seperti di diagram)
        attention_combined = combined + wc + ws
        
        # Channel shuffle untuk feature reorganization
        shuffled = self.channel_shuffle(attention_combined)
        
        # Group Convolution (7x7)
        gconv_out = self.gconv(shuffled)
        
        # Sigmoid untuk mendapatkan attention weights
        attention_weights = torch.sigmoid(gconv_out)
        
        # Apply weights to features (seperti simbol × di diagram)
        weighted_low = low_features * attention_weights
        weighted_high = high_features * (1 - attention_weights)
        
        # Final addition dan projection
        fused = weighted_low + weighted_high
        output = self.proj_conv(fused)
        
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
        norm: bool = True,
        groups: int = 8
    ):
        super().__init__()
        
        # Projections untuk berbagai level feature
        self.conv_low = nn.Conv2d(192, emb_dim, 1)  # S3
        self.conv_mid = nn.Conv2d(384, emb_dim, 1)  # S4
        self.conv_high = nn.Conv2d(768, emb_dim, 1)  # S5
        
        # Self-attention untuk high-level features
        self.self_attention = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout
        )
        
        # Positional encoding
        from positional_encoding import PositionalEncodingsFixed
        self.pos_encoder = PositionalEncodingsFixed(emb_dim)
        
        # Normalization layers untuk self-attention
        self.norm1 = nn.LayerNorm(emb_dim) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(emb_dim) if norm else nn.Identity()
        
        # MLP untuk self-attention
        from mlp import MLP
        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)
        
        # 1x1 Conv setelah self-attention
        self.conv_after_sa = nn.Conv2d(emb_dim, emb_dim, 1)
        
        # Fusion modules untuk top-down pathway
        self.fusion_s5_s4 = ContentGuidedAttention(emb_dim, groups)
        self.fusion_s4_s3 = ContentGuidedAttention(emb_dim, groups)
        
        # 1x1 Convs setelah fusion
        self.conv_after_fusion_s4 = nn.Conv2d(emb_dim, emb_dim, 1)
        self.conv_after_fusion_s3 = nn.Conv2d(emb_dim, emb_dim, 1)
        
        # Fusion modules untuk bottom-up pathway
        self.fusion_s3_s4 = ContentGuidedAttention(emb_dim, groups)
        self.fusion_s4_s5 = ContentGuidedAttention(emb_dim, groups)
        
        # Final projection
        self.final_proj = nn.Conv2d(emb_dim * 3, emb_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first
        self.emb_dim = emb_dim

    def forward(self, s3, s4, s5):
        """
        Forward pass melalui hybrid encoder.
        
        Args:
            s3: Low-level features [B, 192, H, W]
            s4: Mid-level features [B, 384, H/2, W/2]
            s5: High-level features [B, 768, H/4, W/4]
            
        Returns:
            Enhanced features [B, emb_dim, H, W]
        """
        # Convert S3, S4, S5 ke format BCHW jika belum
        if len(s3.shape) == 4 and s3.shape[1] != s3.shape[3]:  # BHWC format
            s3 = s3.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
            s4 = s4.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
            s5 = s5.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
        
        # Get batch size dan dimensi
        bs = s3.size(0)
        h3, w3 = s3.shape[-2:]
        h4, w4 = s4.shape[-2:]
        h5, w5 = s5.shape[-2:]
        
        # Project features ke embedding dimension
        s3 = self.conv_low(s3)     # [B, emb_dim, H, W]
        s4 = self.conv_mid(s4)     # [B, emb_dim, H/2, W/2]
        s5 = self.conv_high(s5)    # [B, emb_dim, H/4, W/4]
        
        # Process S5 dengan self-attention
        s5_flat = s5.flatten(2).permute(2, 0, 1).contiguous()  # [H*W/16, B, emb_dim]
        
        # Generate positional embeddings
        pos_emb = self.pos_encoder(bs, h5, w5, s5.device)
        pos_emb_flat = pos_emb.flatten(2).permute(2, 0, 1).contiguous()  # [H*W/16, B, emb_dim]
        
        # Apply self-attention dengan pre-norm/post-norm handling
        if self.norm_first:
            # Pre-norm approach
            s5_norm = self.norm1(s5_flat)
            q = k = s5_norm + pos_emb_flat
            s5_attn = self.self_attention(q, k, s5_norm, need_weights=False)[0]
            s5_flat = s5_flat + self.dropout(s5_attn)
            
            s5_norm = self.norm2(s5_flat)
            s5_ff = self.mlp(s5_norm)
            s5_flat = s5_flat + self.dropout(s5_ff)
        else:
            # Post-norm approach
            q = k = s5_flat + pos_emb_flat
            s5_attn = self.self_attention(q, k, s5_flat, need_weights=False)[0]
            s5_flat = self.norm1(s5_flat + self.dropout(s5_attn))
            
            s5_ff = self.mlp(s5_flat)
            s5_flat = self.norm2(s5_flat + self.dropout(s5_ff))
        
        # Reshape kembali ke bentuk spatial
        s5_attn = s5_flat.permute(1, 2, 0).reshape(bs, self.emb_dim, h5, w5).contiguous()
        
        # Apply 1x1 Conv setelah self-attention
        s5_processed = self.conv_after_sa(s5_attn)
        
        # Top-down pathway (sesuai dengan Fig 4a)
        # Upsample S5 ke ukuran S4 dan fusion dengan S4
        s5_up = F.interpolate(s5_processed, size=(h4, w4), mode='bilinear', align_corners=True)
        s4_fused_td = self.fusion_s5_s4(s4, s5_up)
        s4_processed = self.conv_after_fusion_s4(s4_fused_td)
        
        # Upsample fused S4 ke ukuran S3 dan fusion dengan S3
        s4_up = F.interpolate(s4_processed, size=(h3, w3), mode='bilinear', align_corners=True)
        s3_fused_td = self.fusion_s4_s3(s3, s4_up)
        s3_processed = self.conv_after_fusion_s3(s3_fused_td)
        
        # Bottom-up pathway (sesuai diagram dengan downsample arrows)
        # Downsample S3 ke ukuran S4 dan fusion dengan S4
        s3_down = F.interpolate(s3_processed, size=(h4, w4), mode='bilinear', align_corners=True)
        s4_fused_bu = self.fusion_s3_s4(s4_processed, s3_down)
        
        # Downsample fused S4 ke ukuran S5 dan fusion dengan S5
        s4_down = F.interpolate(s4_fused_bu, size=(h5, w5), mode='bilinear', align_corners=True)
        s5_fused_bu = self.fusion_s4_s5(s5_processed, s4_down)
        
        # Memastikan semua features memiliki resolusi spatial yang sama (resolusi S3)
        s4_final = F.interpolate(s4_fused_bu, size=(h3, w3), mode='bilinear', align_corners=True)
        s5_final = F.interpolate(s5_fused_bu, size=(h3, w3), mode='bilinear', align_corners=True)
        
        # Concatenate dan project
        concat_features = torch.cat([s3_processed, s4_final, s5_final], dim=1)
        output = self.final_proj(concat_features)
        
        return output