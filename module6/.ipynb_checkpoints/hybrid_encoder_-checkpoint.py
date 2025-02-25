from .mlp import MLP

from torch import nn

class HybridEncoder(nn.Module):
    def __init__(
        self,
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
        
        # High-level Features Self-Attention (hanya untuk S5)
        self.self_attention = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout
        )
        
        # Cross-scale Fusion Module
        self.fusion_module = CrossScaleFusion(
            low_channels=192,   # S3
            mid_channels=384,   # S4 
            high_channels=768,  # S5
            out_channels=emb_dim
        )
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(emb_dim) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(emb_dim) if norm else nn.Identity()
        
        # MLP block
        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)
        
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, s3, s4, s5, pos_emb=None):
        # 1. Apply self-attention only to S5
        if self.norm_first:
            s5_norm = self.norm1(s5)
            s5 = s5 + self.dropout(self.self_attention(
                s5_norm, s5_norm, s5_norm
            )[0])
            s5 = s5 + self.dropout(self.mlp(self.norm2(s5)))
        else:
            s5 = self.norm1(s5 + self.dropout(self.self_attention(
                s5, s5, s5
            )[0]))
            s5 = self.norm2(s5 + self.dropout(self.mlp(s5)))
            
        # 2. Cross-scale fusion
        out = self.fusion_module(s3, s4, s5)
        
        return out

class CrossScaleFusion(nn.Module):
    def __init__(self, low_channels, mid_channels, high_channels, out_channels):
        super().__init__()
        
        # Conv layers untuk menyesuaikan channel dimensions
        self.conv_low = nn.Conv2d(low_channels, out_channels, 1)
        self.conv_mid = nn.Conv2d(mid_channels, out_channels, 1)
        self.conv_high = nn.Conv2d(high_channels, out_channels, 1)
        
        # Content-Guided Attention (CGA)
        self.cga = ContentGuidedAttention(out_channels)
        
    def forward(self, s3, s4, s5):
        # Projecting all features to same channel dimension
        s3 = self.conv_low(s3)
        s4 = self.conv_mid(s4)
        s5 = self.conv_high(s5)
        
        # Upsample s4 dan s5 ke ukuran s3
        s4 = F.interpolate(s4, size=s3.shape[-2:], mode='bilinear', align_corners=True)
        s5 = F.interpolate(s5, size=s3.shape[-2:], mode='bilinear', align_corners=True)
        
        # Apply CGA fusion
        out = self.cga(s3, s4, s5)
        
        return out