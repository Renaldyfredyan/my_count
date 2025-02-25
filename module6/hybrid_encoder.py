from mlp import MLP
import torch
from torch import nn
import torch.nn.functional as F

class ContentGuidedAttention(nn.Module):
    def __init__(self, channels, num_heads=8, dropout=0.1, norm_first=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm_first = norm_first
        
        # Query, Key, Value projections
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        
        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
        # Scaling factor for attention
        self.scale = (channels // num_heads) ** -0.5
        
        # Normalization layers
        self.norm1 = nn.LayerNorm([channels])
        self.norm2 = nn.LayerNorm([channels])
        self.norm3 = nn.LayerNorm([channels])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, s3, s4, s5):
        # Get dimensions
        B, C, H, W = s3.shape
        
        # Pre-process with proper reshape + normalization
        s3_flat = s3.flatten(2).transpose(-1, -2).contiguous()  # [B, H*W, C]
        s4_flat = s4.flatten(2).transpose(-1, -2).contiguous()  # [B, H*W, C]
        s5_flat = s5.flatten(2).transpose(-1, -2).contiguous()  # [B, H*W, C]
        
        if self.norm_first:
            # Pre-norm
            s3_norm = self.norm1(s3_flat)
            s4_norm = self.norm2(s4_flat)
            s5_norm = self.norm3(s5_flat)
            
            # Reshape back for conv operations
            s3_norm = s3_norm.transpose(-1, -2).reshape(B, C, H, W).contiguous()
            s4_norm = s4_norm.transpose(-1, -2).reshape(B, C, s4.shape[2], s4.shape[3]).contiguous()
            s5_norm = s5_norm.transpose(-1, -2).reshape(B, C, s5.shape[2], s5.shape[3]).contiguous()
            
            # Apply projections to normalized features
            q = self.q_proj(s3_norm)
            k = torch.cat([self.k_proj(s4_norm), self.k_proj(s5_norm)], dim=2)
            v = torch.cat([self.v_proj(s4_norm), self.v_proj(s5_norm)], dim=2)
        else:
            # Apply projections directly
            q = self.q_proj(s3)
            k = torch.cat([self.k_proj(s4), self.k_proj(s5)], dim=2)
            v = torch.cat([self.v_proj(s4), self.v_proj(s5)], dim=2)
        
        # Reshape for attention computation
        q = q.flatten(2).transpose(-1, -2).contiguous()   # [B, H*W, C]
        k = k.flatten(2).transpose(-1, -2).contiguous()   # [B, 2*H*W, C]
        v = v.flatten(2).transpose(-1, -2).contiguous()   # [B, 2*H*W, C]
        
        # Multi-head attention split
        head_dim = C // self.num_heads
        q = q.reshape(B, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)  # [B, num_heads, H*W, head_dim]
        k = k.reshape(B, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)  # [B, num_heads, 2*H*W, head_dim]
        v = v.reshape(B, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)  # [B, num_heads, 2*H*W, head_dim]
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, num_heads, H*W, 2*H*W]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v)  # [B, num_heads, H*W, head_dim]
        
        # Reshape back
        out = out.permute(0, 2, 1, 3).reshape(B, H*W, C)  # [B, H*W, C]
        
        # Apply output projection
        if self.norm_first:
            # For pre-norm, reshape, project, and then add to input
            out = out.transpose(-1, -2).reshape(B, C, H, W).contiguous()
            out = self.out_proj(out)
            out = out.flatten(2).transpose(-1, -2).contiguous()  # [B, H*W, C]
            
            # Skip connection back to input space
            s3_flat = s3_flat + out
            
            # Reshape to original format
            out = s3_flat.transpose(-1, -2).reshape(B, C, H, W).contiguous()
        else:
            # For post-norm, reshape and project
            out = out.transpose(-1, -2).reshape(B, C, H, W).contiguous()
            out = self.out_proj(out)
            
            # Skip connection directly in spatial domain
            out = s3 + out
            
            # Apply normalization after skip connection
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
        
        # Add projection layers for different feature levels
        self.conv_high = nn.Conv2d(768, emb_dim, 1)  # Project S5 to emb_dim
        
        # Self-attention for high-level features
        self.self_attention = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout
        )
        
        # Cross-scale Fusion Module
        self.fusion_module = CrossScaleFusion(
            low_channels=192,   # S3
            mid_channels=384,   # S4 
            high_channels=768,  # S5
            out_channels=emb_dim,
            norm_first=norm_first,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(emb_dim) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(emb_dim) if norm else nn.Identity()
        
        # MLP block
        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)
        
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, s3, s4, s5, pos_emb=None):
        # s5 shape: [B, H, W, C]
        # Transpose to format BCHW
        bs = s5.size(0)
        s5 = s5.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
        
        # Project to embedding dimension
        s5 = self.conv_high(s5)  # [B, 768, H, W] -> [B, emb_dim, H, W]
        
        # Reshape for self-attention
        h, w = s5.shape[-2:]
        s5 = s5.flatten(2).transpose(1, 2).contiguous()  # [B, C, H*W] -> [B, H*W, C]
        s5 = s5.transpose(0, 1).contiguous()  # [B, H*W, C] -> [H*W, B, C]

        # Apply self-attention with proper pre-norm/post-norm
        if self.norm_first:
            # Pre-norm architecture
            s5_norm = self.norm1(s5)
            attn_output = self.dropout(self.self_attention(
                s5_norm, s5_norm, s5_norm
            )[0])
            s5 = s5 + attn_output
            
            # Feed-forward with pre-norm
            s5_norm = self.norm2(s5)
            ff_output = self.dropout(self.mlp(s5_norm))
            s5 = s5 + ff_output
        else:
            # Post-norm architecture
            attn_output = self.dropout(self.self_attention(
                s5, s5, s5
            )[0])
            s5 = self.norm1(s5 + attn_output)
            
            # Feed-forward with post-norm
            ff_output = self.dropout(self.mlp(s5))
            s5 = self.norm2(s5 + ff_output)

        # Reshape back
        s5 = s5.transpose(0, 1).contiguous()  # [H*W, B, C] -> [B, H*W, C]
        s5 = s5.transpose(1, 2).contiguous()  # [B, H*W, C] -> [B, C, H*W]
        s5 = s5.reshape(bs, -1, h, w).contiguous()  # [B, C, H*W] -> [B, C, H, W]
                
        # Cross-scale fusion
        out = self.fusion_module(s3, s4, s5)
            
        return out


class CrossScaleFusion(nn.Module):
    def __init__(self, low_channels, mid_channels, high_channels, out_channels, norm_first=False, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Conv layers to adjust channel dimensions
        self.conv_low = nn.Conv2d(low_channels, out_channels, 1)
        self.conv_mid = nn.Conv2d(mid_channels, out_channels, 1)
        
        # Content-Guided Attention (CGA)
        self.cga = ContentGuidedAttention(
            out_channels,
            num_heads=num_heads,
            dropout=dropout,
            norm_first=norm_first
        )
        
    def forward(self, s3, s4, s5):
        # Ensure consistent format
        s3 = s3.permute(0, 3, 1, 2).contiguous()
        s4 = s4.permute(0, 3, 1, 2).contiguous()
        
        # Project to same dimension
        s3 = self.conv_low(s3)
        s4 = self.conv_mid(s4)
        
        # Upsample with consistent format
        s4 = F.interpolate(s4, size=s3.shape[-2:], mode='bilinear', align_corners=True)
        s5 = F.interpolate(s5, size=s3.shape[-2:], mode='bilinear', align_corners=True)
        
        # Content-guided attention fusion
        out = self.cga(s3, s4, s5)
        
        return out.contiguous()