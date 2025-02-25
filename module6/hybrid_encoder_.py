from mlp import MLP
import torch
from torch import nn
import torch.nn.functional as F

class ContentGuidedAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Query, Key, Value projections for each feature level
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        
        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
        # Scaling factor for attention
        self.scale = channels ** -0.5
        
        # Layer normalization for each feature level
        self.norm1 = nn.LayerNorm([channels])
        self.norm2 = nn.LayerNorm([channels])
        self.norm3 = nn.LayerNorm([channels])

    def forward(self, s3, s4, s5):
        # B, C, H, W = s3.shape
        
        # # Normalize features
        # s3 = self.norm1(s3.flatten(2).transpose(-1, -2)).transpose(-1, -2).view(B, C, H, W)
        # s4 = self.norm2(s4.flatten(2).transpose(-1, -2)).transpose(-1, -2).view(B, C, H, W)
        # s5 = self.norm3(s5.flatten(2).transpose(-1, -2)).transpose(-1, -2).view(B, C, H, W)
            
        # Pastikan format dan memory layout konsisten
        B, C, H, W = s3.shape
        
        s3 = self.norm1(s3.flatten(2).transpose(-1, -2)).transpose(-1, -2).reshape(B, C, H, W).contiguous()
        s4 = self.norm2(s4.flatten(2).transpose(-1, -2)).transpose(-1, -2).reshape(B, C, H, W).contiguous()
        s5 = self.norm3(s5.flatten(2).transpose(-1, -2)).transpose(-1, -2).reshape(B, C, H, W).contiguous()
        
        # Compute Q, K, V for each level
        q = self.q_proj(s3)
        k = torch.cat([self.k_proj(s4), self.k_proj(s5)], dim=2)  # Concatenate along height
        v = torch.cat([self.v_proj(s4), self.v_proj(s5)], dim=2)
        
        # Reshape for attention computation
        q = q.flatten(2).transpose(-1, -2).contiguous()   # B, HW, C
        k = k.flatten(2).transpose(-1, -2).contiguous()   # B, 2HW, C
        v = v.flatten(2).transpose(-1, -2).contiguous()   # B, 2HW, C
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)
        
        # Reshape back to spatial dimensions
        out = out.transpose(-1, -2).reshape(B, C, H, W).contiguous() 
        
        # Final projection
        out = self.out_proj(out)
        
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
        
        # Add projection layers
        self.conv_high = nn.Conv2d(768, emb_dim, 1)  # Project S5 ke emb_dim
        
        # Rest of initialization remains same
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
        # s5 shape: [B, H, W, C]
        # Transpose ke format BCHW
        bs = s5.size(0)
        s5 = s5.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
        
        # Project ke embedding dimension yang sesuai
        s5 = self.conv_high(s5)  # [B, 768, H, W] -> [B, emb_dim, H, W]
        
        # Reshape untuk self-attention
        h, w = s5.shape[-2:]
        s5 = s5.flatten(2).transpose(1, 2).contiguous()  # [B, C, H*W] -> [B, H*W, C]
        s5 = s5.transpose(0, 1).contiguous()  # [B, H*W, C] -> [H*W, B, C]

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

        # Reshape kembali
        s5 = s5.transpose(0, 1).contiguous()  # [H*W, B, C] -> [B, H*W, C]
        s5 = s5.transpose(1, 2).contiguous()  # [B, H*W, C] -> [B, C, H*W]
        s5 = s5.reshape(bs, -1, h, w).contiguous()  # [B, C, H*W] -> [B, C, H, W]
                
        # 2. Cross-scale fusion
        out = self.fusion_module(s3, s4, s5)
            
        return out


class CrossScaleFusion(nn.Module):
    def __init__(self, low_channels, mid_channels, high_channels, out_channels):
        super().__init__()
        
        # Conv layers untuk menyesuaikan channel dimensions
        self.conv_low = nn.Conv2d(low_channels, out_channels, 1)
        self.conv_mid = nn.Conv2d(mid_channels, out_channels, 1)
        # Hapus conv_high karena s5 sudah diproyeksi
        # self.conv_high = nn.Conv2d(high_channels, out_channels, 1)  # Tidak perlu lagi
        
        # Content-Guided Attention (CGA)
        self.cga = ContentGuidedAttention(out_channels)
        
    # def forward(self, s3, s4, s5):
    #     # Permute s3 dan s4 ke format BCHW
    #     s3 = s3.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    #     s4 = s4.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    #     # s5 sudah dalam format BCHW dan sudah diproyeksi dari HybridEncoder
        
    #     # Project s3 dan s4 ke dimensi yang sesuai
    #     s3 = self.conv_low(s3)
    #     s4 = self.conv_mid(s4)
    #     # s5 sudah diproyeksi, tidak perlu conv_high
        
    #     # Upsample s4 dan s5 ke ukuran s3
    #     s4 = F.interpolate(s4, size=s3.shape[-2:], mode='bilinear', align_corners=True)
    #     s5 = F.interpolate(s5, size=s3.shape[-2:], mode='bilinear', align_corners=True)
        
    #     # Apply CGA fusion
    #     out = self.cga(s3, s4, s5)
        
    #     return out

    def forward(self, s3, s4, s5):
        # Pastikan format konsisten
        s3 = s3.permute(0, 3, 1, 2).contiguous()
        s4 = s4.permute(0, 3, 1, 2).contiguous()
        
        # Project ke dimensi yang sama
        s3 = self.conv_low(s3)
        s4 = self.conv_mid(s4)
        
        # Upsample dengan format yang konsisten
        s4 = F.interpolate(s4, size=s3.shape[-2:], mode='bilinear', align_corners=True)
        s5 = F.interpolate(s5, size=s3.shape[-2:], mode='bilinear', align_corners=True)
        
        # CGA fusion
        out = self.cga(s3, s4, s5)
        
        return out.contiguous()