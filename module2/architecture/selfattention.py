import torch
from torch import nn
import math

class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = math.sqrt(out_channels // num_heads)
        
        # QKV projection
        self.qkv_proj = nn.Linear(in_channels, out_channels * 3)
        self.out_proj = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        # x shape: [h*w, batch, channel]
        B = x.size(1)  # batch size
        
        # QKV Projection
        qkv = self.qkv_proj(x)  # [h*w, batch, 3*channel]
        qkv = qkv.reshape(x.size(0), B, 3, self.num_heads, -1).permute(2, 1, 3, 0, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [batch, num_heads, h*w, head_dim]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        
        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(x.size(0), B, -1)
        x = self.out_proj(x)
        
        return x