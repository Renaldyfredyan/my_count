import torch
from torch import nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.qkv_proj = nn.Linear(in_channels, out_channels * 3)
        self.out_proj = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        # Reshape input to (batch_size, num_patches, in_channels)
        x = x.flatten(2).transpose(1, 2)
        
        # Apply self-attention
        qkv = self.qkv_proj(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, self.out_channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], self.out_channels)
        
        # Apply output projection
        x = self.out_proj(x)
        
        # Reshape output to (batch_size, out_channels, height, width)
        x = x.transpose(1, 2).view(x.shape[0], self.out_channels, int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])))
        
        return x