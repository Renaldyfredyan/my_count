# import torch
import torch.nn as nn

class FeatureEnhancer(nn.Module):
    def __init__(self, embed_dim=256):
        super(FeatureEnhancer, self).__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, embed_dim, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Feature refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Channel attention
        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights
        
        # Spatial attention
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        x_norm = self.norm(x_flat)
        attn_out, _ = self.spatial_attention(x_norm, x_norm, x_norm)
        x_spatial = attn_out.permute(0, 2, 1).view(B, C, H, W)
        
        # Combine and refine
        enhanced = self.refinement(x_channel + x_spatial)
        return enhanced

# Test functionality
if __name__ == "__main__":
    enhancer = FeatureEnhancer(embed_dim=256).cuda()
    dummy_input = torch.randn(1, 256, 14, 14).cuda()
    enhanced_features = enhancer(dummy_input)
    print("Enhanced Features Shape:", enhanced_features.shape)