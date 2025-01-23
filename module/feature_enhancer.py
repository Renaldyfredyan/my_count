import torch
import torch.nn as nn

class FeatureEnhancer(nn.Module):
    def __init__(self, embed_dim=256):
        super(FeatureEnhancer, self).__init__()
        
        # Self-attention layer for feature enhancement
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        
        # Feedforward network for refinement
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Assume x is [B, C, H, W]
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # Reshape to [B, H*W, C]

        # Self-attention
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_output)  # Residual connection + normalization

        # Feedforward network
        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output)  # Residual connection + normalization

        # Reshape back to [B, C, H, W]
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

# Test the FeatureEnhancer
if __name__ == "__main__":
    enhancer = FeatureEnhancer(embed_dim=256).cuda()
    dummy_input = torch.randn(1, 256, 14, 14).cuda()  # Example input image
    enhanced_features = enhancer(dummy_input)
    print("Enhanced Features Shape:", enhanced_features.shape)
