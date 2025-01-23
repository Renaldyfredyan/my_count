import torch
import torch.nn as nn
import torch.nn.functional as F

class ExemplarFeatureLearning(nn.Module):
    def __init__(self, embed_dim=256, num_iterations=3):
        super(ExemplarFeatureLearning, self).__init__()
        self.num_iterations = num_iterations

        # Cross-Attention layers for iterative updates
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)

        # Feedforward layers for refinement
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # LayerNorm for normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, image_features, exemplar_features):
        # image_features: [B, H*W, C], exemplar_features: [B, N, C]
        for _ in range(self.num_iterations):
            # Cross-attention between image features and exemplar features
            attn_output, _ = self.cross_attention(exemplar_features, image_features, image_features)
            exemplar_features = self.norm1(exemplar_features + attn_output)  # Residual connection + normalization

            # Feedforward network for exemplar refinement
            ff_output = self.feedforward(exemplar_features)
            exemplar_features = self.norm2(exemplar_features + ff_output)  # Residual connection + normalization

        return exemplar_features

# Test the ExemplarFeatureLearning
if __name__ == "__main__":
    learner = ExemplarFeatureLearning(embed_dim=256, num_iterations=3).cuda()
    dummy_image_features = torch.randn(1, 196, 256).cuda()  # Example image features [B, H*W, C]
    dummy_exemplar_features = torch.randn(1, 5, 256).cuda()  # Example exemplar features [B, N, C]

    updated_exemplar_features = learner(dummy_image_features, dummy_exemplar_features)
    print("Updated Exemplar Features Shape:", updated_exemplar_features.shape)
