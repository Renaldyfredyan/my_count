import torch
from torch import nn
from torch.nn import functional as F

class IterativeExemplarFeatureLearning(nn.Module):
    def __init__(self, emb_dim, num_heads, num_iterations):
        super(IterativeExemplarFeatureLearning, self).__init__()
        self.num_iterations = num_iterations
        self.cross_attn = nn.MultiheadAttention(emb_dim, num_heads)
        self.norm = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim * 4, emb_dim)
        )

    def forward(self, exemplar_features, image_features):
        for _ in range(self.num_iterations):
            # Apply cross-attention between exemplar features and image features
            exemplar_features, _ = self.cross_attn(exemplar_features, image_features, image_features)
            exemplar_features = self.norm(exemplar_features)
            
            # Apply feed-forward network to exemplar features
            exemplar_features = self.ff(exemplar_features)
            exemplar_features = self.norm(exemplar_features)

        return exemplar_features