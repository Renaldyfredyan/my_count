import torch
from torch import nn
from torch.nn import functional as F
import math

# class IterativeExemplarFeatureLearning(nn.Module):
#     def __init__(self, emb_dim, num_heads, num_iterations):
#         super(IterativeExemplarFeatureLearning, self).__init__()
#         self.num_iterations = num_iterations
#         self.cross_attn = nn.MultiheadAttention(emb_dim, num_heads)
#         self.norm = nn.LayerNorm(emb_dim)
#         self.ff = nn.Sequential(
#             nn.Linear(emb_dim, emb_dim * 4),
#             nn.ReLU(inplace=True),
#             nn.Linear(emb_dim * 4, emb_dim)
#         )

#     def forward(self, exemplar_features, image_features):
#         for _ in range(self.num_iterations):
#             # Apply cross-attention between exemplar features and image features
#             exemplar_features, _ = self.cross_attn(exemplar_features, image_features, image_features)
#             exemplar_features = self.norm(exemplar_features)
            
#             # Apply feed-forward network to exemplar features
#             exemplar_features = self.ff(exemplar_features)
#             exemplar_features = self.norm(exemplar_features)

#         return exemplar_features

class IterativeExemplarFeatureLearning(nn.Module):
    def __init__(self, emb_dim=256, num_heads=8, num_iterations=2):
        super().__init__()
        
        # Project exemplar features
        self.exemplar_proj = nn.Conv2d(emb_dim * 3, emb_dim, 1)
        
        # Cross attention dengan normalized attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Feature refinement dengan residual
        self.refine_net = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim * 4, emb_dim)
        )
        
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, exemplar_features, image_features):
        batch_size = exemplar_features.size(0)
        
        # Project exemplar features
        exemplar_features = self.exemplar_proj(exemplar_features)
        
        # Reshape untuk attention
        ex_h, ex_w = exemplar_features.shape[-2:]
        im_h, im_w = image_features.shape[-2:]
        
        exemplar_seq = exemplar_features.flatten(2).transpose(1, 2)
        image_seq = image_features.flatten(2).transpose(1, 2)
        
        # Iterative refinement
        for _ in range(2):  # Paper menggunakan 2 iterasi
            # Self attention
            q = self.norm1(exemplar_seq)
            k = self.norm1(image_seq)
            
            # Cross attention dengan scaled dot product
            attn_output, _ = self.cross_attn(q, k, image_seq)
            exemplar_seq = exemplar_seq + self.dropout(attn_output)
            
            # Feature refinement
            refined = self.refine_net(self.norm2(exemplar_seq))
            exemplar_seq = exemplar_seq + self.dropout(refined)
        
        # Reshape kembali ke spatial dimensions
        exemplar_features = exemplar_seq.transpose(1, 2).reshape(
            batch_size, -1, ex_h, ex_w)
            
        return exemplar_features