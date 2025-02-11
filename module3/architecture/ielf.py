import torch
import torch.nn as nn
import torch.nn.functional as F

class ShapeMapping(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.ReLU()
        )

    def forward(self, height_width):
        return self.mlp(height_width)

# class ExemplarFeatureLearning(nn.Module):
#     def __init__(self, embed_dim=256, num_iterations=3):
#         super().__init__()
#         self.num_iterations = num_iterations
        
#         # Shape mapping for exemplar size information
#         self.shape_mapping = ShapeMapping(embed_dim)
        
#         # Multi-head cross attention for each iteration
#         self.cross_attention = nn.ModuleList([
#             nn.MultiheadAttention(embed_dim, 8, batch_first=True)
#             for _ in range(num_iterations)
#         ])
        
#         # Feature refinement for each iteration
#         self.refinement = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(embed_dim, embed_dim * 4),
#                 nn.ReLU(),
#                 nn.Linear(embed_dim * 4, embed_dim),
#                 nn.LayerNorm(embed_dim)
#             ) for _ in range(num_iterations)
#         ])

#     def forward(self, image_features, exemplar_features, exemplar_sizes):
#         B, N, C = exemplar_features.shape
        
#         # Map exemplar sizes to embeddings
#         size_embeddings = self.shape_mapping(exemplar_sizes)  # [B, N, C]
        
#         # Initial feature enhancement with size information
#         exemplar_features = exemplar_features + size_embeddings
        
#         # Iterative refinement
#         for i in range(self.num_iterations):
#             # Cross-attention between image and exemplar features
#             attn_out, _ = self.cross_attention[i](
#                 exemplar_features, image_features, image_features
#             )
            
#             # Refine features
#             exemplar_features = exemplar_features + attn_out
#             exemplar_features = self.refinement[i](exemplar_features)

#         return exemplar_features

class ExemplarFeatureLearning(nn.Module):
    def __init__(self, embed_dim=256, num_iterations=3):
        super().__init__()
        self.num_iterations = num_iterations
        
        # Cross-Attention layers for iterative updates
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)

        # Feedforward layers for refinement
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, image_features, exemplar_features):
        # Remove the exemplar_sizes parameter as we'll handle size information differently
        for _ in range(self.num_iterations):
            # Cross-attention between image features and exemplar features
            attn_output, _ = self.cross_attention(exemplar_features, image_features, image_features)
            exemplar_features = self.norm1(exemplar_features + attn_output)

            # Feedforward network for exemplar refinement
            ff_output = self.feedforward(exemplar_features)
            exemplar_features = self.norm2(exemplar_features + ff_output)

        return exemplar_features