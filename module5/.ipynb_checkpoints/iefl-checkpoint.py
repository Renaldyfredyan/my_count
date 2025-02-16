# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.ops import RoIAlign

# class ShapeMapper(nn.Module):
#     """Maps bounding box to shape embeddings"""
#     def __init__(self, input_dim=4, embed_dim=256):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, embed_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(embed_dim, embed_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(embed_dim, embed_dim)
#         )
    
#     def forward(self, bboxes):
#         """
#         Args:
#             bboxes: [B, K, 4] format: [x1, y1, x2, y2]
#         Returns:
#             shape_embeddings: [B, K, embed_dim]
#         """
#         return self.mlp(bboxes)

# class MHCA(nn.Module):
#     """Multi-Head Cross Attention"""
#     def __init__(self, dim, num_heads=8):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
        
#         self.q_proj = nn.Linear(dim, dim)
#         self.k_proj = nn.Linear(dim, dim)
#         self.v_proj = nn.Linear(dim, dim)
#         self.out_proj = nn.Linear(dim, dim)
        
#     def forward(self, query, key, value):
#         B, N, C = query.shape
        
#         # Project and reshape
#         q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = self.k_proj(key).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = self.v_proj(value).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
#         # Compute attention
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
        
#         # Apply attention to values
#         out = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         out = self.out_proj(out)
#         return out

# class LinearAttention(nn.Module):
#     """Linear attention layer"""
#     def __init__(self, dim):
#         super().__init__()
#         self.linear = nn.Linear(dim, dim)
        
#     def forward(self, x, Fi):
#         """
#         Args:
#             x: exemplar features [B, K, C]
#             Fi: image features [B, H*W, C]
#         """
#         x = self.linear(x)
#         attention = torch.bmm(x, Fi.transpose(1, 2))
#         attention = attention.softmax(dim=-1)
#         out = torch.bmm(attention, Fi)
#         return out

# class FeatureFusion(nn.Module):
#     """Feature fusion module (FF in the paper)"""
#     def __init__(self, dim):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Linear(dim * 2, dim),
#             nn.ReLU(inplace=True)
#         )
        
#     def forward(self, x1, x2):
#         return self.conv(torch.cat([x1, x2], dim=-1))

# class iEFL(nn.Module):
#     def __init__(self, dim=256, num_iterations=2):
#         super().__init__()
        
#         # Components for processing exemplars
#         self.shape_mapper = ShapeMapper(input_dim=4, embed_dim=dim)
#         self.roi_align = RoIAlign(
#             output_size=(7, 7),
#             spatial_scale=1.0/8.0,  # Because input feature map is H/8 x W/8
#             sampling_ratio=2
#         )
        
#         # Iterative adaptation components
#         self.mhca = MHCA(dim=dim)
#         self.linear_attention = LinearAttention(dim=dim)
#         self.feature_fusion = FeatureFusion(dim=dim)
        
#         # Layer normalization
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.norm3 = nn.LayerNorm(dim)
        
#         self.num_iterations = num_iterations
        
#     def forward(self, Fi, bboxes):
#         """
#         Args:
#             Fi: Enhanced image features [B, C, H/8, W/8]
#             bboxes: List of bounding boxes for each image [B, K, 4]
#         Returns:
#             F_E: Enhanced exemplar features
#         """
#         B = Fi.shape[0]
#         H, W = Fi.shape[2:]
        
#         # Reshape image features for attention
#         Fi_flat = Fi.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
#         # Get shape embeddings
#         shape_embed = self.shape_mapper(bboxes)  # [B, K, C]
        
#         # Get RoI features
#         roi_features = []
#         for b in range(B):
#             rois = torch.cat([torch.full((bboxes[b].size(0), 1), b, device=bboxes.device), 
#                             bboxes[b]], dim=1)
#             roi_feat = self.roi_align(Fi[b:b+1], rois)
#             roi_features.append(roi_feat.mean((2, 3)))  # Pool RoI features
#         roi_features = torch.cat(roi_features, dim=0)  # [B*K, C]
#         roi_features = roi_features.view(B, -1, Fi.shape[1])  # [B, K, C]
        
#         # Initialize exemplar features
#         F_exm = shape_embed
        
#         # Iterative adaptation
#         for _ in range(self.num_iterations):
#             # MHCA
#             F_tmp = F_exm
#             F_tmp = self.norm1(F_tmp)
#             F_tmp = self.mhca(F_tmp, roi_features, roi_features)
#             F_exm = F_exm + F_tmp
            
#             # Linear attention with image features
#             F_tmp = self.norm2(F_exm)
#             F_tmp = self.linear_attention(F_tmp, Fi_flat)
#             F_exm = F_exm + F_tmp
            
#             # Feature fusion
#             F_tmp = self.norm3(F_exm)
#             F_tmp = self.feature_fusion(F_tmp, roi_features)
#             F_exm = F_exm + F_tmp
        
#         return F_exm

# if __name__ == "__main__":
#     # Test implementation
#     batch_size = 2
#     num_exemplars = 3
#     dim = 256
#     H, W = 64, 64  # Feature map size after backbone (H/8, W/8)
    
#     # Create dummy inputs
#     Fi = torch.randn(batch_size, dim, H, W)
#     bboxes = torch.rand(batch_size, num_exemplars, 4)  # [x1, y1, x2, y2] format
    
#     # Initialize module
#     iefl = iEFL(dim=dim)
    
#     # Forward pass
#     output = iefl(Fi, bboxes)
    
#     print(f"Input shapes:")
#     print(f"Image features: {Fi.shape}")
#     print(f"Bounding boxes: {bboxes.shape}")
#     print(f"Output shape: {output.shape}  # Should be [B, K, dim]")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

class ShapeMapper(nn.Module):
    """Maps bounding box to shape embeddings"""
    def __init__(self, input_dim=4, embed_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, bboxes):
        return self.mlp(bboxes)

class MHCA(nn.Module):
    """Multi-Head Cross Attention"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, query, key, value):
        B, N, C = query.shape
        
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        return out

class LinearAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        
    def forward(self, x, Fi):
        x = self.linear(x)
        attention = torch.bmm(x, Fi.transpose(1, 2))
        attention = attention.softmax(dim=-1)
        out = torch.bmm(attention, Fi)
        return out

class FeatureFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x1, x2):
        return self.conv(torch.cat([x1, x2], dim=-1))

class iEFL(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        
        self.shape_mapper = ShapeMapper(input_dim=4, embed_dim=dim)
        self.roi_align = RoIAlign(
            output_size=(7, 7),
            spatial_scale=1.0/8.0,
            sampling_ratio=2
        )
        
        # Iterative adaptation components
        self.mhca = MHCA(dim=dim)
        self.linear_attention = LinearAttention(dim=dim)
        self.feature_fusion = FeatureFusion(dim=dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
    def forward(self, Fi, bboxes):
        """
        Args:
            Fi: Enhanced image features [B, C, H/8, W/8]
            bboxes: List of bounding boxes for each image [B, K, 4]
        Returns:
            list of F_E^k: [F_E^1, F_E^2, F_E^3] untuk setiap iterasi
        """
        B = Fi.shape[0]
        H, W = Fi.shape[2:]
        
        # Reshape image features for attention
        Fi_flat = Fi.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Get shape embeddings
        shape_embed = self.shape_mapper(bboxes)  # [B, K, C]
        
        # Get RoI features
        roi_features = []
        for b in range(B):
            rois = torch.cat([torch.full((bboxes[b].size(0), 1), b, device=bboxes.device), 
                            bboxes[b]], dim=1)
            roi_feat = self.roi_align(Fi[b:b+1], rois)
            roi_features.append(roi_feat.mean((2, 3)))
        roi_features = torch.cat(roi_features, dim=0)
        roi_features = roi_features.view(B, -1, Fi.shape[1])
        
        # Initialize exemplar features
        F_exm = shape_embed
        all_features = []
        
        # Perform 2 iterations to get F_E^1, F_E^2, F_E^3
        for _ in range(2):  # 2 iterations sesuai paper
            # 1. F_exm^k = MHCA(F_Si_exm, F_exm, F_exm) + F_Si_exm
            F_tmp = self.norm1(F_exm)
            F_tmp = self.mhca(F_tmp, roi_features, roi_features)
            F_exm = F_exm + F_tmp
            
            # Save intermediate feature
            all_features.append(F_exm.clone())
            
            # 2. F^k = MHCA(F_exm^k, Fi, Fi) + F_exm^k
            F_tmp = self.norm2(F_exm)
            F_tmp = self.linear_attention(F_tmp, Fi_flat)
            F_exm = F_exm + F_tmp
            
            # 3. F_exm = FF(F^k) + F^k
            F_tmp = self.norm3(F_exm)
            F_tmp = self.feature_fusion(F_tmp, roi_features)
            F_exm = F_exm + F_tmp
        
        # Add final feature F_E^3
        all_features.append(F_exm)
        
        return all_features  # [F_E^1, F_E^2, F_E^3]

if __name__ == "__main__":
    # Test implementation
    batch_size = 2
    num_exemplars = 3
    dim = 256
    H, W = 64, 64
    
    # Create dummy inputs
    Fi = torch.randn(batch_size, dim, H, W)
    bboxes = torch.rand(batch_size, num_exemplars, 4)
    
    # Initialize module
    iefl = iEFL(dim=dim)
    
    # Forward pass
    features = iefl(Fi, bboxes)
    
    print(f"\nInput shapes:")
    print(f"Image features: {Fi.shape}")
    print(f"Bounding boxes: {bboxes.shape}")
    print("\nOutput features:")
    for i, feat in enumerate(features, 1):
        print(f"F_E^{i} shape: {feat.shape}")