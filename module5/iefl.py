import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from debug_utils import print_tensor_info, print_gpu_usage

class MHCA(nn.Module):
    """Multi-Head Cross Attention"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divisible by num_heads {num_heads}'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, query, key, value):
        B, N, C = query.shape
        
        # Project
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape with explicit contiguous
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        
        # Scale q instead of scaling attention
        q = q * self.scale
        
        # Compute attention with better numerical stability
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        
        # Combine heads
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, C)
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

class ShapeMapper(nn.Module):
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

class iEFL(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        
        # Components for processing exemplars
        self.shape_mapper = ShapeMapper(input_dim=4, embed_dim=dim)
        self.num_exemplars = 3  # Assuming 3 exemplars as in paper
        
        # Iterative adaptation components
        self.mhca = MHCA(dim=dim)
        self.linear_attention = LinearAttention(dim=dim)
        self.feature_fusion = FeatureFusion(dim=dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def process_roi_features(self, Fi, bboxes):
        # print_gpu_usage("Before ROI processing")
        
        B = Fi.shape[0]
        # Prepare batch indices
        batch_indices = torch.arange(B, device=bboxes.device).repeat_interleave(self.num_exemplars)
        
        # Combine batch indices with bboxes
        rois = torch.cat([
            batch_indices.view(-1, 1),
            bboxes.reshape(-1, 4)
        ], dim=1)
        
        print_tensor_info("ROIs", rois)
        
        # Single RoI Align operation
        roi_features = roi_align(
            Fi,
            boxes=rois,
            output_size=(7, 7),
            spatial_scale=1.0/8.0,
            aligned=True
        )
        
        # print_tensor_info("ROI features before pooling", roi_features)
        
        # Pool features
        roi_features = roi_features.mean((2, 3))  # [B*K, C]
        roi_features = roi_features.view(B, -1, Fi.shape[1])  # [B, K, C]
        
        # print_tensor_info("Final ROI features", roi_features)
        # print_gpu_usage("After ROI processing")
        
        return roi_features
        
    def forward(self, Fi, bboxes):
        B = Fi.shape[0]
        H, W = Fi.shape[2:]
        
        # Reshape image features for attention
        Fi_flat = Fi.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Get shape embeddings
        shape_embed = self.shape_mapper(bboxes)  # [B, K, C]
        # print_tensor_info("Shape embeddings", shape_embed)
        
        # Get RoI features with efficient processing
        roi_features = self.process_roi_features(Fi, bboxes)
        
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
    batch_size = 4  # Try with batch size 4
    num_exemplars = 3
    dim = 256
    H, W = 64, 64
    
    # Create dummy inputs
    Fi = torch.randn(batch_size, dim, H, W).cuda()
    bboxes = torch.rand(batch_size, num_exemplars, 4).cuda()
    
    # Initialize module
    iefl = iEFL(dim=dim).cuda()
    
    # Forward pass
    features = iefl(Fi, bboxes)
    
    print(f"\nInput shapes:")
    print(f"Image features: {Fi.shape}")
    print(f"Bounding boxes: {bboxes.shape}")
    print("\nOutput features:")
    for i, feat in enumerate(features, 1):
        print(f"F_E^{i} shape: {feat.shape}")