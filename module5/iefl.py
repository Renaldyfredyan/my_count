import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

class MHCA(nn.Module):
    """Multi-Head Cross Attention sesuai paper"""
    def __init__(self, dim=256, num_heads=8):
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
        
        # Reshape and scale
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        
        q = q * self.scale
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        
        # Combine heads
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out

class FeatureFusion(nn.Module):
    """Feature Fusion module sesuai paper"""
    def __init__(self, dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x1, x2):
        return self.conv(torch.cat([x1, x2], dim=-1))

class ShapeMapper(nn.Module):
    """Shape mapping MLP sesuai paper"""
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
        # bboxes: [B, K, 4] -> output: [B, K, embed_dim]
        return self.mlp(bboxes)

class iEFL(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        
        # Shape mapper untuk exemplar boxes
        self.shape_mapper = ShapeMapper(input_dim=4, embed_dim=dim)
        self.num_exemplars = 3  # K=3 sesuai paper
        
        # MHCA modules
        self.mhca1 = MHCA(dim=dim)  # Untuk F_exm^k
        self.mhca2 = MHCA(dim=dim)  # Untuk F̂^k
        
        # Feature Fusion
        self.feature_fusion = FeatureFusion(dim=dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def process_roi_features(self, Fi, bboxes):
        """Extract ROI features with proper spatial scale"""
        B = Fi.shape[0]
        batch_indices = torch.arange(B, device=bboxes.device).repeat_interleave(self.num_exemplars)
        
        rois = torch.cat([
            batch_indices.view(-1, 1),
            bboxes.reshape(-1, 4)
        ], dim=1)
        
        # RoIAlign dengan spatial_scale=1/8 sesuai paper
        roi_features = roi_align(
            Fi,                     # [B, 256, 64, 64] dari hybrid encoder
            boxes=rois,            
            output_size=(7, 7),     # Output size sesuai paper
            spatial_scale=1.0/8.0,  # Scale factor karena Fi sudah di-downsample
            aligned=True
        )
        
        # Pool features dan reshape
        roi_features = roi_features.mean((2, 3))  # [B*K, 256]
        roi_features = roi_features.view(B, self.num_exemplars, -1)  # [B, K, 256]
        
        return roi_features

    def forward(self, Fi, bboxes):
        """
        Args:
            Fi: Image features dari hybrid encoder [B, 256, 64, 64]
            bboxes: Exemplar bounding boxes [B, K, 4]
        Returns:
            List of exemplar features [F_E^1, F_E^2, F_E^3]
        """
        B = Fi.shape[0]
        
        # Reshape image features untuk attention
        Fi_flat = Fi.flatten(2).transpose(1, 2)  # [B, H*W, 256]
        
        # Get shape embeddings
        F_exm_S = self.shape_mapper(bboxes)  # [B, K, 256]

        # Get ROI features
        F_exm = self.process_roi_features(Fi, bboxes)  # [B, K, 256]
        
        # Store all iterative features
        all_features = []
        F_exm_k = F_exm_S  # Initialize dengan shape features
        
        # Perform 2 iterations sesuai paper
        for _ in range(2):
            # 1. F_exm^k = MHCA(F_exm^S, F_exm, F_exm) + F_exm^S
            tmp = self.norm1(F_exm_k)
            tmp = self.mhca1(tmp, F_exm, F_exm)
            F_exm_k = F_exm_k + tmp
            
            all_features.append(F_exm_k.clone())
            
            # 2. F̂^k = MHCA(F_exm^k, F_I, F_I) + F_exm^k
            tmp = self.norm2(F_exm_k)
            tmp = self.mhca2(tmp, Fi_flat, Fi_flat)
            F_hat_k = F_exm_k + tmp
            
            # 3. F_exm^S(k+1) = FF(F̂^k) + F̂^k
            tmp = self.norm3(F_hat_k)
            tmp = self.feature_fusion(tmp, F_exm)
            F_exm_k = F_hat_k + tmp
        
        # Add final feature
        all_features.append(F_exm_k)
        
        return all_features  # [F_E^1, F_E^2, F_E^3]

if __name__ == "__main__":
    # Test implementation
    batch_size = 2
    num_exemplars = 3
    dim = 256
    H, W = 64, 64
    
    # Create dummy inputs yang sesuai dengan output hybrid encoder
    Fi = torch.randn(batch_size, dim, H, W).cuda()
    bboxes = torch.rand(batch_size, num_exemplars, 4).cuda()
    
    # Initialize module
    iefl = iEFL(dim=dim).cuda()
    
    # Forward pass
    features = iefl(Fi, bboxes)
    
    print(f"\nInput shapes:")
    print(f"Image features (Fi): {Fi.shape}")
    print(f"Bounding boxes: {bboxes.shape}")
    print("\nOutput features:")
    for i, feat in enumerate(features, 1):
        print(f"F_E^{i} shape: {feat.shape}")