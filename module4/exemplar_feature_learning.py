import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined projections to reduce memory usage
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, query, key, value):
        B, L, C = query.shape
        
        # Project QKV together and split
        qkv = self.qkv_proj(query)  # [B, L, 3*C]
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, L, L]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.out_proj(x)
        
        return x

class ExemplarFeatureLearning(nn.Module):
    def __init__(self, embed_dim=256, num_iterations=2, num_heads=8):
        super().__init__()
        self.num_iterations = num_iterations
        self.embed_dim = embed_dim
        
        # First MHCA: exemplar-exemplar attention
        self.mhca1 = MultiHeadCrossAttention(
            dim=embed_dim,
            num_heads=num_heads
        )
        
        # Second MHCA: image-exemplar attention
        self.mhca2 = MultiHeadCrossAttention(
            dim=embed_dim,
            num_heads=num_heads
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, image_features, exemplar_features, bboxes=None):
        """
        Args:
            image_features: [B, H*W, C]
            exemplar_features: [B, K, C]
            bboxes: Optional
        Returns:
            Updated exemplar features: [B, K, C]
        """
        B, HW, C = image_features.shape
        K = exemplar_features.shape[1]
        
        # Current exemplar features
        F_exm = exemplar_features
        
        for _ in range(self.num_iterations):
            # First MHCA: exemplar-exemplar attention
            F_exm_norm = self.norm1(F_exm)
            F_tmp = self.mhca1(F_exm_norm, F_exm_norm, F_exm_norm)
            F_exm = F_exm + F_tmp
            
            # Second MHCA: image-exemplar attention
            F_exm_norm = self.norm2(F_exm)
            # Process image features in chunks to save memory
            chunk_size = 1024  # Adjust based on GPU memory
            F_tmp = []
            
            for start_idx in range(0, HW, chunk_size):
                end_idx = min(start_idx + chunk_size, HW)
                img_chunk = image_features[:, start_idx:end_idx, :]
                F_chunk = self.mhca2(F_exm_norm, img_chunk, img_chunk)
                F_tmp.append(F_chunk)
                torch.cuda.empty_cache()
            
            F_tmp = torch.mean(torch.stack(F_tmp, dim=0), dim=0)
            
            # Fuse features
            F_cat = torch.cat([F_exm, F_tmp], dim=-1)
            F_fused = self.fusion(F_cat)
            
            # Residual connection
            F_exm = F_exm + F_fused
            
            # Clean up
            del F_tmp, F_cat, F_fused
            torch.cuda.empty_cache()
        
        return F_exm

if __name__ == "__main__":
    # Test implementation
    batch_size = 2
    seq_len = 64 * 64  # Image feature sequence length
    num_exemplars = 3
    embed_dim = 256
    
    # Create dummy inputs
    image_features = torch.randn(batch_size, seq_len, embed_dim)
    exemplar_features = torch.randn(batch_size, num_exemplars, embed_dim)
    
    # Initialize module
    i_efl = ExemplarFeatureLearning(embed_dim=embed_dim)
    
    # Forward pass
    output = i_efl(image_features, exemplar_features)
    
    print(f"Input shapes:")
    print(f"Image features: {image_features.shape}")
    print(f"Exemplar features: {exemplar_features.shape}")
    print(f"Output shape: {output.shape}")