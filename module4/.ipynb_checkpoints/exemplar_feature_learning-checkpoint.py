import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, query, key, value):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        
        # Project and reshape
        q = self.q_proj(query).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        out = self.out_proj(out)
        
        return out

class ExemplarFeatureLearning(nn.Module):
    def __init__(self, embed_dim=256, num_iterations=2):
        super().__init__()
        self.num_iterations = num_iterations
        
        # Shape embedding MLP
        self.shape_embedding = MLP(
            input_dim=4,  # bbox coordinates [x1, y1, x2, y2]
            hidden_dim=256,
            output_dim=embed_dim
        )
        
        # Multi-head cross attention modules
        self.mhca1 = MultiHeadCrossAttention(dim=embed_dim)
        self.mhca2 = MultiHeadCrossAttention(dim=embed_dim)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, image_features, exemplar_features, exemplar_boxes=None):
        """
        Args:
            image_features: Tensor of shape [B, HW, C]
            exemplar_features: Tensor of shape [B, K, C]
            exemplar_boxes: Optional tensor of shape [B, K, 4] for bounding boxes
        Returns:
            Updated exemplar features of shape [B, K, C]
        """
        B, K, C = exemplar_features.shape
        
        # Initialize F_exm with exemplar features
        F_exm = exemplar_features
        
        # If boxes provided, get shape embeddings
        if exemplar_boxes is not None:
            shape_features = self.shape_embedding(exemplar_boxes)
            F_exm = F_exm + shape_features
        
        # Iterative feature enhancement
        for _ in range(self.num_iterations):
            # First MHCA: exemplar-exemplar attention
            F_exm_norm = self.norm1(F_exm)
            F_tmp = self.mhca1(F_exm_norm, F_exm_norm, F_exm_norm)
            F_exm = F_exm + F_tmp
            
            # Second MHCA: image-exemplar attention
            F_exm_norm = self.norm2(F_exm)
            F_tmp = self.mhca2(F_exm_norm, image_features, image_features)
            
            # Fuse features
            F_cat = torch.cat([F_exm, F_tmp], dim=-1)
            F_fused = self.fusion(F_cat)
            
            # Update exemplar features
            F_exm = F_exm + F_fused
        
        return F_exm

if __name__ == "__main__":
    # Test the implementation
    batch_size = 2
    num_exemplars = 3
    image_size = 64 * 64  # For 64x64 feature map
    embed_dim = 256
    
    # Create dummy inputs
    image_features = torch.randn(batch_size, image_size, embed_dim)
    exemplar_features = torch.randn(batch_size, num_exemplars, embed_dim)
    exemplar_boxes = torch.randn(batch_size, num_exemplars, 4)
    
    # Initialize module
    i_efl = ExemplarFeatureLearning(embed_dim=embed_dim)
    
    # Forward pass
    output = i_efl(image_features, exemplar_features, exemplar_boxes)
    
    print(f"Input shapes:")
    print(f"Image features: {image_features.shape}")
    print(f"Exemplar features: {exemplar_features.shape}")
    print(f"Exemplar boxes: {exemplar_boxes.shape}")
    print(f"Output shape: {output.shape}")