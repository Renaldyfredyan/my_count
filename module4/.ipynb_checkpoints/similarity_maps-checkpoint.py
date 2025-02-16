import torch
import torch.nn as nn
import torch.nn.functional as F

class ExemplarImageMatching(nn.Module):
    def __init__(self, embed_dim=256, temperature=0.1):  # Changed feature_dim to embed_dim
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        
        # Feature enhancement for both image and exemplar features
        self.image_enhance = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        self.exemplar_enhance = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Dimension reduction for matching
        self.dim_reduce = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, image_features, exemplar_features):
        """
        Args:
            image_features: Tensor of shape [B, C, H, W]
            exemplar_features: Tensor of shape [B, N, C]
        Returns:
            similarity_maps: Tensor of shape [B, N, H, W]
        """
        B, C, H, W = image_features.shape
        N = exemplar_features.shape[1]
        
        # Enhance image features
        enhanced_image = self.image_enhance(image_features)
        
        # Enhance exemplar features
        enhanced_exemplars = self.exemplar_enhance(exemplar_features)  # [B, N, C]
        
        # Compute similarity maps
        similarity_maps = []
        for i in range(N):
            # Extract current exemplar features
            curr_exemplar = enhanced_exemplars[:, i]  # [B, C]
            
            # Reshape exemplar features to match image feature dimensions
            curr_exemplar = curr_exemplar.view(B, C, 1, 1).expand(-1, -1, H, W)
            
            # Concatenate and reduce dimension
            concat_features = torch.cat([enhanced_image, curr_exemplar], dim=1)
            fused_features = self.dim_reduce(concat_features)
            
            # Compute similarity
            similarity = torch.sum(fused_features ** 2, dim=1)  # [B, H, W]
            
            # Apply temperature scaling
            similarity = (similarity / self.temperature)
            similarity_maps.append(similarity.unsqueeze(1))  # [B, 1, H, W]
        
        # Concatenate all similarity maps
        similarity_maps = torch.cat(similarity_maps, dim=1)  # [B, N, H, W]
        
        # Normalize across spatial dimensions for each exemplar
        similarity_maps = F.softmax(similarity_maps.view(B, N, -1), dim=2).view(B, N, H, W)
        
        return similarity_maps

if __name__ == "__main__":
    # Test implementation
    batch_size = 2
    num_exemplars = 3
    embed_dim = 256
    height, width = 64, 64
    
    # Create dummy inputs
    image_features = torch.randn(batch_size, embed_dim, height, width)
    exemplar_features = torch.randn(batch_size, num_exemplars, embed_dim)
    
    # Initialize module
    matcher = ExemplarImageMatching(embed_dim=embed_dim)
    
    # Forward pass
    similarity_maps = matcher(image_features, exemplar_features)
    
    print(f"Input shapes:")
    print(f"Image features: {image_features.shape}")
    print(f"Exemplar features: {exemplar_features.shape}")
    print(f"Output similarity maps: {similarity_maps.shape}")
    
    # Verify similarity map properties
    print(f"\nSimilarity map properties:")
    print(f"Min value: {similarity_maps.min().item():.6f}")
    print(f"Max value: {similarity_maps.max().item():.6f}")
    print(f"Sum per position: {similarity_maps[0, :, 0, 0].sum().item():.6f}")  # Should be close to 1