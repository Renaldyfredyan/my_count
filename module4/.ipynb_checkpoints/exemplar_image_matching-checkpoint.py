import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityModule(nn.Module):
    def __init__(self, in_channels, hidden_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Dimension reduction for input features if needed
        self.dim_reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Non-local block for capturing long-range dependencies
        self.theta = nn.Conv2d(hidden_dim, hidden_dim // 2, 1)
        self.phi = nn.Conv2d(hidden_dim, hidden_dim // 2, 1)
        self.g = nn.Conv2d(hidden_dim, hidden_dim, 1)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Dimension reduction
        x = self.dim_reduce(x)
        
        # Non-local operations
        theta = self.theta(x).view(batch_size, -1, x.shape[2] * x.shape[3])
        phi = self.phi(x).view(batch_size, -1, x.shape[2] * x.shape[3])
        g = self.g(x).view(batch_size, -1, x.shape[2] * x.shape[3])
        
        # Compute similarity
        similarity = torch.matmul(theta.permute(0, 2, 1), phi)
        similarity = F.softmax(similarity, dim=-1)
        
        # Apply attention
        out = torch.matmul(g, similarity.permute(0, 2, 1))
        out = out.view(batch_size, -1, x.shape[2], x.shape[3])
        
        return self.out_proj(out)

class ExemplarImageMatching(nn.Module):
    def __init__(self, feature_dim=256, temperature=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # Feature enhancement for both image and exemplar features
        self.image_enhance = SimilarityModule(feature_dim)
        self.exemplar_enhance = SimilarityModule(feature_dim)
        
        # Dimension reduction for matching
        self.dim_reduce = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
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
        
        # Reshape and enhance exemplar features
        exemplar_features = exemplar_features.view(B * N, C, 1, 1)
        exemplar_features = F.interpolate(exemplar_features, size=(H, W), mode='bilinear', align_corners=False)
        enhanced_exemplars = self.exemplar_enhance(exemplar_features)
        enhanced_exemplars = enhanced_exemplars.view(B, N, C, H, W)
        
        # Compute similarity maps
        similarity_maps = []
        for i in range(N):
            # Extract current exemplar features
            curr_exemplar = enhanced_exemplars[:, i]  # [B, C, H, W]
            
            # Concatenate and reduce dimension
            concat_features = torch.cat([enhanced_image, curr_exemplar], dim=1)
            fused_features = self.dim_reduce(concat_features)
            
            # Compute similarity
            fused_features = fused_features.view(B, self.feature_dim, -1)  # [B, C, HW]
            similarity = torch.sum(fused_features ** 2, dim=1)  # [B, HW]
            similarity = similarity.view(B, H, W)  # [B, H, W]
            
            # Apply temperature scaling and normalization
            similarity = (similarity / self.temperature).softmax(dim=-1)
            similarity_maps.append(similarity.unsqueeze(1))  # [B, 1, H, W]
        
        # Concatenate all similarity maps
        similarity_maps = torch.cat(similarity_maps, dim=1)  # [B, N, H, W]
        
        # Optional: normalize across exemplars
        similarity_maps = F.softmax(similarity_maps.view(B, N, -1), dim=1).view(B, N, H, W)
        
        return similarity_maps

if __name__ == "__main__":
    # Test implementation
    batch_size = 2
    num_exemplars = 3
    feature_dim = 256
    height, width = 64, 64
    
    # Create dummy inputs
    image_features = torch.randn(batch_size, feature_dim, height, width)
    exemplar_features = torch.randn(batch_size, num_exemplars, feature_dim)
    
    # Initialize module
    matcher = ExemplarImageMatching(feature_dim=feature_dim)
    
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