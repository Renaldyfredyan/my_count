import torch
import torch.nn as nn
import torch.nn.functional as F

class ExemplarImageMatching(nn.Module):
    def __init__(self, embed_dim=256, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.embed_dim = embed_dim
        
        # Regular image enhancement
        self.image_enhance = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Regular exemplar enhancement
        self.exemplar_enhance = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, image_features, exemplar_features):
        """
        Args:
            image_features: [B, C, H, W]
            exemplar_features: [B, N, C]
        Returns:
            similarity_maps: [B, N, H, W]
        """
        B, C, H, W = image_features.shape
        N = exemplar_features.shape[1]
        
        # Enhance features
        image_enhanced = self.image_enhance(image_features)  # [B, C, H, W]
        exemplar_enhanced = self.exemplar_enhance(exemplar_features)  # [B, N, C]
        
        # Reshape for matrix multiplication
        image_flat = image_enhanced.view(B, C, -1)  # [B, C, HW]
        
        # Compute similarity
        similarity = torch.bmm(exemplar_enhanced, image_flat)  # [B, N, HW]
        similarity = similarity.view(B, N, H, W)  # [B, N, H, W]
        
        # Apply temperature and normalize
        similarity = similarity / self.temperature
        similarity = F.softmax(similarity, dim=1)  # Normalize across exemplars
        
        return similarity

if __name__ == "__main__":
    # Test implementation
    B, N, C = 2, 3, 256  # Batch, Num exemplars, Channels
    H, W = 64, 64  # Spatial dimensions
    
    # Create dummy inputs
    image_features = torch.randn(B, C, H, W)
    exemplar_features = torch.randn(B, N, C)
    
    # Initialize module
    matcher = ExemplarImageMatching(embed_dim=C)
    
    # Forward pass
    with torch.no_grad():
        similarity_maps = matcher(image_features, exemplar_features)
        print(f"\nInput shapes:")
        print(f"Image features: {image_features.shape}")
        print(f"Exemplar features: {exemplar_features.shape}")
        print(f"Output similarity maps: {similarity_maps.shape}")
        
        # Verify output properties
        print(f"\nOutput properties:")
        print(f"Min value: {similarity_maps.min().item():.6f}")
        print(f"Max value: {similarity_maps.max().item():.6f}")
        print(f"Sum per position: {similarity_maps[0, :, 0, 0].sum().item():.6f}")  # Should be close to 1