import torch
import torch.nn as nn
import torch.nn.functional as F

class ExemplarImageMatching(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        
    def conv_with_exemplar(self, image_features, exemplar_features):
        """
        Implement convolution dengan exemplar features sebagai kernel
        Args:
            image_features: [B, C, H, W]
            exemplar_features: [B, N, C]  # N adalah jumlah exemplars
        Returns:
            response_maps: [B, N, H, W]  # Harus mempertahankan spatial dimensions H,W
        """
        B, C, H, W = image_features.shape  # H,W digunakan untuk memastikan output shape benar
        N = exemplar_features.shape[1]
        
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        exemplar_features = F.normalize(exemplar_features, dim=2)
        
        kernel = exemplar_features.reshape(-1, C, 1, 1)
        response_maps = []
        
        for b in range(B):
            img_b = image_features[b:b+1]  # [1, C, H, W]
            kernel_b = kernel[b*N:(b+1)*N]  # [N, C, 1, 1]
            
            # Response map akan memiliki shape [1, N, H, W]
            # mempertahankan spatial dimensions H,W dari input
            resp = F.conv2d(img_b, kernel_b)  # Output harus [1, N, H, W]
            
            # Verifikasi output shape
            assert resp.shape == (1, N, H, W), f"Expected shape (1,{N},{H},{W}), got {resp.shape}"
            
            response_maps.append(resp)
        
        # Final shape: [B, N, H, W]
        response_maps = torch.cat(response_maps, dim=0)
        return response_maps
    def forward(self, image_features, exemplar_features):
        """
        Mengimplementasikan alur sesuai paper:
        1. Conv with exemplar kernel (*)
        2. Softmax and multiply (x)
        3. Generate response map (+)
        
        Args:
            image_features: [B, C, H, W]
            exemplar_features: [B, N, C]
        Returns:
            response_maps: [B, N, H, W]
        """
        # Step 1: Convolution dengan exemplar features sebagai kernel
        similarity = self.conv_with_exemplar(image_features, exemplar_features)
        
        # Step 2: Apply softmax dan multiply weights
        weights = F.softmax(similarity, dim=1)  # Softmax across exemplars
        
        # Response maps adalah weights hasil softmax
        response_maps = weights  # [B, N, H, W]
        
        return response_maps

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
    response_maps = matcher(image_features, exemplar_features)
    
    print(f"Input shapes:")
    print(f"Image features: {image_features.shape}")
    print(f"Exemplar features: {exemplar_features.shape}")
    print(f"Output response maps: {response_maps.shape}")
    
    # Verify response map properties
    print(f"\nResponse map properties:")
    print(f"Min value: {response_maps.min().item():.6f}")
    print(f"Max value: {response_maps.max().item():.6f}")