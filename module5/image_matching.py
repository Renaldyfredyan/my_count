import torch
import torch.nn as nn
import torch.nn.functional as F

class ExemplarImageMatching(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        
    # def conv_with_exemplar(self, image_features, exemplar_features):
    #     """
    #     Implement convolution dengan exemplar features sebagai kernel
    #     Args:
    #         image_features: [B, C, H, W]
    #         exemplar_features: [B, N, C]  # N adalah jumlah exemplars
    #     Returns:
    #         response_maps: [B, N, H, W]
    #     """
    #     B, C, H, W = image_features.shape
    #     N = exemplar_features.shape[1]
        
    #     # Reshape exemplar features untuk dijadikan kernel
    #     # [B, N, C] -> [B*N, C, 1, 1]
    #     kernel = exemplar_features.view(B*N, C, 1, 1)
        
    #     # Reshape image features untuk group convolution
    #     # [B, C, H, W] -> [1, B*C, H, W]
    #     img = image_features.view(1, B*C, H, W)
        
    #     # Perform group convolution
    #     # Setiap exemplar akan beroperasi pada channel group yang sesuai
    #     response = F.conv2d(
    #         img,
    #         kernel,
    #         groups=B,  # Setiap batch diproses secara independen
    #         stride=1,
    #         padding=0
    #     )
        
    #     # Reshape hasil kembali ke format [B, N, H, W]
    #     response = response.view(B, N, H, W)
    #     return response

    def conv_with_exemplar(self, image_features, exemplar_features):
        """
        Implement convolution dengan exemplar features sebagai kernel
        Args:
            image_features: [B, C, H, W]
            exemplar_features: [B, N, C]  # N adalah jumlah exemplars
        Returns:
            response_maps: [B, N, H, W]
        """
        B, C, H, W = image_features.shape
        N = exemplar_features.shape[1]
        
        # Reshape exemplar features untuk dijadikan kernel
        # [B, N, C] -> [B*N, C, 1, 1]
        kernel = exemplar_features.reshape(-1, C, 1, 1)
        
        # Process each batch separately to avoid view issues
        response_maps = []
        for b in range(B):
            # Extract single batch
            img_b = image_features[b:b+1]  # [1, C, H, W]
            kernel_b = kernel[b*N:(b+1)*N]  # [N, C, 1, 1]
            
            # Compute response maps for this batch
            resp = F.conv2d(
                img_b,      # [1, C, H, W]
                kernel_b,   # [N, C, 1, 1]
                groups=1    # No grouping needed as we process per batch
            )  # Output: [1, N, H, W]
            response_maps.append(resp)
        
        # Concatenate all batches
        response_maps = torch.cat(response_maps, dim=0)  # [B, N, H, W]
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