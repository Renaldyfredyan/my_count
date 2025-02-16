import torch
import torch.nn as nn
import torch.nn.functional as F

# class DensityRegressionDecoder(nn.Module):
#     def __init__(self, num_exemplars):
#         """
#         Progressive up-sampling regression head untuk menghasilkan density map
#         Args:
#             num_exemplars: Jumlah exemplars, menentukan jumlah input channels
#         """
#         super().__init__()
        
#         # Progressive upsampling blocks
#         self.prog_up1 = nn.Sequential(
#             nn.Conv2d(num_exemplars, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         )
        
#         self.prog_up2 = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         )
        
#         self.prog_up3 = nn.Sequential(
#             nn.Conv2d(64, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         )
        
#         # Final density prediction
#         self.final_conv = nn.Sequential(
#             nn.Conv2d(32, 16, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 1, kernel_size=1),
#             nn.ReLU(inplace=True)  # Ensure non-negative density values
#         )
        
#     def forward(self, response_maps):
#         """
#         Args:
#             response_maps: Response maps dari exemplar-image matching [B, N, H, W]
#                          dimana N adalah jumlah exemplars
#         Returns:
#             density_map: Predicted density map [B, 1, H*8, W*8]
#         """
#         # # Progressive upsampling dan feature processing
#         # x = self.prog_up1(response_maps)    # 2x upsampling
#         # x = self.prog_up2(x)                # 4x upsampling
#         # x = self.prog_up3(x)                # 8x upsampling
        
#         print(f"Decoder input shape: {response_maps.shape}")
        
#         x = self.prog_up1(response_maps)
#         print(f"After prog_up1: {x.shape}")
#         x = self.prog_up2(x)
#         print(f"After prog_up2: {x.shape}")
#         x = self.prog_up3(x)
#         print(f"After prog_up3: {x.shape}")
        
#         # Final density prediction
#         density_map = self.final_conv(x)
        
#         return density_map
    
class DensityRegressionDecoder(nn.Module):
    def __init__(self, num_exemplars):
        super().__init__()
        
        # Progressive upsampling tanpa batch norm
        self.prog_up1 = nn.Sequential(
            nn.Conv2d(num_exemplars, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.prog_up2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.prog_up3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # Final density prediction
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU(inplace=True)  # Ensure non-negative density values
        )
        
    def forward(self, response_maps):
        """
        Args:
            response_maps: Response maps dari exemplar-image matching [B, N, H, W]
                         dimana N adalah jumlah exemplars
        Returns:
            density_map: Predicted density map [B, 1, H*8, W*8]
        """
        # Ensure input is in correct format
        if response_maps.dim() != 4:
            raise ValueError(f"Expected 4D input (B,N,H,W), got {response_maps.dim()}D")
            
        try:
            # Progressive upsampling dan feature processing
            x = self.prog_up1(response_maps)    # 2x upsampling
            x = self.prog_up2(x)                # 4x upsampling
            x = self.prog_up3(x)                # 8x upsampling
            
            # Final density prediction
            density_map = self.final_conv(x)
            
            return density_map
            
        except Exception as e:
            print(f"Error in decoder forward pass:")
            print(f"Input shape: {response_maps.shape}")
            print(f"Device: {response_maps.device}")
            print(f"Dtype: {response_maps.dtype}")
            raise e   

if __name__ == "__main__":
    # Test implementation
    batch_size = 2
    num_exemplars = 3
    height, width = 64, 64  # Input response map size
    
    # Create dummy response maps
    response_maps = torch.rand(batch_size, num_exemplars, height, width)
    
    # Initialize decoder
    decoder = DensityRegressionDecoder(num_exemplars=num_exemplars)
    
    # Forward pass
    with torch.no_grad():
        density_map = decoder(response_maps)
    
    print(f"\nTest summary:")
    print(f"Input response maps shape: {response_maps.shape}")
    print(f"Output density map shape: {density_map.shape}")
    
    # Verify density map properties
    print(f"Min density value: {density_map.min().item():.6f}")
    print(f"Max density value: {density_map.max().item():.6f}")
    print(f"Total count (sum of first density map): {density_map[0].sum().item():.2f}")
    
    # Verify progressive upsampling
    assert density_map.shape[-1] == response_maps.shape[-1] * 8, "Output should be 8x upsampled"
    print("✓ Progressive upsampling verified (8x)")