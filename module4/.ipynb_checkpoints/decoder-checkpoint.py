import torch
import torch.nn as nn
import torch.nn.functional as F

class DensityRegressionDecoder(nn.Module):
    def __init__(self, input_channels=3, min_density_value=1e-7):
        super().__init__()
        
        self.min_density_value = min_density_value
        
        # Input adaptation layer
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Progressive upsampling blocks (if needed)
        self.up_block = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final density prediction
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
    def forward(self, x):
        # Print input shape for debugging
        print(f"Decoder input shape: {x.shape}")
        
        # Initial convolution
        x = self.input_conv(x)
        print(f"After input conv: {x.shape}")
        
        # Feature processing
        x = self.up_block(x)
        print(f"After up block: {x.shape}")
        
        # Final prediction
        density_map = self.final_conv(x)
        print(f"Final density map: {density_map.shape}")
        
        # Ensure non-negativity
        density_map = F.relu(density_map) + self.min_density_value
        
        return density_map

# Test code
if __name__ == "__main__":
    # Test the decoder
    batch_size = 2
    input_channels = 3
    height, width = 512, 512
    
    # Create dummy input
    x = torch.randn(batch_size, input_channels, height, width)
    
    # Initialize decoder
    decoder = DensityRegressionDecoder(input_channels=input_channels)
    
    # Forward pass
    with torch.no_grad():
        output = decoder(x)
    
    print(f"\nTest summary:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Min value: {output.min().item():.6f}")
    print(f"Max value: {output.max().item():.6f}")