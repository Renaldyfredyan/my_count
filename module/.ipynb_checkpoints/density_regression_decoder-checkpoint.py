import torch
import torch.nn as nn
import torch.nn.functional as F

class DensityRegressionDecoder(nn.Module):
    def __init__(self, input_channels=5):
        super(DensityRegressionDecoder, self).__init__()

        # Convolutional layers to refine similarity maps into density maps
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)  # Output a single channel for density map
        )

    def forward(self, similarity_maps):
        # similarity_maps: [B, N, H, W]
        density_maps = self.conv_block(similarity_maps)  # [B, 1, H, W]
        return density_maps

# Test the DensityRegressionDecoder
if __name__ == "__main__":
    decoder = DensityRegressionDecoder(input_channels=5).cuda()
    dummy_similarity_maps = torch.randn(1, 5, 14, 14).cuda()  # Example similarity maps [B, N, H, W]

    density_maps = decoder(dummy_similarity_maps)
    print("Density Maps Shape:", density_maps.shape)
