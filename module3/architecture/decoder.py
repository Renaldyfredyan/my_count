import torch
import torch.nn as nn
import torch.nn.functional as F

class DensityRegressionDecoder(nn.Module):
    def __init__(self, input_channels=5, scales=(0.25, 0.5, 1.0)):
        super().__init__()
        self.scales = scales
        
        # Multi-scale density prediction
        self.density_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, 1)
            ) for _ in scales
        ])
        
        # Feature refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(len(scales), 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, similarity_maps):
        B, N, H, W = similarity_maps.shape
        density_maps = []
        
        # Generate multi-scale density predictions
        for scale, predictor in zip(self.scales, self.density_predictors):
            # Scale input
            scaled_size = (int(H * scale), int(W * scale))
            scaled_maps = F.interpolate(
                similarity_maps, size=scaled_size, mode='bilinear'
            )
            
            # Predict density
            density = predictor(scaled_maps)
            
            # Upsample to original size
            density = F.interpolate(
                density, size=(H, W), mode='bilinear'
            )
            density_maps.append(density)
        
        # Combine multi-scale predictions
        combined = torch.cat(density_maps, dim=1)
        final_density = self.refinement(combined)
        
        return final_density