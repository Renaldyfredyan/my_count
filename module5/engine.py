from backbone import FeatureExtractor
from hybrid_encoder import HybridEncoder
from iefl import iEFL
from image_matching import ExemplarImageMatching
from decoder import DensityRegressionDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F

class FSCModel(nn.Module):
    def __init__(self, num_exemplars=3):
        super().__init__()
        self.num_exemplars = num_exemplars
        
        # Initialize modules
        self.backbone = FeatureExtractor()
        self.hybrid_encoder = HybridEncoder()
        self.iefl = iEFL(dim=256)  # Removed num_iterations karena fixed di iEFL
        self.dropout = nn.Dropout(0.1)
        
        # Create matcher dan decoder untuk setiap iterasi
        self.matchers = nn.ModuleList([
            ExemplarImageMatching(feature_dim=256) 
            for _ in range(3)  # 3 iterasi (0,1,2) sesuai paper
        ])
        self.decoders = nn.ModuleList([
            DensityRegressionDecoder(num_exemplars=num_exemplars)
            for _ in range(3)
        ])
        
 

    def forward(self, image, bboxes):

        # 1. Feature extraction
        backbone_features = self.backbone(image)

        
        # 2. Feature enhancement
        Fi = self.hybrid_encoder(backbone_features)
        Fi = self.dropout(Fi)

        
        # 3. Get exemplar features
        exemplar_features = self.iefl(Fi, bboxes)

            
        # Simpan semua density maps
        density_maps = []
        
        # Iterative process
        for i in range(3):
            # Ensure tensors are on same device
            curr_feat = exemplar_features[i].to(Fi.device)
            
            # Exemplar-Image Matching
            response_maps = self.matchers[i](Fi, curr_feat)
   
            # Generate density map
            density_map = self.decoders[i](response_maps)

            density_maps.append(density_map)
        
        return density_maps

if __name__ == "__main__":
    # Test implementation
    batch_size = 2
    num_exemplars = 3
    image_size = 512
    
    # Create dummy inputs
    image = torch.randn(batch_size, 3, image_size, image_size)
    bboxes = torch.randn(batch_size, num_exemplars, 4)
    
    # Initialize model
    model = FSCModel(num_exemplars=num_exemplars)
    
    # Forward pass
    with torch.no_grad():
        density_maps = model(image, bboxes)
    
    print("\nTest results:")
    print(f"Input image shape: {image.shape}")
    print(f"Input bboxes shape: {bboxes.shape}")
    print("\nDensity maps:")
    for i, dmap in enumerate(density_maps):
        print(f"Iteration {i} density map shape: {dmap.shape}")