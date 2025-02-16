import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import DensityEncoder
from feature_enhancer import FeatureEnhancer
from exemplar_feature_learning import ExemplarFeatureLearning
from similarity_maps import ExemplarImageMatching
from decoder import DensityRegressionDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F

class LowShotCounting(nn.Module):
    def __init__(
        self,
        num_iterations=3,
        embed_dim=256,
        temperature=0.1,
        backbone_type='swin'
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.embed_dim = embed_dim
        
        # Initialize components
        self.encoder = DensityEncoder(min_dim=embed_dim)
        
        self.exemplar_learner = ExemplarFeatureLearning(
            embed_dim=embed_dim,
            num_iterations=num_iterations
        )
        
        self.matcher = ExemplarImageMatching(
            embed_dim=embed_dim,
            temperature=temperature
        )
        
        self.decoders = nn.ModuleList([
            DensityRegressionDecoder(input_channels=num_exemplars) 
            for _ in range(num_iterations)
        ])
        
    def extract_features(self, x):
        """Extract features from input image"""
        # print(f"Input to extract_features: {x.shape}")
        features = self.encoder(x)
        # print(f"Output from encoder: {features.shape}")
        return features

    def process_exemplars(self, exemplars, boxes=None):
        """Process exemplar patches"""
        # print(f"Input to process_exemplars: {exemplars.shape}")
        B, K, C, H, W = exemplars.shape
        
        # Flatten exemplars
        exemplars_flat = exemplars.view(B * K, C, H, W)
        # print(f"Flattened exemplars: {exemplars_flat.shape}")
        
        # Extract features
        exemplar_features = self.encoder(exemplars_flat)
        # print(f"Exemplar features after encoder: {exemplar_features.shape}")
        
        # Global average pooling
        exemplar_features = F.adaptive_avg_pool2d(exemplar_features, (1, 1))
        exemplar_features = exemplar_features.view(B, K, -1)
        # print(f"Final exemplar features: {exemplar_features.shape}")
        
        return exemplar_features

    def forward(self, image, exemplars, boxes=None):
        print("\nStarting forward pass...")
        # print(f"Input image shape: {image.shape}")
        # print(f"Input exemplars shape: {exemplars.shape}")
        
        # Extract features
        image_features = self.extract_features(image)
        B, C, H, W = image_features.shape
        
        # Process exemplars
        exemplar_features = self.process_exemplars(exemplars, boxes)
        # print(f"Processed exemplar features: {exemplar_features.shape}")
        
        current_exemplars = exemplar_features
        all_density_maps = []
        
        # Iterative prediction
        for i in range(self.num_iterations):
            print(f"\nIteration {i+1}/{self.num_iterations}")
            
            # Prepare image features for exemplar learner
            img_features_flat = image_features.permute(0, 2, 3, 1)
            img_features_flat = img_features_flat.reshape(B, H * W, C)
            # print(f"Reshaped image features: {img_features_flat.shape}")
            
            # Update exemplar features
            current_exemplars = self.exemplar_learner(
                img_features_flat,
                current_exemplars,
                boxes
            )
            # print(f"Updated exemplar features: {current_exemplars.shape}")
            
            # Generate similarity maps
            similarity_maps = self.matcher(
                image_features,
                current_exemplars
            )
            # print(f"Similarity maps: {similarity_maps.shape}")
            
            # Predict density map
            density_map = self.decoders[i](similarity_maps)
            print(f"Predicted density map: {density_map.shape}")
            
            # Resize if needed
            if density_map.shape[2:] != image.shape[2:]:
                density_map = F.interpolate(
                    density_map,
                    size=image.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            print(f"Final density map shape: {density_map.shape}")
            
            all_density_maps.append(density_map)
        
        return all_density_maps

# Test code
if __name__ == "__main__":
    print("Creating test data...")
    batch_size = 2
    num_exemplars = 3
    image_size = 512
    exemplar_size = 128
    
    # Create dummy inputs
    image = torch.randn(batch_size, 3, image_size, image_size)
    exemplars = torch.randn(batch_size, num_exemplars, 3, exemplar_size, exemplar_size)
    boxes = torch.randn(batch_size, num_exemplars, 4)
    
    print("\nInitializing model...")
    model = LowShotCounting(num_iterations=3, embed_dim=256)
    
    print("\nRunning forward pass...")
    density_maps = model(image, exemplars, boxes)
    
    print("\nOutput summary:")
    print(f"Number of density maps: {len(density_maps)}")
    for i, dm in enumerate(density_maps):
        print(f"Density map {i} shape: {dm.shape}")