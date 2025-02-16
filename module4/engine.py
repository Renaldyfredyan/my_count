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
        backbone_type='swin',
        num_exemplars=3  # Added num_exemplars parameter
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.embed_dim = embed_dim
        self.num_exemplars = num_exemplars
        
        # Initialize components
        self.encoder = DensityEncoder(min_dim=embed_dim)
        self.feature_enhancer = FeatureEnhancer(dims=(256, 512, 1024))  # Sesuai dengan output encoder
        
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
        # """Extract features from input image"""
        # print(f"Input to extract_features: {x.shape}")
        features = self.encoder(x)
        # print(f"Output from encoder: {features.shape}")
        return features

    def extract_exemplar_patches(self, image, bboxes):
        B, C, H, W = image.shape
        K = bboxes.shape[1]
        patch_size = 128

        # Bboxes sudah dalam format absolut, langsung round dan convert ke long
        bboxes = bboxes.round().long()
        
        patches = []
        for b in range(B):
            batch_patches = []
            for k in range(K):
                x1, y1, x2, y2 = bboxes[b, k]
                
                # Ensure valid coordinates
                x1 = torch.clamp(x1, 0, W-1)
                x2 = torch.clamp(x2, x1+1, W)
                y1 = torch.clamp(y1, 0, H-1)
                y2 = torch.clamp(y2, y1+1, H)
                
                patch = image[b:b+1, :, y1:y2, x1:x2]
                patch = F.interpolate(patch, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                batch_patches.append(patch)
            patches.append(torch.cat(batch_patches, dim=0))
        return torch.stack(patches)
        
    def process_exemplars(self, bboxes, image):
        """Process exemplar patches"""
        # Extract patches from image using bboxes
        exemplar_patches = self.extract_exemplar_patches(image, bboxes)
        # print(f"Extracted exemplar patches shape: {exemplar_patches.shape}")
        
        B, K, C, H, W = exemplar_patches.shape
        
        # Flatten exemplars
        exemplars_flat = exemplar_patches.view(B * K, C, H, W)
        # print(f"Flattened exemplars: {exemplars_flat.shape}")
        
        # Extract features
        exemplar_features = self.encoder(exemplars_flat)
        # print(f"Exemplar features after encoder: {exemplar_features.shape}")
        
        # Global average pooling
        exemplar_features = F.adaptive_avg_pool2d(exemplar_features, (1, 1))
        exemplar_features = exemplar_features.view(B, K, -1)
        # print(f"Final exemplar features: {exemplar_features.shape}")
        
        return exemplar_features

    def forward(self, image, bboxes):
        # Clear cache at the start of forward pass
        torch.cuda.empty_cache()
        
        # print("\nStarting forward pass...")
        # print(f"Input image shape: {image.shape}")
        # print(f"Input bboxes shape: {bboxes.shape}")
        
        # Extract features
        image_features = self.extract_features(image)
        B, C, H, W = image_features.shape
        
        # Process exemplars using bboxes and original image
        exemplar_features = self.process_exemplars(bboxes, image)
        # print(f"Processed exemplar features: {exemplar_features.shape}")
        
        # Clean up intermediate tensors
        del image  # Original image not needed anymore
        torch.cuda.empty_cache()
        
        current_exemplars = exemplar_features
        all_density_maps = []
        
        # Iterative prediction
        for i in range(self.num_iterations):
            print(f"\nIteration {i+1}/{self.num_iterations}")
            
            # Prepare image features for exemplar learner
            img_features_flat = image_features.permute(0, 2, 3, 1)
            img_features_flat = img_features_flat.reshape(B, H * W, C)
            
            # Update exemplar features
            current_exemplars = self.exemplar_learner(
                img_features_flat,
                current_exemplars,
                bboxes
            )
            
            # Clean up flattened features
            del img_features_flat
            torch.cuda.empty_cache()
            
            # Generate similarity maps
            similarity_maps = self.matcher(
                image_features,
                current_exemplars
            )
            
            # Predict density map
            density_map = self.decoders[i](similarity_maps)
            
            # Clean up intermediate results
            del similarity_maps
            torch.cuda.empty_cache()
            
            # Resize if needed
            if density_map.shape[2:] != image_features.shape[2:]:
                density_map = F.interpolate(
                    density_map,
                    size=image_features.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            all_density_maps.append(density_map)
        
        # Final cleanup
        del image_features, current_exemplars
        torch.cuda.empty_cache()
        
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
    boxes = torch.randn(batch_size, num_exemplars, 4)  # Hanya butuh image dan boxes
    
    print("\nInitializing model...")
    model = LowShotCounting(num_iterations=3, embed_dim=256)
    
    print("\nRunning forward pass...")
    density_maps = model(image, boxes)  # Sesuai dengan signature forward(self, image, bboxes)
    
    print("\nOutput summary:")
    print(f"Number of density maps: {len(density_maps)}")
    for i, dm in enumerate(density_maps):
        print(f"Density map {i} shape: {dm.shape}")