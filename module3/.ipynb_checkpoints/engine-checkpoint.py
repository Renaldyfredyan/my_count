import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

# Import our modules
from feature_extraction import FeatureExtractor
from feature_enhancement import FeatureEnhancer
from ielf import ExemplarFeatureLearning
from similarity_maps import ExemplarImageMatching
from decoder import DensityRegressionDecoder

class LowShotCounter(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.emb_dim
        self.num_iterations = args.num_iterations
        
        # Initialize RoIAlign for exemplar feature extraction
        self.roi_align = RoIAlign(
            output_size=(7, 7),  # Standard RoIAlign output size
            spatial_scale=1.0/32.0,  # Assuming 32x downsampling from backbone
            sampling_ratio=2
        )
        
        # Main modules
        self.feature_extractor = FeatureExtractor(embed_dim=self.embed_dim)
        self.feature_enhancer = FeatureEnhancer(embed_dim=self.embed_dim)
        self.exemplar_learner = ExemplarFeatureLearning(
            embed_dim=self.embed_dim,
            num_iterations=self.num_iterations
        )
        self.matcher = ExemplarImageMatching(embed_dim=self.embed_dim)
        self.decoder = DensityRegressionDecoder(input_channels=args.num_objects)

        # Optional projection layers for dimension matching
        self.exemplar_projection = nn.Sequential(
            nn.Linear(self.embed_dim * 7 * 7, self.embed_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dim)
        )

    def extract_exemplar_features(self, image_features, bboxes, batch_size):
        """Extract and process exemplar features using RoIAlign."""
        # Adjust bboxes format for RoIAlign
        batch_idx = torch.arange(batch_size).view(-1, 1).repeat(1, bboxes.size(1))
        batch_idx = batch_idx.view(-1, 1).float().to(bboxes.device)
        bboxes = bboxes.view(-1, 4)
        rois = torch.cat([batch_idx, bboxes], dim=1)

        # Extract features using RoIAlign
        exemplar_features = self.roi_align(image_features, rois)
        
        # Flatten and project features
        exemplar_features = exemplar_features.view(batch_size, -1, self.embed_dim * 7 * 7)
        exemplar_features = self.exemplar_projection(exemplar_features)
        
        return exemplar_features

    def forward(self, images, bboxes):
        """
        Forward pass of the model.
        Args:
            images: Input images [B, C, H, W]
            bboxes: Exemplar bounding boxes [B, K, 4]
        Returns:
            density_map: Predicted density map [B, 1, H, W]
            similarity_maps: Intermediate similarity maps [B, K, H, W]
        """
        batch_size = images.shape[0]

        # Extract and enhance image features
        image_features = self.feature_extractor(images)
        enhanced_features = self.feature_enhancer(image_features)

        # Extract exemplar features using RoIAlign
        exemplar_features = self.extract_exemplar_features(
            enhanced_features, bboxes, batch_size
        )

        # Prepare features for exemplar learning
        B, C, H, W = enhanced_features.shape
        image_features_flat = enhanced_features.view(B, C, -1).permute(0, 2, 1)

        # Iteratively update exemplar features
        updated_exemplars = self.exemplar_learner(
            image_features_flat, 
            exemplar_features
        )

        # Generate similarity maps
        similarity_maps = self.matcher(
            image_features_flat,
            updated_exemplars
        )

        # Generate density map
        density_map = self.decoder(similarity_maps)

        # Ensure output size matches input size
        if density_map.shape[-2:] != images.shape[-2:]:
            density_map = F.interpolate(
                density_map,
                size=images.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        return density_map, similarity_maps

def build_model(args):
    """
    Build and initialize the complete model.
    Args:
        args: Arguments from arg_parser
    Returns:
        model: Initialized model
    """
    model = LowShotCounter(args)
    
    # Initialize weights (optional)
    if hasattr(model, 'initialize_weights'):
        model.initialize_weights()
    
    # Move model to device
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Wrap with DDP if using multiple GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
    
    return model