from .backbone import Backbone
from .transformer import TransformerEncoder
from .ope import OPEModule
from .positional_encoding import PositionalEncodingsFixed
from .regression_head import DensityMapRegressor
from .selfattention import SelfAttention
from .crossscale import CrossScaleFusion
from .iefl import IterativeExemplarFeatureLearning
from .feature_enhancer import FeatureEnhancer
from .exemplar_feature_learning import ExemplarFeatureLearning
from .exemplar_image_matching import ExemplarImageMatching

from .swin_transformer_encoder import HybridEncoder
from torchvision.ops import roi_align

import torch
from torch import nn
from torch.nn import functional as F
import math

class EfficientCounter(nn.Module):
    def __init__(
        self,
        image_size: int,
        emb_dim: int = 256,
        num_heads: int = 8,
        reduction: int = 8,
        num_iterations: int = 2
    ):
        super().__init__()
        
        self.image_size = image_size
        self.emb_dim = emb_dim
        self.reduction = reduction
        
        # Core components
        self.backbone = HybridEncoder(embed_dim=emb_dim)
        self.feature_enhancer = FeatureEnhancer(embed_dim=emb_dim)
        self.exemplar_learner = ExemplarFeatureLearning(
            embed_dim=emb_dim, 
            num_iterations=num_iterations
        )
        self.matcher = ExemplarImageMatching()
        
        # Rest of the initialization...
        
        # Projection layers
        self.response_proj = nn.Conv2d(49, emb_dim, 1)
        
        # Multiple output heads
        self.aux_head1 = nn.Sequential(
            nn.Conv2d(emb_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )
        
        self.aux_head2 = nn.Sequential(
            nn.Conv2d(emb_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )
        
        self.regression_head = nn.Sequential(
            nn.Conv2d(emb_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x, bboxes):
        # Extract features
        features = self.backbone(x)  # [B, 256, H/8, W/8]
        enhanced_features = self.feature_enhancer(features)
        
        # ROI Align 
        batch_size = bboxes.size(0)
        num_boxes = bboxes.size(1)
        batch_idx = torch.arange(batch_size, device=bboxes.device)[:, None].expand(-1, num_boxes).reshape(-1, 1)
        boxes = torch.cat([batch_idx, bboxes.view(-1, 4)], dim=1)
        
        exemplar_features = roi_align(
            enhanced_features,
            boxes,
            output_size=(7, 7),
            spatial_scale=1.0/self.reduction,
            aligned=True
        )
        
        # Reshape untuk learning
        b, c, h, w = enhanced_features.shape
        enhanced_features_flat = enhanced_features.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
        exemplar_features = exemplar_features.view(batch_size, num_boxes, -1)  # [B, N, C]
        
        # Learn exemplar features
        exemplar_features = self.exemplar_learner(enhanced_features_flat, exemplar_features)
        
        # Matching - pastikan dimensi sesuai dengan yang diharapkan ExemplarImageMatching
        similarity_maps = self.matcher(enhanced_features_flat, exemplar_features)
        
        # Project to correct dimension
        response_map = self.response_proj(similarity_maps)  
        
        # Generate predictions
        aux_pred1 = self.aux_head1(response_map)
        aux_pred2 = self.aux_head2(response_map)
        density_map = self.regression_head(response_map)
        
        # Interpolate all outputs
        density_map = F.interpolate(density_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        aux_pred1 = F.interpolate(aux_pred1, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        aux_pred2 = F.interpolate(aux_pred2, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        return density_map, [aux_pred1, aux_pred2], exemplar_features

def build_model(args):
    return EfficientCounter(
        image_size=args.image_size,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        reduction=args.reduction,
        num_iterations=args.num_iterations
    )