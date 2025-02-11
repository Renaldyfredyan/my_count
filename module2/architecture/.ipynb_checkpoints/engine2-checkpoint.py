from .backbone import Backbone
from .transformer import TransformerEncoder
from .ope import OPEModule
from .positional_encoding import PositionalEncodingsFixed
from .regression_head import DensityMapRegressor
from .selfattention import SelfAttention
from .crossscale import CrossScaleFusion
from .iefl import IterativeExemplarFeatureLearning

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
        
        # Core components
        self.backbone = Backbone(reduction=reduction)
        self.i_efl = IterativeExemplarFeatureLearning(
            emb_dim=emb_dim,
            num_heads=num_heads,
            num_iterations=num_iterations
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Conv2d(emb_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU()
        )
        
        self.image_size = image_size
        self.emb_dim = emb_dim

    def forward(self, x, bboxes):
        # Extract features
        features = self.backbone(x)
        
        # ROI align untuk exemplars
        exemplar_features = roi_align(
            features, 
            boxes=bboxes.view(-1, 4),
            output_size=(7, 7),
            spatial_scale=1.0/self.reduction,
            aligned=True
        )
        
        # Enhance exemplar features
        enhanced_exemplars = self.i_efl(exemplar_features, features)
        
        # Generate density map
        response_map = F.conv2d(
            features,
            enhanced_exemplars.view(-1, self.emb_dim, 1, 1),
            padding=0
        )
        
        # Regression to density map
        density_map = self.regression_head(response_map)
        
        return density_map, None, enhanced_exemplars

def build_model(args):
    return EfficientCounter(
        image_size=args.image_size,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        reduction=args.reduction,
        num_iterations=args.num_iterations,
        encoder_type=args.encoder_type
    )