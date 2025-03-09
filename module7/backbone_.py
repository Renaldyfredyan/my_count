import os
import torch
from torch import nn
import torch.nn.functional as F
import timm
import urllib.request

class Backbone(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        reduction: int = 8,
        weights_path: str = "pretrained_weights/groundingdino_swint_ogc.pth",
        requires_grad: bool = False
    ):
        super().__init__()
        
        self.reduction = reduction
        
        # Initialize Swin-T with timm BUT we'll overwrite with GroundingDINO weights
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=0,
            features_only=True,
            out_indices=(1, 2, 3),
            img_size=512
        )
        
        # Download and load GroundingDINO weights
        if pretrained:
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            
            if not os.path.exists(weights_path):
                print(f"GroundingDINO weights not found at {weights_path}. Downloading...")
                try:
                    url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
                    urllib.request.urlretrieve(url, weights_path)
                    print(f"Successfully downloaded GroundingDINO weights to {weights_path}")
                except Exception as e:
                    print(f"Failed to download weights: {e}")
                    raise
            
            print(f"Loading GroundingDINO weights from {weights_path}")
            checkpoint = torch.load(weights_path, map_location="cpu")
            
            # Extract backbone weights from GroundingDINO
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Filter only backbone weights
            backbone_state_dict = {
                k.replace('backbone.', ''): v for k, v in state_dict.items() 
                if k.startswith('backbone.') and k.replace('backbone.', '') in self.backbone.state_dict()
            }
            
            # Load the weights into the timm model
            missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=False)
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
        
        # Set requires_grad for all parameters
        for param in self.backbone.parameters():
            param.requires_grad_(requires_grad)
        
        # Channel dimensions for Swin-T
        self.num_channels = {
            'stage3': 192,
            'stage4': 384,
            'stage5': 768
        }
        self.total_channels = sum(self.num_channels.values())

    def forward_multiscale(self, x):
        """Return multi-scale features (S3, S4, S5)"""
        features = self.backbone(x)
        
        s3, s4, s5 = features
        return s3, s4, s5
    
    def forward_concatenated(self, x):
        """Concatenate multi-scale features after resizing to same resolution"""
        s3, s4, s5 = self.forward_multiscale(x)
        
        # Target size is determined by the reduction factor
        target_size = (x.size(2) // self.reduction, x.size(3) // self.reduction)
        
        # Resize all feature maps to target size
        s3 = F.interpolate(s3, size=target_size, mode='bilinear', align_corners=True)
        s4 = F.interpolate(s4, size=target_size, mode='bilinear', align_corners=True)
        s5 = F.interpolate(s5, size=target_size, mode='bilinear', align_corners=True)
        
        # Concatenate along channel dimension
        concat_features = torch.cat([s3, s4, s5], dim=1)
        
        return concat_features

    def forward(self, x):
        """Default forward returns concatenated features"""
        return self.forward_concatenated(x)