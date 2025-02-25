import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
import os
from typing import Dict


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Dict

def build_swin_transformer(model_path="models/swin_base_patch4_window7_224.ms_in22k.pth"):
    """Build Swin Transformer backbone"""
    if not os.path.exists(model_path):
        print("Downloading pretrained model...")
        model = create_model(
            'swin_base_patch4_window7_224.ms_in22k',
            pretrained=True,
            num_classes=0,
            features_only=True
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    model = create_model(
        'swin_base_patch4_window7_224.ms_in22k',
        pretrained=False,
        num_classes=0,
        features_only=True,
        img_size=512
    )

    if os.path.exists(model_path):
        pretrained_dict = torch.load(model_path, weights_only=True, map_location="cpu")
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
    
    return model

class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_path="models/swin_base_patch4_window7_224.ms_in22k.pth"):
        super().__init__()
        # Initialize Swin Transformer backbone using the provided function
        self.backbone = build_swin_transformer(pretrained_path)
        
        # Freeze first two stages
        self._freeze_early_layers()
        
        # Projectors to match paper dimensions
        self.proj_s3 = nn.Sequential(
            nn.Conv2d(512, 192, 1),     # Stage 3: 512 -> 192 channels (paper spec)
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        
        self.proj_s4 = nn.Sequential(
            nn.Conv2d(1024, 384, 1),    # Stage 4: 1024 -> 384 channels (paper spec)
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        
        self.proj_s5 = nn.Sequential(
            nn.Conv2d(1024, 768, 1),    # Stage 5: 1024 -> 768 channels (paper spec)
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )

    def _freeze_early_layers(self):
        """Freeze first two stages of Swin"""
        for i in range(2):
            layer_name = f'layers_{i}'
            if hasattr(self.backbone, layer_name):
                for param in getattr(self.backbone, layer_name).parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict:
        """
        Extract multi-scale features from input image
        Args:
            x: Input tensor [B, 3, H, W]
        Returns:
            Dictionary containing features matching paper specifications:
            - 'stage3': tensor of shape [B, 192, 64, 64]
            - 'stage4': tensor of shape [B, 384, 32, 32]
            - 'stage5': tensor of shape [B, 768, 16, 16]
        """
        # Get features from backbone
        features = self.backbone(x)
            
        # Convert from [B, H, W, C] to [B, C, H, W]
        s3 = features[2].permute(0, 3, 1, 2)  # Stage 3 features
        s4 = features[3].permute(0, 3, 1, 2)  # Stage 4 features
        s5 = F.avg_pool2d(s4, kernel_size=2, stride=2)  # Stage 5 derived from Stage 4
        
        # Project features to match paper dimensions
        s3 = self.proj_s3(s3)  # [B, 192, H, W]
        s4 = self.proj_s4(s4)  # [B, 384, H/2, W/2]
        s5 = self.proj_s5(s5)  # [B, 768, H/4, W/4]
        
        # Interpolate to desired output sizes
        s3 = F.interpolate(s3, size=(64, 64), mode='bilinear', align_corners=False)
        s4 = F.interpolate(s4, size=(32, 32), mode='bilinear', align_corners=False)
        s5 = F.interpolate(s5, size=(16, 16), mode='bilinear', align_corners=False)
        
        return {
            'stage3': s3,  # [B, 192, 64, 64]
            'stage4': s4,  # [B, 384, 32, 32]
            'stage5': s5   # [B, 768, 16, 16]
        }

def build_backbone(pretrained_path="models/swin_base_patch4_window7_224.ms_in22k.pth"):
    """Build and return the feature extractor"""
    return FeatureExtractor(pretrained_path)

if __name__ == "__main__":
    # Test code
    extractor = FeatureExtractor()
    dummy_input = torch.randn(2, 3, 512, 512)
    features = extractor(dummy_input)
    
    # Print shapes for verification
    for stage_name, feat in features.items():
        print(f"{stage_name} shape: {feat.shape}")