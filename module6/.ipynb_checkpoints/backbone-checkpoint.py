import torch
from torch import nn
import timm
import os

class Backbone(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        reduction: int = 8,
        model_path: str = "models/swin_tiny_patch4_window7_224.ms_in22k.pth",
        requires_grad: bool = True
    ):
        super(Backbone, self).__init__()
        
        # Initialize Swin-T (Tiny)
        self.reduction = reduction
        self.backbone = create_model(
            'swin_tiny_patch4_window7_224.ms_in22k',  # Changed to tiny
            pretrained=False,
            num_classes=0,
            features_only=True,
            img_size=512
        )
        
        # Load pretrained weights if available
        if pretrained:
            if not os.path.exists(model_path):
                print("Downloading pretrained model...")
                temp_model = create_model(
                    'swin_tiny_patch4_window7_224.ms_in22k',  # Changed to tiny
                    pretrained=True,
                    num_classes=0,
                    features_only=True
                )
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(temp_model.state_dict(), model_path)
                print(f"Model saved to {model_path}")
            
            pretrained_dict = torch.load(model_path, map_location="cpu")
            model_dict = self.backbone.state_dict()
            filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(filtered_dict)
            self.backbone.load_state_dict(model_dict, strict=False)

        # Set requires_grad
        for param in self.backbone.parameters():
            param.requires_grad_(requires_grad)

        # Channel dimensions for Swin-T (Tiny)
        self.num_channels = {
            'stage3': 192,  # S3 (sesuai paper)
            'stage4': 384,  # S4 (sesuai paper)
            'stage5': 768   # S5 (sesuai paper)
        }
        self.concatenated_channels = sum(self.num_channels.values())  # Total: 1344

    def forward_multiscale(self, x):
        """Return multi-scale features (S3, S4, S5)"""
        features = self.backbone.forward_features(x)
        
        # Get features from different stages
        s3 = features['stage3']  # [B, 64, 64, 192]
        s4 = features['stage4']  # [B, 32, 32, 384]
        s5 = features['stage5']  # [B, 16, 16, 768]
        
        return s3, s4, s5

    def forward_concatenated(self, x):
        """Return concatenated features (LOCA style)"""
        s3, s4, s5 = self.forward_multiscale(x)
        
        # Calculate target size based on reduction
        size = x.size(-2) // self.reduction, x.size(-1) // self.reduction
        
        # Interpolate all features to the same size
        s3 = nn.functional.interpolate(s3, size=size, mode='bilinear', align_corners=True)
        s4 = nn.functional.interpolate(s4, size=size, mode='bilinear', align_corners=True)
        s5 = nn.functional.interpolate(s5, size=size, mode='bilinear', align_corners=True)
        
        # Concatenate along channel dimension
        x = torch.cat([s3, s4, s5], dim=1)  # Total channels: 192 + 384 + 768 = 1344
        
        return x

    def forward(self, x):
        """Default forward returns concatenated features"""
        return self.forward_concatenated(x)