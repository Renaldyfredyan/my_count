import torch
from torch import nn
import timm
from timm.models import create_model
import os

class Backbone(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        reduction: int = 8,
        model_path: str = "models/swin_tiny_patch4_window7_224.ms_in22k.pth",
        requires_grad: bool = True
    ):
        super().__init__()
        
        # Initialize Swin-T (Tiny)
        self.reduction = reduction
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224.ms_in22k',
            pretrained=False,
            num_classes=0,
            features_only=True,
            out_indices=(1, 2, 3),
            img_size=512  # Set image size ke 512
        )
        
        # Load pretrained weights
        if pretrained:
            if not os.path.exists(model_path):
                print("Downloading pretrained model...")
                temp_model = timm.create_model(
                    'swin_tiny_patch4_window7_224.ms_in22k',
                    pretrained=True,
                    num_classes=0,
                    features_only=True,
                    img_size=512  # Di sini juga
                )
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(temp_model.state_dict(), model_path)
                print(f"Model saved to {model_path}")

            pretrained_dict = torch.load(model_path, weights_only=True, map_location="cpu")
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
        # self.concatenated_channels = sum(self.num_channels.values())  # Total: 1344
        self.total_channels = sum(self.num_channels.values())

    def forward_multiscale(self, x):
        """Return multi-scale features (S3, S4, S5)"""
        features = self.backbone(x)
        
        s3, s4, s5 = features
        return s3, s4, s5
    
    def forward_concatenated(self, x):
        s3, s4, s5 = self.forward_multiscale(x)
        
        # Permute channels ke posisi yang benar (NCHW format)
        s3 = s3.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        s4 = s4.permute(0, 3, 1, 2)
        s5 = s5.permute(0, 3, 1, 2)
        
        size = x.size(-2) // self.reduction, x.size(-1) // self.reduction
        
        # Interpolate
        s3 = nn.functional.interpolate(s3, size=size, mode='bilinear', align_corners=True)
        s4 = nn.functional.interpolate(s4, size=size, mode='bilinear', align_corners=True)
        s5 = nn.functional.interpolate(s5, size=size, mode='bilinear', align_corners=True)
        
        # Concatenate
        x = torch.cat([s3, s4, s5], dim=1)
        
        return x

    def forward(self, x):
        """Default forward returns concatenated features"""
        return self.forward_concatenated(x)