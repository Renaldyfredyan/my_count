import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
import os

def build_swin_transformer(model_path="models/swin_base_patch4_window7_224.ms_in22k.pth"):
    """Build Swin Transformer backbone with proper initialization and weight loading"""
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
    
    # Create model without classification head
    model = create_model(
        'swin_base_patch4_window7_224.ms_in22k',
        pretrained=False,
        num_classes=0,
        features_only=True,  # Enable feature extraction from multiple stages
        img_size=512  # Set model to accept 512x512 input
    )

    # Load pretrained weights
    if os.path.exists(model_path):
        pretrained_dict = torch.load(model_path, weights_only=True, map_location="cpu")
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
    
    return model

class DensityEncoder(nn.Module):
    def __init__(self):
        super(DensityEncoder, self).__init__()
        
        # Swin Transformer backbone
        self.swin_backbone = build_swin_transformer()
        self._freeze_early_layers()
        
        # Proyeksi untuk menghasilkan S3, S4, S5 dengan dimensi yang sesuai (256 channels)
        self.proj_layers = nn.ModuleDict({
            'stage3': nn.Sequential(
                nn.Conv2d(512, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            'stage4': nn.Sequential(
                nn.Conv2d(1024, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        })
        
        # Layer untuk menghasilkan S5 dari S4
        self.s5_generator = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Projector untuk output final
        self.final_proj = nn.Conv2d(256 * 3, 256, 1)
        
    def _freeze_early_layers(self):
        """Freeze first two stages of Swin-T"""
        for i in range(2):
            layer_name = f'layers_{i}'
            if hasattr(self.swin_backbone, layer_name):
                for param in getattr(self.swin_backbone, layer_name).parameters():
                    param.requires_grad = False
    
    def _extract_and_process_features(self, x):
        """Extract features and process them to get S3, S4, S5"""
        # Get features from Swin-T
        features = self.swin_backbone(x)
        
        # Process stage3 and stage4 features (B, H, W, C) -> (B, C, H, W)
        stage3_feat = features[2].permute(0, 3, 1, 2)  # 512 channels
        stage4_feat = features[3].permute(0, 3, 1, 2)  # 1024 channels
        
        print("Stage3 feature shape:", stage3_feat.shape)
        print("Stage4 feature shape:", stage4_feat.shape)
        
        # Project to 256 channels and upsample S3 to 64x64
        S3 = self.proj_layers['stage3'](stage3_feat)  # 32x32x256
        S3 = F.interpolate(S3, size=(64, 64), mode='bilinear', align_corners=False)  # Upsample to 64x64
        
        S4 = self.proj_layers['stage4'](stage4_feat)  # 16x16x256
        S4 = F.interpolate(S4, size=(32, 32), mode='bilinear', align_corners=False)  # Upsample to 32x32
        
        print("S3 shape after projection:", S3.shape)
        print("S4 shape after projection:", S4.shape)
        
        # Resize S4 if needed
        if S4.shape[2:] != (32, 32):
            S4 = F.interpolate(S4, size=(32, 32), mode='bilinear', align_corners=False)
        
        # Generate S5 from S4
        S5 = self.s5_generator(S4)  # 16x16x256
        print("S5 shape:", S5.shape)
        
        return S3, S4, S5

    def forward(self, x):
        """
        Forward pass mengikuti spesifikasi paper
        Input: [B, C, H, W]
        Output: Fi ∈ R64×64×256
        """
        # Original input size
        orig_size = x.shape[2:]
        
        # Resize input to 512x512 for Swin-T
        if orig_size != (512, 512):
            x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        
        # Get S3, S4, S5
        S3, S4, S5 = self._extract_and_process_features(x)
        
        # Upsample S4 dan S5 ke ukuran S3 (64x64)
        S4_up = F.interpolate(S4, size=S3.shape[2:], mode='bilinear', align_corners=False)
        S5_up = F.interpolate(S5, size=S3.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate semua features
        concat_features = torch.cat([S3, S4_up, S5_up], dim=1)  # 64x64x(256*3)
        
        # Project ke dimensi final
        Fi = self.final_proj(concat_features)  # 64x64x256
        
        return Fi

# Test code
if __name__ == "__main__":
    encoder = DensityEncoder()
    dummy_input = torch.randn(1, 3, 512, 512)
    
    with torch.no_grad():
        output = encoder(dummy_input)
        
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be [1, 256, 64, 64]
    assert output.shape == (1, 256, 64, 64), "Output dimensions don't match paper specifications"
    print("✓ Dimensions verified")