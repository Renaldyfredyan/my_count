import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
import os

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
    def __init__(self):
        super().__init__()
        self.backbone = build_swin_transformer()
        self._freeze_early_layers()
        
    def _freeze_early_layers(self):
        """Freeze first two stages of Swin-T"""
        for i in range(2):
            layer_name = f'layers_{i}'
            if hasattr(self.backbone, layer_name):
                for param in getattr(self.backbone, layer_name).parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        """
        Extract multi-scale features from input image
        Args:
            x: Input tensor [B, 3, H, W]
        Returns:
            Dictionary of features from different stages
        """
        # Resize input to 512x512 if needed
        orig_size = x.shape[2:]
        if orig_size != (512, 512):
            x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        
        # Get features from backbone
        features = self.backbone(x)
        
        # Extract and process features from relevant stages
        # Convert from [B, H, W, C] to [B, C, H, W]
        stage3 = features[2].permute(0, 3, 1, 2)  # 512 channels
        stage4 = features[3].permute(0, 3, 1, 2)  # 1024 channels
        
        extracted_features = {
            'stage3': stage3,  # B x 512 x 32 x 32
            'stage4': stage4,  # B x 1024 x 16 x 16
        }
        
        return extracted_features

if __name__ == "__main__":
    # Test code
    extractor = FeatureExtractor()
    dummy_input = torch.randn(2, 3, 512, 512)
    
    features = extractor(dummy_input)
    for stage_name, feat in features.items():
        print(f"{stage_name} shape: {feat.shape}")