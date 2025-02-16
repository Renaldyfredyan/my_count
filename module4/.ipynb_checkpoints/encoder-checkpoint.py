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
    def __init__(self, min_dim=384):
        """
        Enhanced encoder for density map estimation
        Args:
            min_dim: Minimum feature dimension for the decoder
        """
        super(DensityEncoder, self).__init__()
        
        # Swin Transformer backbone
        self.swin_backbone = build_swin_transformer()
        
        # Freeze early layers (optional, can be controlled during training)
        self._freeze_early_layers()
        
        # Multi-scale feature fusion
        # Stage 3 output: 512 channels
        # Stage 4 output: 1024 channels
        self.fusion_layers = nn.ModuleDict({
            'stage3': nn.Sequential(
                nn.Conv2d(512, min_dim, 1),
                nn.BatchNorm2d(min_dim),
                nn.ReLU(inplace=True)
            ),
            'stage4': nn.Sequential(
                nn.Conv2d(1024, min_dim, 1),
                nn.BatchNorm2d(min_dim),
                nn.ReLU(inplace=True)
            )
        })
        
        # Smooth transition layer
        self.smooth = nn.Sequential(
            nn.Conv2d(min_dim * 2, min_dim, 3, padding=1),
            nn.BatchNorm2d(min_dim),
            nn.ReLU(inplace=True)
        )
        
    def _freeze_early_layers(self):
        """Freeze first two stages of Swin-T"""
        # In timm's implementation, layers are named as 'layers_0', 'layers_1', etc.
        for i in range(2):
            layer_name = f'layers_{i}'
            if hasattr(self.swin_backbone, layer_name):
                for param in getattr(self.swin_backbone, layer_name).parameters():
                    param.requires_grad = False
    
    def _extract_features(self, x):
        """Extract and process features from multiple stages"""
        # Get features from different stages
        features = self.swin_backbone(x)
        
        # Process features from stages 3 and 4
        # Features dari Swin Transformer datang dalam format [B, H, W, C]
        # Kita perlu transpose ke format [B, C, H, W]
        stage3_feat = features[2].permute(0, 3, 1, 2)  # 512 channels
        stage4_feat = features[3].permute(0, 3, 1, 2)  # 1024 channels
        
        # Ensure spatial dimensions match through interpolation
        if stage3_feat.shape[2:] != stage4_feat.shape[2:]:
            stage4_feat = F.interpolate(
                stage4_feat,
                size=stage3_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        return stage3_feat, stage4_feat

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape [B, C, H, W]
        Returns:
            Processed features suitable for density map estimation
        """
        # Original input size
        orig_size = x.shape[2:]
        
        # Resize input to match model's expected size (512x512)
        if orig_size != (512, 512):
            x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        
        # Extract multi-scale features
        stage3_feat, stage4_feat = self._extract_features(x)
        
        # Process features through fusion layers
        proc_stage3 = self.fusion_layers['stage3'](stage3_feat)
        proc_stage4 = self.fusion_layers['stage4'](stage4_feat)
        
        # Concatenate and smooth
        fused_features = torch.cat([proc_stage3, proc_stage4], dim=1)
        output = self.smooth(fused_features)
        
        # Ensure output size matches input
        if output.shape[2:] != orig_size:
            output = F.interpolate(output, size=orig_size, mode='bilinear', align_corners=False)
        
        return output

# Test code
if __name__ == "__main__":
    # Test with different input sizes
    encoder = DensityEncoder()
    test_sizes = [(512, 512), (640, 480), (720, 1280)]
    
    print("Testing encoder with various input sizes...")
    print("Note: All inputs will be internally resized to 512x512 for feature extraction")
    print("and then resized back to original dimensions\n")
    
    for size in test_sizes:
        print(f"\nTesting input size: {size}")
        dummy_input = torch.randn(1, 3, size[0], size[1])
        
        with torch.no_grad():
            output = encoder(dummy_input)
            
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output channels: {output.shape[1]}")
        
        # Verify output dimensions
        assert output.shape[2:] == dummy_input.shape[2:], "Output spatial dimensions should match input"
        assert output.shape[1] == 384, "Output should have 384 channels"
        print("✓ Dimensions verified")