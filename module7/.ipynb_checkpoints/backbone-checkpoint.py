import torch
from torch import nn

class BackboneGroundingDINO(nn.Module):
    def __init__(
        self,
        config_path: str = "groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_path: str = "weights/groundingdino_swint_ogc.pth",
        reduction: int = 8,
        requires_grad: bool = True
    ):
        super().__init__()
        self.reduction = reduction
        
        # Load the full GroundingDINO model using the official method
        from groundingdino.inference import load_model
        full_model = load_model(config_path, model_path)
        
        # Extract just the backbone component
        self.backbone = full_model.backbone
        
        # Freeze or unfreeze parameters as needed
        for param in self.backbone.parameters():
            param.requires_grad_(requires_grad)
        
        # Get the output channel dimensions
        # Note: You may need to adjust this based on the actual structure
        # of the GroundingDINO backbone
        self.num_channels = {
            'stage3': 192,  # Adjust these values based on the actual backbone
            'stage4': 384,
            'stage5': 768
        }
        self.total_channels = sum(self.num_channels.values())
        
        print(f"Successfully loaded backbone from GroundingDINO model")

    def forward_multiscale(self, x):
        """Return multi-scale features from the backbone"""
        # Note: You may need to adjust this based on how the GroundingDINO
        # backbone actually returns features
        features = self.backbone(x)
        
        # Check what the backbone actually returns
        if isinstance(features, list) and len(features) >= 3:
            s3, s4, s5 = features[-3:]  # Take the last 3 feature maps
        elif isinstance(features, tuple) and len(features) >= 3:
            s3, s4, s5 = features[-3:]
        elif isinstance(features, dict) and all(k in features for k in ['res3', 'res4', 'res5']):
            s3, s4, s5 = features['res3'], features['res4'], features['res5']
        else:
            # If unsure about the structure, print it out for debugging
            print(f"Unexpected feature structure: {type(features)}")
            if isinstance(features, (list, tuple)):
                print(f"Length: {len(features)}")
                for i, feat in enumerate(features):
                    print(f"Feature {i} shape: {feat.shape}")
            raise ValueError("Unexpected backbone output format. Please modify the code to handle it.")
        
        return s3, s4, s5

    def forward_concatenated(self, x):
        s3, s4, s5 = self.forward_multiscale(x)
        
        # Ensure features have the right format [B, C, H, W]
        # Adjust if needed based on the actual output format
        if s3.dim() == 4 and s3.shape[1] != self.num_channels['stage3']:
            # Features might be in [B, H, W, C] format, so permute
            s3 = s3.permute(0, 3, 1, 2)
            s4 = s4.permute(0, 3, 1, 2)
            s5 = s5.permute(0, 3, 1, 2)

        # Resize features to the target size
        size = (x.size(-2) // self.reduction, x.size(-1) // self.reduction)
        s3 = nn.functional.interpolate(s3, size=size, mode='bilinear', align_corners=True)
        s4 = nn.functional.interpolate(s4, size=size, mode='bilinear', align_corners=True)
        s5 = nn.functional.interpolate(s5, size=size, mode='bilinear', align_corners=True)

        # Concatenate all features
        x = torch.cat([s3, s4, s5], dim=1)
        return x

    def forward(self, x):
        """Default forward returns concatenated features"""
        return self.forward_concatenated(x)


# Example usage
if __name__ == "__main__":
    # Create the backbone
    backbone = BackboneGroundingDINO(
        config_path="groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_path="weights/groundingdino_swint_ogc.pth"
    )
    
    # Test with a sample input
    x = torch.randn(1, 3, 512, 512)
    features = backbone(x)
    print(f"Output features shape: {features.shape}")
    
    # Example of how to use just the multiscale features
    s3, s4, s5 = backbone.forward_multiscale(x)
    print(f"Stage 3 features shape: {s3.shape}")
    print(f"Stage 4 features shape: {s4.shape}")
    print(f"Stage 5 features shape: {s5.shape}")