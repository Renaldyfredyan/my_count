import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
import os


def download_pretrained_model(backbone_type='swinT1k'):
    # Configuration for different backbones
    model_configs = {
        'swinT1k': {
            'model_name': 'swin_tiny_patch4_window7_224',
            'file_name': 'swin_tiny_patch4_window7_224.pth'
        },
        'swinT1K': {
            'model_name': 'swin_tiny_patch4_window7_224.ms_in1k',
            'file_name': 'swin_tiny_patch4_window7_224_1k.pth'
        },
        'swinB1K': {
            'model_name': 'swin_base_patch4_window7_224.ms_in1k',
            'file_name': 'swin_base_patch4_window7_224_1k.pth'
        },
        'swinB22K': {
            'model_name': 'swin_base_patch4_window7_224.ms_in22k',
            'file_name': 'swin_base_patch4_window7_224_22k.pth'
        }
    }

    if backbone_type not in model_configs:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")

    config = model_configs[backbone_type]
    model_dir = "models"
    model_path = os.path.join(model_dir, config['file_name'])

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(model_path):
        print(f"Downloading pre-trained model {backbone_type} using timm...")
        model = create_model(config['model_name'], pretrained=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    else:
        print(f"Model already exists at {model_path}")

    return model_path

def build_swin_transformer(embed_dim=256, backbone_type='swinT1K', model_path=None):
    # Configuration for different backbones
    model_configs = {
        'swinT1K': {
            'model_name': 'swin_tiny_patch4_window7_224',
            'out_dim': 768
        },
        'swinT22K': {
            'model_name': 'swin_tiny_patch4_window7_224.ms_in22k',
            'out_dim': 768
        },
        'swinB1K': {
            'model_name': 'swin_base_patch4_window7_224.ms_in1k',
            'out_dim': 1024
        },
        'swinB22K': {
            'model_name': 'swin_base_patch4_window7_224.ms_in22k',
            'out_dim': 1024
        }
    }

    if backbone_type not in model_configs:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")

    config = model_configs[backbone_type]
    
    # If model_path not provided, use default path
    if model_path is None:
        model_path = f"models/{config['model_name']}.pth"

    # Load the pretrained model weights and prepare the backbone
    if not os.path.exists(model_path):
        print("Downloading pretrained model...")
        model = create_model(
            config['model_name'],
            pretrained=True,
            num_classes=0,  # Remove classification head
            features_only=True  # Get intermediate features
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    else:
        print(f"Model already exists at {model_path}")

    # Create model and load weights
    model = create_model(
        config['model_name'],
        pretrained=False,  # Prevent auto-download
        features_only=True,  # Get intermediate features
        num_classes=0      # Remove classification head
    )

    # Load pretrained weights, ignoring mismatched keys
    pretrained_dict = torch.load(model_path, map_location="cpu")
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    print("Pretrained weights loaded successfully, ignoring mismatched keys.")
    return model


class FeatureExtractor(nn.Module):
    def __init__(self, embed_dim=256, backbone_type='swinT1K'):  # Ubah parameter ini
        super(FeatureExtractor, self).__init__()
        
        # Download pretrained model if needed
        model_path = download_pretrained_model(backbone_type)
        
        # Build backbone
        self.swin_backbone = build_swin_transformer(
            embed_dim=embed_dim,
            backbone_type=backbone_type,
            model_path=model_path
        )
        
        # Get output dimension based on backbone type
        out_dims = {
            'swinT1k': 768,
            'swinT1K': 768,
            'swinB1K': 1024,
            'swinB22K': 1024
        }
        backbone_out_dim = out_dims[backbone_type]
        
        self.feature_proj = nn.Conv2d(backbone_out_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        # Resize input to match Swin Transformer requirements
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features from backbone
        features = self.swin_backbone(x)[-1]  # Take the last feature map
        
        # Log shape for debugging
        # print(f"Feature shape before processing: {features.shape}")
        
        # Reshape features to [B, C, H, W] format
        if features.dim() == 3:  # If shape is [B, HW, C]
            B, L, C = features.shape
            H = W = int(L ** 0.5)
            features = features.reshape(B, H, W, C).permute(0, 3, 1, 2)
        elif features.dim() == 4 and features.shape[1] == 7:  # If shape is [B, H, W, C]
            features = features.permute(0, 3, 1, 2)
            
        # Log shape after processing
        # print(f"Feature shape after processing: {features.shape}")
        
        # Project features to desired embedding dimension
        features = self.feature_proj(features)
        
        return features
# Test functionality
if __name__ == "__main__":
    extractor = FeatureExtractor(embed_dim=256).cuda()
    dummy_input = torch.randn(1, 3, 512, 512).cuda()
    features = extractor(dummy_input)
    print("Extracted Features Shape:", features.shape)