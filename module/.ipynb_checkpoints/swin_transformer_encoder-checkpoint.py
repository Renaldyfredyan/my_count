import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import swin_tiny_patch4_window7_224
from timm.models.helpers import load_checkpoint
import os
import requests
from timm.models import create_model

def download_pretrained_model():
    # Alternative: Use the timm pre-trained model and save it locally
    model_dir = "models"
    model_path = os.path.join(model_dir, "swin_tiny_patch4_window7_224.pth")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(model_path):
        print("Downloading pre-trained model using timm...")
        model = create_model('swin_tiny_patch4_window7_224', pretrained=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    else:
        print(f"Model already exists at {model_path}")

    return model_path

class HybridEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super(HybridEncoder, self).__init__()
        
        # Swin Transformer Backbone (SwinT-Tiny)
        self.swin_backbone = swin_tiny_patch4_window7_224(pretrained=False)
        model_path = download_pretrained_model()
        self.swin_backbone.load_state_dict(torch.load(model_path))  # Load local checkpoint
        self.swin_backbone.head = nn.Identity()  # Remove classification head
        self.swin_proj = nn.Conv2d(768, embed_dim, kernel_size=1)  # Project to embed_dim

    def forward(self, x):
        # Swin Transformer Path
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)  # Resize input
        swin_features = self.swin_backbone.forward_features(x)  # Extract features without head

        # Debug: Print shape of swin_features
        # print("Shape of swin_features before projection:", swin_features.shape)

        # Reshape swin_features to [B, C, H, W]
        swin_features = swin_features.permute(0, 3, 1, 2).contiguous()  # Change to [B, C, H, W]
        swin_features = self.swin_proj(swin_features)  # Shape: [B, embed_dim, H, W]
        return swin_features

# Test the HybridEncoder
if __name__ == "__main__":
    encoder = HybridEncoder(embed_dim=256).cuda()
    dummy_input = torch.randn(1, 3, 512, 512).cuda()  # Example input image
    swin_features = encoder(dummy_input)

    print("Swin Transformer Features Shape:", swin_features.shape)
