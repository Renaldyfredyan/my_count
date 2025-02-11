import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
import os


def download_pretrained_model():
    # Download pretrained model and save locally if not already present
    model_dir = "models"
    model_path = os.path.join(model_dir, "swin_tiny_patch4_window7_224.ms_in22k.pth")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(model_path):
        print("Downloading pre-trained model using timm...")
        model = create_model('swin_tiny_patch4_window7_224.ms_in22k', pretrained=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    else:
        print(f"Model already exists at {model_path}")

    return model_path


def build_swin_transformer(embed_dim=256, model_path="models/swin_tiny_patch4_window7_224.ms_in22k.pth"):
    # Load the pretrained model weights and prepare the backbone
    if not os.path.exists(model_path):
        print("Downloading pretrained model...")
        model = create_model(
            'swin_tiny_patch4_window7_224.ms_in22k',
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    else:
        print(f"Model already exists at {model_path}")

    # Create model and load weights
    model = create_model(
        'swin_tiny_patch4_window7_224.ms_in22k',
        pretrained=False,  # Prevent auto-download
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


class HybridEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super(HybridEncoder, self).__init__()

        # Swin Transformer Backbone (SwinT-Tiny)
        self.swin_backbone = build_swin_transformer(embed_dim)
        self.swin_proj = nn.Conv2d(768, embed_dim, kernel_size=1)  # Project to embed_dim

    def forward(self, x):
        # Swin Transformer Path
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)  # Resize input
        swin_features = self.swin_backbone.forward_features(x)  # Extract features without head

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



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.models import create_model
# import os

# def download_pretrained_model():
#     # Download pretrained model and save locally if not already present
#     model_dir = "models"
#     model_path = os.path.join(model_dir, "swin_tiny_patch4_window7_224.pth")

#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)

#     if not os.path.exists(model_path):
#         print("Downloading pre-trained model using timm...")
#         model = create_model('swin_tiny_patch4_window7_224', pretrained=True)
#         torch.save(model.state_dict(), model_path)
#         print(f"Model saved to {model_path}")
#     else:
#         print(f"Model already exists at {model_path}")

#     return model_path

# def build_swin_transformer(embed_dim=256, model_path="models/swin_tiny_patch4_window7_224.pth"):
#     # Load the pretrained model weights and prepare the backbone
#     if not os.path.exists(model_path):
#         print("Downloading pretrained model...")
#         model = create_model(
#             'swin_tiny_patch4_window7_224',
#             pretrained=True,
#             num_classes=0  # Remove classification head
#         )
#         os.makedirs(os.path.dirname(model_path), exist_ok=True)
#         torch.save(model.state_dict(), model_path)
#         print(f"Model saved to {model_path}")
#     else:
#         print(f"Model already exists at {model_path}")

#     # Create model and load weights
#     model = create_model(
#         'swin_tiny_patch4_window7_224',
#         pretrained=False,  # Prevent auto-download
#         num_classes=0      # Remove classification head
#     )

#     # Load pretrained weights, ignoring mismatched keys
#     pretrained_dict = torch.load(model_path, map_location="cpu")
#     model_dict = model.state_dict()
#     filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     model_dict.update(filtered_dict)
#     model.load_state_dict(model_dict)
#     print("Pretrained weights loaded successfully, ignoring mismatched keys.")
#     return model

# class HybridEncoder(nn.Module):
#     def init(self, embed_dim=256):
#         super(HybridEncoder, self).init()

#         # Swin Transformer Backbone (SwinT-Tiny)
#         self.swin_backbone = build_swin_transformer(embed_dim)
#         self.swin_proj = nn.Conv2d(768, embed_dim, kernel_size=1)  # Project to embed_dim

#     def forward(self, x):
#         # Swin Transformer Path
#         x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)  # Resize input
#         swin_features = self.swin_backbone.forward_features(x)  # Extract features without head

#         # Reshape swin_features to [B, C, H, W]
#         swin_features = swin_features.permute(0, 3, 1, 2).contiguous()  # Change to [B, C, H, W]
#         swin_features = self.swin_proj(swin_features)  # Shape: [B, embed_dim, H, W]
#         return swin_features

# # Test the HybridEncoder
# if name == "main":
#     encoder = HybridEncoder(embed_dim=256).cuda()
#     dummy_input = torch.randn(1, 3, 512, 512).cuda()  # Example input image
#     swin_features = encoder(dummy_input)

#     print("Swin Transformer Features Shape:", swin_features.shape)