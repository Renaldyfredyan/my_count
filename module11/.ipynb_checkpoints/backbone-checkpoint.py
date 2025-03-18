import os
import torch
from torch import nn
import timm
from transformers import AutoModelForZeroShotObjectDetection

class SwinBackbone(nn.Module):
    def __init__(
        self,
        reduction: int = 8,
        requires_grad: bool = False,
        cache_dir: str = "./pretrained_models"
    ):
        super().__init__()
        
        # Setup path untuk menyimpan model
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, 'timm_swin_base_with_gdino_weights.pth')
        
        # Buat model Swin-B dari timm
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=False,
            num_classes=0,
            features_only=True,
            out_indices=(1, 2, 3),
            img_size=512
        )
        
        # Channel dimensions disesuaikan dengan hasil output yang terlihat
        self.num_channels = {
            'stage3': 256,  # S3 (hasil sebenarnya)
            'stage4': 512,  # S4 (hasil sebenarnya)
            'stage5': 1024  # S5 (hasil sebenarnya)
        }
        self.total_channels = sum(self.num_channels.values())  # Sekarang menjadi 1792
        self.reduction = reduction
        
        # Load parameter mapping if it exists, otherwise create new mapping
        if os.path.exists(model_path):
            print(f"Loading pre-mapped weights from {model_path}")
            self.backbone.load_state_dict(torch.load(model_path, weights_only=True))
        # else:
        #     print("Loading GroundingDINO base model untuk mapping...")
        #     # Load GroundingDINO (gunakan model base jika tersedia)
        #     gd_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
        #     gd_backbone = gd_model.model.backbone.conv_encoder.model
            
        #     # Untuk mapping parameter dengan debug yang lebih detail, jalankan debug_backbone.py terlebih dahulu
        #     print("Error: Pre-mapped weights tidak ditemukan!")
        #     print("Jalankan debug_backbone.py terlebih dahulu untuk membuat mapping parameter")
        #     print(f"Weights akan disimpan ke {model_path}")

        # Set requires_grad
        for param in self.backbone.parameters():
            param.requires_grad_(requires_grad)

    def forward_multiscale(self, x):
        """Return multi-scale features (S3, S4, S5)"""
        features = self.backbone(x)
        s3, s4, s5 = features
        return s3, s4, s5
    
    def forward_concatenated(self, x):
        s3, s4, s5 = self.forward_multiscale(x)
        
        # Permute dari BHWC ke BCHW jika diperlukan
        if len(s3.shape) == 4 and s3.shape[1] != s3.shape[3]:  # Format BHWC
            s3 = s3.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            s4 = s4.permute(0, 3, 1, 2)
            s5 = s5.permute(0, 3, 1, 2)
        
        # Get target size based on reduction factor
        size = x.size(-2) // self.reduction, x.size(-1) // self.reduction
        
        # Interpolate features to target size
        s3 = nn.functional.interpolate(s3, size=size, mode='bilinear', align_corners=True)
        s4 = nn.functional.interpolate(s4, size=size, mode='bilinear', align_corners=True)
        s5 = nn.functional.interpolate(s5, size=size, mode='bilinear', align_corners=True)
        
        # Concatenate
        x = torch.cat([s3, s4, s5], dim=1)
        
        return x

    def forward(self, x):
        """Default forward returns concatenated features"""
        return self.forward_concatenated(x)


def test_backbone():
    """Test function for the backbone"""
    backbone = SwinBackbone(reduction=8, requires_grad=False)
    
    # Create dummy input tensor
    x = torch.randn(2, 3, 512, 512)
    
    try:
        # Test forward_multiscale
        s3, s4, s5 = backbone.forward_multiscale(x)
        print(f"S3 shape: {s3.shape}")
        print(f"S4 shape: {s4.shape}")
        print(f"S5 shape: {s5.shape}")
        
        # Test forward_concatenated
        concat = backbone.forward_concatenated(x)
        print(f"Concatenated shape: {concat.shape}")
        print(f"Expected channels: {backbone.total_channels}, Actual: {concat.shape[1]}")
        
    except Exception as e:
        print(f"Error testing backbone: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_backbone()