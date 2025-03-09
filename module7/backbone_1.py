import torch
from torch import nn
import timm
import os

class BackboneGroundingDINO(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        reduction: int = 8,
        model_path: str = "pretrained_weights/groundingdino_swint_ogc.pth",
        requires_grad: bool = True
    ):
        super().__init__()
        self.reduction = reduction

        # Inisialisasi backbone SwinT-Tiny dengan fitur multi-scale.
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=0,
            features_only=True,
            out_indices=(1, 2, 3),
            img_size=512
        )

        if pretrained:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Checkpoint GroundingDINO tidak ditemukan di {model_path}. "
                    "Silakan sediakan file checkpoint yang sesuai."
                )

            # # Muat state_dict dari checkpoint GroundingDINO
            # pretrained_dict = torch.load(model_path, map_location="cpu")
            # model_dict = self.backbone.state_dict()
            # filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # model_dict.update(filtered_dict)
            # self.backbone.load_state_dict(model_dict, strict=False)

            # More detailed loading process
            pretrained_dict = torch.load(model_path, map_location="cpu")
            if "model" in pretrained_dict:  # Handle different checkpoint formats
                pretrained_dict = pretrained_dict["model"]

            # Extract backbone-specific weights (may need to handle prefix differences)
            backbone_dict = {}
            for k, v in pretrained_dict.items():
                if k.startswith("backbone."):  # Adjust based on actual structure
                    backbone_key = k.replace("backbone.", "")
                    backbone_dict[backbone_key] = v

            model_dict = self.backbone.state_dict()
            filtered_dict = {k: v for k, v in backbone_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            # Di BackboneGroundingDINO.__init__
            print(f"Loaded {len(filtered_dict)}/{len(model_dict)} backbone parameters")
            
            model_dict.update(filtered_dict)
            self.backbone.load_state_dict(model_dict, strict=False)

        # Atur agar parameter backbone di-freeze (atau partially freeze) sesuai kebutuhan
        for param in self.backbone.parameters():
            param.requires_grad_(requires_grad)

        # Dimensi channel pada masing-masing stage untuk SwinT-Tiny
        self.num_channels = {
            'stage3': 192,
            'stage4': 384,
            'stage5': 768
        }
        self.total_channels = sum(self.num_channels.values())

    def forward_multiscale(self, x):
        """Mengembalikan fitur multi-scale (S3, S4, S5) dari backbone"""
        features = self.backbone(x)
        s3, s4, s5 = features
        return s3, s4, s5

    def forward_concatenated(self, x):
        s3, s4, s5 = self.forward_multiscale(x)
        # Pastikan output memiliki format [B, C, H, W]
        s3 = s3.permute(0, 3, 1, 2)
        s4 = s4.permute(0, 3, 1, 2)
        s5 = s5.permute(0, 3, 1, 2)

        # Resize fitur ke resolusi (input // reduction)
        size = (x.size(-2) // self.reduction, x.size(-1) // self.reduction)
        s3 = nn.functional.interpolate(s3, size=size, mode='bilinear', align_corners=True)
        s4 = nn.functional.interpolate(s4, size=size, mode='bilinear', align_corners=True)
        s5 = nn.functional.interpolate(s5, size=size, mode='bilinear', align_corners=True)

        # Gabungkan fitur dari ketiga stage
        x = torch.cat([s3, s4, s5], dim=1)
        return x

    def forward(self, x):
        """Forward default mengembalikan fitur yang telah digabungkan"""
        return self.forward_concatenated(x)
