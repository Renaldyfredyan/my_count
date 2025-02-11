import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.ops.misc import FrozenBatchNorm2d
from timm import create_model  # Untuk import Swin Transformer

import os
import torch
from timm import create_model
from pathlib import Path

from torchvision.ops import DeformConv2d  # Kita bisa gunakan ini sebagai base

class DeformableAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.offset_conv = nn.Conv2d(dim, 2*3*3, kernel_size=3, padding=1)  # 2*3*3 untuk x,y offsets
        self.deform_conv = DeformConv2d(dim, dim, kernel_size=3, padding=1)
        
        # Attention components
        self.q_conv = nn.Conv2d(dim, dim, 1)
        self.k_conv = nn.Conv2d(dim, dim, 1)
        self.v_conv = nn.Conv2d(dim, dim, 1)
        self.scale = dim ** -0.5

    def forward(self, s3, s4, s5):
        # Compute deformable offsets
        s3_offset = self.offset_conv(s3)
        s4_offset = self.offset_conv(s4)
        s5_offset = self.offset_conv(s5)
        
        # Apply deformable convolution
        s3_deform = self.deform_conv(s3, s3_offset)
        s4_deform = self.deform_conv(s4, s4_offset)
        s5_deform = self.deform_conv(s5, s5_offset)
        
        # Compute attention
        q3, k3, v3 = self.q_conv(s3_deform), self.k_conv(s3_deform), self.v_conv(s3_deform)
        q4, k4, v4 = self.q_conv(s4_deform), self.k_conv(s4_deform), self.v_conv(s4_deform)
        q5, k5, v5 = self.q_conv(s5_deform), self.k_conv(s5_deform), self.v_conv(s5_deform)
        
        # Scale dot-product attention
        attn3 = (q3 @ k3.transpose(-2, -1)) * self.scale
        attn4 = (q4 @ k4.transpose(-2, -1)) * self.scale
        attn5 = (q5 @ k5.transpose(-2, -1)) * self.scale
        
        # Softmax and matmul with values
        s3_out = (attn3.softmax(dim=-1) @ v3)
        s4_out = (attn4.softmax(dim=-1) @ v4)
        s5_out = (attn5.softmax(dim=-1) @ v5)
        
        return s3_out, s4_out, s5_out

def load_swin_model(pretrained=True, model_dir='pretrained_models'):
    model_name = 'swin_tiny_patch4_window7_224'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = Path(model_dir) / f'{model_name}.pth'
    
    if pretrained:
        if model_path.exists():
            print(f"Loading saved model from {model_path}")
            model = create_model(
                model_name, 
                pretrained=False, 
                num_classes=0,
                img_size=(512, 512)
            )
            state_dict = torch.load(model_path, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Downloading model {model_name}...")
            model = create_model(
                model_name, 
                pretrained=True, 
                num_classes=0,
                img_size=(512, 512)
            )
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    else:
        model = create_model(
            model_name, 
            pretrained=False, 
            num_classes=0,
            img_size=(512, 512)
        )

    def _forward_features(self, x):
        x = self.patch_embed(x)
        
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
            
        return features
    
    model.forward_features = _forward_features.__get__(model)
    
    return model

class HybridEncoder(nn.Module):
    def __init__(self, dim=256, encoder_type='hybrid'):
        super().__init__()
        self.encoder_type = encoder_type
        
        # Projections
        self.conv_s3 = nn.Conv2d(192, dim, 1)
        self.conv_s4 = nn.Conv2d(384, dim, 1)
        self.conv_s5 = nn.Conv2d(768, dim, 1)
        
        if encoder_type == 'standard':
            # Standard transformer layers
            self.transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=dim, nhead=8)
                for _ in range(3)
            ])
            
        elif encoder_type == 'deformable':
            # Deformable attention layers
            self.deform_attn = nn.ModuleList([
                DeformableAttention(dim) for _ in range(3)
            ])
            
        elif encoder_type == 'hybrid':
            # Self attention only for high-level features
            self.self_attn = nn.MultiheadAttention(dim, num_heads=8)
            
            # Cross-scale fusion
            self.fusion = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dim*2, dim, 1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU()
                ) for _ in range(2)
            ])

    def forward(self, features):
        # Project features
        s3, s4, s5 = [conv(f) for f, conv in zip(features, 
                      [self.conv_s3, self.conv_s4, self.conv_s5])]
        
        if self.encoder_type == 'standard':
            # Standard transformer processing
            x = torch.cat([s3, s4, s5], dim=1)
            for layer in self.transformer_layers:
                x = layer(x)
            return x
            
        elif self.encoder_type == 'deformable':
            # Multi-scale deformable attention
            for layer in self.deform_attn:
                s3, s4, s5 = layer(s3, s4, s5)
            return torch.cat([s3, s4, s5], dim=1)
            
        else: # hybrid
            # Self-attention only on high-level features
            b, c, h, w = s5.shape
            s5_flat = s5.flatten(2).permute(2, 0, 1)
            s5_attn = self.self_attn(s5_flat, s5_flat, s5_flat)[0]
            s5 = s5_attn.permute(1, 2, 0).view(b, c, h, w)
            
            # Cross-scale fusion
            p4 = self.fusion[0](torch.cat([
                F.interpolate(s5, size=s4.shape[-2:], mode='bilinear', align_corners=True),
                s4
            ], dim=1))
            
            p3 = self.fusion[1](torch.cat([
                F.interpolate(p4, size=s3.shape[-2:], mode='bilinear', align_corners=True),
                s3
            ], dim=1))
            
            return p3

    
class Backbone(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        encoder_type: str,
        pretrained: bool,
        dilation: bool,
        reduction: int,
        swav: bool,
        requires_grad: bool
    ):
        super(Backbone, self).__init__()
        self.reduction = reduction
        self.encoder_type = encoder_type
        self.hybrid_encoder = HybridEncoder(dim=256, encoder_type=encoder_type)
        
        if backbone_name == 'swinT1k': 
            self.backbone = load_swin_model(
                pretrained=pretrained,
                model_dir='pretrained_models'
            )
            self.num_channels = 256
            
            if requires_grad:
                for param in self.backbone.parameters():
                    param.requires_grad_(True)
            else:
                for param in self.backbone.parameters():
                    param.requires_grad_(False)
                    
    def forward(self, x):
        # Add input validation
        if x is None:
            raise ValueError("Input tensor cannot be None")
            
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected input to be torch.Tensor, got {type(x)}")
            
        # Extract features
        try:
            features = self.backbone.forward_features(x)
            if features is None or len(features) < 3:
                raise ValueError("Backbone returned invalid features")
                
            # Convert features to the correct format
            features = [f.permute(0, 3, 1, 2) for f in features[-3:]]
            
            # Process through hybrid encoder
            x = self.hybrid_encoder(features)
            
            # Final resize based on reduction factor
            size = x.size(-2) // self.reduction, x.size(-1) // self.reduction
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
            
            return x
            
        except Exception as e:
            print(f"Error in backbone forward pass: {str(e)}")
            raise
            
                    
