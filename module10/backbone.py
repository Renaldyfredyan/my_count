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
        model_path = os.path.join(cache_dir, 'timm_swin_with_gdino_weights.pth')
        
        # Buat model Swin-T dari timm
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=0,
            features_only=True,
            out_indices=(1, 2, 3),
            img_size=512
        )
        
        # Channel dimensions
        self.num_channels = {
            'stage3': 192,  # S3
            'stage4': 384,  # S4
            'stage5': 768   # S5
        }
        self.total_channels = sum(self.num_channels.values())
        self.reduction = reduction
        
        # Load parameter mapping if it exists, otherwise create new mapping
        if os.path.exists(model_path):
            print(f"Loading pre-mapped weights from {model_path}")
            self.backbone.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            print("Error...")
            # # Load GroundingDINO
            # gd_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
            # gd_backbone = gd_model.model.backbone.conv_encoder.model
            
            # # Get state dicts
            # timm_state_dict = self.backbone.state_dict()
            # gd_state_dict = gd_backbone.state_dict()
            
            # # Define parameter mapping
            # # This is a starting point and may need adjustment
            # layer_mapping = {
            #     # Patch embedding mappings
            #     "embeddings.patch_embeddings.projection": "patch_embed.proj",
            #     "embeddings.norm": "patch_embed.norm",
                
            #     # Layer mappings - adjust as needed
            #     "encoder.layers.0": "layers.0",
            #     "encoder.layers.1": "layers.1",
            #     "encoder.layers.2": "layers.2",
            #     "encoder.layers.3": "layers.3",
                
            #     # Attention block mappings
            #     "attention.self.query": "blocks.0.attn.qkv",  # Note: in Swin, qkv is combined
            #     "attention.self.key": "blocks.0.attn.qkv",
            #     "attention.self.value": "blocks.0.attn.qkv",
            #     "attention.output.dense": "blocks.0.attn.proj",
            #     "attention.output.LayerNorm": "blocks.0.norm1",
                
            #     # MLP mappings
            #     "intermediate.dense": "blocks.0.mlp.fc1",
            #     "output.dense": "blocks.0.mlp.fc2",
            #     "output.LayerNorm": "blocks.0.norm2",
                
            #     # Norm layers
            #     "hidden_states_norms.stage1": "norm",
            #     "hidden_states_norms.stage2": "norm",
            #     "hidden_states_norms.stage3": "norm",
            #     "hidden_states_norms.stage4": "norm",
            # }
            
            # # Map and load parameters where possible
            # loaded_params = 0
            # total_params = len(timm_state_dict)
            
            # for timm_name in timm_state_dict.keys():
            #     for gd_prefix, timm_prefix in layer_mapping.items():
            #         # Check if current timm parameter matches any mapped prefix
            #         if timm_name.startswith(timm_prefix):
            #             # Try to find corresponding parameter in GroundingDINO
            #             potential_gd_name = timm_name.replace(timm_prefix, gd_prefix)
                        
            #             # Special handling for attention parameters which are structured differently
            #             if "qkv" in timm_name:
            #                 # Swin combines Q, K, V into one parameter, need special handling
            #                 continue
                        
            #             # Check if parameter exists in GroundingDINO state dict
            #             if potential_gd_name in gd_state_dict:
            #                 # Check if shapes match
            #                 if gd_state_dict[potential_gd_name].shape == timm_state_dict[timm_name].shape:
            #                     timm_state_dict[timm_name] = gd_state_dict[potential_gd_name]
            #                     loaded_params += 1
            #                     break
            
            # print(f"Mapped {loaded_params}/{total_params} parameters successfully")
            
            # # Save mapped parameters for future use
            # if loaded_params > 0:
            #     self.backbone.load_state_dict(timm_state_dict, strict=False)
            #     torch.save(self.backbone.state_dict(), model_path)
            #     print(f"Saved mapped parameters to {model_path}")
            # else:
            #     print("Warning: No parameters were successfully mapped!")

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
        if s3.shape[1] != self.num_channels['stage3']:
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