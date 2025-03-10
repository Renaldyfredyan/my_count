import os
import torch
from torch import nn
import timm
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForZeroShotObjectDetection

def visualize_features(features, output_dir="./feature_maps"):
    """Visualisasi feature maps dari backbone"""
    os.makedirs(output_dir, exist_ok=True)
    
    for level, feat in enumerate(features):
        # Ambil satu batch dan rata-ratakan di seluruh channel
        feature_map = feat[0].mean(dim=0).detach().cpu().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(feature_map, cmap='viridis')
        plt.colorbar()
        plt.title(f'Feature Level {level} (S{level+3})')
        plt.savefig(os.path.join(output_dir, f'feature_level_{level}.png'))
        plt.close()
        
        # Simpan juga histogram distribusi nilai
        plt.figure(figsize=(8, 6))
        plt.hist(feature_map.flatten(), bins=50)
        plt.title(f'Feature Level {level} (S{level+3}) - Histogram')
        plt.savefig(os.path.join(output_dir, f'feature_level_{level}_hist.png'))
        plt.close()

def print_param_stats(model, prefix=""):
    """Cetak statistik parameter dari model"""
    trainable_params = 0
    frozen_params = 0
    
    stats = []
    for name, param in model.named_parameters():
        param_data = param.detach().cpu().numpy()
        stat = {
            "name": f"{prefix}.{name}",
            "shape": list(param.shape),
            "min": float(param_data.min()),
            "max": float(param_data.max()),
            "mean": float(param_data.mean()),
            "std": float(param_data.std()),
            "trainable": param.requires_grad
        }
        stats.append(stat)
        
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()
    
    print(f"Trainable params: {trainable_params:,}")
    print(f"Frozen params: {frozen_params:,}")
    
    return stats

class SwinBackbone(nn.Module):
    def __init__(
        self,
        reduction: int = 8,
        requires_grad: bool = False,
        cache_dir: str = "./pretrained_models",
        debug: bool = True  # Tambahkan flag debug
    ):
        super().__init__()
        
        self.debug = debug
        
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
        
        # Cetak struktur backbone
        if debug:
            print(f"Backbone structure:")
            for name, module in self.backbone.named_modules():
                if len(list(module.children())) == 0:  # Hanya cetak leaf modules
                    print(f"  {name}: {module}")
        
        # Load parameter mapping if it exists, otherwise create new mapping
        if os.path.exists(model_path):
            print(f"Loading pre-mapped weights from {model_path}")
            self.backbone.load_state_dict(torch.load(model_path, weights_only=True))
            if debug:
                stats = print_param_stats(self.backbone, "backbone")
        else:
            print("Creating new parameter mapping from GroundingDINO...")
            # Load GroundingDINO
            gd_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
            gd_backbone = gd_model.model.backbone.conv_encoder.model
            
            # Debug: Bandingkan arsitektur
            if debug:
                print("\nGroundingDINO Backbone Structure:")
                for name, module in gd_backbone.named_modules():
                    if len(list(module.children())) == 0:  # Hanya cetak leaf modules
                        print(f"  {name}: {module}")
                
                print("\nParameter stats sebelum mapping:")
                before_stats = print_param_stats(self.backbone, "before")
            
            # Get state dicts
            timm_state_dict = self.backbone.state_dict()
            gd_state_dict = gd_backbone.state_dict()
            
            # Cetak beberapa key dari setiap state dict untuk debugging
            if debug:
                print("\nSample TIMM keys:")
                for i, k in enumerate(timm_state_dict.keys()):
                    if i < 10:  # Cetak 10 key pertama
                        print(f"  {k}: {timm_state_dict[k].shape}")
                
                print("\nSample GroundingDINO keys:")
                for i, k in enumerate(gd_state_dict.keys()):
                    if i < 10:  # Cetak 10 key pertama
                        print(f"  {k}: {gd_state_dict[k].shape}")
            
            # Define parameter mapping
            # This is a starting point and may need adjustment
            layer_mapping = {
                # Patch embedding
                "patch_embed.proj": "embeddings.patch_embeddings.projection",
                "patch_embed.norm": "embeddings.norm",
                
                # Layer normalization mapping - layer 0
                "layers_0.blocks.0.norm1": "encoder.layers.0.blocks.0.layernorm_before",
                "layers_0.blocks.0.norm2": "encoder.layers.0.blocks.0.layernorm_after",
                "layers_0.blocks.1.norm1": "encoder.layers.0.blocks.1.layernorm_before",
                "layers_0.blocks.1.norm2": "encoder.layers.0.blocks.1.layernorm_after",
                
                # Layer normalization mapping - layer 1
                "layers_1.blocks.0.norm1": "encoder.layers.1.blocks.0.layernorm_before",
                "layers_1.blocks.0.norm2": "encoder.layers.1.blocks.0.layernorm_after",
                "layers_1.blocks.1.norm1": "encoder.layers.1.blocks.1.layernorm_before",
                "layers_1.blocks.1.norm2": "encoder.layers.1.blocks.1.layernorm_after",
                
                # Layer normalization mapping - layer 2
                "layers_2.blocks.0.norm1": "encoder.layers.2.blocks.0.layernorm_before",
                "layers_2.blocks.0.norm2": "encoder.layers.2.blocks.0.layernorm_after",
                "layers_2.blocks.1.norm1": "encoder.layers.2.blocks.1.layernorm_before",
                "layers_2.blocks.1.norm2": "encoder.layers.2.blocks.1.layernorm_after",
                "layers_2.blocks.2.norm1": "encoder.layers.2.blocks.2.layernorm_before",
                "layers_2.blocks.2.norm2": "encoder.layers.2.blocks.2.layernorm_after",
                "layers_2.blocks.3.norm1": "encoder.layers.2.blocks.3.layernorm_before",
                "layers_2.blocks.3.norm2": "encoder.layers.2.blocks.3.layernorm_after",
                "layers_2.blocks.4.norm1": "encoder.layers.2.blocks.4.layernorm_before",
                "layers_2.blocks.4.norm2": "encoder.layers.2.blocks.4.layernorm_after",
                "layers_2.blocks.5.norm1": "encoder.layers.2.blocks.5.layernorm_before",
                "layers_2.blocks.5.norm2": "encoder.layers.2.blocks.5.layernorm_after",
                
                # Layer normalization mapping - layer 3
                "layers_3.blocks.0.norm1": "encoder.layers.3.blocks.0.layernorm_before",
                "layers_3.blocks.0.norm2": "encoder.layers.3.blocks.0.layernorm_after",
                "layers_3.blocks.1.norm1": "encoder.layers.3.blocks.1.layernorm_before",
                "layers_3.blocks.1.norm2": "encoder.layers.3.blocks.1.layernorm_after",
                
                # Projection mapping - layer 0
                "layers_0.blocks.0.attn.proj": "encoder.layers.0.blocks.0.attention.output.dense",
                "layers_0.blocks.1.attn.proj": "encoder.layers.0.blocks.1.attention.output.dense",
                
                # Projection mapping - layer 1
                "layers_1.blocks.0.attn.proj": "encoder.layers.1.blocks.0.attention.output.dense",
                "layers_1.blocks.1.attn.proj": "encoder.layers.1.blocks.1.attention.output.dense",
                
                # Projection mapping - layer 2
                "layers_2.blocks.0.attn.proj": "encoder.layers.2.blocks.0.attention.output.dense",
                "layers_2.blocks.1.attn.proj": "encoder.layers.2.blocks.1.attention.output.dense",
                "layers_2.blocks.2.attn.proj": "encoder.layers.2.blocks.2.attention.output.dense",
                "layers_2.blocks.3.attn.proj": "encoder.layers.2.blocks.3.attention.output.dense",
                "layers_2.blocks.4.attn.proj": "encoder.layers.2.blocks.4.attention.output.dense",
                "layers_2.blocks.5.attn.proj": "encoder.layers.2.blocks.5.attention.output.dense",
                
                # Projection mapping - layer 3
                "layers_3.blocks.0.attn.proj": "encoder.layers.3.blocks.0.attention.output.dense",
                "layers_3.blocks.1.attn.proj": "encoder.layers.3.blocks.1.attention.output.dense",
                
                # MLP mapping - layer 0
                "layers_0.blocks.0.mlp.fc1": "encoder.layers.0.blocks.0.intermediate.dense",
                "layers_0.blocks.0.mlp.fc2": "encoder.layers.0.blocks.0.output.dense",
                "layers_0.blocks.1.mlp.fc1": "encoder.layers.0.blocks.1.intermediate.dense",
                "layers_0.blocks.1.mlp.fc2": "encoder.layers.0.blocks.1.output.dense",
                
                # MLP mapping - layer 1
                "layers_1.blocks.0.mlp.fc1": "encoder.layers.1.blocks.0.intermediate.dense",
                "layers_1.blocks.0.mlp.fc2": "encoder.layers.1.blocks.0.output.dense",
                "layers_1.blocks.1.mlp.fc1": "encoder.layers.1.blocks.1.intermediate.dense",
                "layers_1.blocks.1.mlp.fc2": "encoder.layers.1.blocks.1.output.dense",
                
                # MLP mapping - layer 2
                "layers_2.blocks.0.mlp.fc1": "encoder.layers.2.blocks.0.intermediate.dense",
                "layers_2.blocks.0.mlp.fc2": "encoder.layers.2.blocks.0.output.dense",
                "layers_2.blocks.1.mlp.fc1": "encoder.layers.2.blocks.1.intermediate.dense",
                "layers_2.blocks.1.mlp.fc2": "encoder.layers.2.blocks.1.output.dense",
                "layers_2.blocks.2.mlp.fc1": "encoder.layers.2.blocks.2.intermediate.dense",
                "layers_2.blocks.2.mlp.fc2": "encoder.layers.2.blocks.2.output.dense",
                "layers_2.blocks.3.mlp.fc1": "encoder.layers.2.blocks.3.intermediate.dense",
                "layers_2.blocks.3.mlp.fc2": "encoder.layers.2.blocks.3.output.dense",
                "layers_2.blocks.4.mlp.fc1": "encoder.layers.2.blocks.4.intermediate.dense",
                "layers_2.blocks.4.mlp.fc2": "encoder.layers.2.blocks.4.output.dense",
                "layers_2.blocks.5.mlp.fc1": "encoder.layers.2.blocks.5.intermediate.dense",
                "layers_2.blocks.5.mlp.fc2": "encoder.layers.2.blocks.5.output.dense",
                
                # MLP mapping - layer 3
                "layers_3.blocks.0.mlp.fc1": "encoder.layers.3.blocks.0.intermediate.dense",
                "layers_3.blocks.0.mlp.fc2": "encoder.layers.3.blocks.0.output.dense",
                "layers_3.blocks.1.mlp.fc1": "encoder.layers.3.blocks.1.intermediate.dense",
                "layers_3.blocks.1.mlp.fc2": "encoder.layers.3.blocks.1.output.dense",
                
                # Downsample mapping
                "layers_1.downsample.norm": "encoder.layers.0.downsample.norm",
                "layers_1.downsample.reduction": "encoder.layers.0.downsample.reduction",
                "layers_2.downsample.norm": "encoder.layers.1.downsample.norm",
                "layers_2.downsample.reduction": "encoder.layers.1.downsample.reduction",
                "layers_3.downsample.norm": "encoder.layers.2.downsample.norm",
                "layers_3.downsample.reduction": "encoder.layers.2.downsample.reduction"
            }
            # Map and load parameters where possible
            loaded_params = 0
            total_params = len(timm_state_dict)
            qkv_mapped = 0
            qkv_total = 0

            # Debugging: simpan info parameter yang berhasil dimapping
            mapping_info = []

            # Handle QKV parameters first (special case)
            for timm_name in timm_state_dict.keys():
                if "qkv" in timm_name:
                    qkv_total += 1
                    # Extract layer and block information
                    parts = timm_name.split('.')
                    if len(parts) >= 4 and parts[0].startswith("layers_") and parts[2] == "blocks":
                        layer_idx = int(parts[0][7:])  # Extract number after "layers_"
                        block_idx = int(parts[3])      # Block index
                        param_type = parts[-1]         # "weight" or "bias"
                        
                        # Construct corresponding Q, K, V parameter names in GroundingDINO
                        q_name = f"encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.{param_type}"
                        k_name = f"encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.{param_type}"
                        v_name = f"encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.{param_type}"
                        
                        # Check if all three parameters exist
                        if q_name in gd_state_dict and k_name in gd_state_dict and v_name in gd_state_dict:
                            q_param = gd_state_dict[q_name]
                            k_param = gd_state_dict[k_name]
                            v_param = gd_state_dict[v_name]
                            
                            # Combine Q, K, V into a single parameter
                            # Note: Shape should be checked and adjusted as needed
                            if timm_state_dict[timm_name].shape[0] == 3 * q_param.shape[0]:
                                combined = torch.cat([q_param, k_param, v_param], dim=0)
                                
                                # Update TIMM state dict
                                timm_state_dict[timm_name] = combined
                                loaded_params += 1
                                qkv_mapped += 1
                                
                                mapping_info.append({
                                    "timm_name": timm_name,
                                    "gd_name": f"{q_name}, {k_name}, {v_name}",
                                    "timm_shape": list(timm_state_dict[timm_name].shape),
                                    "gd_shape": [list(q_param.shape), list(k_param.shape), list(v_param.shape)],
                                    "status": "QKV mapped successfully"
                                })
            # Regular parameter mapping
            for timm_name in timm_state_dict.keys():
                mapped = False
                
                # Skip QKV parameters yang sudah diproses
                if "qkv" in timm_name:
                    continue
                    
                # Regular parameter mapping
                for timm_prefix, gd_prefix in layer_mapping.items():
                    # Check if current timm parameter matches any mapped prefix
                    if timm_name.startswith(timm_prefix):
                        # Try to find corresponding parameter in GroundingDINO
                        suffix = timm_name[len(timm_prefix):]  # Get the suffix (e.g. ".weight", ".bias")
                        potential_gd_name = gd_prefix + suffix
                        
                        # Check if parameter exists in GroundingDINO state dict
                        if potential_gd_name in gd_state_dict:
                            # Check if shapes match
                            if gd_state_dict[potential_gd_name].shape == timm_state_dict[timm_name].shape:
                                # Copy weights
                                timm_state_dict[timm_name] = gd_state_dict[potential_gd_name]
                                loaded_params += 1
                                
                                # Log mapping info
                                mapping_info.append({
                                    "timm_name": timm_name,
                                    "gd_name": potential_gd_name,
                                    "timm_shape": list(timm_state_dict[timm_name].shape),
                                    "gd_shape": list(gd_state_dict[potential_gd_name].shape),
                                    "status": "Mapped successfully"
                                })
                                mapped = True
                                break
                            else:
                                # Shapes don't match
                                mapping_info.append({
                                    "timm_name": timm_name,
                                    "gd_name": potential_gd_name,
                                    "timm_shape": list(timm_state_dict[timm_name].shape),
                                    "gd_shape": list(gd_state_dict[potential_gd_name].shape),
                                    "status": "Shape mismatch"
                                })
                
                if not mapped and debug:
                    mapping_info.append({
                        "timm_name": timm_name,
                        "gd_name": "No match found",
                        "timm_shape": list(timm_state_dict[timm_name].shape),
                        "gd_shape": None,
                        "status": "No mapping rule"
                    })

            print(f"Mapped {loaded_params}/{total_params} parameters successfully")
            print(f"QKV parameters: {qkv_mapped}/{qkv_total} potential matches found")

            # Debug: Simpan mapping info ke file
            if debug:
                import json
                with open('parameter_mapping_info.json', 'w') as f:
                    json.dump(mapping_info, f, indent=2)

            # Save mapped parameters for future use
            if loaded_params > 0:
                self.backbone.load_state_dict(timm_state_dict, strict=False)
                torch.save(self.backbone.state_dict(), model_path)
                print(f"Saved mapped parameters to {model_path}")
                
                if debug:
                    print("\nParameter stats setelah mapping:")
                    after_stats = print_param_stats(self.backbone, "after")
            else:
                print("Warning: No parameters were successfully mapped!")

        if debug:
            if requires_grad:
                print("Backbone parameters set to trainable (requires_grad=True)")
            else:
                print("Backbone parameters frozen (requires_grad=False)")

    def forward_multiscale(self, x):
        """Return multi-scale features (S3, S4, S5)"""
        if self.debug:
            print(f"Input shape: {x.shape}")
            
            # Cek apakah input memiliki nilai NaN atau Inf
            has_nan = torch.isnan(x).any()
            has_inf = torch.isinf(x).any()
            if has_nan or has_inf:
                print(f"WARNING: Input contains NaN: {has_nan}, Inf: {has_inf}")
                
            # Cetak statistik input
            print(f"Input stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
        
        features = self.backbone(x)
        
        s3, s4, s5 = features
        
        if self.debug:
            print(f"S3 shape: {s3.shape}, min: {s3.min().item():.4f}, max: {s3.max().item():.4f}, mean: {s3.mean().item():.4f}, std: {s3.std().item():.4f}")
            print(f"S4 shape: {s4.shape}, min: {s4.min().item():.4f}, max: {s4.max().item():.4f}, mean: {s4.mean().item():.4f}, std: {s4.std().item():.4f}")
            print(f"S5 shape: {s5.shape}, min: {s5.min().item():.4f}, max: {s5.max().item():.4f}, mean: {s5.mean().item():.4f}, std: {s5.std().item():.4f}")
            
            # Cek untuk NaN/Inf
            for i, feat in enumerate([s3, s4, s5]):
                if torch.isnan(feat).any() or torch.isinf(feat).any():
                    print(f"WARNING: Feature S{i+3} contains NaN/Inf values!")
            
            # Visualisasi fitur
            visualize_features([s3, s4, s5])
        
        return s3, s4, s5
    
    def forward_concatenated(self, x):
        s3, s4, s5 = self.forward_multiscale(x)
        
        # Permute dari BHWC ke BCHW jika diperlukan
        if s3.shape[1] != self.num_channels['stage3']:
            if self.debug:
                print("Permuting feature dimensions from BHWC to BCHW")
            s3 = s3.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            s4 = s4.permute(0, 3, 1, 2)
            s5 = s5.permute(0, 3, 1, 2)
        
        # Get target size based on reduction factor
        size = x.size(-2) // self.reduction, x.size(-1) // self.reduction
        
        if self.debug:
            print(f"Target size after reduction: {size}")
        
        # Interpolate features to target size
        s3 = nn.functional.interpolate(s3, size=size, mode='bilinear', align_corners=True)
        s4 = nn.functional.interpolate(s4, size=size, mode='bilinear', align_corners=True)
        s5 = nn.functional.interpolate(s5, size=size, mode='bilinear', align_corners=True)
        
        # Concatenate
        x = torch.cat([s3, s4, s5], dim=1)
        
        if self.debug:
            print(f"Concatenated feature shape: {x.shape}")
            print(f"Expected channels: {self.total_channels}, Actual: {x.shape[1]}")
        
        return x

    def forward(self, x):
        """Default forward returns concatenated features"""
        return self.forward_concatenated(x)


def test_backbone(requires_grad=False):
    """Test function for the backbone with dummy data"""
    print("\n" + "="*50)
    print("TESTING BACKBONE")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Buat backbone
    backbone = SwinBackbone(reduction=8, requires_grad=requires_grad, debug=True)
    backbone.to(device)
    
    # Pindahkan ke mode eval atau train sesuai requires_grad
    if requires_grad:
        backbone.train()
        print("Backbone in TRAIN mode")
    else:
        backbone.eval()
        print("Backbone in EVAL mode")
    
    # Buat dummy optimizer untuk tes backward jika requires_grad=True
    if requires_grad:
        optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-4)
    
    # Create dummy input tensor (simulasi batch dengan 2 gambar)
    x = torch.randn(2, 3, 512, 512).to(device)
    print(f"Input tensor shape: {x.shape}")
    
    try:
        # Test forward_multiscale
        with torch.set_grad_enabled(requires_grad):
            print("\nTesting forward_multiscale...")
            s3, s4, s5 = backbone.forward_multiscale(x)
            print(f"S3 shape: {s3.shape}")
            print(f"S4 shape: {s4.shape}")
            print(f"S5 shape: {s5.shape}")
        
            # Test forward_concatenated
            print("\nTesting forward_concatenated...")
            concat = backbone.forward_concatenated(x)
            print(f"Concatenated shape: {concat.shape}")
            print(f"Expected channels: {backbone.total_channels}, Actual: {concat.shape[1]}")
        
            # Test backward pass jika requires_grad=True
            if requires_grad:
                print("\nTesting backward pass...")
                # Simulasi loss sederhana (mean dari output)
                loss = concat.mean()
                print(f"Dummy loss: {loss.item()}")
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                # Cek gradients
                grad_norms = []
                for name, param in backbone.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_norms.append((name, grad_norm))
                
                # Cetak 5 gradients dengan norm tertinggi
                print("Top 5 gradients by norm:")
                for name, norm in sorted(grad_norms, key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {name}: {norm:.6f}")
                
                optimizer.step()
                print("Optimizer step completed")
        
    except Exception as e:
        print(f"Error testing backbone: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("BACKBONE TEST COMPLETE")
    print("="*50)


if __name__ == "__main__":
    # Test dengan parameter frozen
    test_backbone(requires_grad=False)
    
    # Test dengan parameter unfrozen
    test_backbone(requires_grad=True)