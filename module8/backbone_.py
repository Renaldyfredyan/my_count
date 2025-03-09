import torch
from torch import nn

class BackboneGroundingDINO(nn.Module):
    def __init__(
        self,
        config_path: str = "/home/renaldy_fredyan/PhDResearch/ELS/module8/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_path: str = "/home/renaldy_fredyan/PhDResearch/ELS/pretrained_weights/groundingdino_swint_ogc.pth",
        reduction: int = 8,
        requires_grad: bool = True
    ):
        super().__init__()
        self.reduction = reduction
        
        # Load the full GroundingDINO model using the official method
        from groundingdino.util.inference import load_model
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
        # Import transformasi yang digunakan oleh GroundingDINO
        import groundingdino.datasets.transforms as T
        from torch.nn.functional import interpolate
        
        # Kita perlu mengonversi input tensor ke format yang diharapkan oleh backbone
        # Di GroundingDINO, input adalah NestedTensor dengan tensors dan mask
        
        # 1. Buat mask yang valid (semua piksel valid, jadi mask = False)
        batch_size, _, height, width = x.shape
        mask = torch.zeros((batch_size, height, width), dtype=torch.bool, device=x.device)
        
        # 2. Import NestedTensor
        from groundingdino.models.GroundingDINO.backbone.backbone import NestedTensor
        
        # 3. Buat NestedTensor
        tensor_list = NestedTensor(tensors=x, mask=mask)
        
        # 4. Forward melalui backbone
        features = self.backbone(tensor_list)
        
        # 5. Proses output sesuai dengan struktur output yang sebenarnya
        # Dari debug, kita tahu bahwa features adalah tuple dengan panjang 2
        
        if isinstance(features, tuple) and len(features) == 2:
            # Features mungkin berisi (output_features, pos)
            # Di mana output_features adalah tensor atau list dari tensor
            output_features, pos = features
            
            # Periksa bentuk dari output_features
            if isinstance(output_features, dict):
                # Ambil 3 layer dari dictionary
                # (Sesuaikan nama layer jika perlu)
                if 'res3' in output_features and 'res4' in output_features and 'res5' in output_features:
                    s3, s4, s5 = output_features['res3'], output_features['res4'], output_features['res5']
                elif '0' in output_features and '1' in output_features and '2' in output_features:
                    s3, s4, s5 = output_features['0'], output_features['1'], output_features['2']
                elif len(output_features) >= 3:
                    # Ambil 3 layer terakhir jika ada
                    layer_names = sorted(list(output_features.keys()))
                    s3, s4, s5 = [output_features[name] for name in layer_names[-3:]]
                else:
                    # Jika kurang dari 3 layer, duplikasi yang terakhir
                    print(f"Warning: Output features has less than 3 layers: {list(output_features.keys())}")
                    layer_names = sorted(list(output_features.keys()))
                    if len(layer_names) == 2:
                        s3, s4 = [output_features[name] for name in layer_names]
                        s5 = s4  # Duplikasi layer terakhir
                    elif len(layer_names) == 1:
                        s3 = output_features[layer_names[0]]
                        s4 = s3  # Duplikasi
                        s5 = s3  # Duplikasi
                    else:
                        raise ValueError(f"Output features has no layers: {output_features}")
            elif isinstance(output_features, (list, tuple)):
                # Jika output adalah list/tuple
                if len(output_features) >= 3:
                    s3, s4, s5 = output_features[-3:]
                elif len(output_features) == 2:
                    s3, s4 = output_features
                    s5 = s4  # Duplikasi
                elif len(output_features) == 1:
                    s3 = output_features[0]
                    s4 = s3  # Duplikasi
                    s5 = s3  # Duplikasi
                else:
                    raise ValueError(f"Output features list is empty: {output_features}")
            elif isinstance(output_features, torch.Tensor):
                # Jika output adalah tensor tunggal, bagi menjadi 3 bagian
                # atau duplikasi untuk mendapatkan 3 output
                # (Ini adalah solusi darurat dan mungkin tidak ideal)
                print(f"Warning: Output features is a single tensor of shape {output_features.shape}")
                
                # Opsi 1: Bagi tensor menjadi 3 bagian sepanjang dimensi channel
                if output_features.shape[1] >= 3:
                    channels = output_features.shape[1]
                    c1, c2 = channels // 3, 2 * (channels // 3)
                    s3 = output_features[:, :c1, :, :]
                    s4 = output_features[:, c1:c2, :, :]
                    s5 = output_features[:, c2:, :, :]
                else:
                    # Opsi 2: Duplikasi tensor yang sama
                    s3 = output_features
                    s4 = output_features
                    s5 = output_features
            else:
                print(f"Unexpected output_features type: {type(output_features)}")
                if hasattr(output_features, '__dict__'):
                    print(f"output_features attributes: {output_features.__dict__}")
                raise ValueError(f"Cannot process output_features of type {type(output_features)}")
        else:
            # Untuk debugging
            print(f"Debug - features type: {type(features)}")
            if isinstance(features, tuple):
                print(f"Debug - features length: {len(features)}")
                for i, item in enumerate(features):
                    print(f"Debug - features[{i}] type: {type(item)}")
                    if isinstance(item, torch.Tensor):
                        print(f"Debug - features[{i}] shape: {item.shape}")
                    elif isinstance(item, (list, tuple)):
                        print(f"Debug - features[{i}] length: {len(item)}")
                    elif isinstance(item, dict):
                        print(f"Debug - features[{i}] keys: {list(item.keys())}")
            
            raise ValueError("Cannot process features of type {type(features)}")
        
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
        config_path="/home/renaldy_fredyan/PhDResearch/ELS/module8/groundingdino/config/GroundingDINO_SwinT_OGC.py",
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