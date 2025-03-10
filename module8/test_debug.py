import os
import sys
import torch
from debug_backbone import SwinBackbone, test_backbone

# Atur environment
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Gunakan GPU pertama jika ada

def main():
    print("Memulai debugging backbone...")
    
    # 1. Test backbone dengan parameter frozen
    print("\n\nMENGUJI BACKBONE DENGAN PARAMETER FROZEN")
    test_backbone(requires_grad=False)
    
    # 2. Test backbone dengan parameter unfrozen
    print("\n\nMENGUJI BACKBONE DENGAN PARAMETER UNFROZEN")
    test_backbone(requires_grad=True)
    
    print("\n\nDebugging selesai!")
    print("Periksa folder ./feature_maps untuk visualisasi feature maps")
    print("Periksa file parameter_mapping_info.json untuk informasi mapping parameter")

if __name__ == "__main__":
    main()