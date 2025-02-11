import torch
import torch.nn as nn
import torch.nn.functional as F

class ExemplarImageMatching(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        
        self.matching_network = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, 1, 1)
        )

    def forward(self, image_features, exemplar_features):
        # image_features is [B, HW, C], reshape it to [B, C, H, W]
        B, L, C = image_features.shape
        H = W = int(L ** 0.5)
        image_features = image_features.permute(0, 2, 1).view(B, C, H, W)
        
        # exemplar_features is [B, N, C]
        _, N, _ = exemplar_features.shape
        
        # Compute similarity using dot product
        exemplar_features = exemplar_features.permute(0, 2, 1)  # [B, C, N]
        similarity = torch.bmm(
            image_features.view(B, C, H*W).transpose(1, 2),  # [B, HW, C]
            exemplar_features  # [B, C, N]
        )  # [B, HW, N]
        
        # Reshape to spatial dimensions
        similarity_maps = similarity.view(B, H, W, N).permute(0, 3, 1, 2)  # [B, N, H, W]
        
        return similarity_maps