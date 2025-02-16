import torch
import torch.nn as nn
import torch.nn.functional as F

class ExemplarImageMatching(nn.Module):
    def __init__(self, temperature=1.0):
        super(ExemplarImageMatching, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, exemplar_features):
        # Ensure dimensions match
        if image_features.size(-1) != exemplar_features.size(-1):
            raise ValueError(
                f"Embedding dimensions must match: "
                f"image_features={image_features.size(-1)}, exemplar_features={exemplar_features.size(-1)}"
            )

        # Normalize features for cosine similarity
        image_features = F.normalize(image_features, p=2, dim=-1)  # [B, H*W, C]
        exemplar_features = F.normalize(exemplar_features, p=2, dim=-1)  # [B, N, C]

        # Compute similarity maps with temperature scaling
        similarity = torch.matmul(image_features, exemplar_features.transpose(1, 2))  # [B, H*W, N]
        similarity = similarity / self.temperature
        
        # Apply softmax along the exemplar dimension
        attention_weights = F.softmax(similarity, dim=-1)  # [B, H*W, N]
        
        # Reshape to spatial dimensions
        similarity_maps = attention_weights.view(
            image_features.size(0), 
            int(image_features.size(1)**0.5), 
            int(image_features.size(1)**0.5),
            exemplar_features.size(1)
        ).permute(0, 3, 1, 2)  # Final shape: [B, N, H, W]

        return similarity_maps

# Test the ExemplarImageMatching
if __name__ == "__main__":
    matcher = ExemplarImageMatching(temperature=0.1).cuda()
    dummy_image_features = torch.randn(1, 196, 256).cuda()  # Example image features [B, H*W, C]
    dummy_exemplar_features = torch.randn(1, 5, 256).cuda()  # Example exemplar features [B, N, C]

    similarity_maps = matcher(dummy_image_features, dummy_exemplar_features)
    print("Similarity Maps Shape:", similarity_maps.shape)
    print("Min value:", similarity_maps.min().item())
    print("Max value:", similarity_maps.max().item())
    print("Sum per position:", similarity_maps[0, :, 0, 0].sum().item())  # Should be close to 1