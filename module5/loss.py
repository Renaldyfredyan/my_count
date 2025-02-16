import torch
import torch.nn as nn
import torch.nn.functional as F

class FSCLoss(nn.Module):
    """
    Loss function sesuai paper:
    L = Lc + λLaux
    dimana:
    - Lc: MSE loss antara predicted dan ground-truth density maps
    - Laux: Loss untuk auxiliary density maps
    - λ = 0.3
    """
    def __init__(self, lambda_aux=0.3):
        super().__init__()
        self.lambda_aux = lambda_aux
        
    def forward(self, pred_density, gt_density, aux_density_maps=None, gt_aux_maps=None):
        """
        Args:
            pred_density: Predicted final density map [B, 1, H, W]
            gt_density: Ground truth density map [B, 1, H, W]
            aux_density_maps: List of auxiliary density maps [P_aux^1, P_aux^2]
            gt_aux_maps: List of ground truth maps for auxiliary supervision [G1, G2]
        """
        # Main density map loss (Lc)
        B, _, H, W = pred_density.shape
        N = H * W
        
        # Compute MSE loss per pixel then average
        Lc = torch.sum((pred_density - gt_density) ** 2, dim=[2, 3]) / N  # [B]
        Lc = Lc.mean()  # Average over batch
        
        # Auxiliary loss (Laux)
        Laux = torch.tensor(0., device=pred_density.device)
        if aux_density_maps is not None and gt_aux_maps is not None:
            for aux_pred, aux_gt in zip(aux_density_maps, gt_aux_maps):
                # Compute MSE loss for each auxiliary map
                aux_loss = torch.sum((aux_pred - aux_gt) ** 2, dim=[2, 3]) / N  # [B]
                Laux += aux_loss.mean()
        
        # Total loss
        total_loss = Lc + self.lambda_aux * Laux
        
        return total_loss, {
            'Lc': Lc.item(),
            'Laux': Laux.item()
        }

if __name__ == "__main__":
    # Test implementation
    batch_size = 2
    height, width = 128, 128
    
    # Create dummy predictions and ground truth
    pred_density = torch.rand(batch_size, 1, height, width)
    gt_density = torch.rand(batch_size, 1, height, width)
    
    # Create dummy auxiliary maps
    aux_pred1 = torch.rand(batch_size, 1, height, width)
    aux_pred2 = torch.rand(batch_size, 1, height, width)
    aux_gt1 = torch.rand(batch_size, 1, height, width)
    aux_gt2 = torch.rand(batch_size, 1, height, width)
    
    # Initialize loss function
    criterion = FSCLoss()
    
    # Test without auxiliary maps
    loss, loss_components = criterion(pred_density, gt_density)
    print("\nTest without auxiliary maps:")
    print(f"Total loss: {loss.item():.4f}")
    print(f"Loss components: {loss_components}")
    
    # Test with auxiliary maps
    loss, loss_components = criterion(
        pred_density, 
        gt_density,
        [aux_pred1, aux_pred2],
        [aux_gt1, aux_gt2]
    )
    print("\nTest with auxiliary maps:")
    print(f"Total loss: {loss.item():.4f}")
    print(f"Loss components: {loss_components}")