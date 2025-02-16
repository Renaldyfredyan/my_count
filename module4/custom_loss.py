import torch
import torch.nn as nn
import torch.nn.functional as F

# class CustomLoss(nn.Module):
#     def __init__(self, alpha=1.0, beta=0.5):
#         super(CustomLoss, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.mse = nn.MSELoss(reduction='mean')
        
#     def forward(self, pred, target):
#         # MSE Loss
#         mse_loss = self.mse(pred, target)
        
#         # Total Count Loss
#         pred_count = pred.sum(dim=(2,3))
#         target_count = target.sum(dim=(2,3))
#         count_loss = F.l1_loss(pred_count, target_count)
        
#         # Smoothness Loss - mendorong density map lebih smooth
#         smoothness_loss = torch.mean(torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])) + \
#                          torch.mean(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]))
        
#         # Combined Loss
#         total_loss = self.alpha * mse_loss + count_loss + self.beta * smoothness_loss
        
#         return total_loss

class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, eps=1e-6):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps  # Added small epsilon for numerical stability
        self.mse = nn.MSELoss(reduction='mean')
        
    def forward(self, pred, target):
        # Add numerical stability
        pred = torch.clamp(pred, min=self.eps)
        
        # MSE Loss with safety check
        mse_loss = self.mse(pred, target)
        
        # Total Count Loss with safety check
        pred_count = pred.sum(dim=(2,3))
        target_count = target.sum(dim=(2,3))
        count_loss = F.l1_loss(pred_count, target_count)
        
        # Smoothness Loss with gradient clipping
        diff_y = torch.clamp(pred[:, :, 1:, :] - pred[:, :, :-1, :], min=-1, max=1)
        diff_x = torch.clamp(pred[:, :, :, 1:] - pred[:, :, :, :-1], min=-1, max=1)
        smoothness_loss = (torch.abs(diff_y).mean() + torch.abs(diff_x).mean()) / 2
        
        # Combined Loss with safety checks
        total_loss = (
            self.alpha * torch.clamp(mse_loss, max=1e3) + 
            torch.clamp(count_loss, max=1e3) + 
            self.beta * torch.clamp(smoothness_loss, max=1e2)
        )
        
        return total_loss



