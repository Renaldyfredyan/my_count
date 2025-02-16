import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss(reduction='mean')
        
    def forward(self, pred, target):
        # MSE Loss
        mse_loss = self.mse(pred, target)
        
        # Total Count Loss
        pred_count = pred.sum(dim=(2,3))
        target_count = target.sum(dim=(2,3))
        count_loss = F.l1_loss(pred_count, target_count)
        
        # Smoothness Loss
        smoothness_loss = torch.mean(torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])) + \
                         torch.mean(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]))
        
        # Combined Loss
        total_loss = self.alpha * mse_loss + count_loss + self.beta * smoothness_loss
        
        return total_loss

class MetricsEvaluator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.mae_sum = 0.0
        self.mse_sum = 0.0
        self.rmse_sum = 0.0
        self.num_samples = 0
    
    @torch.no_grad()
    def update(self, pred, target):
        # Convert to counts
        pred_count = pred.sum(dim=(2,3))
        target_count = target.sum(dim=(2,3))
        
        # Calculate metrics
        mae = torch.abs(pred_count - target_count).mean().item()
        mse = ((pred_count - target_count) ** 2).mean().item()
        rmse = np.sqrt(mse)
        
        # Update running sums
        self.mae_sum += mae
        self.mse_sum += mse
        self.rmse_sum += rmse
        self.num_samples += pred.size(0)
    
    def get_metrics(self):
        if self.num_samples == 0:
            return {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0}
            
        return {
            'mae': self.mae_sum / self.num_samples,
            'mse': self.mse_sum / self.num_samples,
            'rmse': self.rmse_sum / self.num_samples
        }

def print_metrics(phase, metrics):
    print(f"\n{phase} Metrics:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")

# Memory monitoring utilities
def log_gpu_memory(tag=""):
    if torch.cuda.is_available():
        print(f"\nGPU Memory Status {tag}:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}:")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Cached: {cached:.2f} GB")

def profile_memory_usage(model, sample_input):
    """Profile memory usage of a model"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    print("\nProfiling memory usage...")
    log_gpu_memory("Before forward pass")
    
    # Forward pass
    with torch.cuda.amp.autocast():
        output = model(*sample_input)
    
    log_gpu_memory("After forward pass")
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nPeak memory usage: {peak_memory:.2f} GB")
    
    return peak_memory