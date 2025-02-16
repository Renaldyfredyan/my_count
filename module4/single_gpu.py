import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import csv
from time import perf_counter
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import argparse

from data import FSC147Dataset
from metrics_evaluator import CustomLoss, MetricsEvaluator, log_gpu_memory, profile_memory_usage
from engine import LowShotCounting

def train_single_gpu(args):
    # Hyperparameters
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    backbone_learning_rate = args.backbone_learning_rate
    embed_dim = args.embed_dim
    num_iterations = args.num_iterations
    accumulation_steps = args.accumulation_steps
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print configuration
    print(f"\nTraining Configuration:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create checkpoint directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize datasets
    train_dataset = FSC147Dataset(
        data_path=args.data_path,
        img_size=args.img_size,
        split='train',
        tiling_p=0.5
    )
    
    val_dataset = FSC147Dataset(
        data_path=args.data_path,
        img_size=args.img_size,
        split='val',
        tiling_p=0.0
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = LowShotCounting(
        num_iterations=num_iterations,
        embed_dim=embed_dim,
        temperature=0.1,
        backbone_type='swin',
        num_exemplars=3
    ).to(device)
    
    # Enable gradient checkpointing
    model.encoder.swin_backbone.set_grad_checkpointing(True)
    
    # Separate parameters for different learning rates
    backbone_params = []
    non_backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            backbone_params.append(param)
        else:
            non_backbone_params.append(param)
    
    # Optimizer
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": backbone_learning_rate},
        {"params": non_backbone_params, "lr": learning_rate}
    ], weight_decay=1e-4)
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Loss and metrics
    criterion = CustomLoss()
    metrics_evaluator = MetricsEvaluator()
    scaler = GradScaler()
    
    # Training state
    best_val_mae = float('inf')
    start_epoch = 0
    
    # Profile memory usage with a sample batch
    sample_batch = next(iter(train_loader))
    sample_images, sample_exemplars, _ = [x.to(device) for x in sample_batch]
    peak_memory = profile_memory_usage(model, (sample_images, sample_exemplars))
    print(f"\nEstimated peak memory per batch: {peak_memory:.2f} GB")
    
    print("\nStarting training...")
    for epoch in range(start_epoch + 1, epochs + 1):
        start_time = perf_counter()
        
        # Training phase
        model.train()
        metrics_evaluator.reset()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (images, exemplars, density_maps) in enumerate(train_loader):
            images = images.to(device)
            exemplars = exemplars.to(device)
            density_maps = density_maps.to(device)
            
            # Mixed precision training
            with autocast():
                all_density_maps = model(images, exemplars)
                
                # Calculate loss
                loss = 0
                final_density_map = all_density_maps[-1]
                
                # Auxiliary losses
                aux_weight = 0.4
                for density_map in all_density_maps[:-1]:
                    aux_loss = criterion(density_map, density_maps)
                    loss += aux_weight * aux_loss
                
                # Main loss
                main_loss = criterion(final_density_map, density_maps)
                loss += main_loss
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update metrics
            metrics_evaluator.update(final_density_map.detach(), density_maps)
            train_loss += loss.item() * accumulation_steps
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Print progress
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch}/{epochs}][{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
                log_gpu_memory()
        
        # Validation phase
        model.eval()
        metrics_evaluator.reset()
        val_loss = 0.0
        
        with torch.no_grad():
            for val_images, val_exemplars, val_density_maps in val_loader:
                val_images = val_images.to(device)
                val_exemplars = val_exemplars.to(device)
                val_density_maps = val_density_maps.to(device)
                
                with autocast():
                    all_val_density_maps = model(val_images, val_exemplars)
                    final_val_density_map = all_val_density_maps[-1]
                    val_loss += criterion(final_val_density_map, val_density_maps).item()
                
                metrics_evaluator.update(final_val_density_map, val_density_maps)
        
        # Get metrics
        train_metrics = metrics_evaluator.get_metrics()
        val_metrics = metrics_evaluator.get_metrics()
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        is_best = val_metrics['mae'] < best_val_mae
        if is_best:
            best_val_mae = val_metrics['mae']
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_mae': best_val_mae,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, "single_gpu.pth"))
        
        # Calculate epoch time
        epoch_time = perf_counter() - start_time
        
        # Log metrics
        print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s")
        print(f"Train loss: {train_loss/len(train_loader):.4f}")
        print(f"Train MAE: {train_metrics['mae']:.4f}")
        print(f"Val loss: {val_loss/len(val_loader):.4f}")
        print(f"Val MAE: {val_metrics['mae']:.4f}")
        if is_best:
            print("New best model saved!")
        
        # Save to CSV
        with open(os.path.join(checkpoint_dir, 'single_gpu_log.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            if epoch == 1:
                writer.writerow(['Epoch', 'Train_Loss', 'Train_MAE', 'Val_Loss', 'Val_MAE', 'Is_Best'])
            writer.writerow([
                epoch,
                train_loss/len(train_loader),
                train_metrics['mae'],
                val_loss/len(val_loader),
                val_metrics['mae'],
                'Yes' if is_best else 'No'
            ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ELS Counter')
    parser.add_argument('--data_path', type=str, default='/home/renaldy_fredyan/PhDResearch/LOCA/Dataset/',
                      help='Path to FSC147 dataset')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--backbone_learning_rate', type=float, default=1e-5,
                      help='Learning rate for backbone')
    parser.add_argument('--embed_dim', type=int, default=256,
                      help='Embedding dimension')
    parser.add_argument('--num_iterations', type=int, default=3,
                      help='Number of iterations')
    parser.add_argument('--img_size', type=int, default=512,
                      help='Image size')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                      help='Number of gradient accumulation steps')
    
    args = parser.parse_args()
    train_single_gpu(args)