import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import os
import csv
from time import perf_counter
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from data import FSC147Dataset
from custom_loss import CustomLoss
from engine2 import LowShotCounting  # Updated from LowShotObjectCounting

def train():
    # Print version info and setup
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        torch.backends.cudnn.benchmark = True  # Added for performance
        torch.backends.cudnn.deterministic = True  # Added for reproducibility
    
    # Hyperparameters with careful initialization
    epochs = 100
    batch_size = 1
    learning_rate = 1e-5  # Reduced learning rate
    backbone_learning_rate = 1e-6  # Reduced backbone learning rate
    embed_dim = 256
    num_iterations = 3
    grad_clip_value = 0.1  # Added gradient clipping value
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Initialize directories
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize datasets with error checking
    try:
        train_dataset = FSC147Dataset(
            data_path="/home/renaldy_fredyan/PhDResearch/LOCA/Dataset/",
            img_size=512,
            split='train',
            tiling_p=0.5
        )
        val_dataset = FSC147Dataset(
            data_path="/home/renaldy_fredyan/PhDResearch/LOCA/Dataset/",
            img_size=512,
            split='val',
            tiling_p=0.0
        )
        print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    except Exception as e:
        print(f"Error initializing datasets: {str(e)}")
        raise

    # DataLoader setup with error handling
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # Model initialization with gradient scaling
    print("\nInitializing model...")
    model = LowShotCounting(
        num_iterations=num_iterations,
        embed_dim=embed_dim,
        temperature=0.1,
        backbone_type='swin',
        num_exemplars=3
    ).to(device)

    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)  # Apply weight initialization

    # Optimizer setup with parameter groups
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() if 'encoder' in n],
            'lr': backbone_learning_rate,
            'weight_decay': 1e-4
        },
        {
            'params': [p for n, p in model.named_parameters() if 'encoder' not in n],
            'lr': learning_rate,
            'weight_decay': 1e-4
        }
    ]
    
    optimizer = optim.AdamW(param_groups)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = CustomLoss()
    scaler = GradScaler()

    # Training loop
    best_val_mae = float('inf')
    current_loss = float('inf')  # Initialize loss variable

    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        # Training iteration
        for batch_idx, (images, exemplars, density_maps) in enumerate(train_dataloader):
            try:
                # Move data to device
                images = images.to(device, non_blocking=True)
                exemplars = exemplars.to(device, non_blocking=True)
                density_maps = density_maps.to(device, non_blocking=True)

                # Forward pass with autocast
                with autocast('cuda'):
                    all_density_maps = model(images, exemplars)
                    final_density_map = all_density_maps[-1]
                    
                    # Loss calculation with error checking
                    loss = 0
                    aux_weight = 0.4
                    
                    # Check for invalid values in density maps
                    if torch.isnan(final_density_map).any() or torch.isinf(final_density_map).any():
                        print(f"Warning: Invalid values in density map at batch {batch_idx}")
                        continue

                    # Calculate losses with safety checks
                    for i, density_map in enumerate(all_density_maps[:-1]):
                        if not (torch.isnan(density_map).any() or torch.isinf(density_map).any()):
                            aux_loss = criterion(density_map, density_maps)
                            loss += aux_weight * aux_loss

                    main_loss = criterion(final_density_map, density_maps)
                    loss += main_loss
                    current_loss = loss.item()  # Update current_loss

                # Backward pass with gradient scaling
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

                # Check for NaN gradients
                valid_gradients = True
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"Warning: Invalid gradient in {name}")
                            valid_gradients = False
                            break

                if valid_gradients:
                    scaler.step(optimizer)
                    scaler.update()

                # Update metrics
                train_loss += current_loss
                train_mae += torch.abs(final_density_map.sum() - density_maps.sum()).item()

                # Memory cleanup
                del images, exemplars, density_maps, all_density_maps
                torch.cuda.empty_cache()

                # Print progress with safe loss value
                if batch_idx % 10 == 0:
                    print(f"Batch [{batch_idx}/{len(train_dataloader)}] "
                          f"Loss: {current_loss:.4f}")

            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue  # Skip this batch and continue with next
            
        # Validation phase
        print("\nStarting validation...")
        model.eval()
        val_loss = 0.0
        val_mae = 0.0

        with torch.no_grad():
            for val_idx, (val_images, val_exemplars, val_density_maps) in enumerate(val_dataloader):
                val_images = val_images.to(device, non_blocking=True)
                val_exemplars = val_exemplars.to(device, non_blocking=True)
                val_density_maps = val_density_maps.to(device, non_blocking=True)

                all_val_density_maps = model(val_images, val_exemplars)
                final_val_density_map = all_val_density_maps[-1]

                val_loss += criterion(final_val_density_map, val_density_maps).item()
                val_mae += torch.abs(final_val_density_map.sum() - val_density_maps.sum()).item()

                del val_images, val_exemplars, val_density_maps, all_val_density_maps
                torch.cuda.empty_cache()

        # Average metrics
        train_loss /= len(train_dataloader)
        train_mae /= len(train_dataset)
        val_loss /= len(val_dataloader)
        val_mae /= len(val_dataset)

        # Update learning rate
        scheduler.step()

        # Save best model
        end = perf_counter()
        is_best = val_mae < best_val_mae
        
        if is_best:
            best_val_mae = val_mae
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_mae': best_val_mae
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pth"))

        # Log training information
        print(
            f"\nEpoch {epoch} Summary:",
            f"Train loss: {train_loss:.3f}",
            f"Val loss: {val_loss:.3f}",
            f"Train MAE: {train_mae:.3f}",
            f"Val MAE: {val_mae:.3f}",
            f"Time: {end - start:.3f}s",
            "BEST" if is_best else ""
        )
        
        # Save training log
        with open(os.path.join(checkpoint_dir, 'training_log.csv'), mode='a') as file:
            writer = csv.writer(file)
            if epoch == 1:  # Write header for first epoch
                writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Train MAE', 'Val MAE', 'Best'])
            writer.writerow([epoch, train_loss, val_loss, train_mae, val_mae, 'Yes' if is_best else 'No'])

if __name__ == "__main__":
    train()