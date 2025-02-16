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

# Import updated modules
from encoder import DensityEncoder
from exemplar_feature_learning import ExemplarFeatureLearning
from similarity_maps import ExemplarImageMatching
from decoder import DensityRegressionDecoder
from data_loader import ObjectCountingDataset
from custom_loss import CustomLoss
from engine import LowShotCounting  # Updated from LowShotObjectCounting

def train():
    # Print version info for debugging
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
    
    # Hyperparameters
    epochs = 100
    batch_size = 8
    learning_rate = 1e-4
    backbone_learning_rate = 1e-5
    embed_dim = 256  # Make sure this matches with all modules
    num_iterations = 3  # Number of iterative refinements
    
    # Initialize distributed training
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        print(f"Training with {world_size} GPUs")
        print("Initializing directories...")
    
    # Directory to save model checkpoints
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if local_rank == 0:
        print("Setting up datasets...")
    
    # Data preparation
    train_dataset = ObjectCountingDataset(
        data_path="/home/renaldy_fredyan/PhDResearch/LOCA/Dataset/",
        img_size=512,
        split='train',
        tiling_p=0.5
    )
    val_dataset = ObjectCountingDataset(
        data_path="/home/renaldy_fredyan/PhDResearch/LOCA/Dataset/",
        img_size=512,
        split='val',
        tiling_p=0.0
    )

    if local_rank == 0:
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")

    # Use DistributedSampler for distributed training
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    if local_rank == 0:
        print("Initializing model...")
    
    # Model initialization with explicit parameters
    model = LowShotCounting(
        num_iterations=num_iterations,
        embed_dim=embed_dim,
        temperature=0.1,
        backbone_type='swin',
        num_exemplars=3  # Added num_exemplars parameter
    ).to(device)
    
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    if local_rank == 0:
        print("Model initialized successfully")
        # Print model structure for verification
        print("\nModel Structure:")
        print(model)

    # Separate backbone and non-backbone parameters
    backbone_params = []
    non_backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:  # Changed from "backbone" to "encoder"
            backbone_params.append(param)
        else:
            non_backbone_params.append(param)

    # Optimizer with different learning rates
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": backbone_learning_rate},
        {"params": non_backbone_params, "lr": learning_rate}
    ], weight_decay=1e-4)

    # Scheduler for learning rate
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Loss function
    criterion = CustomLoss()
    
    # Mixed precision training
    scaler = GradScaler()

    # Variables for tracking best model
    best_val_mae = float('inf')
    start_epoch = 0

    if local_rank == 0:
        print("\nStarting training...")
        print(f"Total epochs: {epochs}")
        print(f"Batch size per GPU: {batch_size}")
        print(f"Total batch size: {batch_size * world_size}")

    for epoch in range(start_epoch + 1, epochs + 1):
        if local_rank == 0:
            start = perf_counter()
        
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        # Training loop
        for batch_idx, (images, exemplars, density_maps) in enumerate(train_dataloader):
            images = images.to(device)
            exemplars = exemplars.to(device)
            density_maps = density_maps.to(device)

            with autocast('cuda'):
                # Forward pass - get all iterative density maps
                all_density_maps = model(images, exemplars)
                
                # Calculate loss for each iteration
                loss = 0
                final_density_map = all_density_maps[-1]  # Last iteration's output
             
                # Auxiliary losses for intermediate outputs
                aux_weight = 0.4  # You can adjust this weight
                for i, density_map in enumerate(all_density_maps[:-1]):
                    aux_loss = criterion(density_map, density_maps)
                    loss += aux_weight * aux_loss
                
                # Main loss for final output
                main_loss = criterion(final_density_map, density_maps)
                loss += main_loss

            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            # Update metrics - use final density map for MAE
            train_loss += loss.item()
            train_mae += torch.abs(final_density_map.sum() - density_maps.sum()).item()

            # Print progress
            if batch_idx % 20 == 0 and local_rank == 0:
                print(f"Epoch [{epoch}/{epochs}][{batch_idx}/{len(train_dataloader)}] "
                      f"Loss: {loss.item():.4f}")
            
            if batch_idx % 100 == 0 and local_rank == 0:
                print(f"\nDensity map stats:")
                print(f"Min: {final_density_map.min().item():.3f}")
                print(f"Max: {final_density_map.max().item():.3f}")
                print(f"Mean: {final_density_map.mean().item():.3f}")
                print(f"Std: {final_density_map.std().item():.3f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for val_images, val_exemplars, val_density_maps in val_dataloader:
                val_images = val_images.to(device)
                val_exemplars = val_exemplars.to(device)
                val_density_maps = val_density_maps.to(device)

                # Forward pass
                all_val_density_maps = model(val_images, val_exemplars)
                final_val_density_map = all_val_density_maps[-1]
                
                # Compute validation metrics using final output
                val_loss += criterion(final_val_density_map, val_density_maps).item()
                val_mae += torch.abs(final_val_density_map.sum() - val_density_maps.sum()).item()

        # Average metrics
        train_loss /= len(train_dataloader)
        train_mae /= len(train_dataset)
        val_loss /= len(val_dataloader)
        val_mae /= len(val_dataset)

        # Gather metrics dari semua GPU
        if world_size > 1:
            train_metrics = torch.tensor([train_loss, train_mae], device=device)
            val_metrics = torch.tensor([val_loss, val_mae], device=device)
            
            dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
            
            train_metrics /= world_size
            val_metrics /= world_size
            
            train_loss, train_mae = train_metrics.tolist()
            val_loss, val_mae = val_metrics.tolist()

        # Update learning rate
        scheduler.step()

        # Save best model (only on rank 0)
        if local_rank == 0:
            end = perf_counter()
            is_best = val_mae < best_val_mae
            
            if is_best:
                best_val_mae = val_mae
                checkpoint = {
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_mae': best_val_mae
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pth"))

            # Log training information
            print(
                f"Epoch: {epoch}/{epochs}",
                f"Train loss: {train_loss:.3f}",
                f"Val loss: {val_loss:.3f}",
                f"Train MAE: {train_mae:.3f}",
                f"Val MAE: {val_mae:.3f}",
                f"Time: {end - start:.3f}s",
                "BEST" if is_best else ""
            )
            
            # Save training info to CSV
            csv_file = os.path.join(checkpoint_dir, 'training_log.csv')
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, mode='a' if file_exists else 'w', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Train MAE', 'Val MAE', 'Best'])
                writer.writerow([epoch, train_loss, val_loss, train_mae, val_mae, 'Yes' if is_best else 'No'])

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    train()