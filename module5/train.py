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
from loss import FSCLoss  # Menggunakan loss function yang baru
from engine import FSCModel  # Menggunakan nama model yang baru
from debug_utils import print_tensor_info, print_gpu_usage

def train():
    # Print version info for debugging
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
    
    # Hyperparameters
    epochs = 100
    batch_size = 4
    learning_rate = 1e-4
    backbone_learning_rate = 1e-5
    num_exemplars = 3
    
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
    train_dataset = FSC147Dataset(
        data_path="/home/renaldy_fredyan/PhDResearch/LOCA/Dataset/",
        img_size=512,
        split='train',
        num_objects=num_exemplars,
        tiling_p=0.5
    )
    val_dataset = FSC147Dataset(
        data_path="/home/renaldy_fredyan/PhDResearch/LOCA/Dataset/",
        img_size=512,
        split='val',
        num_objects=num_exemplars,
        tiling_p=0.0
    )

    # Use DistributedSampler
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Model initialization
    model = FSCModel(num_exemplars=num_exemplars).to(device)
    
    model = DistributedDataParallel(
        model, 
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )

    # Separate backbone parameters
    backbone_params = []
    non_backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            non_backbone_params.append(param)

    # Optimizer with different learning rates
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": backbone_learning_rate},
        {"params": non_backbone_params, "lr": learning_rate}
    ], weight_decay=1e-4)

    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Loss function sesuai paper (λ=0.3)
    criterion = FSCLoss(lambda_aux=0.3)
    
    # Mixed precision training
    # scaler = GradScaler()

    # Variables for tracking best model
    best_val_mae = float('inf')
    start_epoch = 0

    for epoch in range(start_epoch + 1, epochs + 1):
        if local_rank == 0:
            start = perf_counter()
        
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        # Training loop
        # for batch_idx, (images, exemplars, density_maps) in enumerate(train_dataloader):
        #     torch.cuda.empty_cache()
        #     images = images.to(device)
        #     exemplars = exemplars.to(device)
        #     density_maps = density_maps.to(device)

        for batch_idx, (images, exemplars, density_maps) in enumerate(train_dataloader):
            # print_gpu_usage(f"Start of batch {batch_idx}")

            # Check data validity
            # print(f"\nBatch {batch_idx}:")
            # print(f"Images: shape={images.shape}, dtype={images.dtype}, device={images.device}")
            # print(f"Exemplars: shape={exemplars.shape}, dtype={exemplars.dtype}, device={exemplars.device}")
            # print(f"Density maps: shape={density_maps.shape}, dtype={density_maps.dtype}, device={density_maps.device}")
            
            # Ensure data is on correct device
            images = images.to(device, non_blocking=True)
            exemplars = exemplars.to(device, non_blocking=True)
            density_maps = density_maps.to(device, non_blocking=True)

        #     # with autocast('cuda'):
            
            # Forward pass - get all density maps
            pred_density_maps = model(images, exemplars)
            
            # Calculate loss dengan auxiliary supervision
            loss, loss_components = criterion(
                pred_density_maps[-1],  # Final density map
                density_maps,
                pred_density_maps[:-1],  # Auxiliary density maps
                [density_maps] * len(pred_density_maps[:-1])  # Same GT for all auxiliary maps
            )



            # # Backward pass with gradient scaling
            # optimizer.zero_grad()
            # scaler.scale(loss).backward()
            
            # # Gradient clipping
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # scaler.step(optimizer)
            # scaler.update()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update metrics - use final density map for MAE
            train_loss += loss.item()
            train_mae += torch.abs(pred_density_maps[-1].sum() - density_maps.sum()).item()

            # # Print progress
            # if batch_idx % 20 == 0 and local_rank == 0:
            #     print(f"Epoch [{epoch}/{epochs}][{batch_idx}/{len(train_dataloader)}] "
            #         f"Loss: {loss.item():.4f} (Lc: {loss_components['Lc']:.4f}, Laux: {loss_components['Laux']:.4f})")

    
        del images, exemplars, density_maps
        torch.cuda.empty_cache()
        
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
                pred_val_density_maps = model(val_images, val_exemplars)
                final_val_density_map = pred_val_density_maps[-1]
                
                # Compute validation loss
                val_loss_value, _ = criterion(
                    final_val_density_map,
                    val_density_maps,
                    pred_val_density_maps[:-1],
                    [val_density_maps] * len(pred_val_density_maps[:-1])
                )
                val_loss += val_loss_value.item()
                val_mae += torch.abs(final_val_density_map.sum() - val_density_maps.sum()).item()

        # Average metrics
        train_loss /= len(train_dataloader)
        train_mae /= len(train_dataset)
        val_loss /= len(val_dataloader)
        val_mae /= len(val_dataset)

        # Gather metrics from all GPUs
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
                torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model_module5.pth"))

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
            csv_file = os.path.join(checkpoint_dir, 'training_log_module5.csv')
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