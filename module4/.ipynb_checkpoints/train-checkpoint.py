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

# Import modules
from swin_transformer_encoder import HybridEncoder
from feature_enhancer import FeatureEnhancer
from exemplar_feature_learning import ExemplarFeatureLearning
from similarity_maps import ExemplarImageMatching
# from density_regression_decoder import DensityRegressionDecoder
from density_regression_decoder2 import DensityRegressionDecoder
from data_loader import ObjectCountingDataset
from custom_loss import CustomLoss

    
def train():
    # Hyperparameters
    epochs = 100
    batch_size = 8
    learning_rate = 1e-4
    backbone_learning_rate = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize distributed training
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Directory to save model checkpoints
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

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

    # Model initialization
    model = LowShotObjectCounting().to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Separate backbone and non-backbone parameters
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

    # Scheduler for learning rate
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Loss function
    # criterion = nn.MSELoss()
    criterion = CustomLoss()
    
    # Mixed precision training
    scaler = GradScaler()

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
        
        for images, exemplars, density_maps in train_dataloader:
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
            
        for batch_idx, (images, exemplars, density_map) in enumerate(train_dataloader):
            images = images.to(device)
            exemplars = exemplars.to(device)
            density_map = density_map.to(device)

            # Get all density maps
            all_density_maps = model(images, exemplars)
            final_density_map = all_density_maps[-1]

            # debug
            if batch_idx == 0 and local_rank == 0:
                print("\nFirst batch statistics:")
                print(f"GT density map shape: {density_map.shape}")
                print(f"Pred density map shape: {final_density_map.shape}")
                for i in range(min(3, images.size(0))):
                    gt_sum = density_map[i].sum().item()
                    pred_sum = final_density_map[i].sum().item()
                    print(f"Sample {i}: GT count={gt_sum:.2f}, Pred count={pred_sum:.2f}, Diff={abs(gt_sum-pred_sum):.2f}")
            
            if batch_idx % 100 == 0 and local_rank == 0:
                    print(f"Density map stats:")
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
            
            # All-reduce untuk mengumpulkan dan merata-ratakan metrics dari semua GPU
            dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
            
            # Bagi dengan jumlah GPU untuk mendapatkan rata-rata
            train_metrics /= world_size
            val_metrics /= world_size
            
            train_loss, train_mae = train_metrics.tolist()
            val_loss, val_mae = val_metrics.tolist()

        # Update learning rate
        scheduler.step()

        # Save best model (hanya di rank 0)
        if local_rank == 0:
            end = perf_counter()
            is_best = val_mae < best_val_mae
            
            if is_best:
                best_val_mae = val_mae
                checkpoint = {
                    'epoch': epoch,
                    'model': model.module.state_dict(),  # Ambil state dict dari model asli (bukan DDP)
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_mae': best_val_mae
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model_module4.pth"))

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
            csv_file = os.path.join(checkpoint_dir, 'training_log_module4.csv')
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