import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import os
import csv
from time import perf_counter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from data import FSC147Dataset
# from loss import FSCLoss
from loss import ObjectNormalizedL2Loss
from engine import FSCModel
from debug_utils import print_tensor_info, print_gpu_usage

def get_gradient_norm(model):
    """Calculate the gradient norm for monitoring training stability."""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train():
    # Print version info for debugging
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
    
    # Hyperparameters with adjusted values
    epochs = 100
    batch_size = 4
    learning_rate = 1e-4  # Reduced from 1e-4
    backbone_learning_rate = 0  # Reduced from 1e-5
    num_exemplars = 3
    weight_decay = 1e-4  # Increased from 1e-4
    grad_clip_norm = 0.1  # Reduced from 1.0
    lr_drop = 100
    aux_weight = 0.3
    
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
    
    # Data preparation with augmentation
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
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
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
    
    # Initialize parameters to ensure correct memory layout
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.contiguous()
    
    model = DistributedDataParallel(
        model, 
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        broadcast_buffers=True
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

    # Optimizer with increased weight decay
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": backbone_learning_rate},
        {"params": non_backbone_params, "lr": learning_rate}
    ], weight_decay=weight_decay)

    # Scheduler with longer warm-up
    # scheduler = CosineAnnealingWarmRestarts(
    #     optimizer, 
    #     T_0=15,  # Increased from 10
    #     T_mult=2, 
    #     eta_min=1e-6
    # )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop, gamma=0.25)

    # Loss function
    # criterion = FSCLoss(lambda_aux=0.3)
    criterion = ObjectNormalizedL2Loss()
    
    # Variables for tracking best model and early stopping
    best_val_mae = float('inf')
    start_epoch = 0
    patience = 10
    patience_counter = 0
    min_delta = 0.01

    for epoch in range(start_epoch + 1, epochs + 1):
        if local_rank == 0:
            start = perf_counter()

        train_loss = torch.tensor(0.0).to(device)
        val_loss = torch.tensor(0.0).to(device)
        aux_train_loss = torch.tensor(0.0).to(device)
        aux_val_loss = torch.tensor(0.0).to(device)
        train_ae = torch.tensor(0.0).to(device)
        val_ae = torch.tensor(0.0).to(device)
        
        # Training loop
        for images, exemplars, density_maps in train_dataloader:
            # Ensure data is on correct device
            images = images.to(device, non_blocking=True)
            exemplars = exemplars.to(device, non_blocking=True)
            density_maps = density_maps.to(device, non_blocking=True)

            optimizer.zero_grad()
            out, aux_out = model(images, exemplars)

            # obtain the number of objects in batch
            with torch.no_grad():
                num_objects = density_maps.sum()
                # dist.all_reduce_multigpu([num_objects]) #update pytorch
                dist.all_reduce(num_objects)


            main_loss = criterion(out, density_maps, num_objects)
            aux_loss = sum([
                aux_weight * criterion(aux, density_maps, num_objects) for aux in aux_out
            ])
            loss = main_loss + aux_loss
            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            train_loss += main_loss * images.size(0)
            aux_train_loss += aux_loss * images.size(0)
            running_mae += torch.abs(
                density_maps.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            ).sum()
            
            # # Forward pass
            # pred_density_maps = model(images, exemplars)
            
            # # Calculate loss
            # loss, loss_components = criterion(
            #     pred_density_maps[-1],
            #     density_maps,
            #     pred_density_maps[:-1],
            #     [density_maps] * len(pred_density_maps[:-1])
            # )

            # # Backward pass with gradient clipping
            # optimizer.zero_grad()
            # loss.backward()
            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            # max_grad_norm = max(max_grad_norm, grad_norm.item())
            # optimizer.step()

            # # Update metrics
            # batch_size = images.size(0)
            # train_loss += loss.item() * batch_size
            # running_mae += torch.abs(pred_density_maps[-1].sum() - density_maps.sum()).item()
            # num_samples += batch_size

        del images, exemplars, density_maps
        torch.cuda.empty_cache()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_running_mae = 0.0
        val_num_samples = 0
        
        with torch.no_grad():
            for val_images, val_exemplars, val_density_maps in val_dataloader:
                val_images = val_images.to(device)
                val_exemplars = val_exemplars.to(device)
                val_density_maps = val_density_maps.to(device)
                out, aux_out = model(val_images, val_exemplars)
                with torch.no_grad():
                    num_objects = val_density_maps.sum()
                    dist.all_reduce_multigpu([num_objects])

                main_loss = criterion(out, val_density_maps, num_objects)
                aux_loss = sum([
                    aux_weight * criterion(aux, val_density_maps, num_objects) for aux in aux_out
                ])
                loss = main_loss + aux_loss

                val_loss += main_loss * val_images.size(0)
                aux_val_loss += aux_loss * val_images.size(0)
                val_running_mae += torch.abs(
                    val_density_maps.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
                ).sum()


        # Gather metrics from all GPUs
        if world_size > 1:
            metrics = torch.tensor([
                train_loss, running_mae, float(num_samples),
                val_loss, val_running_mae, float(val_num_samples),
                max_grad_norm
            ], device=device)
            
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            
            train_loss, running_mae, num_samples, val_loss, val_running_mae, val_num_samples, max_grad_norm = metrics.tolist()

        # Calculate final metrics
        train_loss = train_loss / num_samples
        train_mae = running_mae / num_samples
        val_loss = val_loss / val_num_samples
        val_mae = val_running_mae / val_num_samples
        max_grad_norm = max_grad_norm / world_size

        # Update learning rate
        scheduler.step()

        # Early stopping check
        if val_mae < best_val_mae - min_delta:
            best_val_mae = val_mae
            patience_counter = 0
            
            # Save best model (only on rank 0)
            if local_rank == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_mae': best_val_mae
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, "module5_2.pth"))
        else:
            patience_counter += 1

        # Save progress and log (only on rank 0)
        if local_rank == 0:
            end = perf_counter()
            
            print(
                f"Epoch: {epoch}/{epochs}",
                f"Train loss: {train_loss:.3f}",
                f"Val loss: {val_loss:.3f}",
                f"Train MAE: {train_mae:.3f}",
                f"Val MAE: {val_mae:.3f}",
                f"Max Grad Norm: {max_grad_norm:.3f}",
                f"Time: {end - start:.3f}s",
                "BEST" if val_mae == best_val_mae else ""
            )
            
            # Save training info to CSV
            csv_file = os.path.join(checkpoint_dir, 'log_module5_2.csv')
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, mode='a' if file_exists else 'w', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow([
                        'Epoch', 'Train Loss', 'Val Loss', 'Train MAE', 
                        'Val MAE', 'Max Grad Norm', 'Learning Rate', 'Best'
                    ])
                writer.writerow([
                    epoch, train_loss, val_loss, train_mae, val_mae,
                    max_grad_norm, optimizer.param_groups[0]['lr'],
                    'Yes' if val_mae == best_val_mae else 'No'
                ])

        # Early stopping
        if patience_counter >= patience:
            if local_rank == 0:
                print(f"Early stopping triggered after {patience} epochs without improvement")
            break

        # Optional: stop if gradient norm is too high
        if max_grad_norm > 10.0:  # Threshold value
            if local_rank == 0:
                print(f"Training stopped due to high gradient norm: {max_grad_norm}")
            break

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    train()