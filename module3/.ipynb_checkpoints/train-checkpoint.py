import os
import time
import torch
from time import perf_counter
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import SequentialLR, LinearLR, OneCycleLR
from tqdm import tqdm
import os

from utils.data import FSC147Dataset
from engine import build_model
from utils.losses import ObjectNormalizedL2Loss

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss = 0.0
    
    with tqdm(total=len(dataloader), desc=f"Epoch {epoch}", unit="batch") as pbar:
        for images, bboxes, density_maps in dataloader:
            images = images.to(device)
            bboxes = bboxes.to(device)
            density_maps = density_maps.to(device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # Unpack all three returned values
                density_pred, similarity_maps, aux_outputs = model(images, bboxes)
                num_objects = density_maps.sum(dim=(1, 2, 3)) + 1e-6
                loss = criterion(density_pred, density_maps, num_objects)
                loss = loss.mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))
            pbar.update(1)

    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    
    with torch.no_grad():
        for images, bboxes, density_maps in dataloader:
            images = images.to(device)
            bboxes = bboxes.to(device)
            density_maps = density_maps.to(device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # Unpack all three returned values
                density_pred, similarity_maps, aux_outputs = model(images, bboxes)
                num_objects = density_maps.sum(dim=(1, 2, 3)) + 1e-6
                loss = criterion(density_pred, density_maps, num_objects)

            total_loss += loss.mean().item()
            pred_count = density_pred.sum(dim=(1, 2, 3))
            true_count = density_maps.sum(dim=(1, 2, 3))
            mae = torch.abs(pred_count - true_count).mean()
            total_mae += mae.item()

    return total_loss / len(dataloader), total_mae / len(dataloader)

def main(args):
    # Initialize distributed training using environment variables
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    gpu = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Create datasets and dataloaders
    train_dataset = FSC147Dataset(
        data_path=args.data_path,
        img_size=args.image_size,
        split='train',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p
    )
    
    val_dataset = FSC147Dataset(
        data_path=args.data_path,
        img_size=args.image_size,
        split='val',
        num_objects=args.num_objects,
        tiling_p=0.0
    )

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Build model
    model = build_model(args)

    # optimizer = AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    backbone_params = dict()
    non_backbone_params = dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in n:
            backbone_params[n] = p
        else:
            non_backbone_params[n] = p

    optimizer = torch.optim.AdamW(
        [
            {'params': non_backbone_params.values()},
            {'params': backbone_params.values(), 'lr': args.backbone_lr}
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.25)

    if args.resume_training:
        checkpoint = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best = checkpoint['best_val_ae']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_epoch = 0
        best = 10000000000000

    criterion = ObjectNormalizedL2Loss()
    scaler = GradScaler()

    # Training loop
    best_mae = float('inf')
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if rank == 0:
            start = perf_counter()
        train_sampler.set_epoch(epoch)
        
        # Training phase
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, 
            scaler, device, epoch
        )
        
        # Validation phase
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        
        end = perf_counter()
        # Save best model
        if val_mae < best_mae and dist.get_rank() == 0:
            best_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
            }, os.path.join(args.model_path, 'best_model.pth'))

        scheduler.step()
        
        # Print metrics
        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
            print(f"Best MAE: {best_mae:.4f}")
            print(f"Epoch time: {end - start:.3f} seconds")
            print("-" * 50)

    dist.destroy_process_group()

if __name__ == "__main__":
    from utils.arg_parser import get_argparser
    parser = get_argparser()
    args = parser.parse_args()
    main(args)