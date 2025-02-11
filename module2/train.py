from architecture.engine2 import build_model
from utils.data import FSC147Dataset
from utils.arg_parser import get_argparser
from utils.losses import ObjectNormalizedL2Loss

from time import perf_counter
import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F

import numpy as np
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# CHANGE
# scaler = torch.cuda.amp.GradScaler()  # Initialize gradient scaler for mixed precision
# new scaler
scaler = torch.amp.GradScaler()

def compute_loss(pred_map, gt_map, aux_maps=None, lambda_val=0.3):
    # Main loss (L2)
    main_loss = F.mse_loss(pred_map, gt_map)
    
    # Auxiliary loss
    if aux_maps is not None:
        aux_loss = sum(F.mse_loss(aux_map, gt_map) for aux_map in aux_maps)
        return main_loss + lambda_val * aux_loss
    
    return main_loss

def train(args):

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    gpu = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    dist.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank
    )

    model = DistributedDataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )

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

    train = FSC147Dataset(
        args.data_path,
        args.image_size,
        split='train',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot
    )
    val = FSC147Dataset(
        args.data_path,
        args.image_size,
        split='val',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p
    )
    train_loader = DataLoader(
        train,
        sampler=DistributedSampler(train),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val,
        sampler=DistributedSampler(val),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers
    )

    print(rank)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if rank == 0:
            start = perf_counter()
        train_loss = torch.tensor(0.0).to(device).requires_grad_(False)
        val_loss = torch.tensor(0.0).to(device).requires_grad_(False)
        aux_train_loss = torch.tensor(0.0).to(device).requires_grad_(False)
        aux_val_loss = torch.tensor(0.0).to(device).requires_grad_(False)
        train_ae = torch.tensor(0.0).to(device).requires_grad_(False)
        val_ae = torch.tensor(0.0).to(device).requires_grad_(False)

        train_loader.sampler.set_epoch(epoch)
        model.train()
    
        for img, bboxes, density_map in train_loader:
            img, bboxes, density_map = img.to(device), bboxes.to(device), density_map.to(device)
        
            optimizer.zero_grad()
        
            with torch.amp.autocast(device_type='cuda', enabled=True):
                out, aux_out, exemplar_features = model(img, bboxes)
                
                # Interpolate main output
                out = nn.functional.interpolate(out, size=density_map.shape[2:], mode='bilinear', align_corners=False)
                main_loss = criterion(out, density_map, num_objects=args.num_objects)
                
                # Hitung aux_loss
                if aux_out is not None and len(aux_out) > 0:
                    aux_loss = sum([
                        args.aux_weight * criterion(
                            nn.functional.interpolate(aux, size=density_map.shape[2:], mode='bilinear', align_corners=False),
                            density_map, 
                            num_objects=args.num_objects
                        ) for aux in aux_out
                    ])
                else:
                    aux_loss = torch.tensor(0.0, device=out.device)
                    
                # Hitung exemplar_loss
                if exemplar_features is not None and isinstance(exemplar_features, (list, tuple)):
                    exemplar_loss = sum([
                        args.exemplar_weight * criterion(
                            nn.functional.interpolate(exemplar, size=density_map.shape[2:], mode='bilinear', align_corners=False),
                            density_map, 
                            num_objects=args.num_objects
                        ) for exemplar in exemplar_features
                    ])
                else:
                    exemplar_loss = torch.tensor(0.0, device=out.device)
                    
                loss = main_loss + aux_loss + exemplar_loss
        
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
        
            # Gradient clipping (optional)
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)  # Unscale gradients before clipping
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
        
            # Accumulate training metrics
            train_loss += main_loss * img.size(0)
            aux_train_loss += aux_loss * img.size(0)
            train_ae += torch.abs(density_map.sum(dim=(1, 2, 3)) - out.sum(dim=(1, 2, 3))).sum()
        
            # Free unused variables to avoid OOM
            del img, bboxes, density_map, out, aux_out
            torch.cuda.empty_cache()  # Clear GPU cache

        
        model.eval()
        with torch.no_grad():
            for img, bboxes, density_map in val_loader:
                img = img.to(device)
                bboxes = bboxes.to(device)
                density_map = density_map.to(device)
                out, aux_out, exemplar_features = model(img, bboxes)

                # Interpolate main output
                out = nn.functional.interpolate(out, size=density_map.shape[2:], mode='bilinear', align_corners=False)
                main_loss = criterion(out, density_map, num_objects=args.num_objects)
                
                # Hitung aux_loss
                if aux_out is not None and len(aux_out) > 0:
                    aux_loss = sum([
                        args.aux_weight * criterion(
                            nn.functional.interpolate(aux, size=density_map.shape[2:], mode='bilinear', align_corners=False),
                            density_map, 
                            num_objects=args.num_objects
                        ) for aux in aux_out
                    ])
                else:
                    aux_loss = torch.tensor(0.0, device=out.device)
                    
                # Hitung exemplar_loss
                if exemplar_features is not None and isinstance(exemplar_features, (list, tuple)):
                    exemplar_loss = sum([
                        args.exemplar_weight * criterion(
                            nn.functional.interpolate(exemplar, size=density_map.shape[2:], mode='bilinear', align_corners=False),
                            density_map, 
                            num_objects=args.num_objects
                        ) for exemplar in exemplar_features
                    ])
                else:
                    exemplar_loss = torch.tensor(0.0, device=out.device)
                loss = main_loss + aux_loss + exemplar_loss

                val_loss += main_loss * img.size(0)
                aux_val_loss += aux_loss * img.size(0)
                val_ae += torch.abs(
                    density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
                ).sum()
                # Clear unused variables to reduce memory usage
                del img, bboxes, density_map, out, aux_out  # Clear GPU memory
                torch.cuda.empty_cache()  # Free cached GPU memory

        dist.all_reduce(train_loss)
        dist.all_reduce(val_loss)
        dist.all_reduce(aux_train_loss)
        dist.all_reduce(aux_val_loss)
        dist.all_reduce(train_ae)
        dist.all_reduce(val_ae)


        scheduler.step()

        if rank == 0:
            end = perf_counter()
            best_epoch = False
            if val_ae.item() / len(val) < best:
                best = val_ae.item() / len(val)
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_ae': val_ae.item() / len(val)
                }
                torch.save(
                    checkpoint,
                    os.path.join(args.model_path, f'{args.model_name}.pt')
                )
                best_epoch = True

            print(
                f"Epoch: {epoch}",
                f"Train loss: {train_loss.item():.3f}",
                f"Aux train loss: {aux_train_loss.item():.3f}",
                f"Val loss: {val_loss.item():.3f}",
                f"Aux val loss: {aux_val_loss.item():.3f}",
                f"Train MAE: {train_ae.item() / len(train):.3f}",
                f"Val MAE: {val_ae.item() / len(val):.3f}",
                f"Epoch time: {end - start:.3f} seconds",
                'best' if best_epoch else ''
            )

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Efficient Low-Shot', parents=[get_argparser()])
    args = parser.parse_args()
    train(args)