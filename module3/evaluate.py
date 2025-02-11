import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from torch import distributed as dist
from utils.data import FSC147Dataset
from engine import build_model
import os

# @torch.no_grad()
# def evaluate_model(model, dataloader, device):
#     model.eval()
#     ae = torch.tensor(0.0).to(device)
#     se = torch.tensor(0.0).to(device)
    
#     for images, bboxes, density_maps in dataloader:
#         images = images.to(device)
#         bboxes = bboxes.to(device)
#         density_maps = density_maps.to(device)

#         with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
#             # Forward pass
#             density_pred, _, _ = model(images, bboxes)
            
#             # Compute metrics
#             ae += torch.abs(
#                 density_maps.flatten(1).sum(dim=1) - density_pred.flatten(1).sum(dim=1)
#             ).sum()
#             se += ((
#                 density_maps.flatten(1).sum(dim=1) - density_pred.flatten(1).sum(dim=1)
#             ) ** 2).sum()

#     # All reduce across GPUs
#     dist.all_reduce(ae)
#     dist.all_reduce(se)
    
#     return ae, se

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    ae = torch.tensor(0.0).to(device)
    se = torch.tensor(0.0).to(device)
    rank = dist.get_rank()  # Dapatkan rank di dalam fungsi
    
    for i, (images, bboxes, density_maps) in enumerate(dataloader):
        images = images.to(device)
        bboxes = bboxes.to(device)
        density_maps = density_maps.to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            density_pred, _, _ = model(images, bboxes)
            
            pred_count = density_pred.sum(dim=(1, 2, 3))
            true_count = density_maps.sum(dim=(1, 2, 3))
            
            # Cek nilai per batch
            if rank == 0 and i < 2:  # Hanya print 2 batch pertama
                print(f"\nBatch {i}:")
                print(f"Predicted counts: {pred_count}")
                print(f"True counts: {true_count}")
                print(f"Abs diff: {torch.abs(pred_count - true_count)}")
                print(f"Mean abs diff in batch: {torch.abs(pred_count - true_count).mean().item()}")
                print(f"Sum abs diff in batch: {torch.abs(pred_count - true_count).sum().item()}")
            
            ae += torch.abs(pred_count - true_count).sum()
            se += ((pred_count - true_count) ** 2).sum()

            # Cek nilai akumulasi
            if rank == 0 and i < 2:
                print(f"Accumulated ae so far: {ae.item()}")

    dist.all_reduce(ae)
    dist.all_reduce(se)
    
    if rank == 0:
        print("\nFinal accumulated values:")
        print(f"Total ae before division: {ae.item()}")
        print(f"Number of samples: {len(dataloader.dataset)}")
    
    return ae, se

@torch.no_grad()
def main(args):
    # Initialize distributed setup
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

    # Build and load model
    model = build_model(args)
    
    # Load checkpoint
    checkpoint = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'), 
                            map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    # state_dict = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))['model']
    # state_dict = {k if 'module.' in k else 'module.' + k: v for k, v in state_dict.items()}
    # model.load_state_dict(state_dict)

    
    # Create datasets and loaders with DistributedSampler
    for split in ['val', 'test']:
        dataset = FSC147Dataset(
            data_path=args.data_path,
            img_size=args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=0.0
        )
        
        sampler = DistributedSampler(dataset)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # Evaluate
        ae, se = evaluate_model(model, loader, device)
        
        # Print results only on rank 0
        if rank == 0:
            mae = ae.item() / len(dataset)
            rmse = torch.sqrt(se / len(dataset)).item()
            print(f"\n{split.capitalize()} Set Results:")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    from utils.arg_parser import get_argparser
    parser = get_argparser()
    args = parser.parse_args()
    main(args)