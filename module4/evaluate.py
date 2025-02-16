import torch
import os
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from data_loader import ObjectCountingDataset
from train import LowShotObjectCounting

def build_model():
    return LowShotObjectCounting()

@torch.no_grad()
def evaluate(args):
    world_size = int(os.environ.get('WORLD_SIZE', 1))  
    rank = int(os.environ.get('RANK', 0))  
    gpu = int(os.environ.get('LOCAL_RANK', 0))  

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    model = build_model().to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)

    checkpoint = torch.load(os.path.join(args.checkpoint_path, f'{args.model_name}.pth'), map_location=device)
    # if rank == 0:
    #     print("\nCheckpoint info:")
    #     for key in checkpoint.keys():
    #         if key != 'model':  # Skip printing full model state
    #             print(f"{key}: {checkpoint[key]}")
    #     print(f"Best val_mae from checkpoint: {checkpoint.get('best_val_mae', 'Not found')}")

    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    for split in ['val', 'test']:
        dataset = ObjectCountingDataset(args.data_path, args.images_size, split=split, tiling_p=args.tiling_p)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=False, num_workers=args.num_workers)

        if rank == 0:
            print(f"\nStarting evaluation on {split} set:")
            print(f"Dataset size: {len(dataset)}")
            print(f"Batch size: {args.batch_size}")
            print(f"Number of batches: {len(dataloader)}")

        val_mae = 0.0
        val_rmse = 0.0
        total_gt_objects = 0
        total_pred_objects = 0

        # Debug counters
        num_empty_predictions = 0
        num_zero_gt = 0

        for batch_idx, (images, exemplars, density_map) in enumerate(dataloader):
            images = images.to(device)
            exemplars = exemplars.to(device)
            density_map = density_map.to(device)

            # Get all density maps
            all_density_maps = model(images, exemplars)
            final_density_map = all_density_maps[-1]

            # Debug first batch
            if batch_idx == 0 and rank == 0:
                print("\nFirst batch statistics:")
                print(f"GT density map shape: {density_map.shape}")
                print(f"Pred density map shape: {final_density_map.shape}")
                for i in range(min(3, images.size(0))):
                    gt_sum = density_map[i].sum().item()
                    pred_sum = final_density_map[i].sum().item()
                    print(f"Sample {i}: GT count={gt_sum:.2f}, Pred count={pred_sum:.2f}, Diff={abs(gt_sum-pred_sum):.2f}")

            # Calculate batch statistics
            gt_counts = density_map.sum(dim=(1,2,3))
            pred_counts = final_density_map.sum(dim=(1,2,3))
            
            # Update debug counters
            num_empty_predictions += (pred_counts == 0).sum().item()
            num_zero_gt += (gt_counts == 0).sum().item()

            # Accumulate total counts
            total_gt_objects += gt_counts.sum().item()
            total_pred_objects += pred_counts.sum().item()

            # Calculate metrics
            batch_mae = torch.abs(pred_counts - gt_counts).sum().item()
            val_mae += batch_mae
            val_rmse += ((pred_counts - gt_counts) ** 2).sum().item()

        if world_size > 1:
            # Gather metrics
            val_mae_tensor = torch.tensor([val_mae], device=device)
            val_rmse_tensor = torch.tensor([val_rmse], device=device)
            total_gt_tensor = torch.tensor([total_gt_objects], device=device)
            total_pred_tensor = torch.tensor([total_pred_objects], device=device)
            
            dist.all_reduce(val_mae_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_rmse_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_gt_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_pred_tensor, op=dist.ReduceOp.SUM)
            
            val_mae = val_mae_tensor.item()
            val_rmse = val_rmse_tensor.item()
            total_gt_objects = total_gt_tensor.item()
            total_pred_objects = total_pred_tensor.item()

        # Normalize metrics
        val_mae = val_mae / len(dataset)
        val_rmse = (val_rmse / len(dataset)) ** 0.5

        if rank == 0:
            print(f"\nEvaluation results for {split} set:")
            print(f"MAE = {val_mae:.4f}")
            print(f"RMSE = {val_rmse:.4f}")
            print(f"Total GT objects: {total_gt_objects:.2f}")
            print(f"Total predicted objects: {total_pred_objects:.2f}")
            print(f"Average GT objects per image: {total_gt_objects/len(dataset):.2f}")
            print(f"Average predicted objects per image: {total_pred_objects/len(dataset):.2f}")
            print(f"Number of empty predictions: {num_empty_predictions}")
            print(f"Number of zero ground truth: {num_zero_gt}")

        torch.cuda.empty_cache()

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Low-Shot Object Counting Evaluation')
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/", help='Path to the model checkpoint')
    parser.add_argument('--model_name', type=str, default='best_model_module4', help='Model name')
    parser.add_argument('--data_path', type=str, default="/home/renaldy_fredyan/PhDResearch/LOCA/Dataset/", help='Path to the dataset')
    parser.add_argument('--images_size', type=int, default=512, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--tiling_p', type=float, default=0.0, help='Tiling probability')

    args = parser.parse_args()
    evaluate(args)
