import torch
import os
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from torchvision import transforms
from data_loader import ObjectCountingDataset
from train import LowShotObjectCounting

# Build model function (to modularize)
def build_model():
    return LowShotObjectCounting()

@torch.no_grad()
def evaluate(args):
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    gpu = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    # Initialize distributed process group
    dist.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank
    )

    # Load model and wrap with DistributedDataParallel
    model = DistributedDataParallel(
        build_model().to(device),
        device_ids=[gpu],
        output_device=gpu
    )
    checkpoint = torch.load(os.path.join(args.checkpoint_path, f'{args.model_name}.pth'), map_location=device)
    model.load_state_dict(checkpoint['model'])

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    for split in ['val', 'test']:
        dataset = ObjectCountingDataset(
            data_path=args.data_path,
            img_size=args.img_size,
            split=split,
            tiling_p=args.tiling_p
        )

        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler,
            drop_last=False, num_workers=args.num_workers
        )

        mae, mse = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
        total_samples = len(dataset)
        model.eval()

        for images, exemplars, density_maps in dataloader:
            images = images.to(device)
            exemplars = exemplars.to(device)
            density_maps = density_maps.to(device)

            outputs = model(images, exemplars)
            density_map_pred = outputs[0] if isinstance(outputs, tuple) else outputs

            mae += torch.abs(density_map_pred.sum(dim=(1, 2, 3)) - density_maps.sum(dim=(1, 2, 3))).sum()
            mse += ((density_map_pred.sum(dim=(1, 2, 3)) - density_maps.sum(dim=(1, 2, 3))) ** 2).sum()

        # Synchronize results across GPUs
        dist.all_reduce(mae, op=dist.ReduceOp.SUM)
        dist.all_reduce(mse, op=dist.ReduceOp.SUM)

        if rank == 0:
            mae_avg = mae.item() / total_samples
            rmse_avg = torch.sqrt(mse / total_samples).item()
            print(f"{split.capitalize()} set: MAE = {mae_avg:.4f}, RMSE = {rmse_avg:.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Low-Shot Object Counting Evaluation')
    parser.add_argument('--checkpoint_path', type=str, default= "checkpoints/", help='Path to the model checkpoint')
    parser.add_argument('--model_name', type=str, default='best_model', help='Model name')
    parser.add_argument('--data_path', type=str, default= "/home/renaldy_fredyan/PhDResearch/LOCA/Dataset/", help='Path to the dataset')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--tiling_p', type=float, default=0.0, help='Tiling probability')

    args = parser.parse_args()
    evaluate(args)
