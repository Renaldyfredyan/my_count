import torch
import os
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from torchvision import transforms
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
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    for split in ['val', 'test']:
        dataset = ObjectCountingDataset(args.data_path, args.img_size, split=split, tiling_p=args.tiling_p)

        # **Gunakan DistributedSampler hanya jika multi-GPU**
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=False, num_workers=args.num_workers)

        # **Debug: Print jumlah batch per GPU**
        print(f"Rank {rank} - dataset size: {len(dataset)}, batches: {len(dataloader)}")

        ae = torch.tensor(0.0, device=device)
        se = torch.tensor(0.0, device=device)

        local_samples = len(dataloader) * args.batch_size  # Hitung jumlah sampel lokal dengan lebih akurat
        total_samples_tensor = torch.tensor([local_samples], dtype=torch.float32, device=device)
        if world_size > 1:
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        total_samples = int(total_samples_tensor.item()) if world_size > 1 else local_samples

        model.eval()
        for img, exemplars, density_map in dataloader:
            img = img.to(device)
            exemplars = exemplars.to(device)
            density_map = density_map.to(device)

            out, _ = model(img, exemplars)
            ae += torch.abs(density_map.sum(dim=(1, 2, 3)) - out.sum(dim=(1, 2, 3))).sum()
            se += ((density_map.sum(dim=(1, 2, 3)) - out.sum(dim=(1, 2, 3))) ** 2).sum()

        # **Debug: Print sebelum all_reduce**
        print(f"Before all_reduce - Rank {rank}: ae = {ae.item()}, se = {se.item()}")

        if world_size > 1:
            dist.all_reduce(ae, op=dist.ReduceOp.SUM)
            dist.all_reduce(se, op=dist.ReduceOp.SUM)

            # **Bagi hasil dengan jumlah GPU**
            ae /= world_size
            se /= world_size

        # **Debug: Print setelah all_reduce**
        print(f"After all_reduce - Rank {rank}: ae = {ae.item()}, se = {se.item()}")

        if rank == 0:
            mae_avg = ae.item() / total_samples
            rmse_avg = torch.sqrt(se / total_samples).item()
            print(f"{split.capitalize()} set: MAE = {mae_avg:.4f}, RMSE = {rmse_avg:.4f}")

        torch.cuda.empty_cache()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Low-Shot Object Counting Evaluation')
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/", help='Path to the model checkpoint')
    parser.add_argument('--model_name', type=str, default='best_model', help='Model name')
    parser.add_argument('--data_path', type=str, default="/home/renaldy_fredyan/PhDResearch/LOCA/Dataset/", help='Path to the dataset')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--tiling_p', type=float, default=0.0, help='Tiling probability')

    args = parser.parse_args()
    evaluate(args)
