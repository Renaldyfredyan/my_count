import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from data_loader import ObjectCountingDataset
from train import LowShotObjectCounting
from torchvision import transforms

def evaluate_model(checkpoint_path, data_path, img_size=512, batch_size=8, split='test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = ObjectCountingDataset(
        data_path=data_path,
        img_size=img_size,
        split=split,
        tiling_p=0.0
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = LowShotObjectCounting().to(device)

    # Hapus prefix 'module.' jika ada
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # Hapus prefix 'module.'
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()

    mae, mse, total_samples = 0.0, 0.0, 0

    with torch.no_grad():
        for images, exemplars, density_maps in dataloader:
            images = images.to(device)
            exemplars = exemplars.to(device)
            density_maps = density_maps.to(device)

            # Forward pass
            outputs = model(images, exemplars)

            # Unpack outputs (assumes density map is the first element of the tuple)
            density_map = outputs[0] if isinstance(outputs, tuple) else outputs

            # Compute metrics
            batch_mae = torch.abs(density_map.sum(dim=(1, 2, 3)) - density_maps.sum(dim=(1, 2, 3))).cpu().numpy()
            batch_mse = (batch_mae ** 2)

            mae += batch_mae.sum()
            mse += batch_mse.sum()
            total_samples += images.size(0)

    mae /= total_samples
    rmse = np.sqrt(mse / total_samples)
    
    print(f"Current Allocated Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Max Allocated Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"Current Reserved Memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"Max Reserved Memory: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

    print(f"Evaluation Results on {split} set: MAE = {mae:.4f}, RMSE = {rmse:.4f}")

if __name__ == "__main__":
    checkpoint_path = "checkpoints/best_model.pth"
    data_path = "/home/renaldy_fredyan/PhDResearch/LOCA/Dataset/"
    img_size = 512
    batch_size = 8

    print("Evaluating on Validation Set")
    evaluate_model(
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        img_size=img_size,
        batch_size=batch_size,
        split='val'
    )

    print("\nEvaluating on Test Set")
    evaluate_model(
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        img_size=img_size,
        batch_size=batch_size,
        split='test'
    )
