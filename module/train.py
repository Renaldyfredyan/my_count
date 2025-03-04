import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms
import os
from tqdm import tqdm
import time
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import argparse

# Import modules
from swin_transformer_encoder import HybridEncoder
from feature_enhancer import FeatureEnhancer
from exemplar_feature_learning import ExemplarFeatureLearning
from exemplar_image_matching import ExemplarImageMatching
from density_regression_decoder import DensityRegressionDecoder
from data_loader import ObjectCountingDataset
from losses import ObjectNormalizedL2Loss

# Define the full model
class LowShotObjectCounting(nn.Module):
    def __init__(self):
        super(LowShotObjectCounting, self).__init__()
        self.encoder = HybridEncoder(embed_dim=256)
        self.enhancer = FeatureEnhancer(embed_dim=256)
        self.exemplar_learner = ExemplarFeatureLearning(embed_dim=256, num_iterations=3)
        self.matcher = ExemplarImageMatching()
        self.decoder = DensityRegressionDecoder(input_channels=3)  # Assuming 3 exemplars

    def forward(self, image, exemplars):
        # Image feature extraction
        image_features = self.encoder(image)  # [B, 256, H, W]
        enhanced_image_features = self.enhancer(image_features)  # [B, 256, H, W]

        # Reshape for exemplar matching
        B, C, H, W = enhanced_image_features.shape
        image_features_flat = enhanced_image_features.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]

        # Ensure exemplars have the correct embedding dimension
        if exemplars.shape[-1] != 256:
            exemplars = nn.Linear(exemplars.shape[-1], 256).to(exemplars.device)(exemplars)

        # Iteratively update exemplar features
        updated_exemplars = self.exemplar_learner(image_features_flat, exemplars)  # [B, N, C]

        # Exemplar-image matching
        similarity_maps = self.matcher(image_features_flat, updated_exemplars)  # [B, N, H, W]

        # Density regression
        density_map = self.decoder(similarity_maps)  # [B, 1, H, W]
        density_map = nn.functional.interpolate(density_map, size=(512, 512), mode="bilinear", align_corners=False)

        return density_map, similarity_maps

# Training setup
def train():
    parser = argparse.ArgumentParser(description="Low Shot Object Counting Training Script")
    parser.add_argument('--freeze_backbone', action='store_true', help="Freeze backbone during training")
    parser.add_argument('--unfreeze_epoch', type=int, default=5, help="Epoch to unfreeze backbone")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for non-backbone params")
    parser.add_argument('--backbone_learning_rate', type=float, default=1e-5, help="Learning rate for backbone params")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help="Directory to save model checkpoints")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize distributed training
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Directory to save model checkpoints
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Data preparation
    dataset = ObjectCountingDataset(
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

    train_sampler = DistributedSampler(dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

    # Model initialization
    model = LowShotObjectCounting().to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Separate backbone and non-backbone parameters
    backbone_params = []
    non_backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:  # Backbone parameters
            backbone_params.append(param)
        else:  # Non-backbone parameters
            non_backbone_params.append(param)

    # Optimizer with different learning rates
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_learning_rate},
        {"params": non_backbone_params, "lr": args.learning_rate}
    ], weight_decay=1e-4)

    # Scheduler for learning rate
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Loss function
    criterion = ObjectNormalizedL2Loss()
    scaler = GradScaler()  # For mixed precision training

    # Initialize best validation loss
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(args.num_epochs):
        if epoch == 0:  # Untuk mencatat waktu awal pelatihan
            total_start_time = time.perf_counter()
        epoch_start_time = time.perf_counter()
        
        model.train()
        train_sampler.set_epoch(epoch)

        # Logika freeze/unfreeze backbone
        if args.freeze_backbone and epoch < args.unfreeze_epoch:
            for param in backbone_params:
                param.requires_grad = False  # Membekukan backbone
        else:
            for param in backbone_params:
                param.requires_grad = True  # Melatih backbone

        running_loss = 0.0

        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="batch") as pbar:
            for images, exemplars, density_maps in dataloader:
                images = images.to(device, non_blocking=True)
                exemplars = exemplars.to(device, non_blocking=True)
                density_maps = density_maps.to(device, non_blocking=True)

                optimizer.zero_grad()

                with autocast():
                    density_map, _ = model(images, exemplars)
                    num_objects = density_maps.sum(dim=(1, 2, 3))  # Hitung jumlah objek
                    loss = criterion(density_map, density_maps, num_objects=num_objects)
                    loss = loss.mean()  # Pastikan loss adalah skalar

                scaler.scale(loss).backward()

                # Gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (pbar.n + 1))
                pbar.update(1)

        scheduler.step()
        # Catat waktu selesai epoch
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time

        # Tampilkan waktu untuk setiap epoch
        print(f"Epoch [{epoch+1}/{args.num_epochs}] completed in {epoch_duration:.2f} seconds")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_exemplars, val_density_maps in val_dataloader:
                val_images = val_images.to(device, non_blocking=True)
                val_exemplars = val_exemplars.to(device, non_blocking=True)
                val_density_maps = val_density_maps.to(device, non_blocking=True)

                with autocast():
                    val_density_map, _ = model(val_images, val_exemplars)
                    num_objects = val_density_maps.sum(dim=(1, 2, 3))  # Hitung jumlah objek
                    loss_tensor = criterion(val_density_map, val_density_maps, num_objects=num_objects)
                    val_loss += loss_tensor.mean().item()  # Ambil rata-rata loss

        val_loss /= len(val_dataloader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"New best model saved at {best_checkpoint_path} with validation loss: {best_val_loss:.4f}")

    # Catat total waktu pelatihan
    total_end_time = time.perf_counter()
    total_duration = total_end_time - total_start_time
    print(f"Training completed in {total_duration:.2f} seconds.")

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    train()
