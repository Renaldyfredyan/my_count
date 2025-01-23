import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
import time
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

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
    # Hyperparameters
    num_epochs = 10
    batch_size = 8
    learning_rate = 1e-4
    backbone_learning_rate = 1e-5  # Smaller LR for backbone
    lambda_aux = 0.1  # Weight for auxiliary loss
    patience = 3  # Early stopping patience
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directory to save model checkpoints
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

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

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model initialization
    model = LowShotObjectCounting().to(device)

    # Freeze backbone parameters
    # for name, param in model.encoder.backbone.named_parameters():
    #     param.requires_grad = False
    # for name, param in model.encoder.swin_backbone.named_parameters():
    #     param.requires_grad = False

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
        {"params": backbone_params, "lr": backbone_learning_rate},
        {"params": non_backbone_params, "lr": learning_rate}
    ], weight_decay=1e-4)

    # Scheduler for learning rate
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Loss function
    criterion = criterion = nn.MSELoss()

    scaler = GradScaler()  # For mixed precision training

    # Variables for saving the best model
    best_val_loss = float('inf')
    best_checkpoint_path = None

    # Training loop
    total_start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0

        # Optionally unfreeze backbone after a few epochs
        if epoch == 5:
            for name, param in model.encoder.swin_backbone.named_parameters():
                param.requires_grad = True
            print("Backbone unfrozen!")

        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for images, exemplars, density_maps in dataloader:
                images = images.to(device)
                exemplars = exemplars.to(device)
                density_maps = density_maps.to(device)

                with autocast():  # Mixed precision forward pass
                    outputs = model(images, exemplars)

                    # Unpack outputs
                    density_map, _ = outputs

                    # Compute main loss
                    loss = criterion(density_map, density_maps)

                # Backward pass with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # Gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (pbar.n + 1))
                pbar.update(1)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_exemplars, val_density_maps in val_dataloader:
                val_images = val_images.to(device)
                val_exemplars = val_exemplars.to(device)
                val_density_maps = val_density_maps.to(device)

                # Forward pass
                val_outputs = model(val_images, val_exemplars)

                # Unpack outputs
                val_density_map, _ = val_outputs

                # Compute validation loss
                val_main_loss = criterion(val_density_map, val_density_maps)
                val_loss += val_main_loss.item()

        val_loss /= len(val_dataloader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"New best model saved at {best_checkpoint_path}")

        # Step scheduler
        scheduler.step()

        # Log epoch time
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_duration:.2f} seconds, Loss: {running_loss/len(dataloader):.4f}")

    # Log total training time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"Training completed in {total_duration:.2f} seconds.")
    print(f"Best model saved at {best_checkpoint_path} with validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()
