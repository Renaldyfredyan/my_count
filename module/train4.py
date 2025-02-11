import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms
import os
from tqdm import tqdm
import time
from time import perf_counter
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import csv
import math

# Import modules
from swin_transformer_encoder import HybridEncoder
from feature_enhancer import FeatureEnhancer
from exemplar_feature_learning import ExemplarFeatureLearning
from exemplar_image_matching import ExemplarImageMatching
from density_regression_decoder import DensityRegressionDecoder
from data import FSC147Dataset
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
    num_epochs = 100
    batch_size = 8
    learning_rate = 1e-4
    backbone_learning_rate = 1e-5  # Smaller LR for backbone
    patience = 3  # Early stopping patience
    aux_weight = 0.3
    max_grad_norm = 0.1

    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize distributed training
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Directory to save model checkpoints
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Data preparation
    dataset = FSC147Dataset(
        data_path="/home/renaldy_fredyan/PhDResearch/LOCA/Dataset/",
        img_size=512,
        split='train',
        tiling_p=0.5
    )
    val_dataset = FSC147Dataset(
        data_path="/home/renaldy_fredyan/PhDResearch/LOCA/Dataset/",
        img_size=512,
        split='val',
        tiling_p=0.0
    )
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    val_sampler = DistributedSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

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

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": backbone_learning_rate},
        {"params": non_backbone_params, "lr": learning_rate}
    ], weight_decay=1e-4)

    # Scheduler for learning rate
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    '''
    lr ^
    |    /\        /\            /\ 
    |   /  \      /  \          /  \ 
    |  /    \    /    \        /    \ 
    | /      \  /      \      /      \ 
    |/        \/        \    /        \ 
    +---------------------------------> epoch
    0    10    20        40            80
    '''
    # Loss function
    criterion = ObjectNormalizedL2Loss()
    # criterion = nn.MSELoss()
    
    scaler = GradScaler('cuda')  # For mixed precision training

    # Variables for saving the best model
    best_val_loss = float('inf')
    best_checkpoint_path = None
    best = float('inf') 
    start_epoch = 0
    num_objects = 3

    print(local_rank)
    # Training loop
    total_start_time = time.time()
    for epoch in range(start_epoch + 1, num_epochs + 1):
        if local_rank == 0:
            start = perf_counter()
        
        running_loss = 0.0
        train_loss = torch.tensor(0.0).to(device).requires_grad_(False)
        val_loss = torch.tensor(0.0).to(device).requires_grad_(False)
        aux_train_loss = torch.tensor(0.0).to(device).requires_grad_(False)
        aux_val_loss = torch.tensor(0.0).to(device).requires_grad_(False)
        train_ae = torch.tensor(0.0).to(device).requires_grad_(False)
        val_ae = torch.tensor(0.0).to(device).requires_grad_(False)

        model.train()

        #Consistently shuffles (randomizes) data across GPUs on every epoch
        dataloader.sampler.set_epoch(epoch) 
        

       
        for images, exemplars, density_maps in dataloader:
            images = images.to(device)
            exemplars = exemplars.to(device)
            density_maps = density_maps.to(device)

            optimizer.zero_grad() 

            with autocast('cuda'):  # Mixed precision forward pass
                out, aux_out = model(images, exemplars)

                main_loss = criterion(out, density_map, num_objects=num_objects)

                aux_loss = sum([
                    aux_weight * criterion(aux, density_map, num_objects=num_objects)
                    for aux in aux_out
                ])

                loss = main_loss + aux_loss
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()

            '''When using mixed precision training, gradients are scaled to a normal scale first
            This is important because gradient clipping must be done on the actual gradient values, not the scaled ones.
            This function limits the norm (magnitude) of gradients
            If gradient norm > max_grad_norm, the gradient will be scaled down
            Prevents exploding gradients'''
            
            # Gradient clipping (optional)
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)  # Unscale gradients before clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            # Accumulate training metrics
            train_loss += main_loss * img.size(0)
            aux_train_loss += aux_loss * img.size(0)
            train_ae += torch.abs(density_map.sum(dim=(1, 2, 3)) - out.sum(dim=(1, 2, 3))).sum()
        
            # Free unused variables to avoid OOM
            del img, exemplars, density_map, out, aux_out
            torch.cuda.empty_cache()  # Clear GPU cache


        model.eval()
        with torch.no_grad():
            for img, exemplars, density_map in val_dataloader:
                img = img.to(device)
                exemplars = exemplars.to(device)
                density_map = density_map.to(device)
                out, aux_out = model(img, exemplars)
                with torch.no_grad():
                    num_objects = density_map.sum()
                    dist.all_reduce(num_objects)

                main_loss = criterion(out, density_map, num_objects)
                aux_loss = sum([
                    aux_weight * criterion(aux, density_map, num_objects) for aux in aux_out
                ])
                loss = main_loss + aux_loss

                val_loss += main_loss * img.size(0)
                aux_val_loss += aux_loss * img.size(0)
                val_ae += torch.abs(
                    density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
                ).sum()

                # Clear unused variables to reduce memory usage
                del img, exemplars, density_map, out, aux_out  # Clear GPU memory
                torch.cuda.empty_cache()  # Free cached GPU memory
                
                # # Periksa bentuk aux_out dan lakukan penyesuaian jika diperlukan
                # reshaped_aux_out = []
                # for aux in aux_out:
                #     if aux.dim() == 2:
                #         aux = aux.unsqueeze(0).unsqueeze(0)  # Tambahkan dimensi batch dan channel jika perlu
                #     elif aux.dim() == 3:
                #         aux = aux.unsqueeze(0)  # Tambahkan dimensi batch jika perlu
                    
                #     # Periksa jumlah dimensi spasial aux
                #     if aux.size(-1) != density_map.size(-1) or aux.size(-2) != density_map.size(-2):
                #         aux = nn.functional.interpolate(aux, size=density_map.shape[-2:], mode='bilinear', align_corners=False)
                    
                #     reshaped_aux_out.append(aux)
                
                # aux_loss = sum([
                #     aux_weight * criterion(aux, density_map, num_objects) for aux in reshaped_aux_out
                # ])
                # loss = main_loss + aux_loss

                # val_loss += main_loss * img.size(0)
                # aux_val_loss += aux_loss * img.size(0)
                # val_ae += torch.abs(
                #     density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
                # ).sum()

        dist.all_reduce(train_loss)
        dist.all_reduce(val_loss)
        dist.all_reduce(aux_train_loss)
        dist.all_reduce(aux_val_loss)
        dist.all_reduce(train_ae)
        dist.all_reduce(val_ae)

        scheduler.step()


        if local_rank == 0:
            end = perf_counter()
            best_epoch = False
            if val_ae.item() / len(val_dataset) < best:
                best = val_ae.item() / len(val_dataset)
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_ae': val_ae.item() / len(val_dataset)
                }
                torch.save(
                    checkpoint,
                    os.path.join(checkpoint_dir, "_new_best_model.pth")
                )
                best_epoch = True
            print(
                f"Epoch: {epoch}",
                f"Train loss: {running_loss.item():.3f}",
                f"Aux train loss: {aux_train_loss.item():.3f}",
                f"Val loss: {val_loss.item():.3f}",
                f"Aux val loss: {aux_val_loss.item():.3f}",
                f"Train MAE: {train_ae.item() / len(train):.3f}",
                f"Val MAE: {val_ae.item() / len(val):.3f}",
                f"Epoch time: {end - start:.3f} seconds",
                'best' if best_epoch else ''
            )
            # Simpan informasi ke dalam file CSV
            # csv_file = 'training_info.csv'
            csv_file = os.path.join(checkpoint_dir, 'training_info.csv')
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, mode='a' if file_exists else 'w', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Validation MAE', 'Best Epoch'])
                writer.writerow([epoch, running_loss.item() / len(dataloader), val_loss.item() / len(val_dataloader), val_ae.item() / len(val_dataset), 'Yes' if best_epoch else 'No'])

    dist.destroy_process_group()

if __name__ == "__main__":
    train()
