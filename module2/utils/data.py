import os
import json
import argparse

from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

import torch
from torch.utils.data import Dataset
from torchvision.ops import box_convert
from torchvision import transforms as T
from torchvision.transforms import functional as TVF

from tqdm import tqdm


def tiling_augmentation(img, bboxes, density_map, resize, jitter, tile_size, hflip_p):

    def apply_hflip(tensor, apply):
        return TVF.hflip(tensor) if apply else tensor

    def make_tile(x, num_tiles, hflip, hflip_p, jitter=None):
        result = list()
        for j in range(num_tiles):
            row = list()
            for k in range(num_tiles):
                t = jitter(x) if jitter is not None else x
                if hflip[j, k] < hflip_p:
                    t = TVF.hflip(t)
                row.append(t)
            result.append(torch.cat(row, dim=-1))
        return torch.cat(result, dim=-2)

    x_tile, y_tile = tile_size
    y_target, x_target = resize.size
    num_tiles = max(int(x_tile.ceil()), int(y_tile.ceil()))
    # whether to horizontally flip each tile
    hflip = torch.rand(num_tiles, num_tiles)

    img = make_tile(img, num_tiles, hflip, hflip_p, jitter)
    img = resize(img[..., :int(y_tile*y_target), :int(x_tile*x_target)])

    density_map = make_tile(density_map, num_tiles, hflip, hflip_p)
    density_map = density_map[..., :int(y_tile*y_target), :int(x_tile*x_target)]
    original_sum = density_map.sum()
    density_map = resize(density_map)
    density_map = density_map / density_map.sum() * original_sum

    if hflip[0, 0] < hflip_p:
        bboxes[:, [0, 2]] = x_target - bboxes[:, [2, 0]]  # TODO change
    bboxes = bboxes / torch.tensor([x_tile, y_tile, x_tile, y_tile])
    return img, bboxes, density_map


class FSC147Dataset(Dataset):
    def __init__(
        self, data_path, img_size, split='train', num_objects=3,
        tiling_p=0.5, zero_shot=False, return_image_name=False
    ):
        self.split = split
        self.data_path = data_path
        self.horizontal_flip_p = 0.5
        self.tiling_p = tiling_p
        self.img_size = img_size
        self.resize = T.Resize((img_size, img_size))
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        self.num_objects = num_objects
        self.zero_shot = zero_shot
        self.return_image_name = return_image_name  # Parameter baru

        with open(
            os.path.join(self.data_path, 'Train_Test_Val_FSC_147.json'), 'rb'
        ) as file:
            splits = json.load(file)
            self.image_names = splits[split]
        with open(
            os.path.join(self.data_path, 'annotation_FSC147_384.json'), 'rb'
        ) as file:
            self.annotations = json.load(file)

    def __getitem__(self, idx: int):
        img = Image.open(os.path.join(
            self.data_path,
            'images_384_VarV2',
            self.image_names[idx]
        )).convert("RGB")
        w, h = img.size
        if self.split != 'train':
            img = T.Compose([
                T.ToTensor(),
                self.resize,
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(img)
        else:
            img = T.Compose([
                T.ToTensor(),
                self.resize,
            ])(img)

        bboxes = torch.tensor(
            self.annotations[self.image_names[idx]]['box_examples_coordinates'],
            dtype=torch.float32
        )[:3, [0, 2], :].reshape(-1, 4)[:self.num_objects, ...]
        bboxes = bboxes / torch.tensor([w, h, w, h]) * self.img_size

        density_map = torch.from_numpy(np.load(os.path.join(
            self.data_path,
            f'gt_density_map_adaptive_{self.img_size}_{self.img_size}_object_VarV2',
            os.path.splitext(self.image_names[idx])[0] + '.npy',
        ))).unsqueeze(0)

        if self.img_size != 512:
            original_sum = density_map.sum()
            density_map = self.resize(density_map)
            density_map = density_map / density_map.sum() * original_sum

        # data augmentation
        tiled = False
        if self.split == 'train' and torch.rand(1) < self.tiling_p:
            tiled = True
            tile_size = (torch.rand(1) + 1, torch.rand(1) + 1)
            img, bboxes, density_map = tiling_augmentation(
                img, bboxes, density_map, self.resize,
                self.jitter, tile_size, self.horizontal_flip_p
            )

        if self.split == 'train':
            if not tiled:
                img = self.jitter(img)
            img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        if self.split == 'train' and not tiled and torch.rand(1) < self.horizontal_flip_p:
            img = TVF.hflip(img)
            density_map = TVF.hflip(density_map)
            bboxes[:, [0, 2]] = self.img_size - bboxes[:, [2, 0]]

        # Return sesuai kebutuhan
        if self.return_image_name:
            return img, bboxes, density_map, self.image_names[idx]
        return img, bboxes, density_map

    def __len__(self):
        return len(self.image_names)



# def generate_density_maps(data_path, target_size):

#     density_map_path = os.path.join(
#         data_path,
#         f'gt_density_map_adaptive_{self.img_size}_{self.img_size}_object_VarV2'
#     )
#     if not os.path.isdir(density_map_path):
#         os.makedirs(density_map_path)

#     with open(
#         os.path.join(data_path, 'annotation_FSC147_384.json'), 'rb'
#     ) as file:
#         annotations = json.load(file)

#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     for i, (image_name, ann) in enumerate(tqdm(annotations.items())):
#         _, h, w = T.ToTensor()(Image.open(os.path.join(
#             data_path,
#             'images_384_VarV2',
#             image_name
#         ))).size()
#         h_ratio, w_ratio = target_size[0] / h, target_size[1] / w

#         points = (
#             torch.tensor(ann['points'], device=device) *
#             torch.tensor([w_ratio, h_ratio], device=device)
#         ).long()
#         points[:, 0] = points[:, 0].clip(0, target_size[1] - 1)
#         points[:, 1] = points[:, 1].clip(0, target_size[0] - 1)
#         bboxes = box_convert(torch.tensor(
#             ann['box_examples_coordinates'],
#             dtype=torch.float32,
#             device=device
#         )[:3, [0, 2], :].reshape(-1, 4), in_fmt='xyxy', out_fmt='xywh')
#         bboxes = bboxes * torch.tensor([w_ratio, h_ratio, w_ratio, h_ratio], device=device)
#         window_size = bboxes.mean(dim=0)[2:].cpu().numpy()[::-1]

#         dmap = torch.zeros(*target_size)
#         for p in range(points.size(0)):
#             dmap[points[p, 1], points[p, 0]] += 1
#         dmap = gaussian_filter(dmap.cpu().numpy(), window_size / 8)

#         np.save(os.path.join(density_map_path, os.path.splitext(image_name)[0] + '.npy'), dmap)


def generate_density_maps(data_path, target_size):
    density_map_path = os.path.join(
        data_path,
        f'gt_density_map_adaptive_{target_size[0]}_{target_size[1]}_object_VarV2'
    )
    if not os.path.isdir(density_map_path):
        os.makedirs(density_map_path)

    with open(os.path.join(data_path, 'annotation_FSC147_384.json'), 'rb') as file:
        annotations = json.load(file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for image_name, ann in tqdm(annotations.items(), desc="Generating density maps"):
        points = torch.tensor(ann['points'], dtype=torch.float32)
        boxes = torch.tensor(ann['box_examples_coordinates'], dtype=torch.float32)
        
        # Resize points and boxes to target size
        h_scale = target_size[0] / 384
        w_scale = target_size[1] / 384
        points = points * torch.tensor([w_scale, h_scale])
        boxes = boxes * torch.tensor([w_scale, h_scale, w_scale, h_scale])
        
        # Calculate average object size from K-shot boxes
        box_heights = boxes[:, 3] - boxes[:, 1]
        box_widths = boxes[:, 2] - boxes[:, 0]
        avg_size = torch.mean(torch.sqrt(box_heights * box_widths))
        
        # Initialize density map
        density_map = torch.zeros(target_size)
        
        # For each point, calculate adaptive kernel size
        for i, point in enumerate(points):
            x, y = point.long()
            if x >= target_size[1] or y >= target_size[0]:
                continue
                
            # Calculate distances to K nearest points
            distances = torch.sqrt(((points - point)**2).sum(1))
            distances = torch.sort(distances)[0][1:4]  # Get 3 nearest neighbors
            
            # Calculate adaptive sigma based on distances and object size
            if len(distances) > 0:
                sigma = (torch.mean(distances) + avg_size) / 4
            else:
                sigma = avg_size / 4
                
            sigma = max(3, float(sigma))  # Minimum sigma of 3 pixels
            
            # Generate Gaussian kernel
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Create mesh grid for Gaussian
            mesh_x = torch.arange(kernel_size, device=device)
            mesh_y = torch.arange(kernel_size, device=device)
            y_grid, x_grid = torch.meshgrid(mesh_y, mesh_x)
            
            # Calculate Gaussian values
            center = kernel_size // 2
            gaussian = torch.exp(-(
                (x_grid - center)**2 + (y_grid - center)**2
            ) / (2 * sigma**2))
            gaussian = gaussian / gaussian.sum()
            
            # Apply Gaussian to density map
            x, y = int(x), int(y)
            left = max(0, x - center)
            right = min(target_size[1], x + center + 1)
            top = max(0, y - center)
            bottom = min(target_size[0], y + center + 1)
            
            kernel_left = center - (x - left)
            kernel_right = center + (right - x)
            kernel_top = center - (y - top)
            kernel_bottom = center + (bottom - y)
            
            density_map[top:bottom, left:right] += gaussian[
                kernel_top:kernel_bottom,
                kernel_left:kernel_right
            ]
        
        # Save density map
        np.save(
            os.path.join(density_map_path, os.path.splitext(image_name)[0] + '.npy'),
            density_map.numpy()
        )
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Density map generator", add_help=False)
    parser.add_argument(
        '--data_path',
        default='Dataset/',
        type=str
    )
    parser.add_argument('--image_size', default=512, type=int)
    args = parser.parse_args()
    generate_density_maps(args.data_path, (args.image_size, args.image_size))