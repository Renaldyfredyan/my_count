import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as TVF
from scipy.ndimage import gaussian_filter

class ObjectCountingDataset(Dataset):
    def __init__(
        self, data_path, img_size, split='train', num_objects=3,
        tiling_p=0.5, zero_shot=False, return_image_name=False
    ):
        """
        Dataset for FSC147 with tiling augmentation and density map handling.

        Args:
            data_path (str): Path to the dataset base directory.
            img_size (int): Target image size (e.g., 512 for 512x512 images).
            split (str): Dataset split ('train', 'val', or 'test').
            num_objects (int): Number of exemplars to use per image.
            tiling_p (float): Probability of applying tiling augmentation.
            zero_shot (bool): Whether to simulate zero-shot learning.
            return_image_name (bool): Whether to return the image name.
        """
        self.split = split
        self.data_path = data_path
        self.horizontal_flip_p = 0.5
        self.tiling_p = tiling_p
        self.img_size = img_size
        self.resize = T.Resize((img_size, img_size))
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        self.num_objects = num_objects
        self.zero_shot = zero_shot
        self.return_image_name = return_image_name

        # Load dataset splits and annotations
        with open(os.path.join(self.data_path, 'Train_Test_Val_FSC_147.json'), 'r') as file:
            splits = json.load(file)
            self.image_names = splits[split]
        with open(os.path.join(self.data_path, 'annotation_FSC147_384.json'), 'r') as file:
            self.annotations = json.load(file)

    def __getitem__(self, idx: int):
        img_name = self.image_names[idx]

        # Load image
        img = Image.open(os.path.join(
            self.data_path, 'images_384_VarV2', img_name
        )).convert("RGB")
        w, h = img.size

        # Apply transforms
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

        # Load and normalize bounding boxes
        bboxes = torch.tensor(
            self.annotations[img_name]['box_examples_coordinates'],
            dtype=torch.float32
        )[:3, [0, 2], :].reshape(-1, 4)[:self.num_objects, ...]
        bboxes = bboxes / torch.tensor([w, h, w, h]) * self.img_size

        # Load density map
        density_map = torch.from_numpy(np.load(os.path.join(
            self.data_path,
            f'gt_density_map_adaptive_{self.img_size}_{self.img_size}_object_VarV2',
            os.path.splitext(img_name)[0] + '.npy',
        ))).unsqueeze(0)

        if self.img_size != 512:
            original_sum = density_map.sum()
            density_map = self.resize(density_map)
            density_map = density_map / density_map.sum() * original_sum

        # Data augmentation
        if self.split == 'train' and torch.rand(1) < self.tiling_p:
            tile_size = (torch.rand(1) + 1, torch.rand(1) + 1)
            img, bboxes, density_map = self.tiling_augmentation(
                img, bboxes, density_map, self.resize, self.jitter, tile_size, self.horizontal_flip_p
            )

        if self.split == 'train':
            img = self.jitter(img)
            img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        if self.split == 'train' and torch.rand(1) < self.horizontal_flip_p:
            img = TVF.hflip(img)
            density_map = TVF.hflip(density_map)
            bboxes[:, [0, 2]] = self.img_size - bboxes[:, [2, 0]]

        if self.return_image_name:
            return img, bboxes, density_map, img_name
        return img, bboxes, density_map

    def __len__(self):
        return len(self.image_names)

    @staticmethod
    def tiling_augmentation(img, bboxes, density_map, resize, jitter, tile_size, hflip_p):
        """Apply tiling augmentation."""
        def make_tile(x, num_tiles, jitter=None):
            tiles = []
            for _ in range(num_tiles):
                row = [jitter(x) if jitter else x for _ in range(num_tiles)]
                tiles.append(torch.cat(row, dim=-1))
            return torch.cat(tiles, dim=-2)

        num_tiles = max(int(tile_size[0].ceil()), int(tile_size[1].ceil()))
        hflip = torch.rand(num_tiles, num_tiles) < hflip_p

        img = make_tile(img, num_tiles, jitter)
        img = resize(img)

        density_map = make_tile(density_map, num_tiles)
        original_sum = density_map.sum()
        density_map = resize(density_map)
        density_map = density_map / density_map.sum() * original_sum

        bboxes[:, [0, 2]] = img.size(-1) - bboxes[:, [2, 0]] if hflip[0, 0] else bboxes[:, [0, 2]]
        bboxes /= torch.tensor([tile_size[0], tile_size[1], tile_size[0], tile_size[1]])

        return img, bboxes, density_map

# Example usage
if __name__ == "__main__":
    dataset = ObjectCountingDataset(
        data_path="/home/renaldy_fredyan/PhDResearch/LOCA/Dataset/",
        img_size=512,
        split='train'
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    for images, bboxes, density_maps in dataloader:
        print("Images shape:", images.shape)  # [B, C, H, W]
        print("Bounding boxes shape:", bboxes.shape)  # [B, N, 4]
        print("Density maps shape:", density_maps.shape)  # [B, 1, H, W]
