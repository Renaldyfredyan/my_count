from engine import build_model
from data import FSC147Dataset
from arg_parser import get_argparser
import numpy as np
import torch.nn.functional as F

import argparse
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist


def crop(sample, crop_width, crop_height, overlap_width, overlap_height):
    """
    Split the image into overlapping crops.
    """
    h, w = sample.shape[-2], sample.shape[-1]
    
    # Calculate the number of windows needed horizontally and vertically
    n_windows_h = max(1, int(np.ceil((h - overlap_height) / (crop_height - overlap_height))))
    n_windows_w = max(1, int(np.ceil((w - overlap_width) / (crop_width - overlap_width))))
    
    # Calculate step sizes
    step_h = (h - crop_height) / (n_windows_h - 1) if n_windows_h > 1 else 0
    step_w = (w - crop_width) / (n_windows_w - 1) if n_windows_w > 1 else 0
    
    samples_cropped = []
    boundaries_x = []
    boundaries_y = []
    
    for i in range(n_windows_h):
        row_boundaries_x = []
        row_boundaries_y = []
        
        for j in range(n_windows_w):
            start_y = min(int(i * step_h), h - crop_height) if n_windows_h > 1 else 0
            start_x = min(int(j * step_w), w - crop_width) if n_windows_w > 1 else 0
            
            end_y = start_y + crop_height
            end_x = start_x + crop_width
            
            # Extract crop
            crop_sample = sample[:, start_y:end_y, start_x:end_x]
            samples_cropped.append(crop_sample)
            
            # Store boundaries
            row_boundaries_x.append((start_x, end_x))
            row_boundaries_y.append((start_y, end_y))
            
        boundaries_x.append(row_boundaries_x)
        boundaries_y.append(row_boundaries_y)
    
    return samples_cropped, boundaries_x, boundaries_y


def nested_tensor_from_tensor_list(tensor_list):
    """
    Create a nested tensor from a list of tensors.
    """
    # Assuming all tensors are of the same shape
    if len(tensor_list) == 0:
        return torch.empty(0)
    
    if torch.is_tensor(tensor_list[0]):
        # Case 1: All tensors are of the same shape
        return torch.stack(tensor_list)
    else:
        raise ValueError('Not implemented')


def tt_norm(pred_cnt, exemplars, image_size, center_points):
    """
    Apply Test-Time Normalization using exemplars.
    """
    # Simple implementation - this would need to be expanded based on the actual implementation
    if len(exemplars) == 0 or len(center_points) == 0:
        return pred_cnt
        
    avg_exemplar_area = 0
    for exemp in exemplars:
        avg_exemplar_area += (exemp[2] - exemp[0]) * (exemp[3] - exemp[1])
    
    avg_exemplar_area /= len(exemplars)
    
    # Normalize count based on area ratios
    image_area = image_size[0] * image_size[1]
    expected_count = image_area / avg_exemplar_area
    
    # Simple normalization factor
    norm_factor = min(1.5, max(0.5, expected_count / max(1, pred_cnt)))
    
    return pred_cnt * norm_factor


@torch.no_grad()
def evaluate(args):
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    gpu = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    dist.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank
    )

    model = DistributedDataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )
    torch.cuda.empty_cache()

    state_dict = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'), weights_only=True)['model']
    state_dict = {k if 'module.' in k else 'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    for split in ['val', 'test']:
        test = FSC147Dataset(
            args.data_path,
            args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=args.tiling_p,
        )
        test_loader = DataLoader(
            test,
            sampler=DistributedSampler(test),
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers
        )
        ae = torch.tensor(0.0).to(device)
        se = torch.tensor(0.0).to(device)
        model.eval()
        
        for img, bboxes, density_map in test_loader:
            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)
            
            # Standard evaluation without cropping
            if not args.crop and not args.simple_crop:
                out, _ = model(img, bboxes)
                pred_counts = out.flatten(1).sum(dim=1)
                gt_counts = density_map.flatten(1).sum(dim=1)
                
                # Apply TT-Norm if enabled
                if args.exemp_tt_norm:
                    for i in range(len(pred_counts)):
                        # Extract image size and bounding boxes for normalization
                        h, w = img[i].shape[-2], img[i].shape[-1]
                        img_size = torch.tensor([h, w])
                        # Get center points from bounding boxes
                        centers = torch.zeros((bboxes[i].shape[0], 2))
                        for j in range(bboxes[i].shape[0]):
                            if bboxes[i][j].sum() > 0:  # Valid bbox
                                centers[j, 0] = (bboxes[i][j, 0] + bboxes[i][j, 2]) / 2
                                centers[j, 1] = (bboxes[i][j, 1] + bboxes[i][j, 3]) / 2
                        
                        pred_counts[i] = tt_norm(
                            pred_counts[i].cpu().numpy(), 
                            bboxes[i].cpu().numpy(), 
                            img_size.cpu().numpy(), 
                            centers.cpu().numpy()
                        )
                        pred_counts[i] = torch.tensor(pred_counts[i]).to(device)
            
            # Complex cropping strategy
            elif args.crop:
                batch_size = img.shape[0]
                pred_counts = torch.zeros(batch_size).to(device)
                gt_counts = density_map.flatten(1).sum(dim=1)
                
                for b in range(batch_size):
                    sample = img[b]
                    sample_bboxes = bboxes[b]
                    
                    # Calculate average object size from bounding boxes
                    valid_bboxes = sample_bboxes[sample_bboxes.sum(dim=1) > 0]
                    if len(valid_bboxes) == 0:
                        # If no valid bounding boxes, use standard prediction
                        sample_out, _ = model(sample.unsqueeze(0), sample_bboxes.unsqueeze(0))
                        pred_counts[b] = sample_out.flatten().sum()
                        continue
                    
                    # Calculate average object dimensions
                    obj_width = 0
                    obj_height = 0
                    for bbox in valid_bboxes:
                        obj_width += bbox[2] - bbox[0]
                        obj_height += bbox[3] - bbox[1]
                    
                    obj_width = int(obj_width / len(valid_bboxes))
                    obj_height = int(obj_height / len(valid_bboxes))
                    
                    # Define crop dimensions
                    crop_width = 4 * obj_width
                    crop_height = 4 * obj_height
                    overlap_width = round(1.25 * obj_width)
                    overlap_height = round(1.25 * obj_height)
                    
                    h, w = sample.shape[-2], sample.shape[-1]
                    
                    # Adjust crop size if too large
                    crop_width = min(crop_width, w)
                    crop_height = min(crop_height, h)
                    
                    # Create crops
                    samples_cropped, boundaries_x, boundaries_y = crop(
                        sample.unsqueeze(0), crop_width, crop_height, overlap_width, overlap_height
                    )
                    
                    # Process crops in batches
                    num_batches = int(np.ceil(len(samples_cropped) / args.crop_batch_size))
                    pred_cnt = 0
                    
                    for batch_ind in range(num_batches):
                        batch_start = batch_ind * args.crop_batch_size
                        batch_end = min((batch_ind + 1) * args.crop_batch_size, len(samples_cropped))
                        
                        sample_subset = nested_tensor_from_tensor_list(samples_cropped[batch_start:batch_end])
                        bboxes_subset = sample_bboxes.unsqueeze(0).repeat(batch_end - batch_start, 1, 1)
                        
                        crop_out, _ = model(sample_subset, bboxes_subset)
                        
                        for crop_idx, out in enumerate(crop_out):
                            crop_idx_global = batch_start + crop_idx
                            row_ind = crop_idx_global // len(boundaries_x[0])
                            col_ind = crop_idx_global % len(boundaries_x[0])
                            
                            start_x, end_x = boundaries_x[row_ind][col_ind]
                            start_y, end_y = boundaries_y[row_ind][col_ind]
                            
                            # Calculate weight based on position
                            # Center regions get full weight, edges get partial weight
                            weight = 1.0
                            if start_x > 0 and end_x < w and start_y > 0 and end_y < h:
                                # Center region - keep full weight
                                pass
                            elif (start_x == 0 or end_x == w) and (start_y == 0 or end_y == h):
                                # Corner region - 1/4 weight
                                weight = 0.25
                            else:
                                # Edge region - 1/2 weight
                                weight = 0.5
                            
                            crop_count = out.flatten().sum().item()
                            pred_cnt += weight * crop_count
                    
                    pred_counts[b] = pred_cnt
            
            # Simple cropping strategy
            elif args.simple_crop:
                batch_size = img.shape[0]
                pred_counts = torch.zeros(batch_size).to(device)
                gt_counts = density_map.flatten(1).sum(dim=1)
                
                for b in range(batch_size):
                    sample = img[b]
                    sample_bboxes = bboxes[b]
                    
                    # Divide image into 4 quadrants
                    h, w = sample.shape[-2], sample.shape[-1]
                    
                    sample_top_left = F.interpolate(sample.unsqueeze(0)[:, :, :(h//2), :(w//2)], size=(h, w), mode='bilinear')
                    sample_top_right = F.interpolate(sample.unsqueeze(0)[:, :, :(h//2), (w//2):], size=(h, w), mode='bilinear')
                    sample_bottom_left = F.interpolate(sample.unsqueeze(0)[:, :, (h//2):, :(w//2)], size=(h, w), mode='bilinear')
                    sample_bottom_right = F.interpolate(sample.unsqueeze(0)[:, :, (h//2):, (w//2):], size=(h, w), mode='bilinear')
                    
                    samples_quadrants = torch.cat([
                        sample_top_left, sample_top_right, 
                        sample_bottom_left, sample_bottom_right
                    ], dim=0)
                    
                    # Repeat bounding boxes for each quadrant
                    bboxes_quad = sample_bboxes.unsqueeze(0).repeat(4, 1, 1)
                    
                    # Process all quadrants at once
                    quad_out, _ = model(samples_quadrants, bboxes_quad)
                    
                    # Sum the counts from each quadrant
                    pred_cnt = 0
                    for i in range(4):
                        pred_cnt += quad_out[i].flatten().sum().item() * 0.25
                    
                    pred_counts[b] = pred_cnt
                
            # Calculate errors
            ae += torch.abs(gt_counts - pred_counts).sum()
            se += ((gt_counts - pred_counts) ** 2).sum()

        # Gather results from all processes
        dist.all_reduce(ae)
        dist.all_reduce(se)

        if rank == 0:
            print(
                f"{split.capitalize()} set",
                f"MAE: {ae.item() / len(test):.2f}",
                f"RMSE: {torch.sqrt(se / len(test)).item():.2f}",
            )
            print(f"GT count: {gt_counts}")
            print(f"Predicted count: {pred_counts}")
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    dist.destroy_process_group()


def add_cropping_args(parser):
    # Add cropping-related arguments
    parser.add_argument('--crop', action='store_true', help='Use complex cropping strategy for high object counts')
    parser.add_argument('--simple_crop', action='store_true', help='Use simple cropping strategy (4 quadrants)')
    parser.add_argument('--crop_batch_size', type=int, default=10, help='Batch size for processing crops')
    parser.add_argument('--exemp_tt_norm', action='store_true', help='Apply exemplar-based Test-Time Normalization')
    parser.add_argument('--num_select', type=int, default=100, help='Threshold for high object count that triggers cropping')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Efficient', parents=[get_argparser()])
    parser = add_cropping_args(parser)
    args = parser.parse_args()
    evaluate(args)