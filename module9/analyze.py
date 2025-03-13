import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from engine import build_model
from data import FSC147Dataset
from arg_parser import get_argparser

def analyze_counting_errors(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model
    model = build_model(args).to(device)
    
    # Load state dict
    state_dict = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'), 
                           map_location=device)['model']
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    # Daftar gambar yang ingin dikecualikan
    excluded_images = ['7171.jpg', '7611.jpg', '1123.jpg', '865.jpg', '935.jpg', '7656.jpg']
    
    results = {}
    
    for split in ['val', 'test']:
        dataset = FSC147Dataset(
            args.data_path,
            args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=0.0,  # No tiling for clean analysis
            return_image_name=True  # Get image names to track errors
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Process one image at a time for detailed analysis
            shuffle=False,
            num_workers=4
        )
        
        # Storage for analysis
        all_gt_counts = []
        all_pred_counts = []
        all_errors = []
        all_rel_errors = []
        all_image_names = []
        
        # Storage for analysis without excluded images
        filtered_gt_counts = []
        filtered_pred_counts = []
        filtered_errors = []
        filtered_rel_errors = []
        filtered_image_names = []
        
        # Group images by count range for analysis
        count_ranges = {
            "1-10": {"gt": [], "pred": [], "error": [], "rel_error": []},
            "11-50": {"gt": [], "pred": [], "error": [], "rel_error": []},
            "51-100": {"gt": [], "pred": [], "error": [], "rel_error": []},
            "101-500": {"gt": [], "pred": [], "error": [], "rel_error": []},
            ">500": {"gt": [], "pred": [], "error": [], "rel_error": []}
        }
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc=f"Analyzing {split}")):
                if len(batch) == 3:  # Standard dataset without image names
                    img, bboxes, density_map = batch
                    image_name = [f"img_{i}"]  # Use index as placeholder
                else:  # Dataset returns image names
                    img, bboxes, density_map, image_name = batch
                
                img = img.to(device)
                bboxes = bboxes.to(device)
                density_map = density_map.to(device)
                
                # Get prediction
                out, _ = model(img, bboxes)
                
                # Calculate counts
                gt_count = density_map.flatten(1).sum(dim=1).item()
                pred_count = out.flatten(1).sum(dim=1).item()
                
                # Calculate error
                error = abs(gt_count - pred_count)
                rel_error = error / max(1, gt_count) * 100  # % error
                
                # Store results for complete dataset
                all_gt_counts.append(gt_count)
                all_pred_counts.append(pred_count)
                all_errors.append(error)
                all_rel_errors.append(rel_error)
                all_image_names.append(image_name[0])
                
                # Store results for filtered dataset (excluding problematic images)
                if image_name[0] not in excluded_images:
                    filtered_gt_counts.append(gt_count)
                    filtered_pred_counts.append(pred_count)
                    filtered_errors.append(error)
                    filtered_rel_errors.append(rel_error)
                    filtered_image_names.append(image_name[0])
                    
                    # Categorize by count range
                    if gt_count <= 10:
                        count_range = "1-10"
                    elif gt_count <= 50:
                        count_range = "11-50"
                    elif gt_count <= 100:
                        count_range = "51-100"
                    elif gt_count <= 500:
                        count_range = "101-500"
                    else:
                        count_range = ">500"
                    
                    count_ranges[count_range]["gt"].append(gt_count)
                    count_ranges[count_range]["pred"].append(pred_count)
                    count_ranges[count_range]["error"].append(error)
                    count_ranges[count_range]["rel_error"].append(rel_error)
            
        # Calculate statistics for complete dataset
        mae = np.mean(all_errors)
        rmse = np.sqrt(np.mean(np.square(np.array(all_gt_counts) - np.array(all_pred_counts))))
        
        # Calculate statistics for filtered dataset
        filtered_mae = np.mean(filtered_errors)
        filtered_rmse = np.sqrt(np.mean(np.square(np.array(filtered_gt_counts) - np.array(filtered_pred_counts))))
        
        # Calculate over/under counting statistics for filtered dataset
        over_count = sum(1 for p, g in zip(filtered_pred_counts, filtered_gt_counts) if p > g)
        under_count = sum(1 for p, g in zip(filtered_pred_counts, filtered_gt_counts) if p < g)
        exact_count = sum(1 for p, g in zip(filtered_pred_counts, filtered_gt_counts) if round(p) == round(g))
        
        # Find worst cases in filtered dataset
        sorted_errors = sorted(zip(filtered_image_names, filtered_gt_counts, filtered_pred_counts, filtered_errors, filtered_rel_errors), 
                              key=lambda x: x[3], reverse=True)
        
        # Analyze count ranges
        range_stats = {}
        for range_name, data in count_ranges.items():
            if not data["gt"]:  # Skip empty ranges
                continue
            range_stats[range_name] = {
                "count": len(data["gt"]),
                "mae": np.mean(data["error"]) if data["error"] else 0,
                "rmse": np.sqrt(np.mean(np.square(np.array(data["gt"]) - np.array(data["pred"])))) if data["gt"] else 0,
                "rel_error": np.mean(data["rel_error"]) if data["rel_error"] else 0,
                "over_count": sum(1 for p, g in zip(data["pred"], data["gt"]) if p > g),
                "under_count": sum(1 for p, g in zip(data["pred"], data["gt"]) if p < g)
            }
        
        # Store all results
        results[split] = {
            "overall": {
                "mae": mae,
                "rmse": rmse,
                "filtered_mae": filtered_mae,
                "filtered_rmse": filtered_rmse,
                "over_count_pct": over_count / len(filtered_gt_counts) * 100,
                "under_count_pct": under_count / len(filtered_gt_counts) * 100,
                "exact_count_pct": exact_count / len(filtered_gt_counts) * 100
            },
            "range_stats": range_stats,
            "worst_cases": sorted_errors[:20],  # Top 20 worst predictions
            "raw_data": {
                "gt": filtered_gt_counts,
                "pred": filtered_pred_counts,
                "errors": filtered_errors,
                "image_names": filtered_image_names
            },
            "excluded_images": excluded_images
        }
        
        # Print summary
        print(f"\n=== {split.upper()} SET ANALYSIS ===")
        print(f"With all images: MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        print(f"After excluding {', '.join(excluded_images)}: MAE: {filtered_mae:.2f}, RMSE: {filtered_rmse:.2f}")
        print(f"Over-counting: {over_count}/{len(filtered_gt_counts)} images ({over_count/len(filtered_gt_counts)*100:.1f}%)")
        print(f"Under-counting: {under_count}/{len(filtered_gt_counts)} images ({under_count/len(filtered_gt_counts)*100:.1f}%)")
        print(f"Exact count: {exact_count}/{len(filtered_gt_counts)} images ({exact_count/len(filtered_gt_counts)*100:.1f}%)")
        
        print("\nPerformance by count range:")
        for range_name, stats in range_stats.items():
            if stats['count'] > 0:
                print(f"  {range_name}: {stats['count']} images, MAE: {stats['mae']:.2f}, Rel Error: {stats['rel_error']:.1f}%")
                print(f"    Over-counting: {stats['over_count']}/{stats['count']} ({stats['over_count']/stats['count']*100:.1f}%)")
                print(f"    Under-counting: {stats['under_count']}/{stats['count']} ({stats['under_count']/stats['count']*100:.1f}%)")
        
        print("\nTop 5 worst predictions (excluding outliers):")
        for img_name, gt, pred, error, rel_error in sorted_errors[:5]:
            print(f"  {img_name}: GT={gt:.1f}, Pred={pred:.1f}, Error={error:.1f} ({rel_error:.1f}%)")
    
    # Create visualizations
    create_error_visualizations(results, args)
    
    return results

def create_error_visualizations(results, args):
    """Create visualizations to analyze counting errors"""
    os.makedirs("error_analysis", exist_ok=True)
    
    for split, data in results.items():
        # Extract data
        gt_counts = data["raw_data"]["gt"]
        pred_counts = data["raw_data"]["pred"]
        errors = [p - g for g, p in zip(gt_counts, pred_counts)]
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(gt_counts, pred_counts, alpha=0.6, 
                   c=["red" if e > 0 else "blue" for e in errors])
        
        # Add diagonal line (perfect prediction)
        max_val = max(max(gt_counts), max(pred_counts)) * 1.1
        plt.plot([0, max_val], [0, max_val], 'g--')
        
        plt.xlabel('Ground Truth Count')
        plt.ylabel('Predicted Count')
        plt.title(f'{split} Set: Ground Truth vs Predicted Counts')
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend(['Perfect prediction', 'Over-counting', 'Under-counting'])
        
        plt.tight_layout()
        plt.savefig(f'error_analysis/{split}_gt_vs_pred.png')
        plt.close()
        
        # Bar chart of over/under counting by range
        plt.figure(figsize=(12, 6))
        ranges = list(data["range_stats"].keys())
        over_counts = []
        under_counts = []
        
        for r in ranges:
            stats = data["range_stats"][r]
            if stats["count"] > 0:
                over_counts.append(stats["over_count"] / stats["count"] * 100)
                under_counts.append(stats["under_count"] / stats["count"] * 100)
            else:
                over_counts.append(0)
                under_counts.append(0)
        
        x = np.arange(len(ranges))
        width = 0.35
        
        plt.bar(x - width/2, over_counts, width, label='Over-counting')
        plt.bar(x + width/2, under_counts, width, label='Under-counting')
        
        plt.xlabel('Count Range')
        plt.ylabel('Percentage of Images')
        plt.title(f'{split} Set: Over/Under Counting by Range')
        plt.xticks(x, ranges)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'error_analysis/{split}_over_under_by_range.png')
        plt.close()
        
        # Relative error distribution
        plt.figure(figsize=(10, 6))
        rel_errors = [e/max(1, g)*100 for g, p, e in zip(gt_counts, pred_counts, [abs(p-g) for p, g in zip(pred_counts, gt_counts)])]
        
        sns.histplot(rel_errors, bins=30, kde=True)
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Count')
        plt.title(f'{split} Set: Distribution of Relative Errors')
        
        plt.tight_layout()
        plt.savefig(f'error_analysis/{split}_error_distribution.png')
        plt.close()
        
        # Error vs ground truth count
        plt.figure(figsize=(10, 6))
        plt.scatter(gt_counts, errors, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        
        plt.xlabel('Ground Truth Count')
        plt.ylabel('Error (Predicted - Ground Truth)')
        plt.title(f'{split} Set: Error vs Ground Truth Count')
        
        plt.tight_layout()
        plt.savefig(f'error_analysis/{split}_error_vs_gt.png')
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Efficient', parents=[get_argparser()])
    args = parser.parse_args()
    
    # # Default values if not provided
    # if not hasattr(args, 'model_path') or not args.model_path:
    #     args.model_path = './checkpoints/'
    # if not hasattr(args, 'data_path') or not args.data_path:
    #     args.data_path = './Dataset/'
    
    # Run analysis
    results = analyze_counting_errors(args)
    
    # Save results to file for later reference
    # import json
    
    # # Convert non-serializable types
    # for split in results:
    #     results[split]["raw_data"]["gt"] = [float(x) for x in results[split]["raw_data"]["gt"]]
    #     results[split]["raw_data"]["pred"] = [float(x) for x in results[split]["raw_data"]["pred"]]
    #     results[split]["raw_data"]["errors"] = [float(x) for x in results[split]["raw_data"]["errors"]]
        
    #     # Convert worst cases
    #     serializable_worst = []
    #     for case in results[split]["worst_cases"]:
    #         serializable_worst.append([
    #             case[0],  # image name
    #             float(case[1]),  # gt
    #             float(case[2]),  # pred
    #             float(case[3]),  # error
    #             float(case[4])   # rel error
    #         ])
    #     results[split]["worst_cases"] = serializable_worst
    
    # with open('error_analysis/analysis_results.json', 'w') as f:
    #     json.dump(results, f, indent=2)
    
    print("\nAnalysis completed! Results saved to 'error_analysis' folder.")