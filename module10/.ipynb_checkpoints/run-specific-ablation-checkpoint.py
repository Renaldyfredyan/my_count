#!/usr/bin/env python
"""
Script to run a specific ablation experiment rather than the full study.
This allows for running individual ablation experiments separately.
"""

import os
import argparse
import json
from ablation_study_framework import AblationStudy, seed_everything
from arg_parser import get_argparser

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a specific ablation experiment")
    parser.add_argument("--experiment", required=True, help="Name of the experiment to run")
    parser.add_argument("--description", default="", help="Description of the experiment")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_enc_layers", type=int, default=3, help="Number of encoder layers")
    parser.add_argument("--num_iefl_steps", type=int, default=3, help="Number of iEFL iterative steps")
    parser.add_argument("--emb_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--kernel_dim", type=int, default=3, help="Kernel dimension")
    parser.add_argument("--backbone_lr", type=float, default=0, help="Backbone learning rate")
    parser.add_argument("--reduction", type=int, default=8, help="Feature reduction factor")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--pre_norm", action="store_true", help="Use pre-norm architecture")
    parser.add_argument("--aux_weight", type=float, default=0.3, help="Auxiliary loss weight")
    parser.add_argument("--zero_shot", action="store_true", help="Enable zero-shot mode")
    parser.add_argument("--data_path", type=str, default="/path/to/data", help="Path to dataset")
    parser.add_argument("--model_path", type=str, default="./checkpoints", help="Path to save models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    seed_everything(args.seed)
    
    # Get base arguments from the get_argparser function
    base_parser = get_argparser()
    base_args = base_parser.parse_args([])  # Parse empty list to get default values
    
    # Override base args with command line arguments
    base_args.epochs = args.epochs
    base_args.model_name = f"ablation_{args.experiment}"
    base_args.batch_size = args.batch_size
    base_args.lr = args.lr
    base_args.num_enc_layers = args.num_enc_layers
    base_args.num_iefl_iterative_steps = args.num_iefl_steps
    base_args.emb_dim = args.emb_dim
    base_args.num_heads = args.num_heads
    base_args.kernel_dim = args.kernel_dim
    base_args.backbone_lr = args.backbone_lr
    base_args.reduction = args.reduction
    base_args.dropout = args.dropout
    base_args.pre_norm = args.pre_norm
    base_args.aux_weight = args.aux_weight
    base_args.zero_shot = args.zero_shot
    base_args.data_path = args.data_path
    base_args.model_path = args.model_path
    base_args.resume_training = False
    
    # Ensure checkpoint directory exists
    os.makedirs(base_args.model_path, exist_ok=True)
    
    # Initialize ablation study
    study = AblationStudy(base_args)
    
    # Run the experiment
    result = study.run_experiment(
        experiment_name=args.experiment,
        override_args={},  # No overrides since we've already set them above
        description=args.description
    )
    
    # Print results
    print("\nExperiment Results:")
    print(f"Name: {args.experiment}")
    print(f"Description: {args.description}")
    print(f"Test MAE: {result['test_mae']:.3f}")
    print(f"Test RMSE: {result['test_rmse']:.3f}")
    
    # Save results to JSON
    results_file = os.path.join(base_args.model_path, f'ablation_{args.experiment}_results.json')
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    # Handle distributed training
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
        
    main()