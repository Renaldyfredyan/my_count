import os
import argparse
from ablation_study_framework import AblationStudy, seed_everything
from arg_parser import get_argparser

def main():
    # Set random seed for reproducibility
    seed_everything(42)
    
    # Get base arguments
    base_parser = get_argparser()
    base_args = base_parser.parse_args([])  # Parse empty list to get default values
    
    # Override some defaults for the ablation study
    base_args.epochs = 50  # Reduce epochs for faster ablation study
    base_args.resume_training = False
    base_args.model_name = "efficient_ablation"
    
    # Ensure checkpoint directory exists
    os.makedirs(base_args.model_path, exist_ok=True)
    
    # Initialize ablation study
    study = AblationStudy(base_args)
    
    # Define ablation experiments
    experiments = [
        # Baseline model (full model with all components)
        {
            "name": "baseline",
            "override_args": {},
            "description": "Full model with all components"
        },
        
        # Backbone ablations
        {
            "name": "backbone_frozen",
            "override_args": {"backbone_lr": 0.0},
            "description": "Frozen backbone weights"
        },
        {
            "name": "backbone_trainable",
            "override_args": {"backbone_lr": 1e-5},
            "description": "Trainable backbone weights"
        },
        
        # Hybrid encoder ablations
        {
            "name": "no_encoder",
            "override_args": {"num_enc_layers": 0},
            "description": "No hybrid encoder layers"
        },
        {
            "name": "one_encoder_layer",
            "override_args": {"num_enc_layers": 1},
            "description": "One hybrid encoder layer"
        },
        {
            "name": "five_encoder_layers",
            "override_args": {"num_enc_layers": 5},
            "description": "Five hybrid encoder layers"
        },
        
        # IEFL module ablations
        {
            "name": "one_iefl_step",
            "override_args": {"num_iefl_iterative_steps": 1},
            "description": "One iEFL iterative step"
        },
        {
            "name": "five_iefl_steps",
            "override_args": {"num_iefl_iterative_steps": 5},
            "description": "Five iEFL iterative steps"
        },
        
        # Architecture ablations
        {
            "name": "small_emb_dim",
            "override_args": {"emb_dim": 128},
            "description": "Smaller embedding dimension (128)"
        },
        {
            "name": "large_emb_dim",
            "override_args": {"emb_dim": 512},
            "description": "Larger embedding dimension (512)"
        },
        {
            "name": "few_heads",
            "override_args": {"num_heads": 4},
            "description": "Fewer attention heads (4)"
        },
        {
            "name": "many_heads",
            "override_args": {"num_heads": 16},
            "description": "More attention heads (16)"
        },
        
        # Kernel ablations
        {
            "name": "small_kernel",
            "override_args": {"kernel_dim": 1},
            "description": "Smaller kernel dimension (1x1)"
        },
        {
            "name": "large_kernel",
            "override_args": {"kernel_dim": 5},
            "description": "Larger kernel dimension (5x5)"
        },
        
        # Scale ablations
        {
            "name": "small_reduction",
            "override_args": {"reduction": 4},
            "description": "Smaller feature reduction (4)"
        },
        {
            "name": "large_reduction",
            "override_args": {"reduction": 16},
            "description": "Larger feature reduction (16)"
        },
        
        # Normalization ablations
        {
            "name": "post_norm",
            "override_args": {"pre_norm": False},
            "description": "Post-normalization architecture"
        },
        {
            "name": "pre_norm",
            "override_args": {"pre_norm": True},
            "description": "Pre-normalization architecture"
        },
        
        # Regularization ablations
        {
            "name": "high_dropout",
            "override_args": {"dropout": 0.3},
            "description": "Higher dropout rate (0.3)"
        },
        {
            "name": "low_dropout",
            "override_args": {"dropout": 0.0},
            "description": "No dropout"
        },
        
        # Auxiliary loss ablations
        {
            "name": "no_aux_loss",
            "override_args": {"aux_weight": 0.0},
            "description": "No auxiliary loss"
        },
        {
            "name": "high_aux_loss",
            "override_args": {"aux_weight": 0.6},
            "description": "Higher auxiliary loss weight (0.6)"
        },
        
        # Zero-shot ablation
        {
            "name": "zero_shot",
            "override_args": {"zero_shot": True},
            "description": "Zero-shot mode enabled"
        }
    ]
    
    # Run each experiment
    for exp in experiments:
        study.run_experiment(
            experiment_name=exp["name"],
            override_args=exp["override_args"],
            description=exp["description"]
        )
    
    # Summarize results
    summary_path = os.path.join(base_args.model_path, 'ablation_summary.json')
    study.summarize_results(summary_path)
    
    print(f"Ablation study completed! Summary saved to {summary_path}")

if __name__ == "__main__":
    # Handle distributed training
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
        
    main()