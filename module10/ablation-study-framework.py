import os
import argparse
import torch
import numpy as np
import random
import json
from time import perf_counter

from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist

from data import FSC147Dataset
from arg_parser import get_argparser
from losses import ObjectNormalizedL2Loss
from engine import build_model

# For reproducibility
def seed_everything(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AblationStudy:
    def __init__(self, base_args):
        """
        Initialize the ablation study with base arguments
        
        Args:
            base_args: Base arguments for the model
        """
        self.base_args = base_args
        self.results = {}
        
    def run_experiment(self, experiment_name, override_args=None, description=""):
        """
        Run a single experiment with specific configuration
        
        Args:
            experiment_name: Name of the experiment
            override_args: Dictionary of args to override from base_args
            description: Description of what's being tested
        """
        # Make a copy of base args and update with overrides
        args = argparse.Namespace(**vars(self.base_args))
        if override_args:
            for key, value in override_args.items():
                setattr(args, key, value)
        
        # Set model name to avoid overwriting other experiments
        args.model_name = f"ablation_{experiment_name}"
        
        # Initialize distributed training
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        gpu = int(os.environ['LOCAL_RANK'])

        torch.cuda.set_device(gpu)
        device = torch.device(gpu)

        dist.init_process_group(
            backend='nccl', init_method='env://',
            world_size=world_size, rank=rank
        )
        
        # Build model with the modified args
        model = DistributedDataParallel(
            build_model(args).to(device),
            device_ids=[gpu],
            output_device=gpu
        )
        
        # Set up optimizer
        backbone_params = dict()
        non_backbone_params = dict()
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'backbone' in n:
                backbone_params[n] = p
            else:
                non_backbone_params[n] = p

        optimizer = torch.optim.AdamW(
            [
                {'params': non_backbone_params.values()},
                {'params': backbone_params.values(), 'lr': args.backbone_lr}
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.25)
        
        # Loss function
        criterion = ObjectNormalizedL2Loss()
        
        # Data loaders
        train = FSC147Dataset(
            args.data_path,
            args.image_size,
            split='train',
            num_objects=args.num_objects,
            tiling_p=args.tiling_p,
            zero_shot=args.zero_shot
        )
        val = FSC147Dataset(
            args.data_path,
            args.image_size,
            split='val',
            num_objects=args.num_objects,
            tiling_p=args.tiling_p
        )
        test = FSC147Dataset(
            args.data_path,
            args.image_size,
            split='test',
            num_objects=args.num_objects,
            tiling_p=args.tiling_p
        )
        
        train_loader = DataLoader(
            train,
            sampler=DistributedSampler(train),
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=args.num_workers
        )
        val_loader = DataLoader(
            val,
            sampler=DistributedSampler(val),
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers
        )
        test_loader = DataLoader(
            test,
            sampler=DistributedSampler(test),
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers
        )
        
        # Training loop
        best_val_mae = float('inf')
        start_epoch = 0
        
        # Create experiment directory
        if rank == 0:
            os.makedirs(os.path.join(args.model_path, 'ablation_results'), exist_ok=True)
            experiment_dir = os.path.join(args.model_path, 'ablation_results', experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)
            
            # Save experiment config
            with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
                config = {
                    'name': experiment_name,
                    'description': description,
                    'args': vars(args)
                }
                json.dump(config, f, indent=2)
            
            print(f"Starting experiment: {experiment_name}")
            print(f"Description: {description}")
        
        epoch_metrics = []
        for epoch in range(start_epoch + 1, args.epochs + 1):
            if rank == 0:
                start = perf_counter()
            
            train_loss = torch.tensor(0.0).to(device)
            val_loss = torch.tensor(0.0).to(device)
            aux_train_loss = torch.tensor(0.0).to(device)
            aux_val_loss = torch.tensor(0.0).to(device)
            train_ae = torch.tensor(0.0).to(device)
            val_ae = torch.tensor(0.0).to(device)

            train_loader.sampler.set_epoch(epoch)
            model.train()
            
            # Training
            for img, bboxes, density_map in train_loader:
                img = img.to(device)
                bboxes = bboxes.to(device)
                density_map = density_map.to(device)

                optimizer.zero_grad()
                out, aux_out = model(img, bboxes)

                # Obtain the number of objects in batch
                with torch.no_grad():
                    num_objects = density_map.sum()
                    dist.all_reduce(num_objects)

                main_loss = criterion(out, density_map, num_objects)
                aux_loss = sum([
                    args.aux_weight * criterion(aux, density_map, num_objects) for aux in aux_out
                ])
                loss = main_loss + aux_loss
                loss.backward()

                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                train_loss += main_loss * img.size(0)
                aux_train_loss += aux_loss * img.size(0)
                train_ae += torch.abs(
                    density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
                ).sum()
            
            # Validation
            model.eval()
            with torch.no_grad():
                for img, bboxes, density_map in val_loader:
                    img = img.to(device)
                    bboxes = bboxes.to(device)
                    density_map = density_map.to(device)
                    out, aux_out = model(img, bboxes)
                    
                    with torch.no_grad():
                        num_objects = density_map.sum()
                        dist.all_reduce(num_objects)

                    main_loss = criterion(out, density_map, num_objects)
                    aux_loss = sum([
                        args.aux_weight * criterion(aux, density_map, num_objects) for aux in aux_out
                    ])
                    loss = main_loss + aux_loss

                    val_loss += main_loss * img.size(0)
                    aux_val_loss += aux_loss * img.size(0)
                    val_ae += torch.abs(
                        density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
                    ).sum()

            dist.all_reduce(train_loss)
            dist.all_reduce(val_loss)
            dist.all_reduce(aux_train_loss)
            dist.all_reduce(aux_val_loss)
            dist.all_reduce(train_ae)
            dist.all_reduce(val_ae)

            scheduler.step()
            
            train_mae = train_ae.item() / len(train)
            val_mae = val_ae.item() / len(val)
            
            if rank == 0:
                end = perf_counter()
                is_best = val_mae < best_val_mae
                
                if is_best:
                    best_val_mae = val_mae
                    checkpoint = {
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_val_mae': val_mae
                    }
                    torch.save(
                        checkpoint,
                        os.path.join(experiment_dir, f'{args.model_name}.pt')
                    )
                
                # Log metrics
                metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss.item(),
                    'aux_train_loss': aux_train_loss.item(),
                    'val_loss': val_loss.item(),
                    'aux_val_loss': aux_val_loss.item(),
                    'train_mae': train_mae,
                    'val_mae': val_mae,
                    'epoch_time': end - start,
                    'is_best': is_best
                }
                epoch_metrics.append(metrics)
                
                print(
                    f"Experiment: {experiment_name}, Epoch: {epoch}",
                    f"Train loss: {train_loss.item():.3f}",
                    f"Val loss: {val_loss.item():.3f}",
                    f"Train MAE: {train_mae:.3f}",
                    f"Val MAE: {val_mae:.3f}",
                    f"Epoch time: {end - start:.3f} seconds",
                    'best' if is_best else ''
                )
                
                # Save epoch metrics
                with open(os.path.join(experiment_dir, 'metrics.json'), 'w') as f:
                    json.dump(epoch_metrics, f, indent=2)
            
        # Final evaluation on test set after training
        test_ae = torch.tensor(0.0).to(device)
        test_se = torch.tensor(0.0).to(device)
        
        # Load best model
        if rank == 0:
            print(f"Loading best model for test evaluation...")
        
        best_model_path = os.path.join(experiment_dir, f'{args.model_name}.pt')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
        
        model.eval()
        with torch.no_grad():
            for img, bboxes, density_map in test_loader:
                img = img.to(device)
                bboxes = bboxes.to(device)
                density_map = density_map.to(device)
                
                out, _ = model(img, bboxes)
                
                test_ae += torch.abs(
                    density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
                ).sum()
                test_se += ((
                    density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
                ) ** 2).sum()
        
        dist.all_reduce(test_ae)
        dist.all_reduce(test_se)
        
        test_mae = test_ae.item() / len(test)
        test_rmse = torch.sqrt(test_se / len(test)).item()
        
        # Save final results
        if rank == 0:
            final_results = {
                'name': experiment_name,
                'description': description,
                'val_mae': best_val_mae,
                'test_mae': test_mae,
                'test_rmse': test_rmse
            }
            
            with open(os.path.join(experiment_dir, 'results.json'), 'w') as f:
                json.dump(final_results, f, indent=2)
            
            # Store results in the ablation study object
            self.results[experiment_name] = final_results
            
            print(f"Experiment {experiment_name} completed!")
            print(f"Best validation MAE: {best_val_mae:.3f}")
            print(f"Test MAE: {test_mae:.3f}")
            print(f"Test RMSE: {test_rmse:.3f}")
        
        # Clean up
        dist.destroy_process_group()
        
        return final_results
    
    def summarize_results(self, output_path):
        """
        Summarize all experiment results
        
        Args:
            output_path: Path to save summary
        """
        if not self.results:
            print("No experiment results to summarize.")
            return
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Summarize all results in a table
        summary = {
            'experiments': self.results,
            'baseline': None  # Will store info about baseline experiment
        }
        
        # Find baseline experiment if it exists
        if 'baseline' in self.results:
            summary['baseline'] = self.results['baseline']
            baseline_mae = self.results['baseline']['test_mae']
            
            # Calculate relative improvements
            for name, result in self.results.items():
                if name != 'baseline':
                    rel_improvement = ((baseline_mae - result['test_mae']) / baseline_mae) * 100
                    self.results[name]['relative_improvement'] = rel_improvement
        
        # Save summary
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary table
        print("\n" + "="*80)
        print("ABLATION STUDY SUMMARY")
        print("="*80)
        print(f"{'Experiment':<30} {'Description':<30} {'Test MAE':<10} {'Test RMSE':<10} {'Rel. Imp.%':<10}")
        print("-"*80)
        
        for name, result in sorted(self.results.items(), key=lambda x: x[1]['test_mae']):
            rel_imp = result.get('relative_improvement', 'N/A')
            rel_imp_str = f"{rel_imp:.2f}" if isinstance(rel_imp, float) else rel_imp
            
            print(f"{name:<30} {result['description'][:30]:<30} {result['test_mae']:<10.3f} {result['test_rmse']:<10.3f} {rel_imp_str:<10}")
        
        print("="*80)
        
        return summary