import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

def load_ablation_results(summary_path):
    """Load the ablation study summary results"""
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found at {summary_path}")
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    return summary

def create_comparison_table(summary):
    """Create a DataFrame with the comparison of all experiments"""
    data = []
    
    baseline = summary.get('baseline', None)
    baseline_mae = baseline['test_mae'] if baseline else None
    
    for name, result in summary['experiments'].items():
        rel_imp = 'N/A'
        if baseline_mae is not None and name != 'baseline':
            rel_imp = ((baseline_mae - result['test_mae']) / baseline_mae) * 100
        
        data.append({
            'Experiment': name,
            'Description': result['description'],
            'Val MAE': result['val_mae'],
            'Test MAE': result['test_mae'],
            'Test RMSE': result['test_rmse'],
            'Relative Improvement (%)': rel_imp if isinstance(rel_imp, float) else 'N/A'
        })
    
    df = pd.DataFrame(data)
    
    # Sort by Test MAE (better performance at the top)
    df = df.sort_values('Test MAE')
    
    return df

def plot_mae_comparison(df, output_dir):
    """Plot MAE comparison between experiments"""
    plt.figure(figsize=(12, 8))
    
    # Create plot
    ax = sns.barplot(x='Test MAE', y='Experiment', data=df, palette='viridis')
    
    # Highlight baseline
    baseline_row = df[df['Experiment'] == 'baseline']
    if not baseline_row.empty:
        baseline_index = df.index.get_loc(baseline_row.index[0])
        ax.patches[baseline_index].set_facecolor('red')
    
    # Add values to bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_width():.2f}", 
                   (p.get_width(), p.get_y() + p.get_height()/2),
                   ha='left', va='center', fontsize=9, color='black', xytext=(5, 0),
                   textcoords='offset points')
    
    plt.title('MAE Comparison Across Experiments', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'mae_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_relative_improvements(df, output_dir):
    """Plot relative improvements compared to baseline"""
    # Filter out the baseline and experiments with no relative improvement value
    df_filtered = df[(df['Experiment'] != 'baseline') & 
                     (df['Relative Improvement (%)'] != 'N/A')].copy()
    
    if df_filtered.empty:
        print("No relative improvement data to plot")
        return
    
    # Convert to numeric and sort
    df_filtered['Relative Improvement (%)'] = pd.to_numeric(df_filtered['Relative Improvement (%)'])
    df_filtered = df_filtered.sort_values('Relative Improvement (%)', ascending=False)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Set color based on positive/negative improvement
    colors = ['green' if x > 0 else 'red' for x in df_filtered['Relative Improvement (%)']]
    
    ax = sns.barplot(x='Relative Improvement (%)', y='Experiment', data=df_filtered, palette=colors)
    
    # Add a vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add values to bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_width():.2f}%", 
                   (p.get_width(), p.get_y() + p.get_height()/2),
                   ha='left' if p.get_width() > 0 else 'right', 
                   va='center', fontsize=9, 
                   color='black',
                   xytext=(5, 0) if p.get_width() > 0 else (-5, 0),
                   textcoords='offset points')
    
    plt.title('Relative Improvement/Degradation Compared to Baseline (%)', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'relative_improvements.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_component_impact(df, output_dir):
    """
    Create plots showing the impact of specific component types
    using grouped bar charts
    """
    # Group experiments by component type
    component_groups = {
        'Encoder': ['baseline', 'no_encoder', 'one_encoder_layer', 'five_encoder_layers'],
        'iEFL Steps': ['baseline', 'one_iefl_step', 'five_iefl_steps'],
        'Embedding Dimension': ['baseline', 'small_emb_dim', 'large_emb_dim'],
        'Attention Heads': ['baseline', 'few_heads', 'many_heads'],
        'Kernel Size': ['baseline', 'small_kernel', 'large_kernel'],
        'Feature Reduction': ['baseline', 'small_reduction', 'large_reduction'],
        'Normalization': ['baseline', 'pre_norm', 'post_norm'],
        'Dropout': ['baseline', 'low_dropout', 'high_dropout'],
        'Auxiliary Loss': ['baseline', 'no_aux_loss', 'high_aux_loss'],
        'Backbone Training': ['baseline', 'backbone_frozen', 'backbone_trainable']
    }
    
    for group_name, experiments in component_groups.items():
        # Filter data for this group
        df_group = df[df['Experiment'].isin(experiments)].copy()
        
        if len(df_group) <= 1:
            continue
            
        # Sort by experiment name to ensure consistent order
        df_group = df_group.sort_values('Test MAE')
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Set up grouped bar chart
        x = np.arange(len(df_group))
        width = 0.35
        
        # Plot both validation and test MAE
        plt.bar(x - width/2, df_group['Val MAE'], width, label='Validation MAE')
        plt.bar(x + width/2, df_group['Test MAE'], width, label='Test MAE')
        
        # Add labels and formatting
        plt.title(f'Impact of {group_name}', fontsize=16)
        plt.xlabel('Experiment')
        plt.ylabel('MAE')
        plt.xticks(x, df_group['Experiment'], rotation=45, ha='right')
        plt.legend()
        
        # Add value labels
        for i, v in enumerate(df_group['Val MAE']):
            plt.text(i - width/2, v + 0.1, f"{v:.2f}", ha='center', fontsize=9)
        
        for i, v in enumerate(df_group['Test MAE']):
            plt.text(i + width/2, v + 0.1, f"{v:.2f}", ha='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'impact_{group_name.lower().replace(" ", "_")}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def create_learning_curves(results_dir, output_dir):
    """
    Create learning curves for each experiment by reading the metrics.json files
    """
    results_path = Path(results_dir)
    experiment_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    
    plt.figure(figsize=(15, 10))
    
    # Plot validation MAE learning curves
    for exp_dir in experiment_dirs:
        metrics_file = exp_dir / 'metrics.json'
        if not metrics_file.exists():
            continue
            
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        if not metrics:
            continue
            
        epochs = [m['epoch'] for m in metrics]
        val_mae = [m['val_mae'] for m in metrics]
        
        plt.plot(epochs, val_mae, label=exp_dir.name)
    
    plt.title('Validation MAE Learning Curves', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Validation MAE')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_ablation_report(summary_path, output_dir):
    """Generate a comprehensive ablation study report"""
    # Load results
    summary = load_ablation_results(summary_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison table
    df = create_comparison_table(summary)
    
    # Save table as CSV
    df.to_csv(os.path.join(output_dir, 'ablation_comparison.csv'), index=False)
    
    # Generate plots
    plot_mae_comparison(df, output_dir)
    plot_relative_improvements(df, output_dir)
    plot_component_impact(df, output_dir)
    
    # Generate learning curves if metrics are available
    results_dir = os.path.dirname(summary_path) + '/ablation_results'
    if os.path.exists(results_dir):
        create_learning_curves(results_dir, output_dir)
    
    # Generate an HTML report
    html_report = f"""
    <html>
    <head>
        <title>Ablation Study Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333366; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .plot-container {{ margin-bottom: 30px; }}
            .plot-container img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Ablation Study Results</h1>
        
        <h2>Experiment Comparison</h2>
        <table>
            <tr>
                {"".join([f"<th>{col}</th>" for col in df.columns])}
            </tr>
            {"".join([f"<tr>{''.join([f'<td>{cell}</td>' for cell in row])}</tr>" for row in df.values.tolist()])}
        </table>
        
        <h2>Visualization</h2>
        
        <div class="plot-container">
            <h3>MAE Comparison</h3>
            <img src="mae_comparison.png" alt="MAE Comparison">
        </div>
        
        <div class="plot-container">
            <h3>Relative Improvements</h3>
            <img src="relative_improvements.png" alt="Relative Improvements">
        </div>
        
        <h2>Component Impact Analysis</h2>
    """
    
    # Add component impact plots to the HTML
    for group_name in ['Encoder', 'iEFL Steps', 'Embedding Dimension', 'Attention Heads', 
                       'Kernel Size', 'Feature Reduction', 'Normalization', 'Dropout', 
                       'Auxiliary Loss', 'Backbone Training']:
        plot_file = f'impact_{group_name.lower().replace(" ", "_")}.png'
        if os.path.exists(os.path.join(output_dir, plot_file)):
            html_report += f"""
            <div class="plot-container">
                <h3>Impact of {group_name}</h3>
                <img src="{plot_file}" alt="Impact of {group_name}">
            </div>
            """
    
    # Add learning curves if available
    if os.path.exists(os.path.join(output_dir, 'learning_curves.png')):
        html_report += f"""
        <h2>Learning Curves</h2>
        <div class="plot-container">
            <img src="learning_curves.png" alt="Learning Curves">
        </div>
        """
    
    html_report += """
        <h2>Conclusion</h2>
        <p>This ablation study examined the contribution of various components to the model's performance.
        The results show which components are most critical for achieving good performance and
        which settings yield the best results.</p>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(os.path.join(output_dir, 'ablation_report.html'), 'w') as f:
        f.write(html_report)
    
    print(f"Ablation report generated at {output_dir}/ablation_report.html")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument("--summary", required=True, help="Path to the ablation summary JSON file")
    parser.add_argument("--output", default="./ablation_analysis", help="Output directory for the report")
    
    args = parser.parse_args()
    
    generate_ablation_report(args.summary, args.output)