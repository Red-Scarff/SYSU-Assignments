#!/usr/bin/env python3
"""
Visualization script for SSL experiment results
Compare our FixMatch/MixMatch implementations with TorchSSL results
"""

import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# Set font and style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def parse_torchssl_log(log_file):
    """Parse TorchSSL log files"""
    results = []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Use regex to extract evaluation results
    pattern = r'(\d+) iteration.*?eval/top-1-acc.*?(\d+\.\d+)'
    matches = re.findall(pattern, content)
    
    for iteration, accuracy in matches:
        results.append({
            'step': int(iteration),
            'accuracy': float(accuracy) * 100  # Convert to percentage
        })
    
    return results

def parse_our_log(log_file):
    """Parse our implementation log files"""
    results = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if 'Step' in line and 'Test Acc:' in line:
            # Parse format: "Step 1000: Test Acc: 14.01%, Test Loss: 4.6486"
            match = re.search(r'Step (\d+): Test Acc: ([\d.]+)%', line)
            if match:
                step = int(match.group(1))
                accuracy = float(match.group(2))
                results.append({
                    'step': step,
                    'accuracy': accuracy
                })
    
    return results

def load_all_results():
    """Load all experiment results"""
    results = defaultdict(dict)
    
    # Label configurations
    label_configs = [40, 250, 4000]
    
    for n_labels in label_configs:
        # Load TorchSSL FixMatch results
        torchssl_fixmatch_log = f"saved_models/fixmatch_cifar10_{n_labels}_0/log.txt"
        if os.path.exists(torchssl_fixmatch_log):
            results[f'TorchSSL_FixMatch_{n_labels}'] = parse_torchssl_log(torchssl_fixmatch_log)
        
        # Load TorchSSL MixMatch results
        torchssl_mixmatch_log = f"saved_models/mixmatch_cifar10_{n_labels}_0/log.txt"
        if os.path.exists(torchssl_mixmatch_log):
            results[f'TorchSSL_MixMatch_{n_labels}'] = parse_torchssl_log(torchssl_mixmatch_log)
        
        # Load our FixMatch results
        our_fixmatch_log = f"logs/fixmatch_{n_labels}labels_seed42.log"
        if os.path.exists(our_fixmatch_log):
            results[f'Our_FixMatch_{n_labels}'] = parse_our_log(our_fixmatch_log)
        
        # Load our MixMatch results
        our_mixmatch_log = f"logs/mixmatch_{n_labels}labels_seed42.log"
        if os.path.exists(our_mixmatch_log):
            results[f'Our_MixMatch_{n_labels}'] = parse_our_log(our_mixmatch_log)
    
    return results

def plot_training_curves(results):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SSL Algorithm Training Curves Comparison (FixMatch vs MixMatch)', fontsize=16, fontweight='bold')
    
    label_configs = [40, 250, 4000]
    algorithms = ['FixMatch', 'MixMatch']
    
    colors = {'TorchSSL': '#1f77b4', 'Our': '#ff7f0e'}
    
    for i, algorithm in enumerate(algorithms):
        for j, n_labels in enumerate(label_configs):
            ax = axes[i, j]
            
            # Plot TorchSSL results
            torchssl_key = f'TorchSSL_{algorithm}_{n_labels}'
            if torchssl_key in results and results[torchssl_key]:
                data = results[torchssl_key]
                steps = [d['step'] for d in data]
                accs = [d['accuracy'] for d in data]
                ax.plot(steps, accs, label=f'TorchSSL {algorithm}', 
                       color=colors['TorchSSL'], linewidth=2, marker='o', markersize=4)
            
            # Plot our results
            our_key = f'Our_{algorithm}_{n_labels}'
            if our_key in results and results[our_key]:
                data = results[our_key]
                steps = [d['step'] for d in data]
                accs = [d['accuracy'] for d in data]
                ax.plot(steps, accs, label=f'Our {algorithm}', 
                       color=colors['Our'], linewidth=2, marker='s', markersize=4)
            
            ax.set_title(f'{algorithm} - {n_labels} Labels', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Test Accuracy (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('visualization_results/training_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_final_accuracy_comparison(results):
    """Plot final accuracy comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    label_configs = [40, 250, 4000]
    
    # Collect final accuracy data
    fixmatch_data = {'TorchSSL': [], 'Our': []}
    mixmatch_data = {'TorchSSL': [], 'Our': []}
    
    for n_labels in label_configs:
        # FixMatch
        torchssl_key = f'TorchSSL_FixMatch_{n_labels}'
        our_key = f'Our_FixMatch_{n_labels}'
        
        if torchssl_key in results and results[torchssl_key]:
            final_acc = results[torchssl_key][-1]['accuracy']
            fixmatch_data['TorchSSL'].append(final_acc)
        else:
            fixmatch_data['TorchSSL'].append(0)
            
        if our_key in results and results[our_key]:
            final_acc = results[our_key][-1]['accuracy']
            fixmatch_data['Our'].append(final_acc)
        else:
            fixmatch_data['Our'].append(0)
        
        # MixMatch
        torchssl_key = f'TorchSSL_MixMatch_{n_labels}'
        our_key = f'Our_MixMatch_{n_labels}'
        
        if torchssl_key in results and results[torchssl_key]:
            final_acc = results[torchssl_key][-1]['accuracy']
            mixmatch_data['TorchSSL'].append(final_acc)
        else:
            mixmatch_data['TorchSSL'].append(0)
            
        if our_key in results and results[our_key]:
            final_acc = results[our_key][-1]['accuracy']
            mixmatch_data['Our'].append(final_acc)
        else:
            mixmatch_data['Our'].append(0)
    
    # Plot FixMatch comparison
    x = np.arange(len(label_configs))
    width = 0.35
    
    ax1.bar(x - width/2, fixmatch_data['TorchSSL'], width, label='TorchSSL', color='#1f77b4', alpha=0.8)
    ax1.bar(x + width/2, fixmatch_data['Our'], width, label='Our Implementation', color='#ff7f0e', alpha=0.8)
    
    ax1.set_title('FixMatch - Final Accuracy Comparison', fontweight='bold')
    ax1.set_xlabel('Number of Labeled Samples')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(label_configs)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (torch_acc, our_acc) in enumerate(zip(fixmatch_data['TorchSSL'], fixmatch_data['Our'])):
        ax1.text(i - width/2, torch_acc + 1, f'{torch_acc:.1f}%', ha='center', va='bottom')
        ax1.text(i + width/2, our_acc + 1, f'{our_acc:.1f}%', ha='center', va='bottom')
    
    # Plot MixMatch comparison
    ax2.bar(x - width/2, mixmatch_data['TorchSSL'], width, label='TorchSSL', color='#1f77b4', alpha=0.8)
    ax2.bar(x + width/2, mixmatch_data['Our'], width, label='Our Implementation', color='#ff7f0e', alpha=0.8)
    
    ax2.set_title('MixMatch - Final Accuracy Comparison', fontweight='bold')
    ax2.set_xlabel('Number of Labeled Samples')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(label_configs)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (torch_acc, our_acc) in enumerate(zip(mixmatch_data['TorchSSL'], mixmatch_data['Our'])):
        ax2.text(i - width/2, torch_acc + 1, f'{torch_acc:.1f}%', ha='center', va='bottom')
        ax2.text(i + width/2, our_acc + 1, f'{our_acc:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('visualization_results/final_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(results):
    """Create results summary table"""
    label_configs = [40, 250, 4000]
    
    # Create summary data
    summary_data = []
    
    for n_labels in label_configs:
        row = {'Labels': n_labels}
        
        # FixMatch results
        torchssl_key = f'TorchSSL_FixMatch_{n_labels}'
        our_key = f'Our_FixMatch_{n_labels}'
        
        if torchssl_key in results and results[torchssl_key]:
            row['TorchSSL_FixMatch'] = f"{results[torchssl_key][-1]['accuracy']:.2f}%"
        else:
            row['TorchSSL_FixMatch'] = "N/A"
            
        if our_key in results and results[our_key]:
            row['Our_FixMatch'] = f"{results[our_key][-1]['accuracy']:.2f}%"
        else:
            row['Our_FixMatch'] = "N/A"
        
        # MixMatch results
        torchssl_key = f'TorchSSL_MixMatch_{n_labels}'
        our_key = f'Our_MixMatch_{n_labels}'
        
        if torchssl_key in results and results[torchssl_key]:
            row['TorchSSL_MixMatch'] = f"{results[torchssl_key][-1]['accuracy']:.2f}%"
        else:
            row['TorchSSL_MixMatch'] = "N/A"
            
        if our_key in results and results[our_key]:
            row['Our_MixMatch'] = f"{results[our_key][-1]['accuracy']:.2f}%"
        else:
            row['Our_MixMatch'] = "N/A"
        
        summary_data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(summary_data)
    df.to_csv('visualization_results/results_summary.csv', index=False)
    
    print("📊 Experiment Results Summary:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"📄 Detailed results saved to: results_summary.csv")
    
    return df

def main():
    print("🎯 Starting SSL experiment results analysis...")
    
    # Load all results
    print("📂 Loading experiment data...")
    results = load_all_results()
    
    # Create output directory
    os.makedirs('visualization_results', exist_ok=True)
    
    if not results:
        print("❌ No experiment results found!")
        return
    
    print(f"✅ Successfully loaded {len(results)} experiment results")
    
    # Generate visualization charts
    print("📈 Generating training curves comparison...")
    plot_training_curves(results)
    
    print("📊 Generating final accuracy comparison...")
    plot_final_accuracy_comparison(results)
    
    print("📋 Generating results summary table...")
    create_summary_table(results)
    
    print("\n🎉 Visualization completed! Generated files:")
    print("  - training_curves_comparison.png: Training curves comparison")
    print("  - final_accuracy_comparison.png: Final accuracy comparison")
    print("  - results_summary.csv: Results summary table")
    
    print(f"\n📁 All files saved in: {os.getcwd()}/visualization_results")

if __name__ == "__main__":
    main()
